import os
import numpy as np
import joblib
import shap

# Import local ML modules
from ml.preprocessing.audio import AudioPreprocessingPipeline
from ml.feature_extraction.acoustic import extract_all_acoustic_features
from ml.feature_extraction.embeddings import extract_wav2vec2_embeddings, project_embedding_2d

class VitaVoicePredictor:
    def __init__(self, checkpoints_dir="ml/checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        self.scaler = None
        self.reducer = None
        self.pca_vis = None
        self.model = None
        self.feature_names = None
        self.model_type = None
        self.background_data = None
        self.wavlm_intel = None
        self.transform_config = None  # Log transform config saved during training
        self.loaded = False
        self.pipeline = AudioPreprocessingPipeline()
        
    def load_models(self):
        """
        Loads the trained model checkpoints, scalers, reducers, and SHAP background data.
        """
        try:
            scaler_path = os.path.join(self.checkpoints_dir, "scaler.joblib")
            reducer_path = os.path.join(self.checkpoints_dir, "reducer.joblib")
            pca_vis_path = os.path.join(self.checkpoints_dir, "pca_model.joblib")
            model_path = os.path.join(self.checkpoints_dir, "classifier_model.joblib")
            feats_path = os.path.join(self.checkpoints_dir, "feature_names.joblib")
            type_path = os.path.join(self.checkpoints_dir, "model_type.joblib")
            bg_path = os.path.join(self.checkpoints_dir, "background_data.joblib")
            
            # Check if critical files exist
            if not all(os.path.exists(p) for p in [scaler_path, reducer_path, pca_vis_path, model_path, feats_path]):
                print("Warning: Model checkpoints not found. Model must be trained first.")
                return False
                
            self.scaler = joblib.load(scaler_path)
            self.reducer = joblib.load(reducer_path)
            self.pca_vis = joblib.load(pca_vis_path)
            self.model = joblib.load(model_path)
            self.feature_names = joblib.load(feats_path)

            # Load optional benchmarking configs
            self.model_type = joblib.load(type_path) if os.path.exists(type_path) else "svm"
            self.background_data = joblib.load(bg_path) if os.path.exists(bg_path) else None

            # Load log transform config (saved during training for train/inference consistency)
            transform_path = os.path.join(self.checkpoints_dir, "transform_config.joblib")
            if os.path.exists(transform_path):
                self.transform_config = joblib.load(transform_path)
            else:
                self.transform_config = None
            
            # Load WavLM references
            try:
                from ml.inference.wavlm_intelligence import WavLMIntelligenceLayer
                refs_path = os.path.join(self.checkpoints_dir, "wavlm_references.joblib")
                self.wavlm_intel = WavLMIntelligenceLayer(references_path=refs_path)
            except Exception as e:
                print(f"Error loading WavLMIntelligenceLayer: {e}")
                
            self.loaded = True
            return True
        except Exception as e:
            print(f"Error loading checkpoints: {e}")
            return False
            
    def predict_audio(self, audio_path, recording_quality=None):
        """
        Runs the full inference, explainability (SHAP), and certainty calibration pipeline on a WAV file.
        """
        if not self.loaded:
            success = self.load_models()
            if not success:
                raise RuntimeError("Failed to load prediction models. Ensure the model is trained.")
                
        # 1. Preprocess audio
        y, sr = self.pipeline.preprocess_audio(audio_path)
        
        # 2. Extract clinical features
        cli_feats = extract_all_acoustic_features(y, sr)
        
        # Align clinical features according to trained feature list
        cli_vec = np.array([cli_feats[k] for k in self.feature_names])
        
        # 3. Extract WavLM Base embeddings
        w2v_emb = extract_wav2vec2_embeddings(y, sr)
        
        # 4. Dimensionality reduction of embeddings (using PCA or UMAP)
        if self.reducer is not None:
            w2v_reduced = self.reducer.transform(w2v_emb.reshape(1, -1))[0]
        else:
            w2v_reduced = w2v_emb
            
        # 5. Skip concatenation for classification (use robust clinical features only)
        # 6. Apply log transforms (must match training pipeline)
        cli_vec_transformed = cli_vec.copy().astype(np.float64)
        if self.transform_config is not None:
            for idx in self.transform_config.get('log_transform_indices', []):
                if idx < len(cli_vec_transformed):
                    cli_vec_transformed[idx] = np.log1p(max(cli_vec_transformed[idx], 0))

        # 7. Standardize scale
        features_scaled = self.scaler.transform(cli_vec_transformed.reshape(1, -1))
        
        # 7. Predict risk score
        probabilities = self.model.predict_proba(features_scaled)[0]
        risk_score = float(probabilities[1])
        status = int(self.model.predict(features_scaled)[0])
        
        # 8. Confidence Calibration
        certainty_score = 2 * abs(risk_score - 0.5)
        if certainty_score >= 0.70:
            certainty_label = "High Certainty"
            calibration_confidence = "Risk profile is highly defined"
        elif certainty_score >= 0.35:
            certainty_label = "Moderate Certainty"
            calibration_confidence = "Risk profile is moderately defined"
        else:
            certainty_label = "Low Certainty"
            calibration_confidence = "Borderline classification / high variance"
            
        # 9. Explainability: SHAP values calculation
        shap_explanations = self.compute_shap_explanations(features_scaled)
        
        # 10. Project to 2D for visualization
        pca_x, pca_y = project_embedding_2d(w2v_emb, self.pca_vis)
        
        # 11. Format response clinical parameters for report
        report_metrics = {
            'fo_mean': float(cli_feats['MDVP:Fo(Hz)']),
            'fhi': float(cli_feats['MDVP:Fhi(Hz)']),
            'flo': float(cli_feats['MDVP:Flo(Hz)']),
            'jitter_pct': float(cli_feats['MDVP:Jitter(%)']),
            'jitter_abs': float(cli_feats['MDVP:Jitter(Abs)']),
            'shimmer_local': float(cli_feats['MDVP:Shimmer']),
            'shimmer_db': float(cli_feats['MDVP:Shimmer(dB)']),
            'hnr': float(cli_feats['HNR']),
            'nhr': float(cli_feats['NHR']),
            'energy': float(cli_feats['Energy']),
            'formants': [float(cli_feats['F1']), float(cli_feats['F2']), float(cli_feats['F3'])]
        }
        
        # Setup default/on-the-fly recording quality if not provided
        if recording_quality is None:
            try:
                from app.recording_quality import analyze_recording_quality
                recording_quality = analyze_recording_quality(audio_path)
            except Exception:
                recording_quality = {
                    "duration_seconds": float(len(y) / sr),
                    "background_noise_pct": 10.0,
                    "snr_db": 25.0,
                    "speech_coverage_pct": 80.0,
                    "silence_ratio_pct": 20.0,
                    "clipping_detected": False,
                    "mic_status": "Good",
                    "quality_score": 4,
                    "quality_stars": "★★★★☆",
                    "suitable_for_analysis": True,
                    "quality_warning": None
                }
                
        # 12. Run WavLM Intelligence Layer
        if self.wavlm_intel is None:
            from ml.inference.wavlm_intelligence import WavLMIntelligenceLayer
            refs_path = os.path.join(self.checkpoints_dir, "wavlm_references.joblib")
            self.wavlm_intel = WavLMIntelligenceLayer(references_path=refs_path)
            
        quality_info = self.wavlm_intel.verify_quality(w2v_emb, recording_quality)
        similarity_info = self.wavlm_intel.compute_similarity(w2v_emb, self.reducer)
        ood_info = self.wavlm_intel.detect_ood(w2v_emb)
        decision_info = self.wavlm_intel.run_decision_engine(
            clinical_risk=risk_score,
            clinical_confidence=certainty_score,
            quality_info=quality_info,
            ood_info=ood_info,
            similarity_info=similarity_info
        )
        
        return {
            'risk_score': risk_score,
            'status': status,
            'embedding_coords': [pca_x, pca_y],
            'clinical_metrics': report_metrics,
            'confidence_calibration': {
                'risk_probability': risk_score,
                'certainty_score': certainty_score,
                'certainty_label': certainty_label,
                'calibration_confidence': calibration_confidence
            },
            'shap_explanation': shap_explanations,
            'wavlm_quality': quality_info,
            'wavlm_similarity': similarity_info,
            'wavlm_ood': ood_info,
            'decision_engine': decision_info
        }
        
    def compute_shap_explanations(self, features_scaled):
        """
        Computes SHAP values using TreeExplainer (exact, fast) for tree-based models,
        falling back to KernelExplainer for non-tree models.
        """
        try:
            # Resolve to base estimator for TreeSHAP
            # CalibratedClassifierCV wraps the raw model — unwrap it
            base_model = self.model
            if hasattr(base_model, 'calibrated_classifiers_'):
                # Use the first calibrator's base estimator for TreeSHAP
                base_model = base_model.calibrated_classifiers_[0].estimator
            elif hasattr(base_model, 'estimator'):
                base_model = base_model.estimator

            # Also try loading the raw (uncalibrated) model saved alongside the calibrated one
            raw_model_path = os.path.join(self.checkpoints_dir, "classifier_model_raw.joblib")
            if os.path.exists(raw_model_path):
                base_model = joblib.load(raw_model_path)

            model_type = self.model_type or ''
            is_tree_model = any(t in model_type.lower() for t in [
                'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'balanced_rf'
            ])

            if is_tree_model:
                # TreeSHAP — exact, millisecond-speed for tree models
                explainer = shap.TreeExplainer(base_model)
                raw_shap = explainer.shap_values(features_scaled, check_additivity=False)

                # For multi-output (RF returns list of [class0_shap, class1_shap])
                if isinstance(raw_shap, list) and len(raw_shap) >= 2:
                    shap_vals = raw_shap[1][0]  # class=1 (PD), first sample
                elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
                    shap_vals = raw_shap[0, :, 1]  # first sample, PD class
                elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
                    shap_vals = raw_shap[0]  # first sample
                else:
                    shap_vals = raw_shap[0] if hasattr(raw_shap, '__len__') else raw_shap

            else:
                # Fallback KernelExplainer for non-tree models (e.g. calibrated SVM)
                if self.background_data is None:
                    return []
                def predict_proba_pd(x):
                    try:
                        return self.model.predict_proba(x)[:, 1]
                    except Exception:
                        return np.zeros(len(x))
                bg_summary = shap.kmeans(self.background_data, 10)
                explainer = shap.KernelExplainer(predict_proba_pd, bg_summary)
                shap_vals = explainer.shap_values(features_scaled, silent=True)[0]

            # Human-readable labels (updated to include nonlinear Oxford features)
            readable_labels = {
                'MDVP:Fo(Hz)': 'Average Pitch (F0)',
                'MDVP:Fhi(Hz)': 'Maximum Pitch (Fhi)',
                'MDVP:Flo(Hz)': 'Minimum Pitch (Flo)',
                'MDVP:Jitter(%)': 'Pitch Jitter (%)',
                'MDVP:Jitter(Abs)': 'Pitch Jitter (Abs)',
                'MDVP:RAP': 'Pitch Jitter (RAP)',
                'MDVP:PPQ': 'Pitch Jitter (PPQ)',
                'Jitter:DDP': 'Pitch Jitter (DDP)',
                'MDVP:Shimmer': 'Amplitude Shimmer (%)',
                'MDVP:Shimmer(dB)': 'Amplitude Shimmer (dB)',
                'Shimmer:APQ3': 'Amplitude Shimmer (APQ3)',
                'Shimmer:APQ5': 'Amplitude Shimmer (APQ5)',
                'MDVP:APQ': 'Amplitude Shimmer (APQ)',
                'Shimmer:DDA': 'Amplitude Shimmer (DDA)',
                'NHR': 'Noise-to-Harmonics (NHR)',
                'HNR': 'Harmonics-to-Noise (HNR)',
                'Energy': 'RMS Vocal Energy',
                'F1': 'Formant F1 Frequency',
                'F2': 'Formant F2 Frequency',
                'F3': 'Formant F3 Frequency',
                'Spectral_Centroid': 'Spectral Centroid',
                'Spectral_Bandwidth': 'Spectral Bandwidth',
                'Zero_Crossing_Rate': 'Zero-Crossing Rate',
                # Nonlinear dynamical complexity features (highest PD discriminators)
                'RPDE': 'Recurrence Period Density Entropy',
                'DFA': 'Detrended Fluctuation Analysis (DFA)',
                'spread1': 'Nonlinear Pitch Variation (spread1)',
                'spread2': 'Nonlinear Pitch Variation (spread2)',
                'D2': 'Correlation Dimension (D2)',
                'PPE': 'Pitch Period Entropy (PPE)',
            }

            shap_results = []
            shap_arr = np.array(shap_vals).flatten()
            for i, feat_name in enumerate(self.feature_names):
                if i >= len(shap_arr):
                    break
                shap_val = float(shap_arr[i])
                label = readable_labels.get(feat_name, feat_name)
                impact = "increase" if shap_val > 0 else "decrease"
                shap_results.append({
                    'feature_name': feat_name,
                    'label': label,
                    'shap_value': shap_val,
                    'impact': impact,
                    'abs_value': abs(shap_val)
                })

            shap_results = sorted(shap_results, key=lambda x: x['abs_value'], reverse=True)
            return shap_results[:5]

        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            return []
