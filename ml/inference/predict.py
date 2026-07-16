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
            
            self.loaded = True
            return True
        except Exception as e:
            print(f"Error loading checkpoints: {e}")
            return False
            
    def predict_audio(self, audio_path):
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
        # 6. Standardize scale
        features_scaled = self.scaler.transform(cli_vec.reshape(1, -1))
        
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
            'shap_explanation': shap_explanations
        }
        
    def compute_shap_explanations(self, features_scaled):
        """
        Runs SHAP on the scaled fused features and maps it back to clinical biomarkers.
        """
        if self.background_data is None:
            return []
            
        try:
            # We wrap the predict_proba function to return only the probability of Parkinson's
            def predict_probability_parkinsons(x):
                return self.model.predict_proba(x)[:, 1]
                
            # Perform Kernel SHAP using a kmeans summary of 10 points for speed
            bg_summary = shap.kmeans(self.background_data, 10)
            explainer = shap.KernelExplainer(predict_probability_parkinsons, bg_summary)
            
            # Compute SHAP values for the current input, shape [1, 49]
            shap_vals = explainer.shap_values(features_scaled, silent=True)[0]
            
            # Map the first len(feature_names) SHAP values to the clinical feature keys
            # The remaining elements correspond to the reduced neural components
            num_cli = len(self.feature_names)
            cli_shap_values = shap_vals[:num_cli]
            
            # Human-readable labels mapping
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
                'Zero_Crossing_Rate': 'Zero-Crossing Rate'
            }
            
            shap_results = []
            for i, feat_name in enumerate(self.feature_names):
                shap_val = float(cli_shap_values[i])
                label = readable_labels.get(feat_name, feat_name)
                
                # Check contribution direction
                impact = "increase" if shap_val > 0 else "decrease"
                
                shap_results.append({
                    'feature_name': feat_name,
                    'label': label,
                    'shap_value': shap_val,
                    'impact': impact,
                    'abs_value': abs(shap_val)
                })
                
            # Sort by absolute contribution and return top 5
            shap_results = sorted(shap_results, key=lambda x: x['abs_value'], reverse=True)
            return shap_results[:5]
            
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            # Safe fallback: return empty list
            return []
