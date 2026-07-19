import os
import numpy as np
import joblib

class WavLMIntelligenceLayer:
    """
    WavLM-assisted intelligence layer that acts as a clinical reasoning assistant.
    Decoupled from primary Random Forest classification to preserve SHAP explainability.
    """
    def __init__(self, references_path="ml/checkpoints/wavlm_references.joblib"):
        self.loaded = False
        if os.path.exists(references_path):
            try:
                refs = joblib.load(references_path)
                self.healthy_centroid = refs['healthy_centroid']
                self.parkinsons_centroid = refs['parkinsons_centroid']
                self.train_embeddings = refs['train_embeddings']
                self.train_labels = refs['train_labels']
                self.ood_detector = refs['ood_detector']
                self.ood_threshold = refs['ood_threshold']
                self.mean_healthy_reduced = refs['mean_healthy_reduced']
                self.mean_parkinsons_reduced = refs['mean_parkinsons_reduced']
                self.cov_healthy_inv = refs['cov_healthy_inv']
                self.cov_parkinsons_inv = refs['cov_parkinsons_inv']
                self.loaded = True
            except Exception as e:
                print(f"Error loading WavLM references: {e}")
        else:
            print(f"Warning: WavLM references not found at {references_path}. Running with dummy references.")
            self._setup_dummy_references()

    def _setup_dummy_references(self):
        """Sets up fallback dummy references for testing/safety."""
        self.healthy_centroid = np.zeros(768)
        self.parkinsons_centroid = np.zeros(768)
        self.train_embeddings = np.zeros((10, 768))
        self.train_labels = np.zeros(10)
        
        # Simple dummy class for OOD
        class DummyOOD:
            def score_samples(self, X):
                return np.zeros(len(X))
        self.ood_detector = DummyOOD()
        self.ood_threshold = -1.0
        self.mean_healthy_reduced = np.zeros(16)
        self.mean_parkinsons_reduced = np.zeros(16)
        self.cov_healthy_inv = np.eye(16)
        self.cov_parkinsons_inv = np.eye(16)

    def verify_quality(self, wavlm_emb, dsp_quality):
        """
        1. Recording Quality Verification using WavLM and DSP metrics.
        Returns overall Quality Score, Recording Reliability, and Re-record recommendations.
        """
        # Cosine similarity to training population
        norm_emb = wavlm_emb / (np.linalg.norm(wavlm_emb) + 1e-10)
        norm_train = self.train_embeddings / (np.linalg.norm(self.train_embeddings, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(norm_train, norm_emb)
        mean_top5_similarity = float(np.mean(np.sort(similarities)[-5:]))
        
        # DSP metrics
        snr_db = dsp_quality.get('snr_db', 0.0)
        bg_noise = dsp_quality.get('background_noise_pct', 0.0)
        clipping = dsp_quality.get('clipping_detected', False)
        duration = dsp_quality.get('duration_seconds', 0.0)
        speech_coverage = dsp_quality.get('speech_coverage_pct', 0.0)
        
        too_noisy = snr_db < 12.0 or bg_noise > 35.0
        clipped = clipping
        insufficient_speech = speech_coverage < 30.0 or duration < 5.0
        
        # Out-of-distribution quality check using WavLM similarity bounds
        ood_quality = mean_top5_similarity < 0.72
        # If WavLM similarity is low but DSP parameters look fine, mic response is abnormal
        abnormal_mic = mean_top5_similarity < 0.78 and not (too_noisy or clipped or insufficient_speech)
        
        # Compute overall quality score (1 to 5 stars)
        score = 5.0
        reasons = []
        if too_noisy:
            score -= 1.5
            reasons.append("high background noise")
        if clipped:
            score -= 2.0
            reasons.append("signal clipping")
        if insufficient_speech:
            score -= 1.5
            reasons.append("insufficient speech duration/coverage")
        if ood_quality:
            score -= 1.5
            reasons.append("out-of-distribution acoustic profile")
        if abnormal_mic:
            score -= 1.0
            reasons.append("abnormal microphone response")
            
        score = max(1.0, min(5.0, score))
        score_stars = "★" * int(round(score)) + "☆" * (5 - int(round(score)))
        
        # Determine reliability tier
        if score >= 4.0:
            reliability = "High"
        elif score >= 2.5:
            reliability = "Moderate"
        else:
            reliability = "Low"
            
        re_record = score < 3.0
        if re_record:
            rec_msg = "Re-recording is recommended in a quiet environment using a standard calibrated microphone."
            if reasons:
                rec_msg += f" Reasons: {', '.join(reasons)}."
        else:
            rec_msg = "Recording quality is verified and suitable for diagnostic screening."
            
        return {
            "quality_score": round(score, 1),
            "quality_stars": score_stars,
            "recording_reliability": reliability,
            "re_record_recommended": re_record,
            "re_record_recommendation": rec_msg,
            "reasons": reasons,
            "wavlm_similarity_to_train": round(mean_top5_similarity, 4),
            "too_noisy": too_noisy,
            "clipped": clipped,
            "insufficient_speech": insufficient_speech,
            "out_of_distribution": ood_quality,
            "abnormal_mic": abnormal_mic
        }

    def compute_similarity(self, wavlm_emb, reducer=None):
        """
        2. Embedding Similarity Engine.
        Computes cosine similarities and Mahalanobis distances to reference groups.
        """
        norm_emb = wavlm_emb / (np.linalg.norm(wavlm_emb) + 1e-10)
        norm_h = self.healthy_centroid / (np.linalg.norm(self.healthy_centroid) + 1e-10)
        norm_p = self.parkinsons_centroid / (np.linalg.norm(self.parkinsons_centroid) + 1e-10)
        
        sim_healthy = float(np.dot(norm_emb, norm_h))
        sim_parkinsons = float(np.dot(norm_emb, norm_p))
        
        # Calibrated similarity score to Parkinson's group (softmax with temp=0.05)
        temp = 0.05
        exp_p = np.exp(sim_parkinsons / temp)
        exp_h = np.exp(sim_healthy / temp)
        similarity_score = float(exp_p / (exp_p + exp_h)) * 100.0
        
        nearest_cluster = "Parkinson's Cluster" if sim_parkinsons > sim_healthy else "Healthy Cluster"
        
        # Calculate cosine similarity difference
        sim_diff = abs(sim_parkinsons - sim_healthy)
        if sim_diff >= 0.04:
            embedding_confidence = "High"
        elif sim_diff >= 0.015:
            embedding_confidence = "Moderate"
        else:
            embedding_confidence = "Low"
            
        # Mahalanobis Distance in 16D space
        d_mahalanobis_healthy = 0.0
        d_mahalanobis_parkinsons = 0.0
        if reducer is not None:
            try:
                e_reduced = reducer.transform(wavlm_emb.reshape(1, -1))[0]
                diff_h = e_reduced - self.mean_healthy_reduced
                d_mahalanobis_healthy = float(np.sqrt(np.dot(np.dot(diff_h, self.cov_healthy_inv), diff_h.T)))
                
                diff_p = e_reduced - self.mean_parkinsons_reduced
                d_mahalanobis_parkinsons = float(np.sqrt(np.dot(np.dot(diff_p, self.cov_parkinsons_inv), diff_p.T)))
            except Exception as e:
                print(f"Error computing Mahalanobis distance: {e}")
                
        return {
            "similarity_score": round(similarity_score, 1),
            "sim_healthy": round(sim_healthy, 4),
            "sim_parkinsons": round(sim_parkinsons, 4),
            "nearest_cluster": nearest_cluster,
            "embedding_confidence": embedding_confidence,
            "mahalanobis_healthy": round(d_mahalanobis_healthy, 2),
            "mahalanobis_parkinsons": round(d_mahalanobis_parkinsons, 2)
        }

    def detect_ood(self, wavlm_emb):
        """
        3. Out-of-Distribution Detection.
        Determines if the sample is statistically far from the training population.
        """
        try:
            raw_score = float(self.ood_detector.score_samples(wavlm_emb.reshape(1, -1))[0])
            is_ood = raw_score < self.ood_threshold
            
            # Map raw score to ood_score from 0 to 100
            if raw_score >= self.ood_threshold:
                # In distribution: map to 0-50 range
                diff = max(1e-5, raw_score - self.ood_threshold)
                ood_score = max(0.0, min(50.0, 50.0 - (diff / 2.0) * 50.0))
            else:
                # Out of distribution: map to 50-100 range
                diff = self.ood_threshold - raw_score
                ood_score = min(100.0, 50.0 + (diff / 0.5) * 50.0)
        except Exception as e:
            print(f"Error executing OOD detector: {e}")
            raw_score = 0.0
            is_ood = False
            ood_score = 0.0
            
        return {
            "is_ood": is_ood,
            "ood_score": round(ood_score, 1),
            "raw_ood_score": round(raw_score, 4),
            "message": "This recording differs significantly from the reference population." if is_ood else "In-distribution"
        }

    def run_decision_engine(self, clinical_risk, clinical_confidence, quality_info, ood_info, similarity_info):
        """
        4. Clinical Decision Engine.
        Fuses clinical biomarker risk, quality verification, OOD flags, and embedding similarity
        to produce a robust clinical diagnostic decision.
        """
        final_risk = clinical_risk
        final_confidence_score = clinical_confidence * 100.0
        
        # Initial status based on clinical risk
        if final_risk >= 0.65:
            status = 1
            status_label = "Elevated Risk"
        elif final_risk <= 0.35:
            status = 0
            status_label = "Low Risk"
        else:
            status = 2
            status_label = "Borderline / Moderate Risk"
            
        re_record_recommended = quality_info['re_record_recommended']
        is_ood = ood_info['is_ood']
        ood_score = ood_info['ood_score']
        sim_score = similarity_info['similarity_score']
        
        # Rule 1: Poor Quality Recording
        if re_record_recommended or quality_info['quality_score'] <= 2.0:
            final_confidence_label = "Low Confidence (Poor Quality)"
            final_confidence_score = min(final_confidence_score, 30.0)
            decision_reasoning = (
                "The recording quality is insufficient for a reliable analysis. "
                "The presence of " + ", ".join(quality_info['reasons']) + " interferes with the acoustic biomarkers."
            )
            recommendation = (
                "Recommend re-recording in a quiet environment. Due to poor recording quality, "
                "a reliable assessment cannot be performed."
            )
            
        # Rule 2: Out of Distribution (OOD)
        elif is_ood or ood_score > 70:
            final_confidence_label = "Low Confidence (Out-of-Distribution)"
            final_confidence_score = min(final_confidence_score, 40.0)
            decision_reasoning = (
                "This recording differs significantly from the reference population. The acoustic characteristics "
                "do not match the training distribution (possibly due to background noise, microphone difference, "
                "age/gender differences, or language variance). Use caution when interpreting the clinical risk score."
            )
            recommendation = (
                "The vocal profile is atypical relative to our reference database. "
                "A clinical consultation is advised, and you should re-record in a quiet environment "
                "to rule out ambient noise or microphone interference."
            )
            
        # Rule 3: Borderline risk AND ambiguous similarity (Inconclusive)
        elif status == 2 and 45.0 <= sim_score <= 55.0:
            status = 3  # Inconclusive status
            final_confidence_label = "Low Confidence (Inconclusive)"
            final_confidence_score = min(final_confidence_score, 50.0)
            decision_reasoning = (
                "Vocal biomarkers are in the borderline range and neural embedding similarity is ambiguous, "
                "leading to an inconclusive assessment. A repeat screening is recommended."
            )
            recommendation = (
                "The screening result is inconclusive. Please rest your vocal cords, ensure a completely quiet "
                "environment, and repeat the recording in 24 hours."
            )
            
        # Rule 4: Congruence case (Strong Parkinson's agreement)
        elif status == 1 and clinical_confidence >= 0.70 and quality_info['quality_score'] >= 4.0 and sim_score > 65.0:
            final_confidence_label = "Very High Confidence"
            final_confidence_score = min(99.0, max(final_confidence_score, 95.0))
            decision_reasoning = (
                "There is strong congruence between the clinical acoustic biomarkers and the WavLM neural embedding similarity. "
                "The voice profile shows clear patterns associated with Parkinsonian dysphonia."
            )
            recommendation = (
                "Consider consulting a neurologist for a comprehensive clinical evaluation. "
                "This screening result suggests vocal patterns that warrant professional assessment. This is not a diagnosis."
            )
            
        # Rule 5: Congruence case (Strong Healthy agreement)
        elif status == 0 and clinical_confidence >= 0.70 and quality_info['quality_score'] >= 4.0 and sim_score < 35.0:
            final_confidence_label = "Very High Confidence"
            final_confidence_score = min(99.0, max(final_confidence_score, 95.0))
            decision_reasoning = (
                "There is strong congruence between the clinical acoustic biomarkers and the WavLM neural embedding similarity. "
                "The voice profile shows stable vocal fold control and normal harmonic ratios."
            )
            recommendation = (
                "Continue monitoring voice health. No immediate action is necessary. "
                "Consider periodic screening every 1–3 months for wellness tracking."
            )
            
        # Default Case: Divergence or standard confidence
        else:
            # Check for conflict/divergence
            if status == 1 and sim_score < 35.0:
                final_confidence_label = "Moderate Confidence (Divergent Evidence)"
                final_confidence_score = max(50.0, final_confidence_score - 15.0)
                decision_reasoning = (
                    "The primary classifier indicates elevated risk based on clinical biomarkers (e.g. jitter/shimmer), "
                    "but WavLM embedding similarity suggests a normal voice profile. This divergence reduces overall confidence."
                )
                recommendation = (
                    "Vocal biomarkers suggest some instability, but neural embeddings resemble a healthy profile. "
                    "We recommend repeating the screening in 24 hours under quiet conditions."
                )
            elif status == 0 and sim_score > 65.0:
                final_confidence_label = "Moderate Confidence (Divergent Evidence)"
                final_confidence_score = max(50.0, final_confidence_score - 15.0)
                decision_reasoning = (
                    "The primary classifier indicates low risk based on clinical biomarkers, but WavLM embedding "
                    "similarity suggests patterns resembling the Parkinsonian cluster. This divergence reduces overall confidence."
                )
                recommendation = (
                    "The primary clinical metrics are within normal ranges, but neural embedding similarity is atypical. "
                    "We recommend repeating the screening in 24 hours under quiet conditions."
                )
            else:
                # Agreeing standard cases
                final_confidence_label = "High" if final_confidence_score >= 70 else ("Moderate" if final_confidence_score >= 35 else "Low")
                decision_reasoning = (
                    "The assessment is primarily based on the Random Forest classifier trained on 48 clinical vocal biomarkers. "
                    f"The WavLM similarity engine provides supporting evidence ({similarity_info['nearest_cluster']}) in agreement."
                )
                if status == 1:
                    recommendation = (
                        "Consider consulting a neurologist for a comprehensive clinical evaluation. "
                        "This screening result suggests vocal patterns that may warrant professional assessment. This is not a diagnosis."
                    )
                elif status == 0:
                    recommendation = (
                        "Continue monitoring voice health. No immediate action is necessary. "
                        "Consider periodic screening every 1–3 months for wellness tracking."
                    )
                else:
                    recommendation = (
                        "Consider repeating the recording in a quieter environment to rule out environmental factors. "
                        "If vocal irregularities persist across multiple screenings, consult a neurologist for further evaluation."
                    )
                    
        return {
            "status": status,
            "status_label": status_label,
            "risk_score": final_risk,
            "confidence_score": round(final_confidence_score, 1),
            "confidence_label": final_confidence_label,
            "decision_reasoning": decision_reasoning,
            "recommendation": recommendation
        }
