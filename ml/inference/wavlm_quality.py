import numpy as np

class WavLMQualityAuditor:
    def __init__(self, references_path="ml/checkpoints/wavlm_references.joblib"):
        self.references_path = references_path

    def audit_quality(self, wavlm_embedding: np.ndarray, train_embeddings: np.ndarray, dsp_quality: dict) -> dict:
        """
        Audits recording consistency, room reverb, microphone response, 
        and acoustic fidelity by fusing DSP statistics with neural density.
        """
        reasons = []
        score = 5.0
        
        # 1. Retrieve DSP parameters
        snr_db = dsp_quality.get('snr_db', 0.0)
        bg_noise = dsp_quality.get('background_noise_pct', 0.0)
        clipping = dsp_quality.get('clipping_detected', False)
        duration = dsp_quality.get('duration_seconds', 0.0)
        speech_coverage = dsp_quality.get('speech_coverage_pct', 0.0)
        
        # 2. Compute WavLM density check (similarity to top reference points)
        mean_top_similarity = 1.0
        if train_embeddings is not None and len(train_embeddings) > 0:
            norm_emb = wavlm_embedding / (np.linalg.norm(wavlm_embedding) + 1e-10)
            norm_train = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-10)
            similarities = np.dot(norm_train, norm_emb)
            # Take average of the 5 closest training points
            mean_top_similarity = float(np.mean(np.sort(similarities)[-5:]))
            
        # 3. Apply Quality Deductions
        too_noisy = snr_db < 12.0 or bg_noise > 35.0
        clipped = clipping
        insufficient_speech = speech_coverage < 30.0 or duration < 5.0
        
        # If WavLM similarity is low, indicates abnormal microphone or massive room reverberation
        abnormal_mic = mean_top_similarity < 0.78 and not (too_noisy or clipped or insufficient_speech)
        ood_acoustic = mean_top_similarity < 0.72
        
        if too_noisy:
            score -= 1.5
            reasons.append("high background noise")
        if clipped:
            score -= 2.0
            reasons.append("signal clipping")
        if insufficient_speech:
            score -= 1.5
            reasons.append("insufficient speech duration/coverage")
        if ood_acoustic:
            score -= 1.5
            reasons.append("out-of-distribution acoustic profile")
        if abnormal_mic:
            score -= 1.0
            reasons.append("abnormal microphone response or reverb")
            
        # Bound score
        score = max(1.0, min(5.0, score))
        score_stars = "★" * int(round(score)) + "☆" * (5 - int(round(score)))
        
        # Determine reliability tier
        if score >= 4.0:
            reliability = "High"
        elif score >= 2.5:
            reliability = "Moderate"
        else:
            reliability = "Low"
            
        re_record_recommended = score < 3.0
        if re_record_recommended:
            rec_msg = "Re-recording is recommended in a quiet environment using a standard calibrated microphone."
            if reasons:
                rec_msg += f" Reasons identified: {', '.join(reasons)}."
        else:
            rec_msg = "Recording quality is verified and suitable for clinical screening."
            
        return {
            "overall_score": round(score, 1),
            "quality_stars": score_stars,
            "recording_reliability": reliability,
            "re_record_recommended": re_record_recommended,
            "re_record_recommendation": rec_msg,
            "reasons": reasons,
            "wavlm_similarity_to_train": round(mean_top_similarity, 4)
        }
