import os
import joblib
import numpy as np

class WavLMOODDetector:
    def __init__(self, checkpoints_dir="ml/checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        self.ood_detector = None
        self.ood_threshold = 0.0
        self.loaded = False
        self.load_model()

    def load_model(self):
        refs_path = os.path.join(self.checkpoints_dir, "wavlm_references.joblib")
        if os.path.exists(refs_path):
            try:
                refs = joblib.load(refs_path)
                self.ood_detector = refs.get('ood_detector')
                self.ood_threshold = refs.get('ood_threshold', -1.0)
                if self.ood_detector is not None:
                    self.loaded = True
            except Exception as e:
                print(f"Error loading OOD detector: {e}")
                self.loaded = False
        else:
            self.loaded = False

    def detect_ood(self, wavlm_embedding: np.ndarray) -> dict:
        """
        Determines if the speech sample is statistically outside the reference training distribution.
        """
        if not self.loaded or self.ood_detector is None:
            return {
                "is_ood": False,
                "ood_probability": 0.0,
                "raw_ood_score": 0.0,
                "reason": "OOD detector not trained/loaded. Defaulting to in-distribution."
            }
            
        try:
            # Reshape embedding for the model
            emb_reshaped = wavlm_embedding.reshape(1, -1)
            raw_score = float(self.ood_detector.score_samples(emb_reshaped)[0])
            is_ood = raw_score < self.ood_threshold
            
            # Map raw score to a 0-100% OOD Probability curve
            if raw_score >= self.ood_threshold:
                # In-distribution: map distance to 0-50%
                diff = max(1e-5, raw_score - self.ood_threshold)
                # Squeezed sigmoid-like mapping
                ood_prob = max(0.0, min(50.0, 50.0 - (diff / 2.0) * 50.0))
            else:
                # Out-of-distribution: map distance to 50-100%
                diff = self.ood_threshold - raw_score
                ood_prob = min(100.0, 50.0 + (diff / 0.5) * 50.0)
                
            # Formulate diagnostic explanation
            if is_ood:
                if ood_prob > 85.0:
                    reason = "Critical acoustic mismatch. Extremely high deviation from training cohort (potential non-human speech, extreme room noise, or signal corruption)."
                else:
                    reason = "Moderate acoustic drift detected. Voice characteristics vary from the reference group (potential microphone difference, language dialect, or speaker age/gender outlier)."
            else:
                reason = "Acoustic parameters align with reference training cohort distributions."
                
            return {
                "is_ood": is_ood,
                "ood_probability": round(ood_prob, 1),
                "raw_ood_score": round(raw_score, 4),
                "reason": reason
            }
            
        except Exception as e:
            return {
                "is_ood": False,
                "ood_probability": 0.0,
                "raw_ood_score": 0.0,
                "reason": f"Error running OOD detector: {str(e)}"
            }
