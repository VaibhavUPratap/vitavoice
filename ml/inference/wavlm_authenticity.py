import os
import numpy as np
import librosa

class RecordingAuthenticityAuditor:
    def __init__(self, checkpoints_dir="ml/checkpoints"):
        self.checkpoints_dir = checkpoints_dir

    def verify_authenticity(self, audio_path: str, wavlm_embedding: np.ndarray) -> dict:
        """
        Audits the audio recording for synthetic generation, replay attacks, 
        clipping, extreme compression, or corruption.
        """
        warnings = []
        score = 100.0
        
        try:
            # Load audio for DSP checks
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            return {
                "authenticity_score": 0.0,
                "confidence": "Low",
                "warnings": [f"Audio file could not be read: {str(e)}"],
                "is_authentic": False
            }
            
        # 1. Zero signal / Corruption check
        if len(y) == 0 or np.all(np.abs(y) < 0.0001):
            return {
                "authenticity_score": 0.0,
                "confidence": "High",
                "warnings": ["Audio is silent or corrupted."],
                "is_authentic": False
            }
            
        # 2. Digital Clipping Saturation Check
        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(y) >= clipping_threshold)
        clipping_ratio = clipped_samples / len(y)
        if clipping_ratio > 0.001:  # More than 0.1% of samples clipped
            penalty = min(40.0, clipping_ratio * 100.0 * 10.0)
            score -= penalty
            warnings.append(f"Digital saturation clipping detected ({clipping_ratio*100:.2f}% clipped).")

        # 3. Compression & Bandwidth Check
        # Synthetic/compressed audio often lacks high frequencies
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_centroid = float(np.mean(spectral_centroids))
        
        # Heavy compression (e.g. low-rate MP3/OGG) will drop spectral bandwidth
        spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        mean_bandwidth = float(np.mean(spectral_bandwidths))
        
        if mean_bandwidth < 1500:  # Unnaturally narrow frequency bandwidth
            score -= 25.0
            warnings.append("Extremely compressed or low-fidelity audio channel.")

        # 4. Replay Attack Detection via Spectral Flatness & Spectral Roll-off
        # Replayed audio has high noise floor and specific resonance spikes
        flatness = librosa.feature.spectral_flatness(y=y)
        mean_flatness = float(np.mean(flatness))
        
        # Roll-off checks frequency bounds
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        mean_rolloff = float(np.mean(rolloff))
        
        # Replay signatures: low rolloff + very high room noise
        if mean_rolloff < 2500 and mean_flatness < 0.0005:
            score -= 30.0
            warnings.append("Acoustic anomalies typical of speakers/room replay attacks detected.")

        # 5. Synthetic / AI Generated Speech Check (Robotic Monotone)
        # Synthetic speech often exhibits zero pitch jitter/variation (perfect monotone)
        # or unnatural pitch transitions.
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
            )
            f0_voiced = f0[voiced_flag]
            if len(f0_voiced) > 10:
                f0_std = np.std(f0_voiced)
                # Unnaturally steady pitch is a major sign of synthetic/TTS systems
                if f0_std < 4.0:
                    score -= 45.0
                    warnings.append("Robotic voice pattern: Unnaturally stable pitch detected (possible TTS synth).")
                elif f0_std > 120.0:
                    score -= 20.0
                    warnings.append("Highly unstable vocal frequency: possible synthetic modulation artifact.")
        except Exception:
            # Fallback if pitch extraction fails
            pass
            
        # Ensure score bounds
        score = max(0.0, min(100.0, score))
        
        # Determine confidence of the auditor
        if score > 85.0 or score < 40.0:
            confidence = "High"
        else:
            confidence = "Moderate"
            
        is_authentic = score >= 65.0
        
        return {
            "authenticity_score": round(score, 1),
            "confidence": confidence,
            "warnings": warnings,
            "is_authentic": is_authentic
        }
