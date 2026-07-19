import numpy as np
from datetime import datetime

class ClinicalDecisionEngine:
    def __init__(self):
        pass

    def run_rules(
        self,
        svm_risk: float,
        svm_trust: dict,
        quality_info: dict,
        ood_info: dict,
        similarity_info: dict,
        authenticity_info: dict,
        history: list = None,
        baseline_embedding: np.ndarray = None,
        current_embedding: np.ndarray = None
    ) -> dict:
        """
        Fuses all clinical diagnostic parameters and checks to produce a unified, safe assessment.
        Allows inconclusive status and quality warnings instead of forcing binary outcomes.
        """
        risk_score = svm_risk
        trust_score = svm_trust['trust_score']
        trust_level = svm_trust['trust_level']
        
        # Default category mapping
        if risk_score >= 0.65:
            status = 1
            status_label = "Elevated Risk"
        elif risk_score <= 0.35:
            status = 0
            status_label = "Low Risk"
        else:
            status = 2
            status_label = "Borderline / Moderate Risk"

        re_record_recommended = quality_info.get('re_record_recommended', False) or (quality_info.get('overall_score', 5.0) < 2.5)
        is_spoofed = not authenticity_info.get('is_authentic', True) or (authenticity_info.get('authenticity_score', 100.0) < 65.0)
        is_ood = ood_info.get('is_ood', False) or (ood_info.get('ood_probability', 0.0) > 75.0)
        
        # Calculate embedding drift if baseline is available
        drift_val = 0.0
        significant_shift = False
        drift_msg = "Stable voice baseline compared to historical measurements."
        
        if baseline_embedding is not None and current_embedding is not None:
            # Cosine distance to baseline
            norm_curr = current_embedding / (np.linalg.norm(current_embedding) + 1e-10)
            norm_base = baseline_embedding / (np.linalg.norm(baseline_embedding) + 1e-10)
            cosine_sim = np.dot(norm_curr, norm_base)
            drift_val = float(1.0 - cosine_sim)
            
            # Clinical drift threshold (0.08 indicates significant structural voice drift)
            if drift_val > 0.08:
                significant_shift = True
                drift_msg = "WARNING: Voice characteristics have changed significantly compared to previous baseline recordings."

        # Apply Clinical Decision Rules
        
        # Rule 1: Spoofed or Robotic Audio
        if is_spoofed:
            status = 4  # Authenticity Failure
            status_label = "Assessment Suspended (Authenticity Failure)"
            decision_reasoning = (
                "The clinical screening was aborted due to high probability of synthetic/AI-generated speech "
                "or digital replay spoofing artifacts. Authentic human voice is required."
            )
            recommendation = (
                "Please perform the recording live without filters, digital synthesizers, or speaker playback loopbacks."
            )
            trust_score = min(trust_score, 15.0)
            trust_level = "Low"

        # Rule 2: Unusable Quality Recording
        elif re_record_recommended:
            status = 5  # Poor Quality
            status_label = "Assessment Suspended (Poor Quality)"
            reasons = quality_info.get('reasons', [])
            reason_str = f" due to: {', '.join(reasons)}" if reasons else ""
            decision_reasoning = (
                f"The recording quality is insufficient for a reliable screening analysis{reason_str}. "
                "Handcrafted vocal perturbation biomarkers are highly sensitive to background noise and signal clipping."
            )
            recommendation = (
                "Please repeat the recording in a quiet room using a standard calibrated microphone. "
                "Ensure you hold the vowel 'ah' steadily for at least 5-10 seconds."
            )
            trust_score = min(trust_score, 20.0)
            trust_level = "Low"

        # Rule 3: Extreme Out-of-Distribution (OOD)
        elif is_ood:
            status = 2  # Borderline/OOD
            status_label = "Borderline / Atypical Profile"
            decision_reasoning = (
                "This voice recording falls outside our standard clinical training database distribution. "
                "This may stem from a rare microphone signature, heavy background acoustic echoes, language mismatch, "
                "or extreme age/gender features. The clinical risk prediction is less reliable."
            )
            recommendation = (
                "The vocal profile is atypical relative to our database. A clinical consultation is advised, "
                "and you should re-record in a quiet room to rule out environment/microphone bias."
            )
            trust_score = min(trust_score, 35.0)
            trust_level = "Low"

        # Rule 4: Borderline prediction + Ambiguous neural cohort similarity (Inconclusive)
        elif status == 2 and 40.0 <= similarity_info.get('similarity_parkinsons', 50.0) <= 60.0:
            status = 3  # Inconclusive
            status_label = "Inconclusive Assessment"
            decision_reasoning = (
                "Acoustic feature calculations reside in the borderline range, and the WavLM neural similarity "
                "engine reports ambiguous cohort matching (near 50%). A definitive risk classification cannot be made."
            )
            recommendation = (
                "The screening result is inconclusive. Please rest your vocal cords, ensure a quiet room, "
                "and repeat the screening in 24 hours."
            )
            trust_score = min(trust_score, 50.0)
            trust_level = "Low"

        # Rule 5: Strong Congruence (Elevated Risk)
        elif status == 1 and trust_level == "High" and similarity_info.get('similarity_parkinsons', 0.0) > 65.0:
            decision_reasoning = (
                "Strong agreement observed between the clinical acoustic features (Jitter/Shimmer) "
                f"and the WavLM neural similarity matching ({similarity_info['similarity_parkinsons']}% pathology similarity). "
                "The vocal profile exhibits patterns typical of Parkinsonian dysphonia."
            )
            recommendation = (
                "Consider consulting a neurologist or speech-language pathologist for a comprehensive clinical evaluation. "
                "This screening is a pre-clinical wellness aid and does not constitute a diagnosis."
            )
            if significant_shift:
                decision_reasoning += f" Note: {drift_msg}"

        # Rule 6: Strong Congruence (Low Risk)
        elif status == 0 and trust_level == "High" and similarity_info.get('similarity_healthy', 0.0) > 65.0:
            decision_reasoning = (
                "Strong agreement observed between the clinical acoustic features and the WavLM neural similarity matching "
                f"({similarity_info['similarity_healthy']}% healthy similarity). The vocal profile shows normal vocal fold control."
            )
            recommendation = (
                "Continue monitoring voice health. No immediate actions are required. "
                "Consider screening periodically (every 1-3 months) for wellness tracking."
            )
            if significant_shift:
                decision_reasoning += f" Note: {drift_msg}"

        # Rule 7: Divergence (e.g. SVM elevated, but WavLM healthy)
        elif status == 1 and similarity_info.get('similarity_parkinsons', 50.0) < 35.0:
            status = 2  # Downgrade to borderline due to conflict
            status_label = "Divergent / Borderline Risk"
            decision_reasoning = (
                "The primary classifier flags elevated risk based on clinical biomarkers (jitter/shimmer), "
                "but the WavLM embedding matches the healthy cohort closely. This divergence lowers the overall reliability."
            )
            recommendation = (
                "Acoustic parameters indicate mild vocal instability, but neural characteristics align with a healthy cohort. "
                "We recommend repeating the recording in 24 hours under quiet conditions."
            )
            trust_score = max(35.0, trust_score - 15.0)
            trust_level = "Medium"

        elif status == 0 and similarity_info.get('similarity_parkinsons', 50.0) > 65.0:
            status = 2  # Upgrade to borderline due to conflict
            status_label = "Divergent / Borderline Risk"
            decision_reasoning = (
                "The primary classifier reports low risk, but the WavLM embedding indicates similarities to "
                "the Parkinsonian cohort. This conflict reduces overall diagnostic confidence."
            )
            recommendation = (
                "Handcrafted features indicate a stable voice, but deep neural representation matches a pathology cohort. "
                "We recommend repeating the screening in 24-48 hours under quiet conditions."
            )
            trust_score = max(35.0, trust_score - 15.0)
            trust_level = "Medium"

        # Default standard cases
        else:
            decision_reasoning = (
                f"The voice assessment is based on a calibrated SVM classifier. The WavLM neural similarity engine "
                f"provides supporting evidence in agreement ({similarity_info['nearest_cluster']})."
            )
            if status == 1:
                recommendation = "Consider consulting a medical professional for a clinical speech evaluation."
            elif status == 0:
                recommendation = "Continue monitoring voice health. Retest periodically."
            else:
                recommendation = "Consider repeating the recording in a quieter room to eliminate ambient noise bias."
            if significant_shift:
                decision_reasoning += f" Note: {drift_msg}"

        return {
            "status": status,
            "status_label": status_label,
            "risk_score": risk_score,
            "trust_score": round(trust_score, 1),
            "trust_level": trust_level,
            "decision_reasoning": decision_reasoning,
            "recommendation": recommendation,
            "longitudinal_drift": {
                "baseline_drift": round(drift_val, 4),
                "significant_shift_detected": significant_shift,
                "drift_status": drift_msg
            }
        }
