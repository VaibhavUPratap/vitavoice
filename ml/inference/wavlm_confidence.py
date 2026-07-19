import numpy as np

class WavLMConfidenceAuditor:
    def __init__(self):
        pass

    def evaluate_trust(
        self,
        svm_risk: float,
        quality_score: float,        # 1.0 to 5.0
        ood_probability: float,      # 0.0 to 100.0
        similarity_matching_score: float, # 0.0 to 100.0 (PD similarity)
        authenticity_score: float,   # 0.0 to 100.0
    ) -> dict:
        """
        Calculates a trust score and level by auditing all pipeline checks.
        Does NOT alter the risk prediction score itself.
        """
        reasons = []
        
        # 1. Calculate base prediction margin-based confidence
        # Maximum difference from threshold (0.5)
        certainty = 2.0 * abs(svm_risk - 0.5)  # 0.0 to 1.0
        base_conf = 50.0 + (certainty * 45.0)  # Maps 0.0-1.0 to 50.0-95.0%
        
        trust_score = base_conf
        
        # 2. Check for alignment between SVM prediction and WavLM nearest cohort
        svm_pd_pred = svm_risk >= 0.5
        wavlm_pd_pred = similarity_matching_score >= 50.0
        
        if svm_pd_pred != wavlm_pd_pred:
            trust_score -= 15.0
            reasons.append("divergence between clinical acoustic features and neural speech patterns")

        # 3. Apply Quality Auditor Penalties
        if quality_score < 4.0:
            penalty = (4.0 - quality_score) * 12.0
            trust_score -= penalty
            reasons.append(f"reduced recording quality (Audit Score: {quality_score:.1f} Stars)")
            
        # 4. Apply Authenticity Penalties
        if authenticity_score < 85.0:
            penalty = (85.0 - authenticity_score) * 0.6
            trust_score -= penalty
            reasons.append("audio authenticity warning flags raised")

        # 5. Apply OOD Detection Penalties
        if ood_probability > 25.0:
            penalty = (ood_probability - 25.0) * 0.5
            trust_score -= penalty
            reasons.append("speech recording differs from reference cohort distributions (OOD warning)")

        # 6. Safety overrides (severe quality or spoofing failures drop trust to floor)
        if quality_score < 2.5:
            trust_score = min(30.0, trust_score)
            reasons.append("CRITICAL: poor recording quality makes analysis unreliable")
        if authenticity_score < 65.0:
            trust_score = min(25.0, trust_score)
            reasons.append("CRITICAL: high probability of synthetic audio or spoofing attack")
        if ood_probability > 75.0:
            trust_score = min(35.0, trust_score)
            reasons.append("CRITICAL: input falls completely outside model's training space")
            
        # Clamp bounds
        trust_score = max(10.0, min(99.0, trust_score))
        
        # Determine Trust Level
        if trust_score >= 75.0:
            trust_level = "High"
        elif trust_score >= 45.0:
            trust_level = "Medium"
        else:
            trust_level = "Low"
            
        return {
            "trust_score": round(trust_score, 1),
            "trust_level": trust_level,
            "reasons": reasons
        }
