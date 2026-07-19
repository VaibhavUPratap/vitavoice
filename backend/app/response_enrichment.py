"""
Response Enrichment for VitaVoice.

Post-processing module that takes the existing predict_audio() result dict
and enriches it with confidence labels, natural-language explanations,
biomarker statuses, recommendations, and responsible AI text.

Does NOT modify any ML pipeline code. Operates purely on the result dict.
"""


# ─── Clinical reference ranges ──────────────────────────────────────────────

BIOMARKER_REFERENCES = {
    "fo_mean": {
        "label": "Average Pitch (F0)",
        "unit": "Hz",
        "normal_range": "85–255",
        "threshold_low": 85.0,
        "threshold_high": 255.0,
        "status_fn": lambda v: "Normal" if 85 <= v <= 255 else ("Low" if v < 85 else "High"),
    },
    "jitter_pct": {
        "label": "Pitch Jitter (Local)",
        "unit": "%",
        "normal_range": "< 1.04",
        "threshold": 1.04,
        "status_fn": lambda v: "Normal" if v <= 1.04 else "Elevated",
    },
    "shimmer_local": {
        "label": "Amplitude Shimmer",
        "unit": "%",
        "normal_range": "< 3.80",
        "threshold": 0.038,   # stored as fraction; multiply by 100 for display
        "is_fraction": True,
        "status_fn": lambda v: "Normal" if (v * 100) <= 3.80 else "Elevated",
    },
    "hnr": {
        "label": "Harmonics-to-Noise (HNR)",
        "unit": "dB",
        "normal_range": "> 20.0",
        "threshold": 20.0,
        "status_fn": lambda v: "Normal" if v >= 20.0 else "Reduced",
    },
    "nhr": {
        "label": "Noise-to-Harmonics (NHR)",
        "unit": "",
        "normal_range": "< 0.19",
        "threshold": 0.19,
        "status_fn": lambda v: "Normal" if v <= 0.19 else "Elevated",
    },
    "energy": {
        "label": "RMS Vocal Energy",
        "unit": "",
        "normal_range": "Variable",
        "status_fn": lambda v: "Normal",
    },
}


RESPONSIBLE_AI_TEXT = (
    "This application is intended for preliminary screening only. "
    "It is not a medical diagnosis. Voice recordings are processed only for "
    "analysis purposes. Long-term storage of recordings is disabled. "
    "Consult a qualified neurologist for clinical diagnosis."
)

RESPONSIBLE_AI_POINTS = [
    "This application is intended for preliminary screening only.",
    "It is not a medical diagnosis.",
    "Voice recordings are processed only for analysis.",
    "Long-term storage of recordings is disabled.",
    "Consult a qualified neurologist for clinical diagnosis.",
]


# ─── Public API ──────────────────────────────────────────────────────────────

def enrich_response(result: dict) -> dict:
    """
    Takes the raw predict_audio() result and returns enrichment fields.
    
    Parameters
    ----------
    result : dict
        The dict returned by VitaVoicePredictor.predict_audio().
    
    Returns
    -------
    dict
        Enrichment fields to merge into the API response.
    """
    risk_score = result.get("risk_score", 0.5)
    confidence_cal = result.get("confidence_calibration", {})
    shap_explanation = result.get("shap_explanation", [])
    clinical_metrics = result.get("clinical_metrics", {})
    
    decision_engine = result.get("decision_engine", {})

    # Confidence & reliability (Overridden by Decision Engine if available)
    confidence_score = decision_engine.get("confidence_score", _compute_confidence_score(confidence_cal))
    confidence_label = decision_engine.get("confidence_label", _compute_confidence_label(confidence_score))
    prediction_reliability = _compute_prediction_reliability(confidence_score)

    # Directional biomarkers from SHAP
    top_biomarkers = _extract_top_biomarkers(shap_explanation)

    # Natural language explanation (Overridden by Decision Engine)
    explanation = decision_engine.get("decision_reasoning", _generate_explanation(shap_explanation, risk_score))

    # Clinical recommendation (Overridden by Decision Engine)
    recommendation = decision_engine.get("recommendation", _generate_recommendation(risk_score))

    # Biomarker statuses
    biomarker_statuses = _compute_biomarker_statuses(clinical_metrics)

    return {
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "prediction_reliability": prediction_reliability,
        "top_biomarkers": top_biomarkers,
        "natural_language_explanation": explanation,
        "recommendation": recommendation,
        "responsible_ai": RESPONSIBLE_AI_TEXT,
        "responsible_ai_points": RESPONSIBLE_AI_POINTS,
        "biomarker_statuses": biomarker_statuses,
    }


# ─── Confidence ──────────────────────────────────────────────────────────────

def _compute_confidence_score(confidence_cal: dict) -> float:
    """
    Derive a 0–100 confidence score from the existing certainty_score.
    The certainty_score is already computed in predict.py line 97 as:
        certainty_score = 2 * abs(risk_score - 0.5)
    We map it to a percentage.
    """
    certainty = confidence_cal.get("certainty_score", 0.5)
    # Scale to 0-100 with a minimum floor so we never show 0%
    score = max(50.0, min(99.0, certainty * 100.0))
    return round(score, 1)


def _compute_confidence_label(score: float) -> str:
    """Map confidence score to a human-readable label."""
    if score >= 90:
        return "Very High"
    elif score >= 70:
        return "High"
    elif score >= 55:
        return "Moderate"
    else:
        return "Low"


def _compute_prediction_reliability(score: float) -> str:
    """Map confidence score to a reliability tier."""
    if score >= 75:
        return "High"
    elif score >= 55:
        return "Moderate"
    else:
        return "Low"


# ─── Biomarker Extraction ───────────────────────────────────────────────────

def _extract_top_biomarkers(shap_explanation: list) -> list:
    """
    Transform SHAP explanation into directional biomarker indicators.
    Example: [{"label": "Pitch Jitter (%)", "direction": "↑", "descriptor": "Increased Jitter"}]
    """
    if not shap_explanation:
        return []

    results = []
    for feat in shap_explanation[:5]:
        label = feat.get("label", feat.get("feature_name", "Unknown"))
        shap_value = feat.get("shap_value", 0)

        # Determine direction
        if shap_value > 0:
            direction = "↑"
            # Generate descriptor based on label
            descriptor = _positive_descriptor(label)
        else:
            direction = "↓"
            descriptor = _negative_descriptor(label)

        results.append({
            "label": label,
            "direction": direction,
            "descriptor": descriptor,
            "shap_value": round(shap_value, 4),
        })

    return results


def _positive_descriptor(label: str) -> str:
    """Generate a human-readable descriptor for a feature that increases risk."""
    label_lower = label.lower()
    if "jitter" in label_lower:
        return "Increased Jitter"
    elif "shimmer" in label_lower:
        return "Elevated Shimmer"
    elif "hnr" in label_lower or "harmonics-to-noise" in label_lower:
        return "Reduced Harmonic-to-Noise Ratio"
    elif "nhr" in label_lower or "noise-to-harmonics" in label_lower:
        return "Elevated Noise-to-Harmonics Ratio"
    elif "pitch" in label_lower or "f0" in label_lower or "fo" in label_lower:
        return "Altered Pitch Characteristics"
    elif "energy" in label_lower:
        return "Altered Vocal Energy"
    elif "spectral" in label_lower:
        return "Altered Spectral Profile"
    elif "formant" in label_lower:
        return "Shifted Formant Frequency"
    elif "zero" in label_lower:
        return "Elevated Zero-Crossing Rate"
    else:
        return f"Elevated {label}"


def _negative_descriptor(label: str) -> str:
    """Generate a human-readable descriptor for a feature that decreases risk."""
    label_lower = label.lower()
    if "jitter" in label_lower:
        return "Reduced Jitter"
    elif "shimmer" in label_lower:
        return "Reduced Shimmer"
    elif "hnr" in label_lower or "harmonics-to-noise" in label_lower:
        return "Stable Harmonic-to-Noise Ratio"
    elif "nhr" in label_lower or "noise-to-harmonics" in label_lower:
        return "Reduced Noise-to-Harmonics Ratio"
    elif "pitch" in label_lower or "f0" in label_lower or "fo" in label_lower:
        return "Stable Pitch Characteristics"
    elif "energy" in label_lower:
        return "Stable Vocal Energy"
    elif "spectral" in label_lower:
        return "Normal Spectral Profile"
    elif "formant" in label_lower:
        return "Normal Formant Frequency"
    elif "zero" in label_lower:
        return "Normal Zero-Crossing Rate"
    else:
        return f"Reduced {label}"


# ─── Natural Language Explanation ────────────────────────────────────────────

def _generate_explanation(shap_explanation: list, risk_score: float) -> str:
    """Generate a plain-English explanation from SHAP features."""
    if not shap_explanation or len(shap_explanation) == 0:
        if risk_score < 0.35:
            return (
                "The screening analysis did not identify significant vocal biomarkers "
                "associated with neurological speech patterns. Your voice profile appears "
                "within normal parameters."
            )
        return (
            "The screening analysis was performed but detailed biomarker attribution "
            "is currently unavailable."
        )

    # Build descriptors for top 3 features
    top_features = shap_explanation[:3]
    descriptors = []
    for feat in top_features:
        label = feat.get("label", "")
        shap_value = feat.get("shap_value", 0)
        label_lower = label.lower()

        if shap_value > 0:
            if "jitter" in label_lower:
                descriptors.append("increased vocal jitter (pitch instability)")
            elif "shimmer" in label_lower:
                descriptors.append("elevated shimmer (amplitude variation)")
            elif "hnr" in label_lower or "harmonics" in label_lower:
                descriptors.append("altered harmonic-to-noise ratio")
            elif "nhr" in label_lower:
                descriptors.append("elevated noise-to-harmonics ratio")
            elif "pitch" in label_lower or "f0" in label_lower or "fo" in label_lower:
                descriptors.append("altered pitch characteristics")
            elif "energy" in label_lower:
                descriptors.append("altered vocal energy patterns")
            elif "spectral" in label_lower:
                descriptors.append("altered spectral characteristics")
            elif "formant" in label_lower:
                descriptors.append("shifted formant frequencies")
            else:
                descriptors.append(f"altered {label.lower()}")
        else:
            if "hnr" in label_lower or "harmonics" in label_lower:
                descriptors.append("reduced harmonicity")
            elif "jitter" in label_lower:
                descriptors.append("reduced pitch jitter")
            elif "shimmer" in label_lower:
                descriptors.append("reduced amplitude shimmer")
            else:
                descriptors.append(f"reduced {label.lower()}")

    if not descriptors:
        return "Detailed biomarker attribution is currently unavailable."

    # Join with proper English grammar
    if len(descriptors) == 1:
        feature_text = descriptors[0]
    elif len(descriptors) == 2:
        feature_text = f"{descriptors[0]} and {descriptors[1]}"
    else:
        feature_text = f"{descriptors[0]}, {descriptors[1]}, and {descriptors[2]}"

    if risk_score >= 0.65:
        context = (
            "which are commonly associated with neuromotor speech characteristics "
            "found in Parkinsonian dysphonia"
        )
    elif risk_score >= 0.35:
        context = (
            "which may indicate minor speech irregularities or transient vocal factors "
            "such as fatigue or environmental conditions"
        )
    else:
        context = (
            "however, the overall vocal profile remains within normal clinical parameters"
        )

    return (
        f"The screening result was primarily influenced by {feature_text}, "
        f"{context}."
    )


# ─── Recommendation ─────────────────────────────────────────────────────────

def _generate_recommendation(risk_score: float) -> str:
    """Generate risk-stratified screening recommendation."""
    if risk_score < 0.35:
        return (
            "Continue monitoring voice health. No immediate action is necessary. "
            "Consider periodic screening every 1–3 months for wellness tracking."
        )
    elif risk_score < 0.65:
        return (
            "Consider repeating the recording in a quieter environment to rule out "
            "environmental factors. If vocal irregularities persist across multiple "
            "screenings, consult a neurologist for further evaluation."
        )
    else:
        return (
            "Consider consulting a neurologist for a comprehensive clinical evaluation. "
            "This screening result suggests vocal patterns that may warrant professional "
            "assessment. This is not a diagnosis."
        )


# ─── Biomarker Statuses ─────────────────────────────────────────────────────

def _compute_biomarker_statuses(clinical_metrics: dict) -> list:
    """
    Compute status labels for each biomarker.
    Returns a list of dicts with label, value, unit, reference_range, and status.
    """
    statuses = []

    for key, ref in BIOMARKER_REFERENCES.items():
        raw_value = clinical_metrics.get(key)
        if raw_value is None:
            continue

        value = float(raw_value)
        display_value = value

        # Shimmer is stored as a fraction, display as percentage
        if ref.get("is_fraction"):
            display_value = value * 100.0

        status = ref["status_fn"](value)  # type: ignore[operator]

        statuses.append({
            "key": key,
            "label": ref["label"],
            "value": round(display_value, 3),
            "unit": ref["unit"],
            "reference_range": ref["normal_range"],
            "status": status,
        })

    # Add MFCC profile summary (not a single metric, so we flag as Normal by default)
    statuses.append({
        "key": "mfcc_profile",
        "label": "MFCC Profile",
        "value": "—",
        "unit": "",
        "reference_range": "Variable",
        "status": "Normal",
    })

    return statuses
