def generate_clinical_report(metrics, risk_score, confidence_calibration=None, shap_explanation=None):
    """
    Generates a structured, plain-language health insight report based on
    acoustic metrics and risk scores.
    """
    # Healthy thresholds (standard clinical ranges for sustained "ah")
    # Reference: Max Little et al. (2008)
    jitter_threshold = 1.04     # % (local jitter)
    shimmer_threshold = 3.80    # % (local shimmer)
    hnr_threshold = 20.0        # dB (Harmonics-to-Noise Ratio)
    
    jitter_pct = metrics['jitter_pct']
    shimmer_pct = metrics['shimmer_local'] * 100.0  # Convert to percent
    hnr = metrics['hnr']
    f0 = metrics['fo_mean']
    
    # Assess status of each biomarker
    jitter_status = "Elevated" if jitter_pct > jitter_threshold else "Normal"
    shimmer_status = "Elevated" if shimmer_pct > shimmer_threshold else "Normal"
    hnr_status = "Low (Breathiness)" if hnr < hnr_threshold else "Normal (Stable)"
    
    # Fundamental frequency analysis
    f0_range_status = "Within range"
    if f0 > 0:
        if f0 < 80:
            f0_range_status = "Low (Deep voice/dysphonia potential)"
        elif f0 > 280:
            f0_range_status = "High (Tension potential)"
            
    # Risk category
    if risk_score < 0.35:
        risk_category = "Low Risk"
        risk_description = (
            "Your voice profile shows high stability and clarity. The vocal cord vibration "
            "frequencies, amplitude changes, and signal-to-noise ratios are within normal "
            "clinical ranges. No significant voice biomarkers for Parkinson's disease were detected."
        )
        recommendations = [
            "Maintain voice hygiene by staying hydrated.",
            "Repeat the voice screening periodically if you experience vocal fatigue.",
            "This report is for screening purposes; always consult a doctor if you feel unwell."
        ]
    elif risk_score < 0.65:
        risk_category = "Borderline / Moderate Risk"
        risk_description = (
            "Your voice profile shows mild irregularities. Some markers such as pitch jitter "
            "or amplitude shimmer are slightly outside standard thresholds, which can be caused by "
            "temporary factors (e.g., vocal fatigue, hoarseness, cold, dehydration) or early voice biomarkers. "
            "We recommend repeating the test under quiet conditions."
        )
        recommendations = [
            "Rest your vocal cords and drink plenty of water.",
            "Ensure a completely quiet environment and repeat the test in 24 hours.",
            "Monitor for any other physical symptoms like muscle stiffness or resting tremors."
        ]
    else:
        risk_category = "Elevated Risk"
        risk_description = (
            "Your voice profile demonstrates a pattern of dysphonia (vocal instability) "
            "frequently associated with neuromotor vocal impairments. Significant variations in cycle-to-cycle "
            "frequency (Jitter) and loudness (Shimmer), combined with reduced voice harmonics (HNR), "
            "indicate vocal tremor and breathiness which are common voice biomarkers of Parkinson's disease."
        )
        recommendations = [
            "Schedule a clinical evaluation with a primary care physician or neurologist.",
            "Consult an Otolaryngologist (ENT) or Speech-Language Pathologist for a professional voice assessment.",
            "Consider tracking your vocal stability weekly to establish a clinical trend."
        ]

    # Summarize the top SHAP features in the plain-language clinical description
    if shap_explanation and len(shap_explanation) > 0:
        top_feats = []
        for feat in shap_explanation[:3]:  # Top 3 features
            direction = "increased" if feat['shap_value'] > 0 else "decreased"
            top_feats.append(f"{feat['label']} ({direction} risk)")
        shap_summary = " Explainable AI (SHAP) analysis indicates that the top features contributing to this assessment are " + ", ".join(top_feats) + "."
        risk_description += shap_summary

    # Compile structured markdown sections
    report = {
        'risk_category': risk_category,
        'summary': risk_description,
        'biomarker_analysis': [
            {
                'name': 'vocal_fundamental_frequency',
                'label': 'Average Pitch (F0)',
                'value': f"{f0:.2f} Hz",
                'status': f0_range_status,
                'explanation': "Represents the speed of vocal fold vibration. Healthy male ranges are typically 85-155Hz, and female ranges are 165-255Hz."
            },
            {
                'name': 'pitch_jitter',
                'label': 'Pitch Jitter (Local)',
                'value': f"{jitter_pct:.3f}%",
                'status': jitter_status,
                'explanation': f"Measures frequency instability from cycle to cycle. Normal threshold is < {jitter_threshold}%. Elevated values indicate micro-tremors in vocal cord control."
            },
            {
                'name': 'amplitude_shimmer',
                'label': 'Amplitude Shimmer',
                'value': f"{shimmer_pct:.3f}%",
                'status': shimmer_status,
                'explanation': f"Measures loudness instability from cycle to cycle. Normal threshold is < {shimmer_threshold}%. High shimmer indicates difficulty maintaining constant vocal pressure."
            },
            {
                'name': 'hnr',
                'label': 'Harmonics-to-Noise (HNR)',
                'value': f"{hnr:.2f} dB",
                'status': hnr_status,
                'explanation': f"Measures the ratio of pure vocal harmonic tone to background/breathiness noise. Normal is > {hnr_threshold} dB. Lower values indicate hoarseness or breathy speech."
            }
        ],
        'recommendations': recommendations,
        'disclaimer': (
            "VitaVoice is an AI-powered voice biomarker screening tool designed for educational, "
            "research, and wellness purposes only. It is not an FDA-cleared diagnostic device "
            "and does not provide a definitive medical diagnosis for Parkinson's disease or "
            "any other neurological condition. Vocal abnormalities can stem from transient physical "
            "states like common colds, environmental allergies, or acid reflux. Please seek "
            "professional medical advice for any health-related concerns."
        ),
        'confidence_calibration': confidence_calibration,
        'shap_explanation': shap_explanation
    }
    
    return report
