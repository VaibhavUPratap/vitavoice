# VitaVoice Ethical Considerations & Responsible AI Guide

This document outlines the ethical standards, privacy protections, and safety guidelines built into the **VitaVoice Vocal Biomarker Screening Engine**.

---

## 1. Safety and Clinical Boundaries

### The Pre-Clinical Screening Distinction
VitaVoice is engineered and marketed strictly as a **pre-clinical screening aid** rather than a medical diagnostic system.
- **Diagnostic Limitation**: A diagnostic system provides a definitive medical conclusion (e.g., "This patient has Parkinson's Disease").
- **Screening Aid**: VitaVoice identifies vocal fold vibrations, shimmer, jitter, and spectral stability irregularities that are statistically correlated with vocal tremors (dysphonia). These irregularities indicate a elevated risk pattern that warrants professional medical evaluation, but do not diagnose the underlying cause.

### Prevention of False Reassurance and Alarmism
To manage risk and prevent alarmism:
1. **Calibration Check**: The application requires room acoustics calibration. Noisy environments can skew features and trigger incorrect risk assessments.
2. **Explicit Consent & Disclaimers**: Before running the AI screening, the user must check a box agreeing that transient physical conditions (cold, fatigue, laryngitis, acid reflux) affect voice biomarkers.
3. **Structured Recommendations**: Risk scores are accompanied by clear recommendations. Elevated risk profiles recommend scheduling a clinical evaluation with an Otolaryngologist (ENT) or Neurologist rather than self-diagnosing.

---

## 2. Data Privacy & Patient Confidentiality

Vocal recordings contain biometric information. To ensure maximum patient privacy:
1. **Immediate Deletion**: Raw audio uploads are deleted immediately after the ML feature extraction is complete. The system does not write raw patient audio to persistent storage.
2. **Ephemeral Reports**: Generated PDF reports and charts are kept in a static directory and automatically pruned by an async background worker after 1 hour.
3. **No PII Tracking**: The backend processes anonymous audio streams. The API does not associate recordings with names, email addresses, or other personally identifiable information (PII).

---

## 3. Transparency & Interpretability (Explainable AI)

Black-box machine learning in medicine can hide biases and reduce clinician trust. VitaVoice enforces transparency by:
- **Interpretability via SHAP**: Utilizing SHAP (SHapley Additive exPlanations) to display the top 5 contributing acoustic biomarkers on the results screen. This allows clinicians to understand exactly which characteristics (e.g., amplitude shimmer or pitch jitter) drove the risk estimation.
- **Certainty Calibration**: Providing a calibrated certainty score (e.g., "High Certainty" vs "Low Certainty"). This signals to the user when a vocal pattern lies near the decision boundary (low certainty) or is highly defined.
- **Acoustic Explanations**: Presenting clear, clinical definitions for all vocal biomarkers (F0, Jitter, Shimmer, HNR) on the dashboard cards, making the results understandable to non-expert users.
