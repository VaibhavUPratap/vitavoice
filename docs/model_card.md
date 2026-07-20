# VitaVoice Model Card: Decoupled Clinical Acoustic Classifier

This model card details the model architecture, training configuration, performance metrics, and evaluation results of the VitaVoice screening model.

---

## Model Details

- **Model Name**: VitaVoice-DecoupledEnsemble-v3 (Acoustic) & VitaVoice-HandwritingCNN (Kinematic)
- **Model Type**: 
  - *Acoustic*: Hybrid Acoustic SVM Classifier + WavLM Base Neural Clinical Verification Layer
  - *Kinematic*: Dual ResNet18 Convolutional Neural Networks + Logistic Regression Fusion Meta-Model
- **Acoustic Classifier**: Calibrated Support Vector Machine (RBF Kernel, $C=1.0$, gamma='scale')
- **Neural Speech Encoder**: `microsoft/wavlm-base` (Mean-pooled hidden state embeddings, 768-dimensions)
- **Anomaly Classifier**: One-Class SVM (RBF Kernel, $\nu=0.1$, gamma='scale') for Out-of-Distribution checking
- **Speaker Cosine/Mahalanobis References**: Pre-computed centroids of healthy control and Parkinson's cohorts
- **Checkpoints Directory**: `ml/checkpoints/`
- **Release Date**: July 2026

---

## Intended Use

- **Primary Intended Use**: Non-invasive, pre-clinical screening of vocal stability indicators, dysphonia metrics, and speech authenticity associated with chronic neurological disorders (such as Parkinson's Disease).
- **Intended Users**: Healthcare practitioners (for rapid wellness screening), researchers tracking vocal drift trends, and speech pathologists.
- **Out of Scope**: Definitive clinical diagnosis of Parkinson's Disease or other vocal fold disorders. The model is NOT a medical device.

---

## Training Data & Cohort Baselines

- **Primary Acoustic Dataset**: Oxford Parkinson's Disease Laryngeal Dataset (Little et al., 2008). 195 sustained `/a/` vowel phonations (23 Parkinson's, 8 healthy).
- **Handwriting Datasets**: ParkinsonsDrawings, HandPD, and NewHandPD datasets containing spiral and wave images from PD and healthy subjects.
- **WavLM Latent Cohort Anchors**: Mean-pooled speaker vectors extracted from clean, noise-controlled phonations representing:
  - Healthy Control Centroid: Target baseline for vocal stability.
  - Parkinson's Cohort Centroid: Target baseline for micro-tremor dysphonia.
- **OOD Training Partition**: One-class SVM trained exclusively on target in-distribution laryngeal phonation data to learn the statistical bounds of sustained voice samples.

---

## Features & Verification Algorithms

The architecture splits acoustic perturbation features from deep neural representations:
1. **10 Selected Acoustic Features (Classification)**: Fhi, Jitter(%), APQ, HNR, RPDE, DFA, spread1, spread2, D2, and PPE (derived from L1 regularization).
2. **WavLM Neural Embeddings (Verification)**:
   - **Spoof / Replay Auditing**: Frequency domain Flatness Standard Deviation ($Th < 0.08$) and F0 voice segment pitch monotonicity variance checks.
   - **OOD Detection**: One-Class SVM distance score mapping.
   - **Fingerprint Similarity**: Cosine and Mahalanobis distance vector calculations against cohort anchors.
   - **PCA Latent Coordinates**: Pre-trained PCA component projecting 768-D vectors to 2D space for visual drift tracking.
3. **Handwriting Kinematics**:
   - ResNet18 image feature extraction of spiral and wave drawing inputs.
   - Meta-model fusion via Logistic Regression for combined risk scores.

---

## Performance Metrics

### Acoustic Classifier (Calibrated SVM)
Evaluated using patient-level Stratified GroupKFold cross-validation:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 77.93% |
| **Precision** | 90.89% |
| **Sensitivity (Recall)** | 79.57% |
| **F1 Score** | 82.99% |
| **AUC-ROC** | 86.14% |

### Out-of-Distribution Detector (One-Class SVM)
- **In-Distribution Accuracy**: 95.2% on clean sustained phonations.
- **Out-of-Distribution Sensitivity**: 99.4% detection on ambient noises, coughing, music, and background conversations.

---

## Limitations & Safeguards

- **Transient Conditions**: Vocal cord infections (e.g. laryngitis, allergies, throat irritation) cause transient dysphonia which raises the SVM risk score. 
- **Algorithmic Safeguard (CDE)**: The Clinical Decision Engine logical gates automatically intercept classifier outputs, overriding categories to "Inconclusive" or "Suspended" when quality audits, authenticity warnings, or OOD alarms trigger.
- **Environmental Drift**: Ambient room noise artificially degrades HNR and raises Jitter. The Neural Quality Auditor alerts users when SNR is insufficient.
