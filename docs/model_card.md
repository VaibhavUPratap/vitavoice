# VitaVoice Model Card: Decoupled Clinical Acoustic Classifier

This model card details the model architecture, training configuration, performance metrics, and evaluation results of the VitaVoice screening model.

---

## Model Details

- **Model Name**: VitaVoice-CalibratedSVM-v3
- **Model Type**: Decoupled Clinical Acoustic Classifier (with WavLM Base Neural 2D Visualizer Mapping)
- **Primary Algorithm**: Calibrated Support Vector Machine (RBF Kernel)
- **Pretrained Neural Encoder**: `microsoft/wavlm-base` (Mean-pooled hidden state embeddings, 768-dimensions; used exclusively for 2D visual cohort cluster plotting)
- **Checkpoints Directory**: `ml/checkpoints/`
- **Release Date**: July 2026

---

## Intended Use

- **Primary Intended Use**: Non-invasive, pre-clinical screening of vocal dysphonia and stability indicators associated with chronic neuromotor disorders (such as Parkinson's Disease).
- **Intended Users**: Healthcare practitioners (for wellness screening), research groups, and patients tracking vocal health over time.
- **Out of Scope**: Definitive medical diagnosis of Parkinson's Disease or other neurological conditions. The model is NOT a replacement for standard clinical diagnostic tools (such as laryngoscopy, unified Parkinson's disease rating scale assessments, or neurologist evaluations).

---

## Training Data

- **Primary Dataset**: Oxford Parkinson's Disease Laryngeal/Vocal Dataset (Little et al., 2008).
- **Cohort Composition**: 195 voice recordings from 31 subjects (23 diagnosed with Parkinson's Disease, 8 healthy controls).
- **Phonation Protocol**: Sustained vowel phonation of the letter "ah" (vowel `/a/`) recorded at a constant pitch and volume.
- **Data Partitions**: 5-Fold Stratified Cross-Validation on the full cohort.

---

## Features & Preprocessing

The model decouples the classification feature space from the visualization embedding space:
1. **10 Selected Clinical Acoustic Classification Features**: Optimized using L1-regularized Logistic Regression to eliminate collinear redundancy (features include Fhi, Jitter(%), APQ, HNR, RPDE, DFA, spread1, spread2, D2, and PPE).
2. **WavLM Base Neural Coordinates (Visualization Only)**: A 768-dimensional WavLM Base vector compressed to 2 coordinates via PCA for visual cohort mapping on the UI.

---

## Performance Metrics

Evaluation using strict 5-Fold Stratified GroupKFold Cross-Validation (patient-level splits to prevent leakage):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 77.93% |
| **Precision** | 90.89% |
| **Sensitivity (Recall)** | 79.57% |
| **F1 Score** | 82.99% |
| **AUC-ROC** | 86.14% |

---

## Limitations & Biases

- **Ambient Noise**: The model's feature extraction is highly sensitive to room acoustics. Environment calibration is required; noise RMS > 0.05 will degrade performance.
- **Transient Conditions**: Acute physical states affecting vocal cords (e.g. throat infections, colds, dehydration, allergies, fatigue, laryngitis, acid reflux) can mimic neuromotor tremor indicators and cause false positive elevations.
- **Cohort Diversity**: The training set is dominated by English-speaking adult cohorts. Variations in accent, age, gender, and regional vocal patterns have not been fully cross-validated.
