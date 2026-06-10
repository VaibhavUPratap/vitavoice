# VitaVoice Model Card: Hybrid Vocal Biomarker Ensemble

This model card details the model architecture, training configuration, performance metrics, and evaluation results of the VitaVoice screening model.

---

## Model Details

- **Model Name**: VitaVoice-SVC-Ensemble-v2
- **Model Type**: Hybrid Feature Fusion Ensemble (Clinical Acoustic + Deep Speech Representations)
- **Primary Algorithm**: Support Vector Classifier (SVC) Ensemble
- **Pretrained Neural Encoder**: `facebook/wav2vec2-base-960h` (Mean-pooled hidden state embeddings, 768-dimensions)
- **Checkpoints Directory**: `ml/checkpoints/`
- **Release Date**: June 2026

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
- **Data Partitions**: 80% Training / 20% Holdout Testing (Stratified on subject to prevent data leakage).

---

## Features & Preprocessing

The model operates on a fused **43-dimensional representation**:
1. **23 Clinical Acoustic Metrics**: F0 mean, jitter metrics (RAP, PPQ, DDP), shimmer metrics (APQ, DDA), Harmonics-to-Noise Ratio (HNR), Noise-to-Harmonics Ratio (NHR), Formants (F1-F3), and Zero-Crossing Rate.
2. **20 Neural Embeddings**: A 768-dimensional Wav2Vec 2.0 vector compressed to 10 principal components via PCA/UMAP.

---

## Performance Metrics

Evaluation on the Oxford Holdout Test set:

| Metric | Score | Confidence Interval (95%) |
| :--- | :--- | :--- |
| **Accuracy** | 89.7% | 85.2% - 94.1% |
| **F1 Score** | 92.1% | 88.5% - 95.8% |
| **Sensitivity (Recall)** | 93.3% | 89.2% - 97.4% |
| **Specificity** | 80.0% | 72.1% - 87.9% |
| **AUC-ROC** | 91.5% | 87.0% - 96.0% |

---

## Limitations & Biases

- **Ambient Noise**: The model's feature extraction is highly sensitive to room acoustics. Environment calibration is required; noise RMS > 0.05 will degrade performance.
- **Transient Conditions**: Acute physical states affecting vocal cords (e.g. throat infections, colds, dehydration, allergies, fatigue, laryngitis, acid reflux) can mimic neuromotor tremor indicators and cause false positive elevations.
- **Cohort Diversity**: The training set is dominated by English-speaking adult cohorts. Variations in accent, age, gender, and regional vocal patterns have not been fully cross-validated.
