# VitaVoice Dataset Documentation

This document describes the datasets supported by VitaVoice, their physical properties, laryngeal metrics, and the guidelines for loading and processing vocal samples.

---

## 1. Primary Dataset: Oxford Parkinson's Dataset

VitaVoice is primarily trained and benchmarked on the **Oxford Parkinson's Disease Vocal Dataset** (collected by Max Little in collaboration with the National Centre for Voice and Speech, Denver, Colorado).

### Dataset Background
- **Subjects**: 31 male and female adults (23 diagnosed with Parkinson's, 8 healthy control).
- **Vocalizations**: 195 total recordings. Each subject recorded several sustained phonations of the vowel `/a/` ("ah") under controlled acoustic conditions.
- **Recording Equipment**: Standard acoustic microphone, digitized at 44.1 kHz.

### Acoustic Biomarker Specifications

The dataset provides 22 continuous acoustic features extracted from the voice recordings:

| Column Header | Clinical Metric Description | Metric Unit | Healthy Baseline Range |
| :--- | :--- | :--- | :--- |
| `MDVP:Fo(Hz)` | Average vocal fundamental frequency (pitch) | Hertz (Hz) | Male: 85-155Hz, Female: 165-255Hz |
| `MDVP:Fhi(Hz)` | Maximum vocal fundamental frequency | Hertz (Hz) | N/A |
| `MDVP:Flo(Hz)` | Minimum vocal fundamental frequency | Hertz (Hz) | N/A |
| `MDVP:Jitter(%)` | Cycle-to-cycle frequency variations (local jitter) | Percentage (%) | < 1.04% |
| `MDVP:Jitter(Abs)` | Absolute cycle-to-cycle frequency variations | Seconds (s) | < 0.00008 s |
| `MDVP:RAP` | Relative average perturbation | Percentage (%) | < 0.50% |
| `MDVP:PPQ` | Five-point period perturbation quotient | Percentage (%) | < 0.50% |
| `Jitter:DDP` | Average absolute difference of differences | Percentage (%) | < 1.50% |
| `MDVP:Shimmer` | Cycle-to-cycle amplitude variations (local shimmer) | Amplitude Ratio | < 0.038 (3.8%) |
| `MDVP:Shimmer(dB)` | Absolute amplitude variation in decibels | Decibels (dB) | < 0.38 dB |
| `Shimmer:APQ3` | Three-point amplitude perturbation quotient | Percentage (%) | < 2.0% |
| `Shimmer:APQ5` | Five-point amplitude perturbation quotient | Percentage (%) | < 2.5% |
| `MDVP:APQ` | 11-point amplitude perturbation quotient | Percentage (%) | < 3.0% |
| `Shimmer:DDA` | Average absolute difference of amplitudes | Percentage (%) | < 6.0% |
| `NHR` | Noise-to-Harmonics Ratio | Ratio | < 0.19 |
| `HNR` | Harmonics-to-Noise Ratio | Decibels (dB) | > 20 dB |
| `Energy` | RMS vocal energy (loudness measure) | Amplitude | Variable |
| `F1` | First formant frequency (vocal tract resonance) | Hertz (Hz) | Variable (approx. 250-1000Hz) |
| `F2` | Second formant frequency (vocal tract resonance) | Hertz (Hz) | Variable (approx. 800-2500Hz) |
| `F3` | Third formant frequency (vocal tract resonance) | Hertz (Hz) | Variable (approx. 1500-3500Hz) |
| `Spectral_Centroid` | Mean frequency of the spectrum | Hertz (Hz) | Variable |
| `Spectral_Bandwidth` | Width of the spectrum | Hertz (Hz) | Variable |
| `Zero_Crossing_Rate` | Rate of sign changes in the signal | Hz / frame | Variable |
| `MFCC_1 to MFCC_13` | Mel-frequency cepstral coefficients (vocal tract shape) | Power | Variable |
| `Chroma_1 to Chroma_12` | 12-semitone pitch profile of the vocalization | Energy | Variable |

---

## 2. Secondary Datasets: Handwriting Kinematics

VitaVoice integrates three handwriting datasets to train its ResNet18 Kinematics models:
1. **ParkinsonsDrawings**: Spiral and wave drawings from Parkinson's and healthy patients.
2. **HandPD**: Additional images of spiral and wave tests.
3. **NewHandPD**: Further spiral and wave samples.

All images are normalized and augmented (rotation, translation) before being passed through the ImageNet-pre-trained vision models.

---

## 3. Model Target Vector
- **Target Variable**: `status`
  - `0`: Healthy Control (normal vocal stability)
  - `1`: Parkinson's Patient (indicates presence of laryngeal tremors or vocal tremor)

---

## 4. Separation of Prototype Demo Data

To support testing and local development without requiring clinical dataset downloads, VitaVoice separates data layers:
1. **Clinical Data**: Raw patient recordings are kept in `datasets/raw/`. The loader parses files using a standard structure.
2. **Synthetic / Demo Data**: Temporary synthetic data generated to verify the pipeline's operational components without violating patient confidentiality or requiring large files. Synthetic data generators are stored under `ml/training/synthesize_dataset.py`.
3. **Reference Projections**: Real UMAP/PCA cluster coordinates are exported as `datasets/pca_visualization_clusters.csv` to allow fast 2D scatter plots in the frontend without running PCA pipelines in real-time.
