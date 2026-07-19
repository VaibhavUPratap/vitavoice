# 🎙️ VitaVoice — AI-Powered Vocal Biomarker Health Screener

VitaVoice is an advanced, non-invasive health screening application that analyzes voice recordings to identify vocal instabilities and dysphonia patterns associated with neurological conditions (e.g., Parkinson's disease).

By combining **clinical acoustic digital signal processing** with **modern transformer-based deep speech representation models (WavLM Base)**, VitaVoice provides rapid, explainable health screenings for researchers and clinical enthusiasts.

---

> [!WARNING]
> **Medical Disclaimer:** VitaVoice is intended for educational, research, and wellness screening purposes only. It is not an FDA-cleared diagnostic device and does not provide formal medical diagnoses. Vocal changes can stem from common allergies, vocal fatigue, colds, or acid reflux. Please consult a qualified healthcare professional (ENT or Neurologist) for medical concerns.

---

## ✨ Key Features

- **Interactive Vocal Signal & Feature Analyzer**: Live canvas-less SVG spectrogram visualizer on the home page responding directly to mouse gestures with real-time frequency/amplitude tooltips.
- **WavLM AI Intelligence Layer**: Bypasses simplistic concatenation by treating WavLM Base embeddings as an independent, multi-modal validation layer:
  - **Voice Fingerprint Similarity**: Calculates cosine and Mahalanobis distances against target clinical cohorts to run $k$-nearest neighbors (KNN) cluster lookups.
  - **Recording Authenticity & Spoof Auditor**: Analyzes spectral flatness anomalies and monotone pitch metrics to prevent robotic/synthetic speech injection and replay loops.
  - **Out-of-Distribution (OOD) Detector**: Employs a calibrated One-Class SVM boundary to flag statistical voice anomalies.
  - **Neural Quality Auditor**: Checks background noise, RMS limits, and speech coverage ratios against reference clusters.
  - **Confidence & Trust Auditor**: Combines classifier decision margins, quality scores, spoof risk, and historical baseline variance to output an overall Trust Level (High, Medium, Low).
- **Clinical Decision Engine (CDE)**: Integrates expert rules that prevent forced classifications by overriding risk categories with "Inconclusive" or suspending screening in case of failed quality/spoof metrics.
- **SQLite Patient Timeline & Longitudinal Trajectory**: Tracks patient screenings over time. Computes baseline voice drift vectors and renders a visual chronological path connecting coordinate drift on the 2D cluster map.
- **Explainable AI (SHAP)**: Provides a feature-by-feature SHAP importance analysis showing which vocal biomarkers contribute positively or negatively to the risk score.
- **PDF Report Generator**: Generates 3-page clinical-grade PDF reports containing patient timeline summaries, recording quality stats, cohort similarity scores, SHAP explanations, and responsible AI guardrails.

---

## ⚙️ High-Level Architecture

```text
                           ┌──────────────────────────┐
                           │   User Voice Recording   │
                           └─────────────┬────────────┘
                                         │
                                         ▼
                           ┌──────────────────────────┐
                           │   Audio Preprocessing    │
                           │(16kHz, Resampling, Noise)│
                           └─────────────┬────────────┘
                                         │
                                         ▼
                           ┌──────────────────────────┐
                           │ Recording Quality Check  │
                           │  (SNR, Noise, Clipping)  │
                           └─────────────┬────────────┘
                                         │
                                         ▼
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
          ┌───────────────────────────┐       ┌───────────────────────────┐
          │  Clinical Perturbation    │       │    WavLM Base Embedding   │
          │ (Jitter, Shimmer, HNR...) │       │  (microsoft/wavlm-base)   │
          └─────────────┬─────────────┘       └─────────────┬─────────────┘
                        │                                   │
                        ▼                                   ▼
          ┌───────────────────────────┐       ┌───────────────────────────┐
          │   10 Selected Features    │       │ WavLM Intelligence Layer  │
          └─────────────┬─────────────┘       │(Spoof, OOD, Quality, Sim) │
                        │                     └─────────────┬─────────────┘
                        ▼                                   │
          ┌───────────────────────────┐                     │
          │ Calibrated SVM Classifier │                     │
          └─────────────┬─────────────┘                     │
                        │                                   │
                        └─────────────────┬─────────────────┘
                                          │ (Classifier Margin & Audits)
                                          ▼
                           ┌──────────────────────────┐
                           │ Clinical Decision Engine │ (Logical overrides)
                           └─────────────┬────────────┘
                                         │
                                         ▼
                           ┌──────────────────────────┐
                           │  Patient DB Timeline &   │ (SQLite Timeline
                           │   Longitudinal Drift     │  & PCA trajectory)
                           └─────────────┬────────────┘
                                         │
                                         ▼
                           ┌──────────────────────────┐
                           │   Response Enrichment     │ (Confidence, SHAP,
                           │    & Clinical Reports    │  & 3-page PDF File)
                           └──────────────────────────┘
```

---

## 🚀 Quick Start (Running Locally)

To run the application, you need to start the FastAPI backend and the Vite frontend separately.

### 1. Start the FastAPI Backend
Initialize your virtual environment, install the dependencies, and run the development server:
```bash
# Navigate to the backend directory
cd backend

# Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the development server
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
The backend API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

**Note:** If port 8000 is already in use, use a different port:
```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

### 2. Start the Vite Frontend
In a new terminal window, navigate to the `frontend` directory, install packages, and run the Vite development server:
```bash
cd frontend
npm install
npm run dev
```
The frontend dashboard will be available at [http://localhost:5173](http://localhost:5173).

---

## 📖 Technical Documentation

For detailed local setup, directories, architecture, and development guidelines, refer to the documents below:

- **Local Installation & Setup**: [developer_guide.md](docs/developer_guide.md)
- **Architecture Details**: [architecture.md](docs/architecture.md)
- **Dataset Reference (Oxford Parkinson's)**: [dataset_info.md](docs/dataset_info.md)
- **Model Details**: [model_card.md](docs/model_card.md)
- **Responsible AI Framework**: [responsible_ai.md](docs/responsible_ai.md)
