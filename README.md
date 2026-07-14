# 🎙️ VitaVoice — AI-Powered Vocal Biomarker Health Screener

VitaVoice is an advanced, non-invasive health screening application that analyzes voice recordings to identify vocal instabilities and dysphonia patterns associated with neurological conditions (e.g., Parkinson's disease).

By combining **clinical acoustic digital signal processing** with **modern transformer-based deep speech representation models (Wav2Vec 2.0)**, VitaVoice provides rapid, explainable health screenings for researchers and clinical enthusiasts.

---

> [!WARNING]
> **Medical Disclaimer:** VitaVoice is intended for educational, research, and wellness screening purposes only. It is not an FDA-cleared diagnostic device and does not provide formal medical diagnoses. Vocal changes can stem from common allergies, vocal fatigue, colds, or acid reflux. Please consult a qualified healthcare professional (ENT or Neurologist) for medical concerns.

---

## ✨ Key Features

- **Interactive Vocal Signal & Feature Analyzer**: Live canvas-less SVG spectrogram visualizer on the home page responding directly to mouse gestures with real-time frequency/amplitude tooltips.
- **Hybrid Acoustic & Neural Pipeline**: Extracts 20 classical acoustic perturbation biomarkers (Jitter, Shimmer, HNR, F0 fundamental frequencies) and 16 high-dimensional Wav2Vec 2.0 transformer embedding components.
- **Explainable AI (SHAP)**: Provides a feature-by-feature SHAP importance analysis showing which vocal biomarkers contribute positively or negatively to the risk score.
- **Interactive PCA Embedding Space**: Visualizes the patient's voice coordinate relative to reference healthy control and pathology cohorts on a 2D cluster map.
- **PDF Report Generator**: Generates clean, clinical-grade PDF reports with patient summaries, biomarker charts, and recommendation paths.

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
                          │ (16kHz, Resampling, Noise)│
                          └─────────────┬────────────┘
                                        │
                      ┌─────────────────┴─────────────────┐
                      ▼                                   ▼
        ┌───────────────────────────┐       ┌───────────────────────────┐
        │  Clinical Perturbation    │       │   Wav2Vec 2.0 Embedding   │
        │ (Jitter, Shimmer, HNR...) │       │  (facebook/wav2vec-base)  │
        └─────────────┬─────────────┘       └─────────────┬─────────────┘
                      │                                   │ (768-dim PCA)
                      ▼                                   ▼
        ┌───────────────────────────┐       ┌───────────────────────────┐
        │   20 Acoustic Features    │       │     16 PCA Components     │
        └─────────────┬─────────────┘       └─────────────┬─────────────┘
                      │                                   │
                      └─────────────────┬─────────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   36-Feature Biomap      │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   SVM Ensemble Model     │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   Clinical Report, SHAP  │
                          │   & 2D PCA Cluster Map   │
                          └──────────────────────────┘
```

---

## 🚀 Quick Start (Running Locally)

To run the application, you need to start the FastAPI backend and the Vite frontend separately.

### 1. Start the FastAPI Backend
Initialize your virtual environment, install the dependencies, and run the development server:
```bash
# Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the development server
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
The backend API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

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

- **Local Installation & Setup**: [developer_guide.md](file:///d:/Projects/vitavoice/docs/developer_guide.md)
- **Architecture Details**: [architecture.md](file:///d:/Projects/vitavoice/docs/architecture.md)
- **Dataset Reference (Oxford Parkinson's)**: [dataset_info.md](file:///d:/Projects/vitavoice/docs/dataset_info.md)
- **Model Details**: [model_card.md](file:///d:/Projects/vitavoice/docs/model_card.md)
- **Responsible AI Framework**: [responsible_ai.md](file:///d:/Projects/vitavoice/docs/responsible_ai.md)
