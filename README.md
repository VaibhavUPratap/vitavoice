# VitaVoice – AI-Powered Voice-Based Health Risk Screening

VitaVoice is an AI-powered healthcare application that analyzes sustained vowel vocal samples (e.g., a sustained "ah" sound) to screen for voice abnormalities that act as clinical biomarkers for chronic conditions like Parkinson's disease.

By combining **domain-specific clinical acoustics** (perturbation and spectral noise ratios) with **modern deep speech representation models** (Wav2Vec 2.0), VitaVoice provides a rapid, non-invasive, and explainable health risk screening.

> [!WARNING]
> **Medical Disclaimer:** VitaVoice is intended for educational, research, and wellness screening purposes only. It is not an FDA-cleared diagnostic device and does not provide formal medical diagnoses. Vocal changes can stem from common allergies, vocal fatigue, colds, or acid reflux. Please consult a qualified healthcare professional (ENT or Neurologist) for medical concerns.

---

## System Architecture

```text
                               ┌─────────────────────────┐
                               │   User Voice Recording  │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │    Audio Preprocessing  │
                               │   (16kHz, Resampling,   │
                               │     Noise Reduction)    │
                               └────────────┬────────────┘
                                            │
                           ┌────────────────┴────────────────┐
                           ▼                                 ▼
             ┌───────────────────────────┐     ┌───────────────────────────┐
             │ Clinical Feature Pipeline │     │  Wav2Vec 2.0 Pipeline     │
             │  (F0, Jitter, Shimmer,    │     │  (facebook/wav2vec-base)  │
             │       HNR, MFCCs)         │     └─────────────┬─────────────┘
             └─────────────┬─────────────┘                   │ (768-dim embedding)
                           │ (20 features)                   ▼
                           ▼                   ┌───────────────────────────┐
             ┌───────────────────────────┐     │     PCA Dimensionality    │
             │   Feature Normalization   │     │         Reduction         │
             │     (StandardScaler)      │     └─────────────┬─────────────┘
             └─────────────┬─────────────┘                   │ (16 components)
                           │                                 ▼
                           └────────────────┬────────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │  Concatenated Features  │
                               │      (36 features)      │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │   SV Ensemble Model     │
                               │ (Probability Classifier)│
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │   AI Health Report &    │
                               │    Visual Analytics     │
                               └─────────────────────────┘
```

## Vocal Biomarkers Explained

- **Pitch (F0):** Fundamental frequency representing vocal fold vibration speed. Healthy male range: 85–155Hz, female: 165–255Hz.
- **Jitter (local):** Cycle-to-cycle frequency variations. Healthy threshold is `< 1.04%`. Elevated values indicate micro-tremors in vocal cord control.
- **Shimmer (local):** Cycle-to-cycle amplitude (loudness) variations. Healthy threshold is `< 3.80%`. High shimmer indicates instability in vocal cord pressure.
- **Harmonics-to-Noise Ratio (HNR):** Ratio of pure vocal harmonics to background/breathy noise. Healthy threshold is `> 20.0 dB`. Lower values indicate vocal breathiness or hoarseness.

---

## Directory Structure

```text
vitavoice/
├── backend/                  # FastAPI Web Server
│   ├── app/
│   │   ├── main.py           # API Endpoints
│   │   ├── config.py         # Application settings
│   │   └── report_generator.py # Clinical report engine
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/                 # Vite + React + TypeScript SPA
│   ├── src/
│   │   ├── components/
│   │   │   ├── AudioRecorder.tsx # Canvas visualizer & WAV capture
│   │   │   └── Dashboard.tsx     # Radial gauge, graphs, 2D PCA cluster map
│   │   ├── App.tsx           # State routing & loader pages
│   │   └── index.css         # Tailwind v4.0 glassmorphic styling
│   ├── Dockerfile
│   └── index.html
│
├── ml/                       # Machine Learning Codebase
│   ├── preprocessing/
│   │   └── audio.py          # Resampling, Noise reduction, trimming
│   ├── feature_extraction/
│   │   ├── acoustic.py       # Pitch, Jitter, Shimmer, HNR, MFCCs, Formants
│   │   └── embeddings.py     # Wav2Vec 2.0 feature extractor
│   ├── inference/
│   │   └── predict.py        # Inference pipeline loader
│   ├── training/
│   │   ├── synthesize_dataset.py # Additive vocal synthesizer for Oxford dataset
│   │   └── train.py          # 5-fold cross-validation classifier training
│   └── checkpoints/          # Trained model weights & scalers
│
└── datasets/
    ├── parkinsons.data       # Oxford Parkinson's voice features
    └── parkinsons.names      # Dataset documentation metadata
```

---

## Getting Started

### Method 1: Local Installation

#### 1. Setup Backend & ML models

1. Navigate to the root directory and create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Run the model training script. This script automatically synthesizes vocal waveforms matching the Oxford features, extracts embeddings, and trains the model:
   ```bash
   python ml/training/train.py
   ```
4. Launch the FastAPI server:
   ```bash
   python -m uvicorn backend.app.main:app --reload
   ```
   The backend will be available at `http://localhost:8000`.

#### 2. Setup Frontend

1. Navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Launch Vite Dev server:
   ```bash
   npm run dev
   ```
   The frontend dashboard will be available at `http://localhost:5173`.

---

### Method 2: Docker Compose (Production Build)

Launch the complete multi-container setup in one command:
```bash
docker-compose up --build
```
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

---

## Reference Dataset

The models are trained using the **Oxford Parkinson's Disease Detection Dataset** created by Max Little of the University of Oxford.
- **Samples:** 196 recordings from 31 subjects (23 with Parkinson's, 8 healthy).
- **Core target:** Discriminate healthy people from those with Parkinson's (status: 0 or 1).
- **Paper Citation:** Little MA, McSharry PE, Hunter EJ, Ramig LO (2008), *'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease'*, IEEE Transactions on Biomedical Engineering.
