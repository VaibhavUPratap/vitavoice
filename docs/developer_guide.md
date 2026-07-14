# VitaVoice Developer Guide

Welcome to the VitaVoice developer guide. This document details how to set up the development environment, run test suites, write new endpoints, and extend the machine learning pipeline.

---

## 1. Directory Structure

```text
vitavoice/
├── backend/                  # FastAPI Application
│   ├── app/
│   │   ├── config.py         # Application configurations
│   │   ├── main.py           # Core FastAPI app & Middlewares
│   │   ├── pdf_generator.py  # ReportLab PDF compiler
│   │   └── report_generator.py # Clinical markdown reports
│   └── requirements.txt      # Python dependencies
├── datasets/                 # Oxford & Raw Clinical Datasets
├── docs/                     # Technical & Ethical Documentation
├── frontend/                 # React SPA (Vite + TS)
│   ├── src/
│   │   ├── components/
│   │   │   ├── AudioRecorder.tsx # Calibration & Mic level canvas
│   │   │   └── Dashboard.tsx     # Recharts & UMAP coordinate maps
│   │   ├── App.tsx           # Global states & landing page
│   │   └── index.css         # Styling system & Themes
│   └── package.json
├── ml/                       # Machine Learning Pipeline
│   ├── feature_extraction/   # Acoustic features & speech embeddings
│   ├── preprocessing/        # Resampling, gating noise reduction
│   ├── training/             # Ensemble trainers & synthetic data
│   └── checkpoints/          # Trained model binary joblibs
└── tests/                    # Automation and unit tests
```

---

## 2. Setting Up the Development Environment

### Backend Setup (Python)
1. Initialize virtual environment:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate` (or .\venv\Scripts\activate)
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the development server:
   ```bash
   python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

### Frontend Setup (Node.js)
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install Node packages:
   ```bash
   npm install
   ```
3. Run Vite server:
   ```bash
   npm run dev
   ```
   Open `http://localhost:5173` in your browser.

---

## 3. Code Verification & Quality Standards

To maintain clinical research standards, all code must pass linters, type checks, and pytest suites.

### Running Pytest
Run the test suite with coverage tracking:
```bash
.\venv\Scripts\python.exe -m pytest --cov=ml --cov=backend tests/
```

### Static Type Checking (Mypy)
Verify static types for Python modules:
```bash
mypy --ignore-missing-imports backend/ ml/
```

### Code Style Checking (Flake8)
Verify PEP8 coding standards:
```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

### TypeScript Validation
Verify types on the frontend:
```bash
cd frontend
npx tsc --noEmit
```

---

## 4. Extending the Machine Learning Pipeline

If you wish to add new features (e.g., spectral flatness or new speech foundation embeddings):
1. **Acoustic Extraction**: Update `extract_all_acoustic_features` in `ml/feature_extraction/acoustic.py`.
2. **Model Training**: Add the feature name to the lists in `ml/training/train.py` and run a model retraining.
3. **Training Models**: Model retraining can be triggered directly on the UI via the "Train AI Models Now" warning banner, or by sending a POST request to `/api/v1/train`.
