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
   If port 8000 is already in use, specify a different port:
   ```bash
   python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
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

To maintain clinical standards, all code must pass linters, type checks, backend unit tests, and end-to-end browser automation suites.

### 3.1 Running Backend Pytests
Run the unit test suite (ignoring browser/direct-upload tests to avoid local rate-limiting or uvicorn requirements):
```bash
.\venv\Scripts\python.exe -m pytest --ignore=tests/test_frontend.py --ignore=tests/test_results_page.py --ignore=tests/test_direct_upload.py
```

### 3.2 Running Playwright Integration Tests
To verify the end-to-end visual integration between the FastAPI backend and React frontend (including file uploads and latent cluster visualizer rendering), use the automated server test wrapper:
```bash
.\venv\Scripts\python.exe .agents\skills\webapp-testing\scripts\with_server.py --server ".\venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000" --port 8000 -- .\venv\Scripts\python.exe -m pytest tests/test_results_page.py -s
```
*Note: This script launches uvicorn on port 8000, wait-checks for readiness, executes the Playwright chrome tests, captures browser console outputs, and saves page screenshots to `tests/screenshots/results_page.png` before terminating the server process.*

### 3.3 Static Type Checking & Linters
Verify static types for Python modules:
```bash
mypy --ignore-missing-imports backend/ ml/
```
Verify PEP8 style guidelines:
```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```
Verify TypeScript files on the frontend:
```bash
cd frontend
npx tsc --noEmit
```

---

## 4. Extending the SQLite Database & CDE Rules

### 4.1 Modifying Database Schema
The patient timeline schema is managed in [patient_db.py](file:///d:/Projects/vitavoice/backend/app/services/patient_db.py). The DB utilizes an automated sqlite dynamic migration scheme:
- If you need to track new diagnostic parameters (e.g. spoofy probability, raw audio paths), add columns to the `CREATE TABLE` script inside `PatientDBService._init_db()`.
- Baseline voice drift calculations are processed in `PatientDBService.save_screening` and query historical coordinates to map drift magnitude vectors.

### 4.2 Expanding Clinical Decision Engine (CDE) Logic
The logic gates reside in [decision_engine.py](file:///d:/Projects/vitavoice/backend/app/services/decision_engine.py):
- Override rules evaluate classifier probability margins, OOD bounds, spoof checks, and historical tracking drift.
- If you add a new diagnostic check, register the condition in `ClinicalDecisionEngine.run_rules` and ensure it returns appropriate override status keys.

---

## 5. Clinical Copilot & RAG Integration

The backend includes a dedicated router `/api/v1/analysis` in `backend/app/main.py` for clinician copilot features:
- **Baseline Fallback**: The `/copilot-insight` endpoint currently generates rule-based descriptions of the acoustic features.
- **RAG Upgrades**: To implement full Retrieval-Augmented Generation:
  1. Initialize a vector database client (e.g., ChromaDB or Qdrant) within `backend/app/main.py` or a helper module.
  2. Embed clinical research papers using a sentence transformer model.
  3. Query the vector database with extracted user biomarkers.
  4. Pass the retrieved context along with patient acoustic scores to an LLM provider (e.g., OpenAI API or a local Ollama server running Llama-3) to return customized diagnostic insights.
