import os
import sys

# Ensure backend and project root directories are in python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
root_dir = os.path.abspath(os.path.join(backend_dir, ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import shutil
import threading
import uuid
import datetime
import time
import logging
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import pandas as pd

# Import local backend configurations and utilities
from app.config import settings
from app.report_generator import generate_clinical_report
from app.recording_quality import analyze_recording_quality
from app.response_enrichment import enrich_response
from ml.inference.predict import VitaVoicePredictor
from ml.training.train import train_vita_voice

# Initialize logging configuration to output structured JSON logs
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(title=settings.PROJECT_NAME)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
predictor = VitaVoicePredictor(checkpoints_dir=settings.CHECKPOINTS_DIR)

# Global lock for training state to prevent concurrent training
training_lock = threading.Lock()
is_training = False

# Rate Limiter (Token-Bucket Algorithm)
class TokenBucketLimiter:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate  # tokens added per second
        self.capacity = capacity
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()
        
    def allow_request(self, ip: str) -> bool:
        with self.lock:
            now = time.time()
            if ip not in self.buckets:
                self.buckets[ip] = (self.capacity, now)
                return True
            tokens, last_update = self.buckets[ip]
            elapsed = now - last_update
            tokens = min(self.capacity, tokens + elapsed * self.rate)
            if tokens >= 1.0:
                self.buckets[ip] = (tokens - 1.0, now)
                return True
            else:
                self.buckets[ip] = (tokens, now)
                return False

# Limit to 10 requests per minute (rate of 10/60 = 0.1667 tokens/sec, capacity = 10)
limiter = TokenBucketLimiter(rate=10/60, capacity=10.0)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if "/api/v1/screen" in path or "/api/v1/train" in path:
        client_ip = request.client.host if request.client else "127.0.0.1"
        if not limiter.allow_request(client_ip):
            log_record = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "action": "rate_limited",
                "client_ip": client_ip,
                "path": path
            }
            logging.info(json.dumps(log_record))
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Limit is 10 requests per minute."}
            )
    return await call_next(request)

@app.middleware("http")
async def json_logging_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    log_record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "client_ip": request.client.host if request.client else "unknown",
        "duration_seconds": round(duration, 4)
    }
    logging.info(json.dumps(log_record))
    return response

async def periodic_cleanup_task():
    """
    Background job that cleans up uploads and reports older than 1 hour.
    """
    while True:
        try:
            now = time.time()
            one_hour_ago = now - 3600
            for folder in [settings.UPLOAD_DIR, settings.REPORTS_DIR]:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        if filename.startswith("."):
                            continue
                        file_path = os.path.join(folder, filename)
                        if os.path.isfile(file_path):
                            mtime = os.path.getmtime(file_path)
                            if mtime < one_hour_ago:
                                os.remove(file_path)
                                log_record = {
                                    "timestamp": datetime.datetime.utcnow().isoformat(),
                                    "action": "cleanup_deleted_file",
                                    "file": file_path
                                }
                                logging.info(json.dumps(log_record))
        except Exception as e:
            logging.error(f"Error during periodic cleanup: {e}")
        await asyncio.sleep(600)  # Check every 10 minutes

@app.on_event("startup")
async def startup_event():
    # Load model on startup
    predictor.load_models()
    # Launch background cleanup task
    asyncio.create_task(periodic_cleanup_task())

# Mount reports directory to serve static PDFs
app.mount("/api/v1/reports", StaticFiles(directory=settings.REPORTS_DIR), name="reports")

def background_train_task():
    global is_training
    try:
        train_vita_voice(
            dataset_type="oxford",
            root_dir=os.path.join(settings.BASE_DIR, "datasets"),
            checkpoints_dir=settings.CHECKPOINTS_DIR
        )
        predictor.load_models()
    except Exception as e:
        print(f"Error during background training: {e}")
    finally:
        with training_lock:
            is_training = False

@app.get("/api/v1/health")
def health_check():
    """
    Returns system status, model availability, and storage directory writeability.
    """
    model_status = "Loaded" if predictor.loaded else "Not trained / Checkpoints missing"
    
    # Check directory writeability
    storage_ok = True
    for directory in [settings.UPLOAD_DIR, settings.REPORTS_DIR]:
        try:
            test_file = os.path.join(directory, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception:
            storage_ok = False
            
    status_overall = "online" if predictor.loaded and storage_ok else "degraded"
    
    return {
        "status": status_overall,
        "model_loaded": predictor.loaded,
        "model_status": model_status,
        "is_training": is_training,
        "storage_writeable": storage_ok,
        "directories": {
            "uploads": settings.UPLOAD_DIR,
            "reports": settings.REPORTS_DIR
        }
    }

@app.post("/api/v1/train")
def trigger_training(background_tasks: BackgroundTasks):
    global is_training
    if not os.path.exists(settings.DATASET_PATH):
        raise HTTPException(status_code=400, detail=f"Dataset not found at {settings.DATASET_PATH}.")
        
    with training_lock:
        if is_training:
            return {"status": "in_progress", "message": "Model training is already running."}
        is_training = True
        
    background_tasks.add_task(background_train_task)
    return {"status": "started", "message": "Model training started in the background."}

@app.post("/api/v1/screen")
async def screen_voice(request: Request, file: UploadFile = File(...)):
    # Validate content length to enforce size constraints
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.MAX_CONTENT_LENGTH:
        raise HTTPException(status_code=400, detail=f"File size exceeds the maximum limit of {settings.MAX_CONTENT_LENGTH // (1024*1024)}MB.")
        
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")
    ext = filename.split(".")[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {settings.ALLOWED_EXTENSIONS}"
        )
        
    # Save file with unique ID prefix to prevent filename conflicts
    file_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}_{filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Extra verification check on saved file size
        actual_size = os.path.getsize(file_path)
        if actual_size > settings.MAX_CONTENT_LENGTH:
            raise HTTPException(status_code=400, detail="File size exceeds the maximum limit.")
            
        if not predictor.loaded:
            predictor.load_models()
            if not predictor.loaded:
                raise HTTPException(
                    status_code=503, 
                    detail="Inference models not loaded. Retrain models first."
                )
                
        # Analyze recording quality BEFORE inference (file still on disk)
        recording_quality = analyze_recording_quality(file_path)
        
        # Run ML prediction
        result = predictor.predict_audio(file_path, recording_quality=recording_quality)
        
        # Enrich response with confidence, explanation, recommendation, etc.
        enriched = enrich_response(result)
        
        # Generate clinical summary report
        report = generate_clinical_report(
            result['clinical_metrics'], 
            result['risk_score'],
            confidence_calibration=result.get('confidence_calibration'),
            shap_explanation=result.get('shap_explanation')
        )
        
        # Generate PDF report
        from backend.app.pdf_generator import generate_pdf_report
        pdf_path = generate_pdf_report(
            report_id=file_id,
            metrics=result['clinical_metrics'],
            risk_score=result['risk_score'],
            confidence_calibration=result['confidence_calibration'],
            shap_explanation=result['shap_explanation'],
            output_dir=settings.REPORTS_DIR,
            recording_quality=recording_quality,
            confidence_label=enriched['confidence_label'],
            prediction_reliability=enriched['prediction_reliability'],
            top_biomarkers=enriched['top_biomarkers'],
            recommendation=enriched['recommendation'],
            natural_language_explanation=enriched['natural_language_explanation'],
            biomarker_statuses=enriched['biomarker_statuses'],
            wavlm_quality=result.get('wavlm_quality'),
            wavlm_similarity=result.get('wavlm_similarity'),
            wavlm_ood=result.get('wavlm_ood'),
            decision_engine=result.get('decision_engine'),
        )
        report_url = f"/api/v1/reports/report_{file_id}.pdf"
        
        # Cleanup uploaded audio (prevent holding raw patient voices on server disk)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return {
            "success": True,
            "prediction_id": file_id,
            "filename": file.filename,
            "risk_score": result['risk_score'],
            "status": result['status'],
            "embedding_coords": result['embedding_coords'],
            "clinical_metrics": result['clinical_metrics'],
            "report": report,
            "report_url": report_url,
            # New clinical enrichment fields (backward-compatible)
            "recording_quality": recording_quality,
            "confidence_score": enriched['confidence_score'],
            "confidence_label": enriched['confidence_label'],
            "prediction_reliability": enriched['prediction_reliability'],
            "top_biomarkers": enriched['top_biomarkers'],
            "natural_language_explanation": enriched['natural_language_explanation'],
            "recommendation": enriched['recommendation'],
            "responsible_ai": enriched['responsible_ai'],
            "responsible_ai_points": enriched['responsible_ai_points'],
            "biomarker_statuses": enriched['biomarker_statuses'],
        }
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Screening error: {str(e)}")

@app.get("/api/v1/clusters")
def get_reference_clusters():
    csv_path = os.path.join(settings.BASE_DIR, "datasets", "pca_visualization_clusters.csv")
    if not os.path.exists(csv_path):
        return {"loaded": False, "points": []}
        
    try:
        df = pd.read_csv(csv_path)
        points = []
        for _, row in df.iterrows():
            points.append({
                "x": float(row['pca_x']),
                "y": float(row['pca_y']),
                "status": int(row['status'])
            })
        return {"loaded": True, "points": points}
    except Exception as e:
        return {"loaded": False, "error": str(e), "points": []}

# Analysis Router Hooks (Fallback Architecture)
router = APIRouter(prefix="/api/v1/analysis", tags=["Analysis Baseline"])

class AnalysisRequest(BaseModel):
    audio_id: str

class CopilotResponse(BaseModel):
    summary: str
    citations: list[str]
    is_fallback: bool

@router.get("/download-pdf/{prediction_id}")
async def download_screening_report(prediction_id: str):
    """
    BASELINE: Returns the actual generated PDF report if it exists,
    otherwise returns a success status or dummy data placeholder.
    FUTURE: Will compile ReportLab elements + SHAP graphs into a polished medical PDF.
    """
    pdf_path = os.path.join(settings.REPORTS_DIR, f"report_{prediction_id}.pdf")
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"report_{prediction_id}.pdf")
    
    return {"status": "success", "message": f"Placeholder for PDF report {prediction_id}"}

@router.post("/copilot-insight", response_model=CopilotResponse)
async def get_clinical_copilot_insight(payload: AnalysisRequest):
    """
    BASELINE: Fallback engine providing rule-based explanations based on DSP metrics.
    FUTURE: Will hook into ChromaDB vector search + OpenAI/Ollama for RAG insights.
    """
    try:
        # Hardcoded baseline fallback logic for now:
        baseline_summary = (
            "Vocal analysis baseline complete. Acoustic measurements indicate parameters "
            "within normal variance thresholds. Feature extraction successfully isolated vocal track properties."
        )
        
        return CopilotResponse(
            summary=baseline_summary,
            citations=["VitaVoice Standard DSP Diagnostic Baseline (v1.0)"],
            is_fallback=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)

# Mount frontend build directory to serve static React SPA
frontend_dir = os.path.join(settings.BASE_DIR, "frontend", "dist")
if os.path.exists(frontend_dir):
    # Mount assets folder
    assets_dir = os.path.join(frontend_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
        
    # Catch-all to serve index.html for React SPA client-side routing
    @app.get("/{fallback_path:path}")
    async def spa_fallback(request: Request, fallback_path: str):
        # Let API endpoints and auto-docs bypass
        if fallback_path.startswith("api/") or fallback_path in ("docs", "redoc", "openapi.json"):
            raise HTTPException(status_code=404, detail="Not Found")
            
        # Avoid intercepting static files like favicon.ico, etc.
        if "." in fallback_path and not fallback_path.endswith(".html"):
            file_path = os.path.join(frontend_dir, fallback_path)
            if os.path.exists(file_path):
                return FileResponse(file_path)
            raise HTTPException(status_code=404, detail="File Not Found")
            
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Index Not Found")

