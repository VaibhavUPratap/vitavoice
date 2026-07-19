import os
import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from app.services.patient_db import PatientDBService
from ml.feature_extraction.embeddings import project_embedding_2d
from app.config import settings

router = APIRouter(prefix="/api/v1/patients", tags=["Patients Timeline"])
db_service = PatientDBService()

# Load PCA visualizer model for Digital Twin projection
pca_model_path = os.path.join(settings.CHECKPOINTS_DIR, "pca_model.joblib")
pca_model = None
if os.path.exists(pca_model_path):
    try:
        pca_model = joblib.load(pca_model_path)
    except Exception as e:
        print(f"Error loading PCA model for patients API: {e}")

@router.get("/{patient_id}/timeline")
def get_patient_timeline(patient_id: str):
    """
    Retrieves the historical timeline of voice screenings for a patient.
    """
    try:
        history = db_service.get_patient_history(patient_id)
        return {
            "patient_id": patient_id,
            "total_screenings": len(history),
            "timeline": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/{patient_id}/digital-twin")
def get_patient_digital_twin(patient_id: str):
    """
    Returns coordinate sets [x, y] to plot the patient's voice trajectory over time.
    """
    try:
        trajectory = db_service.get_patient_trajectory(patient_id)
        
        if not trajectory:
            return {
                "patient_id": patient_id,
                "trajectory": []
            }
            
        if pca_model is None:
            raise HTTPException(status_code=503, detail="PCA visualizer model not loaded on server.")
            
        coords_trajectory = []
        for point in trajectory:
            emb = point['embedding']
            try:
                # Project 768-D to 2D
                pca_x, pca_y = project_embedding_2d(emb, pca_model)
                coords_trajectory.append({
                    "x": float(pca_x),
                    "y": float(pca_y),
                    "timestamp": point['timestamp']
                })
            except Exception as e:
                print(f"Failed to project embedding: {e}")
                
        return {
            "patient_id": patient_id,
            "trajectory": coords_trajectory
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
