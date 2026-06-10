import os

class Settings:
    PROJECT_NAME: str = "VitaVoice API"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    UPLOAD_DIR = os.path.join(BASE_DIR, "backend", "uploads")
    REPORTS_DIR = os.path.join(BASE_DIR, "backend", "reports")
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, "ml", "checkpoints")
    DATASET_PATH = os.path.join(BASE_DIR, "datasets", "parkinsons.data")
    
    # Model configuration
    ALLOWED_EXTENSIONS = {"wav", "mp3", "webm", "ogg"}
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB

settings = Settings()

# Ensure folders exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.REPORTS_DIR, exist_ok=True)
os.makedirs(settings.CHECKPOINTS_DIR, exist_ok=True)
