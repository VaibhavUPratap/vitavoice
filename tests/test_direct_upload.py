import os
import sys
import io
import pytest
from fastapi.testclient import TestClient

# Add project root to python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.main import app
from tests.test_results_page import generate_dummy_wav

def test_direct_api_upload():
    client = TestClient(app)
    
    mock_wav_path = "tests/mock_voice_direct.wav"
    generate_dummy_wav(mock_wav_path)
    
    print("Sending POST request to /api/v1/screen...")
    try:
        with open(mock_wav_path, "rb") as f:
            file_payload = {"file": ("voice_sample.wav", f, "audio/wav")}
            response = client.post("/api/v1/screen", files=file_payload)
            
        print("Response status code:", response.status_code)
        assert response.status_code == 200
        data = response.json()
        print("Keys returned:", list(data.keys()))
        assert data["success"] is True
        assert "decision_engine" in data
        print("Decision Status Label:", data["decision_engine"]["status_label"])
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
