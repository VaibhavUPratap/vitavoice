import os
import sys
import pytest
import shutil
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add project root to python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.main import app, limiter
from backend.app.config import settings

client = TestClient(app)

def test_health_check_endpoint():
    # Mock predictor loading status
    with patch('backend.app.main.predictor') as mock_predictor:
        mock_predictor.loaded = True
        
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_status" in data
        assert "storage_writeable" in data
        assert "directories" in data

def test_screen_invalid_file_format():
    # Upload an unsupported .txt file format
    file_payload = {"file": ("test.txt", io.BytesIO(b"dummy text content"), "text/plain")}
    response = client.post("/api/v1/screen", files=file_payload)
    
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_screen_file_size_exceeded():
    # Enforce content size constraints by setting a small limit and checking header validation
    large_payload = {"file": ("test.wav", io.BytesIO(b"0" * (settings.MAX_CONTENT_LENGTH + 100)), "audio/wav")}
    
    # We test it with headers mimicking larger content length
    headers = {"content-length": str(settings.MAX_CONTENT_LENGTH + 100)}
    response = client.post("/api/v1/screen", files=large_payload, headers=headers)
    
    assert response.status_code == 400
    assert "exceeds the maximum limit" in response.json()["detail"]

@patch('backend.app.main.predictor')
@patch('backend.app.pdf_generator.generate_pdf_report')
@patch('backend.app.report_generator.generate_clinical_report')
def test_screen_success_mock(mock_report_gen, mock_pdf_gen, mock_predictor):
    # Setup mocks
    mock_predictor.loaded = True
    
    dummy_metrics = {
        'fo_mean': 150.0, 'fhi': 160.0, 'flo': 140.0,
        'jitter_pct': 0.5, 'jitter_abs': 0.00003, 'shimmer_local': 0.02, 'shimmer_db': 0.18,
        'hnr': 25.0, 'nhr': 0.005, 'energy': 0.1, 'formants': [500.0, 1500.0, 2500.0]
    }
    
    mock_predictor.predict_audio.return_value = {
        'risk_score': 0.25,
        'status': 0,
        'embedding_coords': [0.5, -0.2],
        'clinical_metrics': dummy_metrics,
        'confidence_calibration': {
            'risk_probability': 0.25,
            'certainty_score': 0.5,
            'certainty_label': 'Moderate Certainty',
            'calibration_confidence': 'Risk profile is moderately defined'
        },
        'shap_explanation': []
    }
    
    mock_report_gen.return_value = {
        'risk_category': 'Low Risk',
        'summary': 'Vocal features are stable.',
        'biomarker_analysis': [],
        'recommendations': [],
        'disclaimer': 'Safe fallback notice',
        'confidence_calibration': None,
        'shap_explanation': None
    }
    
    mock_pdf_gen.return_value = os.path.join(settings.REPORTS_DIR, "report_mock.pdf")
    
    # Create files for test
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)
    with open(os.path.join(settings.REPORTS_DIR, "report_mock.pdf"), "w") as f:
        f.write("mock pdf contents")
        
    try:
        # Create a small dummy wav file (about 1s at 16kHz is 32000 bytes, but here we just send small wave bytes)
        wav_data = io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
        file_payload = {"file": ("test.wav", wav_data, "audio/wav")}
        
        response = client.post("/api/v1/screen", files=file_payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["risk_score"] == 0.25
        assert "report_url" in data
        
    finally:
        # Cleanup mock pdf
        mock_pdf_path = os.path.join(settings.REPORTS_DIR, "report_mock.pdf")
        if os.path.exists(mock_pdf_path):
            os.remove(mock_pdf_path)

def test_rate_limiting_endpoint():
    # Make a request using an IP
    # Clear the bucket list first
    limiter.buckets = {}
    
    # Bucket capacity is 10. We send 11 requests rapidly.
    # The 11th request should be rate-limited (status 429).
    wav_payload = {"file": ("test.wav", io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"), "audio/wav")}
    
    # We will mock predictor.predict_audio to avoid errors when it succeeds
    with patch('backend.app.main.predictor') as mock_predictor:
        mock_predictor.loaded = True
        mock_predictor.predict_audio.return_value = {
            'risk_score': 0.1, 'status': 0, 'embedding_coords': [0, 0],
            'clinical_metrics': {'fo_mean': 150, 'fhi': 160, 'flo': 140, 'jitter_pct': 0.1, 'jitter_abs': 0.00001, 'shimmer_local': 0.01, 'shimmer_db': 0.1, 'hnr': 25, 'nhr': 0.001, 'energy': 0.1, 'formants': [500, 1500, 2500]},
            'confidence_calibration': {}, 'shap_explanation': []
        }
        
        # Mock pdf generator and report generator to speed up
        with patch('backend.app.pdf_generator.generate_pdf_report', return_value="mock.pdf"):
            with patch('backend.app.report_generator.generate_clinical_report', return_value={}):
                
                status_codes = []
                for _ in range(12):
                    # Reset seek of bytes io
                    wav_payload["file"][1].seek(0)
                    res = client.post("/api/v1/screen", files=wav_payload)
                    status_codes.append(res.status_code)
                
                # Check that we got at least one 429
                assert 429 in status_codes
                # The first few requests should be 200
                assert status_codes[0] == 200

def test_download_screening_report_placeholder():
    response = client.get("/api/v1/analysis/download-pdf/missing_prediction_id")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Placeholder for PDF" in data["message"]

def test_download_screening_report_file_exists():
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)
    test_id = "test_download_id"
    test_pdf_path = os.path.join(settings.REPORTS_DIR, f"report_{test_id}.pdf")
    with open(test_pdf_path, "w") as f:
        f.write("dummy pdf content")
        
    try:
        response = client.get(f"/api/v1/analysis/download-pdf/{test_id}")
        assert response.status_code == 200
        assert response.content == b"dummy pdf content"
    finally:
        if os.path.exists(test_pdf_path):
            os.remove(test_pdf_path)

def test_get_clinical_copilot_insight():
    payload = {"audio_id": "test_audio_id"}
    response = client.post("/api/v1/analysis/copilot-insight", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "citations" in data
    assert data["is_fallback"] is True

if __name__ == "__main__":
    pytest.main([__file__])
