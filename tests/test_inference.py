import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Add project root to python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.inference.predict import VitaVoicePredictor

@pytest.fixture
def mock_predictor():
    predictor = VitaVoicePredictor(checkpoints_dir="ml/checkpoints")
    
    # Create mock sub-components
    predictor.scaler = MagicMock()
    # Scaler returns a 1D array of same shape on transform
    predictor.scaler.transform = MagicMock(side_effect=lambda x: x)
    
    predictor.reducer = MagicMock()
    # Reducer converts 768 features to 10 features
    predictor.reducer.transform = MagicMock(return_value=np.zeros((1, 10)))
    
    predictor.pca_vis = MagicMock()
    predictor.pca_vis.transform = MagicMock(return_value=np.array([[1.5, -0.8]]))
    
    predictor.model = MagicMock()
    predictor.model.predict_proba = MagicMock(return_value=np.array([[0.25, 0.75]]))
    predictor.model.predict = MagicMock(return_value=np.array([1]))
    
    predictor.feature_names = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
        'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
        'NHR', 'HNR', 'Energy', 'F1', 'F2', 'F3',
        'Spectral_Centroid', 'Spectral_Bandwidth', 'Zero_Crossing_Rate'
    ]
    predictor.model_type = "svm"
    predictor.background_data = np.zeros((20, len(predictor.feature_names) + 10))
    predictor.compute_shap_explanations = MagicMock(return_value=[{
        'feature_name': 'MDVP:Fo(Hz)',
        'label': 'Average Pitch (F0)',
        'shap_value': 0.05,
        'impact': 'increase',
        'abs_value': 0.05
    }])
    predictor.loaded = True
    
    return predictor

def test_load_models_missing():
    # Test that it returns False when checkpoints are missing
    predictor = VitaVoicePredictor(checkpoints_dir="non_existent_checkpoints_path_12345")
    assert predictor.load_models() is False
    assert predictor.loaded is False

@patch('ml.inference.predict.extract_all_acoustic_features')
@patch('ml.inference.predict.extract_wav2vec2_embeddings')
@patch('ml.preprocessing.audio.AudioPreprocessingPipeline.preprocess_audio')
def test_predictor_predict_audio(mock_preprocess, mock_embeddings, mock_acoustic, mock_predictor):
    # Setup mocks
    mock_preprocess.return_value = (np.zeros(16000), 16000)
    mock_embeddings.return_value = np.zeros(768)
    
    dummy_acoustic = {
        'MDVP:Fo(Hz)': 150.0, 'MDVP:Fhi(Hz)': 160.0, 'MDVP:Flo(Hz)': 140.0,
        'MDVP:Jitter(%)': 0.5, 'MDVP:Jitter(Abs)': 0.00003, 'MDVP:RAP': 0.001, 'MDVP:PPQ': 0.001, 'Jitter:DDP': 0.003,
        'MDVP:Shimmer': 0.02, 'MDVP:Shimmer(dB)': 0.18, 'Shimmer:APQ3': 0.01, 'Shimmer:APQ5': 0.01, 'MDVP:APQ': 0.015, 'Shimmer:DDA': 0.03,
        'NHR': 0.005, 'HNR': 25.0, 'Energy': 0.1, 'F1': 500.0, 'F2': 1500.0, 'F3': 2500.0,
        'Spectral_Centroid': 1200.0, 'Spectral_Bandwidth': 1000.0, 'Zero_Crossing_Rate': 0.05
    }
    # Add other mock keys just in case
    for idx in range(1, 13):
        dummy_acoustic[f'MFCC_{idx}'] = 0.0
        dummy_acoustic[f'Chroma_{idx}'] = 0.0
    
    mock_acoustic.return_value = dummy_acoustic
    
    # Run prediction
    result = mock_predictor.predict_audio("dummy_path.wav")
    
    # Verify outputs
    assert result['risk_score'] == 0.75
    assert result['status'] == 1
    assert result['embedding_coords'] == [1.5, -0.8]
    assert result['clinical_metrics']['jitter_pct'] == 0.5
    assert result['clinical_metrics']['hnr'] == 25.0
    assert result['clinical_metrics']['fo_mean'] == 150.0
    assert result['confidence_calibration']['certainty_label'] == "Moderate Certainty"
    assert len(result['shap_explanation']) > 0
    assert result['shap_explanation'][0]['feature_name'] in dummy_acoustic

if __name__ == "__main__":
    pytest.main([__file__])
