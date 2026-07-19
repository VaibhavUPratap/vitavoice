import os
import sys
import pytest
import numpy as np

# Add project root to python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.inference.wavlm_intelligence import WavLMIntelligenceLayer

@pytest.fixture
def intel_layer():
    # Instantiate with fallback dummy references
    layer = WavLMIntelligenceLayer(references_path="non_existent_references_path_1234")
    # Setup mock centroids and embeddings
    layer.healthy_centroid = np.zeros(768)
    layer.healthy_centroid[0] = 1.0
    layer.parkinsons_centroid = np.zeros(768)
    layer.parkinsons_centroid[1] = 1.0
    
    # 5 training embeddings close to the test domain
    layer.train_embeddings = np.zeros((5, 768))
    for idx in range(5):
        if idx < 3:
            layer.train_embeddings[idx][0] = 1.0
            layer.train_embeddings[idx][1] = 0.02
        else:
            layer.train_embeddings[idx][0] = 0.98
            layer.train_embeddings[idx][1] = 0.05
    layer.train_labels = np.array([0, 0, 0, 1, 1])
    
    # Mock OOD detector
    class MockOOD:
        def __init__(self):
            self.score = 5.0
        def score_samples(self, X):
            return np.array([self.score])
            
    layer.ood_detector = MockOOD()
    layer.ood_threshold = 3.0
    
    layer.mean_healthy_reduced = np.ones(16) * 0.1
    layer.mean_parkinsons_reduced = np.ones(16) * 0.2
    layer.cov_healthy_inv = np.eye(16)
    layer.cov_parkinsons_inv = np.eye(16)
    
    layer.loaded = True
    return layer

def test_quality_verification_good(intel_layer):
    """Test quality verification for a good quality recording."""
    dsp_quality = {
        "snr_db": 25.0,
        "background_noise_pct": 5.0,
        "clipping_detected": False,
        "duration_seconds": 12.0,
        "speech_coverage_pct": 80.0
    }
    # An embedding identical to the healthy centroid
    emb = np.zeros(768)
    emb[0] = 1.0
    res = intel_layer.verify_quality(emb, dsp_quality)
    
    assert res['quality_score'] == 5.0
    assert res['recording_reliability'] == "High"
    assert res['re_record_recommended'] is False

def test_quality_verification_bad(intel_layer):
    """Test quality verification with penalties (clipping, noise)."""
    dsp_quality = {
        "snr_db": 8.0,
        "background_noise_pct": 45.0,
        "clipping_detected": True,
        "duration_seconds": 3.0,
        "speech_coverage_pct": 20.0
    }
    # Embedding with low similarity
    emb = np.random.normal(size=768)
    res = intel_layer.verify_quality(emb, dsp_quality)
    
    assert res['quality_score'] < 3.0
    assert res['re_record_recommended'] is True
    assert "signal clipping" in res['reasons']
    assert "high background noise" in res['reasons']

def test_embedding_similarity(intel_layer):
    """Test the similarity engine calculation."""
    # Embedding close to Parkinson's centroid
    emb = np.zeros(768)
    emb[1] = 1.0
    res = intel_layer.compute_similarity(emb, reducer=None)
    
    assert res['nearest_cluster'] == "Parkinson's Cluster"
    assert res['similarity_score'] > 50.0
    assert res['embedding_confidence'] == "High"

def test_ood_detector_in_distribution(intel_layer):
    """Test OOD detection for normal samples."""
    intel_layer.ood_detector.score = 6.0
    res = intel_layer.detect_ood(np.ones(768))
    assert res['is_ood'] is False
    assert res['ood_score'] < 50.0

def test_ood_detector_out_of_distribution(intel_layer):
    """Test OOD detection for anomalous samples."""
    intel_layer.ood_detector.score = 2.0
    res = intel_layer.detect_ood(np.ones(768))
    assert res['is_ood'] is True
    assert res['ood_score'] >= 50.0

def test_clinical_decision_engine_poor_quality(intel_layer):
    """Rule 1: Poor quality triggers re-record recommend."""
    quality_info = {
        're_record_recommended': True,
        'quality_score': 2.0,
        'reasons': ['high background noise'],
        'recording_reliability': 'Low'
    }
    ood_info = {'is_ood': False, 'ood_score': 10.0}
    similarity_info = {'similarity_score': 90.0, 'nearest_cluster': "Parkinson's Cluster"}
    
    res = intel_layer.run_decision_engine(
        clinical_risk=0.85,
        clinical_confidence=0.8,
        quality_info=quality_info,
        ood_info=ood_info,
        similarity_info=similarity_info
    )
    
    assert "Low Confidence (Poor Quality)" in res['confidence_label']
    assert res['confidence_score'] <= 30.0
    assert "Recommend re-recording" in res['recommendation']

def test_clinical_decision_engine_ood(intel_layer):
    """Rule 2: OOD sample triggers caution flag."""
    quality_info = {
        're_record_recommended': False,
        'quality_score': 4.5,
        'reasons': [],
        'recording_reliability': 'High'
    }
    ood_info = {'is_ood': True, 'ood_score': 85.0}
    similarity_info = {'similarity_score': 90.0, 'nearest_cluster': "Parkinson's Cluster"}
    
    res = intel_layer.run_decision_engine(
        clinical_risk=0.85,
        clinical_confidence=0.8,
        quality_info=quality_info,
        ood_info=ood_info,
        similarity_info=similarity_info
    )
    
    assert "Low Confidence (Out-of-Distribution)" in res['confidence_label']
    assert "differs significantly from the reference population" in res['decision_reasoning']

def test_clinical_decision_engine_inconclusive(intel_layer):
    """Rule 3: Borderline risk + ambiguous similarity triggers inconclusive."""
    quality_info = {
        're_record_recommended': False,
        'quality_score': 4.5,
        'reasons': [],
        'recording_reliability': 'High'
    }
    ood_info = {'is_ood': False, 'ood_score': 10.0}
    similarity_info = {'similarity_score': 50.0, 'nearest_cluster': 'Parkinson\'s Cluster'}
    
    res = intel_layer.run_decision_engine(
        clinical_risk=0.5,
        clinical_confidence=0.4,
        quality_info=quality_info,
        ood_info=ood_info,
        similarity_info=similarity_info
    )
    
    assert res['status'] == 3
    assert "Inconclusive" in res['confidence_label']

def test_clinical_decision_engine_congruent_parkinsons(intel_layer):
    """Rule 4: High risk + congruent similarity triggers Very High Confidence."""
    quality_info = {
        're_record_recommended': False,
        'quality_score': 4.5,
        'reasons': [],
        'recording_reliability': 'High'
    }
    ood_info = {'is_ood': False, 'ood_score': 10.0}
    similarity_info = {'similarity_score': 85.0, 'nearest_cluster': "Parkinson's Cluster"}
    
    res = intel_layer.run_decision_engine(
        clinical_risk=0.85,
        clinical_confidence=0.8,
        quality_info=quality_info,
        ood_info=ood_info,
        similarity_info=similarity_info
    )
    
    assert res['confidence_label'] == "Very High Confidence"
    assert res['confidence_score'] >= 95.0
    assert "strong congruence" in res['decision_reasoning']
