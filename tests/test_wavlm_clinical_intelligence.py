import os
import sys
import pytest
import numpy as np

# Add project root to python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.inference.wavlm_fingerprint import WavLMFingerprintEngine
from ml.inference.wavlm_authenticity import RecordingAuthenticityAuditor
from ml.inference.wavlm_ood import WavLMOODDetector
from ml.inference.wavlm_quality import WavLMQualityAuditor
from ml.inference.wavlm_confidence import WavLMConfidenceAuditor
from backend.app.services.decision_engine import ClinicalDecisionEngine
from backend.app.services.patient_db import PatientDBService

@pytest.fixture
def mock_references():
    class References:
        healthy_centroid = np.zeros(768)
        healthy_centroid[0] = 1.0
        parkinsons_centroid = np.zeros(768)
        parkinsons_centroid[1] = 1.0
        
        train_embeddings = np.zeros((5, 768))
        train_embeddings[0][0] = 1.0   # Healthy
        train_embeddings[1][0] = 0.95  # Healthy
        train_embeddings[2][0] = 0.90  # Healthy
        train_embeddings[3][1] = 1.0   # PD
        train_embeddings[4][1] = 0.95  # PD
        train_labels = np.array([0, 0, 0, 1, 1])
        
        mean_healthy_reduced = np.ones(16) * 0.1
        mean_parkinsons_reduced = np.ones(16) * 0.2
        cov_healthy_inv = np.eye(16)
        cov_parkinsons_inv = np.eye(16)
        
        class MockOOD:
            def score_samples(self, X):
                # Returns high score if first element is non-zero
                return np.array([5.0 if X[0][0] > 0.5 else 1.0])
        ood_detector = MockOOD()
        ood_threshold = 3.0
    return References()

@pytest.fixture
def fingerprint_engine(mock_references):
    engine = WavLMFingerprintEngine(checkpoints_dir="non_existent")
    engine.healthy_centroid = mock_references.healthy_centroid
    engine.parkinsons_centroid = mock_references.parkinsons_centroid
    engine.train_embeddings = mock_references.train_embeddings
    engine.train_labels = mock_references.train_labels
    engine.mean_healthy_reduced = mock_references.mean_healthy_reduced
    engine.mean_parkinsons_reduced = mock_references.mean_parkinsons_reduced
    engine.cov_healthy_inv = mock_references.cov_healthy_inv
    engine.cov_parkinsons_inv = mock_references.cov_parkinsons_inv
    engine.loaded = True
    return engine

@pytest.fixture
def ood_detector(mock_references):
    detector = WavLMOODDetector(checkpoints_dir="non_existent")
    detector.ood_detector = mock_references.ood_detector
    detector.ood_threshold = mock_references.ood_threshold
    detector.loaded = True
    return detector

def test_fingerprint_similarity_matching(fingerprint_engine):
    # Match PD centroid closely
    emb = np.zeros(768)
    emb[1] = 1.0
    res = fingerprint_engine.compute_similarity(emb)
    
    assert res['nearest_cluster'] == "Parkinson's Cluster"
    assert res['similarity_parkinsons'] > 90.0
    assert res['embedding_confidence'] == "High"

def test_fingerprint_nearest_neighbors(fingerprint_engine):
    # Embedding identical to PD centroid
    emb = np.zeros(768)
    emb[1] = 1.0
    neighbors = fingerprint_engine.find_nearest_neighbors(emb, k=2)
    
    assert len(neighbors) == 2
    assert neighbors[0]['label'] == 1
    assert "Pathology Cohort" in neighbors[0]['cohort']

def test_ood_detector_logic(ood_detector):
    # In-distribution
    emb_in = np.zeros(768)
    emb_in[0] = 1.0
    res_in = ood_detector.detect_ood(emb_in)
    assert res_in['is_ood'] is False
    assert res_in['ood_probability'] < 50.0
    
    # Out-of-distribution
    emb_out = np.zeros(768)
    emb_out[0] = 0.0
    res_out = ood_detector.detect_ood(emb_out)
    assert res_out['is_ood'] is True
    assert res_out['ood_probability'] >= 50.0

def test_quality_auditor_calculations():
    auditor = WavLMQualityAuditor()
    train_embs = np.ones((5, 768))
    emb = np.ones(768)
    
    # Excellent quality DSP parameters
    dsp_ok = {
        "snr_db": 30.0,
        "background_noise_pct": 2.0,
        "clipping_detected": False,
        "duration_seconds": 15.0,
        "speech_coverage_pct": 90.0
    }
    res_ok = auditor.audit_quality(emb, train_embs, dsp_ok)
    assert res_ok['overall_score'] >= 4.5
    assert res_ok['recording_reliability'] == "High"
    assert res_ok['re_record_recommended'] is False
    
    # Bad quality DSP parameters
    dsp_bad = {
        "snr_db": 5.0,
        "background_noise_pct": 55.0,
        "clipping_detected": True,
        "duration_seconds": 2.0,
        "speech_coverage_pct": 10.0
    }
    res_bad = auditor.audit_quality(emb, train_embs, dsp_bad)
    assert res_bad['overall_score'] <= 2.5
    assert res_bad['re_record_recommended'] is True
    assert "signal clipping" in res_bad['reasons']

def test_confidence_auditor_combinations():
    auditor = WavLMConfidenceAuditor()
    
    # High confidence scenario
    res_high = auditor.evaluate_trust(
        svm_risk=0.90,
        quality_score=4.8,
        ood_probability=5.0,
        similarity_matching_score=92.0,
        authenticity_score=97.0
    )
    assert res_high['trust_level'] == "High"
    assert res_high['trust_score'] >= 80.0
    
    # Quality failure drops trust level to Low
    res_low = auditor.evaluate_trust(
        svm_risk=0.90,
        quality_score=2.0,
        ood_probability=5.0,
        similarity_matching_score=92.0,
        authenticity_score=97.0
    )
    assert res_low['trust_level'] == "Low"
    assert res_low['trust_score'] <= 30.0

def test_decision_engine_rules():
    engine = ClinicalDecisionEngine()
    
    # 1. Quality failure overrides risk classification
    quality_bad = {'overall_score': 2.0, 're_record_recommended': True, 'reasons': ['signal clipping']}
    auth_ok = {'authenticity_score': 98.0, 'is_authentic': True}
    ood_ok = {'is_ood': False, 'ood_probability': 5.0}
    sim_ok = {'similarity_parkinsons': 90.0, 'nearest_cluster': "Parkinson's Cluster"}
    svm_trust = {'trust_score': 85.0, 'trust_level': "High"}
    
    res = engine.run_rules(
        svm_risk=0.88,
        svm_trust=svm_trust,
        quality_info=quality_bad,
        ood_info=ood_ok,
        similarity_info=sim_ok,
        authenticity_info=auth_ok
    )
    assert "Suspended" in res['status_label']
    assert "recording" in res['recommendation']
    
    # 2. Inconclusive assessment on borderline + ambiguous matching
    quality_ok = {'overall_score': 4.5, 're_record_recommended': False, 'reasons': []}
    sim_ambig = {'similarity_parkinsons': 50.0, 'similarity_healthy': 50.0, 'nearest_cluster': 'Healthy Cluster'}
    res_inc = engine.run_rules(
        svm_risk=0.5,
        svm_trust={'trust_score': 45.0, 'trust_level': "Medium"},
        quality_info=quality_ok,
        ood_info=ood_ok,
        similarity_info=sim_ambig,
        authenticity_info=auth_ok
    )
    assert res_inc['status'] == 3
    assert "Inconclusive" in res_inc['status_label']

def test_patient_db_service_integration(tmp_path):
    # Use temporary file to isolate testing
    db_file = os.path.join(tmp_path, "test_patients.db")
    db = PatientDBService(db_path=db_file)
    
    patient_id = "pat_test_123"
    screening_id = "scr_abc"
    embedding = np.ones(768)
    embedding[0] = 0.5
    
    metrics = {"hnr": 22.0, "jitter_pct": 0.45}
    reasoning = "Normal acoustic biomarkers."
    rec = "wellness monitoring."
    
    # Save screening (will implicitly initialize baseline)
    db.save_screening(
        patient_id=patient_id,
        screening_id=screening_id,
        risk_score=0.12,
        trust_score=88.5,
        trust_level="High",
        embedding=embedding,
        clinical_metrics=metrics,
        decision_reasoning=reasoning,
        recommendation=rec
    )
    
    # Verify baseline is stored
    base = db.get_patient_baseline(patient_id)
    assert base is not None
    assert base.shape == (768,)
    assert base[0] == 0.5
    
    # Verify history
    history = db.get_patient_history(patient_id)
    assert len(history) == 1
    assert history[0]['screening_id'] == screening_id
    assert history[0]['clinical_metrics']['hnr'] == 22.0
    
    # Verify trajectory
    traj = db.get_patient_trajectory(patient_id)
    assert len(traj) == 1
    assert traj[0]['embedding'].shape == (768,)
