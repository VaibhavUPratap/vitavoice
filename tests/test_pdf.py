import os
import sys
import pytest
import shutil

# Add project root to python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.pdf_generator import generate_pdf_report

def test_generate_pdf_report():
    report_id = "test_run_999"
    metrics = {
        'fo_mean': 150.25,
        'fhi': 165.12,
        'flo': 135.84,
        'jitter_pct': 0.45,
        'jitter_abs': 0.00003,
        'shimmer_local': 0.025,
        'shimmer_db': 0.22,
        'hnr': 24.5,
        'nhr': 0.004,
        'energy': 0.08,
        'formants': [550.0, 1600.0, 2450.0]
    }
    risk_score = 0.22
    confidence_calibration = {
        'risk_probability': 0.22,
        'certainty_score': 0.56,
        'certainty_label': 'Moderate Certainty',
        'calibration_confidence': 'Risk profile is moderately defined'
    }
    shap_explanation = [
        {'feature_name': 'MDVP:Jitter(%)', 'label': 'Pitch Jitter (%)', 'shap_value': -0.015, 'impact': 'decrease', 'abs_value': 0.015},
        {'feature_name': 'HNR', 'label': 'Harmonics-to-Noise (HNR)', 'shap_value': -0.012, 'impact': 'decrease', 'abs_value': 0.012},
        {'feature_name': 'MDVP:Fo(Hz)', 'label': 'Average Pitch (F0)', 'shap_value': 0.005, 'impact': 'increase', 'abs_value': 0.005}
    ]
    
    output_dir = "tests/temp_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        pdf_path = generate_pdf_report(
            report_id=report_id,
            metrics=metrics,
            risk_score=risk_score,
            confidence_calibration=confidence_calibration,
            shap_explanation=shap_explanation,
            output_dir=output_dir
        )
        
        # Verify file creation
        assert os.path.exists(pdf_path), "PDF file was not created"
        assert os.path.getsize(pdf_path) > 0, "PDF file is empty"
        assert pdf_path.endswith("report_test_run_999.pdf")
        
    finally:
        # Cleanup
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

if __name__ == "__main__":
    pytest.main([__file__])
