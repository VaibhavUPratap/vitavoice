import os
import numpy as np
import soundfile as sf
import sys

# Add project root to python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocessing.audio import AudioPreprocessingPipeline
from ml.feature_extraction.acoustic import extract_all_acoustic_features

def test_preprocessing_and_extraction():
    print("Initializing test run...")
    temp_wav = "tests/temp_test.wav"
    os.makedirs("tests", exist_ok=True)
    
    # 1. Synthesize a dummy 3-second vowel-like signal (150Hz pitch)
    sr = 16000
    duration = 3.0
    t = np.arange(0, duration, 1.0/sr)
    
    # Add minor jitter and shimmer
    f0 = 150.0
    # Modulate phase for Jitter
    phase = 2 * np.pi * f0 * t + 0.05 * np.sin(2 * np.pi * 5.0 * t) 
    # Modulate amplitude for Shimmer
    amplitude = 0.5 + 0.04 * np.sin(2 * np.pi * 3.0 * t)
    
    # Simple glottal-like buzz (fundamental + 3 harmonics)
    signal = amplitude * (
        np.sin(phase) + 
        0.5 * np.sin(2 * phase) + 
        0.25 * np.sin(3 * phase) +
        0.125 * np.sin(4 * phase)
    )
    
    # Add background white noise (simulating ~25dB HNR)
    noise = np.random.normal(0, 0.01, len(signal))
    signal_with_noise = signal + noise
    
    # Save dummy wav
    sf.write(temp_wav, signal_with_noise, sr)
    print(f"Generated test audio at '{temp_wav}'")
    
    try:
        # 2. Test Preprocessing
        print("Testing preprocessing...")
        pipeline = AudioPreprocessingPipeline()
        y, processed_sr = pipeline.preprocess_audio(temp_wav)
        assert processed_sr == 16000, f"Expected 16kHz sample rate, got {processed_sr}"
        assert len(y) > 0, "Preprocessed audio signal is empty"
        print("[OK] Preprocessing successfully verified.")
        
        # 3. Test Feature Extraction
        print("Testing clinical feature extraction...")
        features = extract_all_acoustic_features(y, processed_sr)
        
        # Verify key clinical features are present and physical
        required_keys = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
            'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
            'NHR', 'HNR', 'Energy', 'F1', 'F2', 'F3',
            'MFCC_1', 'MFCC_13', 'Spectral_Centroid', 'Spectral_Bandwidth', 'Zero_Crossing_Rate',
            'Chroma_1', 'Chroma_12'
        ]
        
        for key in required_keys:
            assert key in features, f"Missing required feature key: {key}"
            
        print(f"[OK] Feature extraction successfully verified. Extracted {len(features)} parameters.")
        print(f"  F0 Mean:   {features['MDVP:Fo(Hz)']:.2f} Hz")
        print(f"  Jitter:    {features['MDVP:Jitter(%)']:.3f} %")
        print(f"  Shimmer:   {features['MDVP:Shimmer'] * 100.0:.3f} %")
        print(f"  HNR:       {features['HNR']:.2f} dB")
        
    finally:
        # Cleanup
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            print("Removed temporary test file.")
            
    print("\nAll unit tests passed successfully!")

if __name__ == "__main__":
    test_preprocessing_and_extraction()
