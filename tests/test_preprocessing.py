import os
import sys
import numpy as np

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocessing.audio import (
    AudioResampler,
    SpectralNoiseReducer,
    VoiceActivityDetector,
    SilenceTrimmer,
    LoudnessNormalizer,
    AudioPreprocessingPipeline
)

def test_audio_resampler():
    print("Testing AudioResampler...")
    # Generate 1-second 22050Hz signal
    sr_orig = 22050
    t = np.linspace(0, 1.0, sr_orig)
    y_orig = np.sin(2 * np.pi * 440 * t)
    
    resampler = AudioResampler(target_sr=16000)
    y_res, sr_res = resampler.process(y_orig, sr_orig)
    
    assert sr_res == 16000
    assert len(y_res) == 16000
    print("[OK] AudioResampler verified.")

def test_loudness_normalizer():
    print("Testing LoudnessNormalizer...")
    y_raw = np.array([0.1, -0.2, 0.5, -0.4, 0.3])
    
    normalizer = LoudnessNormalizer(target_peak=0.95)
    y_norm = normalizer.process(y_raw)
    
    assert abs(np.max(np.abs(y_norm)) - 0.95) < 1e-6
    assert y_norm[2] == 0.95
    assert y_norm[1] == -0.38 # (-0.2 / 0.5) * 0.95
    print("[OK] LoudnessNormalizer verified.")

def test_silence_trimmer():
    print("Testing SilenceTrimmer...")
    sr = 16000
    # 0.5s silence, 1s tone, 0.5s silence
    y = np.concatenate([
        np.zeros(int(0.5 * sr)),
        np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr)),
        np.zeros(int(0.5 * sr))
    ])
    
    trimmer = SilenceTrimmer(top_db=30)
    y_trimmed = trimmer.process(y, sr)
    
    # Trimmed size should be smaller than original
    assert len(y_trimmed) < len(y)
    # The active tone part should remain
    assert len(y_trimmed) >= sr
    print("[OK] SilenceTrimmer verified.")

def test_voice_activity_detector():
    print("Testing VoiceActivityDetector...")
    sr = 16000
    
    # Voiced signal: highly periodic tone (low flatness, high energy)
    t = np.linspace(0, 0.5, int(0.5 * sr))
    voiced = np.sin(2 * np.pi * 150 * t)
    
    # Unvoiced signal: pure noise (high flatness, low energy)
    unvoiced = np.random.normal(0, 0.005, int(0.5 * sr))
    
    y = np.concatenate([voiced, unvoiced])
    
    # energy_threshold=0.02, flatness_threshold=0.30
    vad = VoiceActivityDetector(energy_threshold=0.02, flatness_threshold=0.30)
    y_voiced = inline_vad_test(y, sr, vad)
    
    # VAD should successfully extract the periodic section and discard the noise
    # Unfiltered concatenated frame length would be ~31,200 samples.
    # Voiced-only concatenated frame length should be around ~16,320.
    assert len(y_voiced) < 20000
    assert len(y_voiced) > 0
    print("[OK] VoiceActivityDetector verified.")

def inline_vad_test(y, sr, vad):
    return vad.process(y, sr)

if __name__ == "__main__":
    test_audio_resampler()
    test_loudness_normalizer()
    test_silence_trimmer()
    test_voice_activity_detector()
    print("\nAll modular preprocessing unit tests passed successfully!")
