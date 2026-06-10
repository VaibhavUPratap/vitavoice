import numpy as np
import librosa
import soundfile as sf

class AudioResampler:
    """
    Handles resampling of audio waveforms to a target sample rate.
    """
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        
    def process(self, y, sr):
        if sr == self.target_sr:
            return y, sr
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
        return y_resampled, self.target_sr

class SpectralNoiseReducer:
    """
    Applies spectral gating to reduce steady-state background noise.
    """
    def __init__(self, noise_clip_duration=0.4, prop_decrease=0.85):
        self.noise_clip_duration = noise_clip_duration
        self.prop_decrease = prop_decrease
        
    def process(self, y, sr):
        # Short-time Fourier transform (STFT)
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        stft_abs = np.abs(stft)
        
        # Estimate noise from the first part of the audio
        noise_frames = int(self.noise_clip_duration * sr / 512)
        if noise_frames < 2:
            noise_frames = 2
            
        noise_stft = stft[:, :noise_frames]
        noise_mean = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
        noise_std = np.std(np.abs(noise_stft), axis=1, keepdims=True)
        
        # Noise threshold: mean + 2 * std
        threshold = noise_mean + 2 * noise_std
        
        # Create mask: attenuate frequencies below threshold
        mask = stft_abs > threshold
        mask = mask.astype(float)
        
        # Apply mask with soft gating
        stft_clean = stft * (mask + (1 - mask) * (1 - self.prop_decrease))
        
        # Inverse STFT
        y_clean = librosa.istft(stft_clean, hop_length=512)
        return y_clean

class VoiceActivityDetector:
    """
    Filters out unvoiced and silent sections to extract active voiced phonation.
    Uses Short-Time RMS Energy and Spectral Flatness.
    """
    def __init__(self, frame_ms=30, stride_ms=15, energy_threshold=0.015, flatness_threshold=0.15):
        self.frame_ms = frame_ms
        self.stride_ms = stride_ms
        self.energy_threshold = energy_threshold
        self.flatness_threshold = flatness_threshold
        
    def process(self, y, sr):
        frame_len = int(self.frame_ms * sr / 1000)
        hop_len = int(self.stride_ms * sr / 1000)
        
        # Frame the signal
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
        
        voiced_frames = []
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            # 1. Compute RMS energy
            rms = np.sqrt(np.mean(frame**2))
            
            # 2. Compute Spectral Flatness (flatness closer to 1 means noise/whisper, closer to 0 means harmonic speech)
            flatness = librosa.feature.spectral_flatness(y=frame)[0][0]
            
            # Voiced speech has high energy and low flatness (harmonicity)
            if rms > self.energy_threshold and flatness < self.flatness_threshold:
                voiced_frames.append(frame)
                
        if len(voiced_frames) == 0:
            # Fallback: if VAD filters everything, return the original signal to prevent crashes
            return y
            
        # Reconstruct voiced audio (overlap-add or simple concatenation)
        # For vocal biomarkers, simple concatenation of speech frames is standard
        y_voiced = np.concatenate(voiced_frames)
        return y_voiced

class SilenceTrimmer:
    """
    Removes leading and trailing silence.
    """
    def __init__(self, top_db=30):
        self.top_db = top_db
        
    def process(self, y, sr):
        y_trimmed, _ = librosa.effects.trim(y, top_db=self.top_db)
        return y_trimmed

class LoudnessNormalizer:
    """
    Normalizes peak amplitude to prevent distance-to-microphone biases.
    """
    def __init__(self, target_peak=0.95):
        self.target_peak = target_peak
        
    def process(self, y):
        if len(y) == 0:
            return y
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y_normalized = (y / max_val) * self.target_peak
            return y_normalized
        return y

class AudioPreprocessingPipeline:
    """
    Orchestration pipeline executing all audio preprocessing stages.
    """
    def __init__(self, target_sr=16000):
        self.resampler = AudioResampler(target_sr)
        self.noise_reducer = SpectralNoiseReducer()
        self.vad = VoiceActivityDetector()
        self.trimmer = SilenceTrimmer()
        self.normalizer = LoudnessNormalizer()
        
    def preprocess_audio(self, file_path, output_path=None):
        # 1. Load original audio
        y_raw, sr_raw = librosa.load(file_path, sr=None, mono=True)
        
        # 2. Resample
        y_res, sr_res = self.resampler.process(y_raw, sr_raw)
        
        # 3. Noise reduction
        y_denoised = self.noise_reducer.process(y_res, sr_res)
        
        # 4. Voice Activity Detection
        y_voiced = self.vad.process(y_denoised, sr_res)
        
        # 5. Trim silence
        y_trimmed = self.trimmer.process(y_voiced, sr_res)
        
        # 6. Normalize loudness
        y_normalized = self.normalizer.process(y_trimmed)
        
        # Save output if specified
        if output_path:
            sf.write(output_path, y_normalized, sr_res)
            
        return y_normalized, sr_res
