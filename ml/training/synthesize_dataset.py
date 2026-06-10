import os
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, lfilter

def synthesize_voice_cycle(f0_mean, jitter_pct, shimmer_local, hnr_db, duration=3.0, sr=16000):
    """
    Synthesizes a vowel phonation (vocal tract filtered glottal-like pulse train)
    using physical parameter mappings of F0, Jitter, Shimmer, and HNR.
    """
    dt = 1.0 / sr
    total_samples = int(duration * sr)
    
    # 1. Generate cycle-by-cycle periods T_i and amplitudes A_i
    mean_period = 1.0 / f0_mean
    periods = []
    amplitudes = []
    
    current_time = 0.0
    # Add random walk or gaussian jitter/shimmer
    while current_time < duration:
        # Jitter: cycle period perturbation
        # jitter_pct is MDVP:Jitter(%), so deviation standard deviation is proportional
        jitter_dev = (jitter_pct / 100.0) * mean_period
        t_i = mean_period + np.random.normal(0, jitter_dev)
        # Keep period physically bounds (50Hz to 600Hz)
        t_i = np.clip(t_i, 1.0/600.0, 1.0/50.0)
        
        # Shimmer: amplitude perturbation
        # shimmer_local is MDVP:Shimmer, which is absolute shimmer amplitude perturbation
        shimmer_dev = shimmer_local * 0.5
        a_i = 1.0 + np.random.normal(0, shimmer_dev)
        a_i = np.clip(a_i, 0.1, 2.0)
        
        periods.append(t_i)
        amplitudes.append(a_i)
        current_time += t_i
        
    # 2. Synthesize glottal pulse train
    y = np.zeros(total_samples)
    idx = 0
    
    for t_i, a_i in zip(periods, amplitudes):
        cycle_samples = int(t_i * sr)
        if cycle_samples <= 0:
            continue
        if idx + cycle_samples >= total_samples:
            break
            
        # Glottal pulse approximation: Liljencrants-Fant (LF) model simplified
        # We can use a windowed sine wave segment or Rosenberg glottal pulse
        # Rosenberg pulse: y(t) = 3*(t/Tp)^2 - 2*(t/Tp)^3 during open phase, then decays
        tp = int(0.6 * cycle_samples) # Open phase
        tr = int(0.15 * cycle_samples) # Return phase
        
        cycle_waveform = np.zeros(cycle_samples)
        for t_s in range(cycle_samples):
            if t_s < tp:
                cycle_waveform[t_s] = 3 * (t_s / tp)**2 - 2 * (t_s / tp)**3
            elif t_s < tp + tr:
                cycle_waveform[t_s] = 1 - ((t_s - tp) / tr)
            # remaining closed phase is 0
            
        # Scale amplitude
        y[idx : idx + cycle_samples] = a_i * cycle_waveform
        idx += cycle_samples
        
    # 3. Add white noise according to HNR
    # HNR = 10 * log10(signal_power / noise_power)
    # => noise_power = signal_power * 10^(-HNR/10)
    sig_power = np.mean(y**2)
    if sig_power > 0:
        noise_power = sig_power * (10 ** (-hnr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        y_noise = y + noise
    else:
        y_noise = y
        
    # 4. Formant resonator filter (Vocal Tract model for vowel "ah")
    # Formants for "ah": F1=730Hz, F2=1090Hz, F3=2440Hz
    # Standard bandwidths: B1=50Hz, B2=70Hz, B3=110Hz
    formants = [730.0, 1090.0, 2440.0]
    bandwidths = [50.0, 70.0, 110.0]
    
    y_filtered = y_noise
    for f, b in zip(formants, bandwidths):
        # Design a 2nd order resonator bandpass filter
        r = np.exp(-np.pi * b / sr)
        theta = 2 * np.pi * f / sr
        a = [1.0, -2.0 * r * np.cos(theta), r * r]
        b_coef = [1.0 - r] # normalization
        y_filtered = lfilter(b_coef, a, y_filtered)
        
    # Normalize final waveform
    max_val = np.max(np.abs(y_filtered))
    if max_val > 0:
        y_filtered = y_filtered / max_val
        
    return y_filtered

def synthesize_dataset(data_path="datasets/parkinsons.data", output_dir="datasets/synthesized_wavs"):
    """
    Reads the CSV dataset and synthesizes a WAV file for each row.
    """
    print(f"Reading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Synthesizing voice samples (this will take a few seconds)...")
    for idx, row in df.iterrows():
        # Get patient name/recording identifier
        name = row['name']
        
        # Extract features
        f0_mean = row['MDVP:Fo(Hz)']
        jitter_pct = row['MDVP:Jitter(%)']
        shimmer_local = row['MDVP:Shimmer']
        hnr_db = row['HNR']
        
        # Synthesize audio
        y = synthesize_voice_cycle(f0_mean, jitter_pct, shimmer_local, hnr_db)
        
        # Save file
        out_file = os.path.join(output_dir, f"{name}.wav")
        sf.write(out_file, y, 16000)
        
    print(f"Synthesis complete. Created {len(df)} voice samples in '{output_dir}'.")

if __name__ == "__main__":
    synthesize_dataset()
