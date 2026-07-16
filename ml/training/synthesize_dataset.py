import os
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, lfilter

def synthesize_voice_cycle(f0_mean, jitter_pct, shimmer_local, hnr_db, duration=3.0, sr=16000):
    """
    Synthesizes a vowel phonation (vocal tract filtered glottal-like pulse train)
    using physical parameter mappings of F0, Jitter, Shimmer, and HNR.
    Avoids quantization noise by performing continuous phase-based lookup.
    """
    total_samples = int(duration * sr)
    t = np.arange(total_samples) / sr
    
    mean_period = 1.0 / f0_mean
    periods = []
    amplitudes = []
    
    current_time = 0.0
    while current_time < duration + 0.5:
        # In parkinsons.data, jitter_pct is stored as a fraction (e.g. 0.00784)
        jitter_dev = jitter_pct * mean_period
        t_i = mean_period + np.random.normal(0, jitter_dev)
        t_i = np.clip(t_i, 1.0/600.0, 1.0/50.0)
        
        # shimmer_local is stored as a fraction (e.g. 0.04374)
        shimmer_dev = shimmer_local * 0.5
        a_i = 1.0 + np.random.normal(0, shimmer_dev)
        a_i = np.clip(a_i, 0.1, 2.0)
        
        periods.append(t_i)
        amplitudes.append(a_i)
        current_time += t_i
        
    t_starts = np.cumsum([0.0] + periods)
    
    # Continuous phase lookup: find which cycle each sample belongs to
    cycle_indices = np.searchsorted(t_starts, t) - 1
    cycle_indices = np.clip(cycle_indices, 0, len(periods) - 1)
    
    t_start = t_starts[cycle_indices]
    t_i_arr = np.array(periods)[cycle_indices]
    a_i_arr = np.array(amplitudes)[cycle_indices]
    
    t_rel = t - t_start
    theta = t_rel / t_i_arr
    
    # Rosenberg glottal pulse waveform
    y = np.zeros(total_samples)
    
    # Open phase (theta < 0.6)
    open_mask = (theta < 0.6)
    y[open_mask] = 3 * (theta[open_mask] / 0.6)**2 - 2 * (theta[open_mask] / 0.6)**3
    
    # Return phase (0.6 <= theta < 0.75)
    return_mask = (theta >= 0.6) & (theta < 0.75)
    y[return_mask] = 1.0 - ((theta[return_mask] - 0.6) / 0.15)
    
    # Closed phase (theta >= 0.75) is already 0
    
    # Scale amplitudes
    y = y * a_i_arr
    
    # 3. Add white noise according to HNR
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
        r = np.exp(-np.pi * b / sr)
        theta_f = 2 * np.pi * f / sr
        a = [1.0, -2.0 * r * np.cos(theta_f), r * r]
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
