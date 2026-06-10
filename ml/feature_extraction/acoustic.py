import numpy as np
import librosa
from scipy.signal import find_peaks, lfilter

def estimate_f0(y, sr, fmin=75, fmax=500):
    """
    Estimates the fundamental frequency (F0) contour using Librosa's YIN.
    """
    f0 = librosa.yin(y, sr=sr, fmin=fmin, fmax=fmax)
    # Replace NaNs or infinite values with 0
    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    voiced_f0 = f0[f0 > 0]
    
    if len(voiced_f0) == 0:
        return 0.0, 0.0, 0.0, np.zeros_like(f0)
        
    fo_mean = np.mean(voiced_f0)
    fhi = np.max(voiced_f0)
    flo = np.min(voiced_f0)
    return fo_mean, fhi, flo, f0

def extract_pitch_periods(y, sr, mean_f0):
    """
    Extracts cycle-by-cycle pitch periods (T_i) and amplitudes (A_i) from voiced segments.
    Applies a bandpass filter around mean F0 and finds peaks.
    """
    if mean_f0 <= 0 or len(y) < 100:
        return np.array([]), np.array([])
        
    # Bandpass filter the signal around F0 to isolate the fundamental component
    # Center frequency is mean_f0. Passband is [0.7 * F0, 1.4 * F0]
    nyquist = sr / 2
    low = max(50.0, 0.7 * mean_f0)
    high = min(nyquist - 10, 1.4 * mean_f0)
    
    # Simple butterworth filter using scipy.signal
    # To keep dependencies light, we can use a basic windowed sync filter or scipy filter
    from scipy.signal import butter, filtfilt
    try:
        b, a = butter(2, [low / nyquist, high / nyquist], btype='band')
        y_filtered = filtfilt(b, a, y)
    except Exception:
        # Fallback to unfiltered signal if butterworth fails
        y_filtered = y
        
    # Find peaks representing the cycle pulses
    # Min distance between peaks is approximately 80% of mean period
    min_dist_samples = int(0.8 * sr / mean_f0)
    peaks, _ = find_peaks(y_filtered, distance=min_dist_samples, prominence=0.01)
    
    if len(peaks) < 5:
        return np.array([]), np.array([])
        
    # Pitch periods T_i (in seconds)
    periods = np.diff(peaks) / sr
    
    # Amplitudes A_i (peak heights)
    amplitudes = y_filtered[peaks]
    
    return periods, amplitudes

def calculate_jitter(periods):
    """
    Calculates Jitter metrics from cycle periods.
    """
    if len(periods) < 5:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    mean_period = np.mean(periods)
    if mean_period == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    # 1. Jitter (Abs) in seconds
    jitter_abs = np.mean(np.abs(np.diff(periods)))
    
    # 2. Jitter (%)
    jitter_pct = (jitter_abs / mean_period) * 100
    
    # 3. RAP (Relative Average Perturbation)
    # Average difference from 3-point moving average
    rap_numerator = []
    for i in range(1, len(periods) - 1):
        local_avg = (periods[i-1] + periods[i] + periods[i+1]) / 3
        rap_numerator.append(abs(periods[i] - local_avg))
    rap = np.mean(rap_numerator) / mean_period if len(rap_numerator) > 0 else 0.0
    
    # 4. PPQ (Pitch Period Perturbation Quotient, 5-point)
    ppq_numerator = []
    for i in range(2, len(periods) - 2):
        local_avg = (periods[i-2] + periods[i-1] + periods[i] + periods[i+1] + periods[i+2]) / 5
        ppq_numerator.append(abs(periods[i] - local_avg))
    ppq = np.mean(ppq_numerator) / mean_period if len(ppq_numerator) > 0 else 0.0
    
    # 5. DDP (Difference of Differences of Periods)
    # Defined as 3 * RAP in standard Praat/OpenSMILE
    ddp = 3 * rap
    
    return jitter_pct, jitter_abs, rap, ppq, ddp

def calculate_shimmer(amplitudes):
    """
    Calculates Shimmer metrics from peak amplitudes.
    """
    # Filter out zeros or negatives
    amplitudes = amplitudes[amplitudes > 0]
    if len(amplitudes) < 5:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    mean_amp = np.mean(amplitudes)
    if mean_amp == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    # 1. Shimmer (local)
    shimmer_local = np.mean(np.abs(np.diff(amplitudes))) / mean_amp
    
    # 2. Shimmer (dB)
    shimmer_db_terms = []
    for i in range(len(amplitudes) - 1):
        val = abs(20 * np.log10(amplitudes[i+1] / amplitudes[i]))
        shimmer_db_terms.append(val)
    shimmer_db = np.mean(shimmer_db_terms)
    
    # 3. APQ3 (3-point Amplitude Perturbation Quotient)
    apq3_num = []
    for i in range(1, len(amplitudes) - 1):
        local_avg = (amplitudes[i-1] + amplitudes[i] + amplitudes[i+1]) / 3
        apq3_num.append(abs(amplitudes[i] - local_avg))
    apq3 = np.mean(apq3_num) / mean_amp if len(apq3_num) > 0 else 0.0
    
    # 4. APQ5 (5-point Amplitude Perturbation Quotient)
    apq5_num = []
    for i in range(2, len(amplitudes) - 2):
        local_avg = (amplitudes[i-2] + amplitudes[i-1] + amplitudes[i] + amplitudes[i+1] + amplitudes[i+2]) / 5
        apq5_num.append(abs(amplitudes[i] - local_avg))
    apq5 = np.mean(apq5_num) / mean_amp if len(apq5_num) > 0 else 0.0
    
    # 5. APQ (11-point Amplitude Perturbation Quotient)
    apq_num = []
    for i in range(5, len(amplitudes) - 5):
        local_avg = np.mean(amplitudes[i-5:i+6])
        apq_num.append(abs(amplitudes[i] - local_avg))
    apq = np.mean(apq_num) / mean_amp if len(apq_num) > 0 else 0.0
    
    # 6. DDA
    # Defined as 3 * APQ3
    dda = 3 * apq3
    
    return shimmer_local, shimmer_db, apq3, apq5, apq, dda

def calculate_hnr_nhr(y, sr, mean_f0):
    """
    Computes Harmonics-to-Noise Ratio (HNR) and Noise-to-Harmonics Ratio (NHR)
    using autocorrelative method.
    """
    if mean_f0 <= 0:
        return 0.0, 99.0
        
    # Calculate autocorrelation
    r = librosa.autocorrelate(y, max_size=int(sr/50)) # up to 50Hz period
    
    # Find peak corresponding to F0 period
    min_lag = int(sr / 500) # 500Hz limit
    max_lag = int(sr / 75)  # 75Hz limit
    
    if len(r) <= max_lag:
        return 15.0, 0.03 # Default fallback values
        
    r_search = r[min_lag:max_lag]
    peak_lag = np.argmax(r_search) + min_lag
    
    r_xx = r[0]
    r_ac = r[peak_lag]
    
    if r_xx == r_ac or r_xx <= 0 or (r_xx - r_ac) <= 0:
        return 15.0, 0.03 # Fallback
        
    # NHR = (R(0) - R(tau)) / R(tau)
    nhr = (r_xx - r_ac) / r_ac
    # HNR = 10 * log10( R(tau) / (R(0) - R(tau)) )
    hnr = 10 * np.log10(r_ac / (r_xx - r_ac))
    
    # Bound values to reasonable physical limits
    hnr = np.clip(hnr, 0, 40)
    nhr = np.clip(nhr, 0.001, 10.0)
    
    return hnr, nhr

def estimate_formants(y, sr):
    """
    Estimates the first three formants (F1, F2, F3) using LPC root-finding.
    """
    # Pre-emphasis filter to boost higher frequencies
    y_pre = lfilter([1, -0.97], 1, y)
    
    # Find LPC coefficients (order should be 2 + sr / 1000)
    n_coeff = 2 + int(sr / 1000)
    a = librosa.lpc(y_pre, order=n_coeff)
    
    # Find roots of the LPC polynomial
    roots = np.roots(a)
    roots = [r for r in roots if np.imag(r) >= 0]
    
    # Convert roots to angles/frequencies
    freqs = sorted([np.arctan2(np.imag(r), np.real(r)) * sr / (2 * np.pi) for r in roots])
    
    # Find formants (frequencies > 250Hz)
    formants = [f for f in freqs if f > 250]
    
    # Fill in defaults if not found
    f1 = formants[0] if len(formants) > 0 else 500.0
    f2 = formants[1] if len(formants) > 1 else 1500.0
    f3 = formants[2] if len(formants) > 2 else 2500.0
    
    return f1, f2, f3

def extract_all_acoustic_features(y, sr):
    """
    Runs the complete acoustic biomarker extraction pipeline.
    Returns a dictionary of clinical parameters and MFCC array.
    """
    # 1. Pitch F0
    fo_mean, fhi, flo, f0_contour = estimate_f0(y, sr)
    
    # 2. Jitter & Shimmer
    periods, amplitudes = extract_pitch_periods(y, sr, fo_mean)
    jit_pct, jit_abs, rap, ppq, ddp = calculate_jitter(periods)
    shim, shim_db, apq3, apq5, apq, dda = calculate_shimmer(amplitudes)
    
    # 3. HNR & NHR
    hnr, nhr = calculate_hnr_nhr(y, sr, fo_mean)
    
    # 4. Formants
    f1, f2, f3 = estimate_formants(y, sr)
    
    # 5. MFCCs (Mel-frequency cepstral coefficients)
    # Extract 13 MFCCs, mean and std across frames
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    
    # 6. RMS Energy
    rms = librosa.feature.rms(y=y)
    energy_mean = np.mean(rms)
    
    # 7. New Spectral Features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = np.mean(bandwidth)
    
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = np.mean(zcr)
    
    # 8. Chroma Features (12 pitch classes)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    chroma_means = np.mean(chroma, axis=1)
    
    # Compile Oxford format dictionary matching parkinsons.data columns
    features = {
        'MDVP:Fo(Hz)': fo_mean,
        'MDVP:Fhi(Hz)': fhi,
        'MDVP:Flo(Hz)': flo,
        'MDVP:Jitter(%)': jit_pct,
        'MDVP:Jitter(Abs)': jit_abs,
        'MDVP:RAP': rap,
        'MDVP:PPQ': ppq,
        'Jitter:DDP': ddp,
        'MDVP:Shimmer': shim,
        'MDVP:Shimmer(dB)': shim_db,
        'Shimmer:APQ3': apq3,
        'Shimmer:APQ5': apq5,
        'MDVP:APQ': apq,
        'Shimmer:DDA': dda,
        'NHR': nhr,
        'HNR': hnr,
        'Energy': energy_mean,
        'F1': f1,
        'F2': f2,
        'F3': f3,
        'Spectral_Centroid': centroid_mean,
        'Spectral_Bandwidth': bandwidth_mean,
        'Zero_Crossing_Rate': zcr_mean,
    }
    
    # Add MFCCs explicitly
    for i, m_val in enumerate(mfcc_means):
        features[f'MFCC_{i+1}'] = m_val
        
    # Add Chroma features explicitly
    for i, c_val in enumerate(chroma_means):
        features[f'Chroma_{i+1}'] = c_val
        
    return features
