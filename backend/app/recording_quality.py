"""
Recording Quality Analyzer for VitaVoice.

Analyzes WAV audio files for recording quality metrics before inference.
Uses only librosa and numpy — does NOT modify or depend on the ML pipeline.
"""
import numpy as np
import librosa


def analyze_recording_quality(file_path: str, target_sr: int = 16000) -> dict:
    """
    Analyzes a WAV/MP3 audio file and returns quality metrics.
    
    Parameters
    ----------
    file_path : str
        Absolute path to the audio file.
    target_sr : int
        Target sample rate for analysis.
    
    Returns
    -------
    dict
        Recording quality metrics including noise, SNR, duration,
        speech coverage, silence ratio, clipping, and overall score.
    """
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception:
        return _default_quality_result()

    duration_seconds = len(y) / sr

    # --- Background noise & SNR ---
    frame_length = int(0.025 * sr)   # 25 ms frames
    hop_length = int(0.010 * sr)     # 10 ms hop
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    if len(rms) == 0:
        return _default_quality_result(duration=duration_seconds)

    # Classify frames as speech or silence using adaptive threshold
    rms_sorted = np.sort(rms)
    noise_floor = np.mean(rms_sorted[:max(1, int(len(rms_sorted) * 0.10))])
    speech_threshold = max(noise_floor * 3.0, 0.005)

    speech_frames = rms > speech_threshold
    speech_coverage = float(np.mean(speech_frames)) * 100.0
    silence_ratio = 100.0 - speech_coverage

    # Background noise percentage (relative to peak signal)
    peak_rms = np.max(rms) if np.max(rms) > 0 else 1e-10
    background_noise_pct = float(noise_floor / peak_rms) * 100.0
    background_noise_pct = min(100.0, max(0.0, background_noise_pct))

    # Signal-to-Noise Ratio
    if noise_floor > 0:
        speech_rms = np.mean(rms[speech_frames]) if np.any(speech_frames) else peak_rms
        snr_db = float(20.0 * np.log10(speech_rms / noise_floor))
        snr_db = max(0.0, min(60.0, snr_db))
    else:
        snr_db = 40.0  # Very clean recording

    # --- Clipping detection ---
    clipping_threshold = 0.99
    clipped_samples = int(np.sum(np.abs(y) >= clipping_threshold))
    clipping_ratio = clipped_samples / max(1, len(y))
    clipping_detected = clipping_ratio > 0.001  # More than 0.1% clipped

    # --- Microphone status ---
    if peak_rms < 0.001:
        mic_status = "No Signal"
    elif clipping_detected:
        mic_status = "Clipping Detected"
    elif snr_db < 10:
        mic_status = "Noisy"
    else:
        mic_status = "Good"

    # --- Overall quality score (1-5 stars) ---
    score = _compute_quality_score(
        snr_db=snr_db,
        speech_coverage=speech_coverage,
        duration=duration_seconds,
        clipping_detected=clipping_detected,
        background_noise_pct=background_noise_pct
    )

    # Suitability assessment
    suitable_for_analysis = score >= 2
    quality_warning = None
    if score <= 2:
        quality_warning = (
            "Recording quality is below the recommended threshold. "
            "Results may be less reliable."
        )

    return {
        "duration_seconds": round(duration_seconds, 1),
        "background_noise_pct": round(background_noise_pct, 1),
        "snr_db": round(snr_db, 1),
        "speech_coverage_pct": round(speech_coverage, 1),
        "silence_ratio_pct": round(silence_ratio, 1),
        "clipping_detected": clipping_detected,
        "mic_status": mic_status,
        "quality_score": score,
        "quality_stars": _score_to_stars(score),
        "suitable_for_analysis": suitable_for_analysis,
        "quality_warning": quality_warning,
    }


def _compute_quality_score(
    snr_db: float,
    speech_coverage: float,
    duration: float,
    clipping_detected: bool,
    background_noise_pct: float,
) -> int:
    """Compute a 1-5 star quality score from component metrics."""
    points = 0.0

    # SNR contribution (max 30 points)
    if snr_db >= 25:
        points += 30
    elif snr_db >= 15:
        points += 20
    elif snr_db >= 8:
        points += 10
    else:
        points += 2

    # Speech coverage contribution (max 25 points)
    if speech_coverage >= 60:
        points += 25
    elif speech_coverage >= 40:
        points += 18
    elif speech_coverage >= 20:
        points += 10
    else:
        points += 2

    # Duration contribution (max 20 points)
    if duration >= 10:
        points += 20
    elif duration >= 5:
        points += 12
    elif duration >= 2:
        points += 6
    else:
        points += 1

    # Noise floor contribution (max 15 points)
    if background_noise_pct <= 10:
        points += 15
    elif background_noise_pct <= 25:
        points += 10
    elif background_noise_pct <= 50:
        points += 5
    else:
        points += 1

    # Clipping penalty (max 10 points)
    if not clipping_detected:
        points += 10
    else:
        points += 0

    # Map 0-100 to 1-5 stars
    if points >= 85:
        return 5
    elif points >= 65:
        return 4
    elif points >= 45:
        return 3
    elif points >= 25:
        return 2
    else:
        return 1


def _score_to_stars(score: int) -> str:
    """Convert a 1-5 integer score to a star string."""
    filled = "★" * score
    empty = "☆" * (5 - score)
    return filled + empty


def _default_quality_result(duration: float = 0.0) -> dict:
    """Return a safe default quality result when analysis fails."""
    return {
        "duration_seconds": round(duration, 1),
        "background_noise_pct": 0.0,
        "snr_db": 0.0,
        "speech_coverage_pct": 0.0,
        "silence_ratio_pct": 100.0,
        "clipping_detected": False,
        "mic_status": "Unknown",
        "quality_score": 1,
        "quality_stars": "★☆☆☆☆",
        "suitable_for_analysis": False,
        "quality_warning": "Unable to analyze recording quality.",
    }
