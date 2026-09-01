"""
Audio preprocessing for KWS pipeline.

Loads raw WAV/FLAC audio files, normalizes, and prepares them for
MFCC feature extraction. All audio is converted to 16 kHz mono float32.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE: int = 16_000
TARGET_DURATION_S: float = 1.0
TARGET_SAMPLES: int = int(TARGET_SAMPLE_RATE * TARGET_DURATION_S)  # 16000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file and return (samples, sample_rate).

    Supports WAV, FLAC, OGG and any format supported by soundfile/librosa.

    Args:
        path: Absolute or relative path to the audio file.

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is float32
        with values in [-1, 1] and sample_rate is the native sample rate.
    """
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        # Fallback to librosa for unsupported formats
        audio, sr = librosa.load(path, sr=None, mono=True, dtype=np.float32)
        return audio, sr

    # Convert multi-channel to mono by averaging
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    return audio, sr


def resample(audio: np.ndarray, from_sr: int, to_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample audio from `from_sr` to `to_sr`.

    Args:
        audio: 1-D float32 audio array.
        from_sr: Source sample rate in Hz.
        to_sr: Target sample rate in Hz. Defaults to TARGET_SAMPLE_RATE.

    Returns:
        Resampled float32 audio array.
    """
    if from_sr == to_sr:
        return audio
    return librosa.resample(audio, orig_sr=from_sr, target_sr=to_sr)


def normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to peak amplitude of 1.0.

    If the audio is silent (all zeros), returns it unchanged.

    Args:
        audio: 1-D float32 audio array.

    Returns:
        Amplitude-normalized float32 array with values in [-1, 1].
    """
    peak = np.abs(audio).max()
    if peak < 1e-8:
        return audio
    return audio / peak


def trim_or_pad(audio: np.ndarray, target_samples: int = TARGET_SAMPLES) -> np.ndarray:
    """Trim or zero-pad audio to exactly `target_samples` length.

    Trimming takes the first `target_samples` samples.
    Padding appends zeros at the end.

    Args:
        audio: 1-D float32 audio array.
        target_samples: Desired length in samples.

    Returns:
        Float32 array of length exactly `target_samples`.
    """
    length = len(audio)
    if length >= target_samples:
        return audio[:target_samples]
    pad_width = target_samples - length
    return np.pad(audio, (0, pad_width), mode="constant")


def preprocess(path: str, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Full preprocessing pipeline for a single audio file.

    Steps:
        1. Load audio
        2. Resample to target sample rate
        3. Convert to mono
        4. Trim or pad to 1 second
        5. Normalize amplitude to [-1, 1]

    Args:
        path: Path to the audio file.
        sample_rate: Target sample rate. Defaults to 16 000 Hz.

    Returns:
        Preprocessed float32 audio array of shape (16000,).
    """
    audio, sr = load_audio(path)
    if sr != sample_rate:
        audio = resample(audio, from_sr=sr, to_sr=sample_rate)
    audio = trim_or_pad(audio, target_samples=int(sample_rate * TARGET_DURATION_S))
    audio = normalize(audio)
    return audio.astype(np.float32)
