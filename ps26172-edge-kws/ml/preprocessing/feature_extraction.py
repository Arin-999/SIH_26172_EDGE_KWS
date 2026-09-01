"""
Log-Mel MFCC feature extraction for the KWS pipeline.

Produces (49, 40, 1) float32 feature maps matching the DS-CNN input.
Parameters are driven by ml/training/config.yaml:
  - sample_rate: 16000
  - frame_length_ms: 30  → 480 samples
  - frame_stride_ms: 10  → 160 samples
  - num_coeffs: 40
  - num_frames: 49
"""

from __future__ import annotations

import numpy as np
import librosa


# ---------------------------------------------------------------------------
# Default feature parameters (must match config.yaml)
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16_000
FRAME_LENGTH_MS: int = 30
FRAME_STRIDE_MS: int = 10
NUM_MFCC: int = 40
NUM_FRAMES: int = 49
N_FFT: int = 512
N_MELS: int = 40


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_mfcc(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = NUM_MFCC,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    frame_length_ms: int = FRAME_LENGTH_MS,
    frame_stride_ms: int = FRAME_STRIDE_MS,
    num_frames: int = NUM_FRAMES,
) -> np.ndarray:
    """Extract MFCC features from a 1-second audio clip.

    Computes MFCCs using librosa, then normalizes per utterance (zero-mean,
    unit-variance) and reshapes to the DS-CNN input format.

    Args:
        audio: 1-D float32 audio array (should be exactly `sample_rate` samples).
        sample_rate: Sample rate in Hz.
        n_mfcc: Number of MFCC coefficients.
        n_fft: FFT window size in samples.
        n_mels: Number of Mel filterbank bins.
        frame_length_ms: STFT window length in ms.
        frame_stride_ms: STFT hop length in ms.
        num_frames: Expected number of time frames. Output will be trimmed/padded.

    Returns:
        Float32 array of shape (num_frames, n_mfcc, 1), normalized.
    """
    hop_length = int(sample_rate * frame_stride_ms / 1000)   # 160
    win_length = int(sample_rate * frame_length_ms / 1000)   # 480

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        window="hamming",
        center=True,
    )
    # mfcc shape: (n_mfcc, time_frames)

    # Transpose to (time_frames, n_mfcc)
    mfcc = mfcc.T

    # Trim or pad along time axis to exactly num_frames
    mfcc = _fix_frame_count(mfcc, num_frames)

    # Per-utterance mean-variance normalization
    mfcc = _normalize(mfcc)

    # Add channel dimension → (num_frames, n_mfcc, 1)
    mfcc = mfcc[:, :, np.newaxis]

    return mfcc.astype(np.float32)


def audio_to_features(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convenience wrapper: raw audio → MFCC feature map.

    This is the primary entry point used by the training pipeline and
    the personalization module.

    Args:
        audio: 1-D float32 audio array (16 000 samples at 16 kHz).
        sample_rate: Sample rate in Hz.

    Returns:
        Float32 array of shape (49, 40, 1).
    """
    return extract_mfcc(audio, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fix_frame_count(mfcc: np.ndarray, target: int) -> np.ndarray:
    """Trim or zero-pad the time axis of an MFCC array.

    Args:
        mfcc: 2-D array of shape (time, coeffs).
        target: Desired number of time frames.

    Returns:
        Array of shape (target, coeffs).
    """
    n = mfcc.shape[0]
    if n >= target:
        return mfcc[:target, :]
    pad = np.zeros((target - n, mfcc.shape[1]), dtype=mfcc.dtype)
    return np.concatenate([mfcc, pad], axis=0)


def _normalize(mfcc: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-utterance mean-variance normalization.

    Args:
        mfcc: 2-D array of shape (frames, coeffs).
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized array of same shape.
    """
    mean = mfcc.mean()
    std = mfcc.std()
    return (mfcc - mean) / (std + eps)
