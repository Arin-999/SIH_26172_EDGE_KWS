"""
PCM audio reconstruction and validation utilities.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
MIN_DURATION_S = 0.2
MAX_DURATION_S = 30.0


def pcm_bytes_to_numpy(raw: bytes, dtype: np.dtype = np.int16) -> np.ndarray:
    """Convert raw PCM bytes to a numpy float32 array.

    Args:
        raw: Raw bytes containing 16-bit signed PCM samples (little-endian).
        dtype: Source dtype. Default int16.

    Returns:
        Float32 array with values in [-1.0, 1.0].
    """
    samples = np.frombuffer(raw, dtype=dtype)
    # Normalize int16 → float32
    if dtype == np.int16:
        return samples.astype(np.float32) / 32768.0
    return samples.astype(np.float32)


def numpy_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert a float32 numpy array back to int16 PCM bytes.

    Args:
        audio: Float32 array with values in [-1.0, 1.0].

    Returns:
        Raw bytes containing 16-bit signed PCM samples.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def validate_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    min_duration_s: float = MIN_DURATION_S,
    max_duration_s: float = MAX_DURATION_S,
) -> list[str]:
    """Validate a float32 audio array for ASR input quality.

    Args:
        audio: Float32 audio array.
        sample_rate: Expected sample rate.
        min_duration_s: Minimum acceptable duration in seconds.
        max_duration_s: Maximum acceptable duration in seconds.

    Returns:
        List of issue strings. Empty list = valid.
    """
    issues: list[str] = []

    duration_s = len(audio) / sample_rate
    if duration_s < min_duration_s:
        issues.append(f"Audio too short: {duration_s:.3f}s < {min_duration_s}s")
    if duration_s > max_duration_s:
        issues.append(f"Audio too long: {duration_s:.1f}s > {max_duration_s}s")

    peak = float(np.abs(audio).max())
    if peak < 1e-6:
        issues.append("Audio is silent (peak < 1e-6)")
    if peak > 1.0:
        issues.append(f"Audio is clipped (peak={peak:.3f} > 1.0)")

    # Check for mostly-silence (RMS < -50 dBFS)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-4:
        issues.append(f"Audio RMS too low: {rms:.6f} (likely silence)")

    return issues
