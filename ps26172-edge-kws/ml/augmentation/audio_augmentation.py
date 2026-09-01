"""
Audio augmentation pipeline for KWS training.

Applies on-the-fly augmentations to 1-second 16 kHz float32 audio arrays
to improve model robustness. Uses the `audiomentations` library.

Augmentations applied:
  - Additive Gaussian noise (SNR 5–20 dB)
  - Time shift (±100 ms)
  - Speed perturbation (0.9–1.1×)
  - Room impulse response convolution (if IR files available)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

try:
    import audiomentations as am
    _HAS_AUDIOMENTATIONS = True
except ImportError:
    _HAS_AUDIOMENTATIONS = False

SAMPLE_RATE: int = 16_000


# ---------------------------------------------------------------------------
# Augmentation pipeline builder
# ---------------------------------------------------------------------------


def build_augmenter(
    noise_min_snr_db: float = 5.0,
    noise_max_snr_db: float = 20.0,
    time_shift_max_ms: float = 100.0,
    speed_min: float = 0.9,
    speed_max: float = 1.1,
    ir_dir: str | None = None,
    p: float = 0.5,
) -> "am.Compose | _FallbackAugmenter":
    """Build an audiomentations augmentation pipeline.

    Args:
        noise_min_snr_db: Minimum SNR for additive noise (dB).
        noise_max_snr_db: Maximum SNR for additive noise (dB).
        time_shift_max_ms: Maximum time shift in milliseconds.
        speed_min: Minimum speed perturbation factor.
        speed_max: Maximum speed perturbation factor.
        ir_dir: Optional path to directory of room impulse response WAV files.
        p: Probability that each augmentation is applied.

    Returns:
        Callable augmenter that accepts (audio: np.ndarray, sample_rate: int)
        and returns augmented np.ndarray of the same shape.
    """
    if not _HAS_AUDIOMENTATIONS:
        return _FallbackAugmenter(
            noise_min_snr_db=noise_min_snr_db,
            noise_max_snr_db=noise_max_snr_db,
            time_shift_max_ms=time_shift_max_ms,
            speed_min=speed_min,
            speed_max=speed_max,
        )

    transforms = [
        am.AddGaussianNoise(
            min_snr_db=noise_min_snr_db,
            max_snr_db=noise_max_snr_db,
            p=p,
        ),
        am.Shift(
            min_shift=-time_shift_max_ms / 1000.0,
            max_shift=time_shift_max_ms / 1000.0,
            shift_unit="fraction",
            rollover=False,
            p=p,
        ),
        am.TimeStretch(
            min_rate=speed_min,
            max_rate=speed_max,
            leave_length_unchanged=True,
            p=p,
        ),
    ]

    if ir_dir is not None and Path(ir_dir).exists():
        transforms.append(
            am.ApplyImpulseResponse(
                ir_paths=ir_dir,
                p=p * 0.5,  # Apply RIR less frequently
            )
        )

    return am.Compose(transforms)


def augment(
    audio: np.ndarray,
    augmenter: "am.Compose | _FallbackAugmenter",
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Apply augmentation to a single audio clip.

    Args:
        audio: 1-D float32 array of shape (16000,).
        augmenter: Pipeline built by `build_augmenter`.
        sample_rate: Sample rate in Hz.

    Returns:
        Augmented float32 array of same shape as input.
    """
    out = augmenter(samples=audio, sample_rate=sample_rate)
    # Ensure output length is preserved
    n = len(audio)
    if len(out) > n:
        out = out[:n]
    elif len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Fallback augmenter (no audiomentations dependency)
# ---------------------------------------------------------------------------


class _FallbackAugmenter:
    """Minimal numpy-only augmenter as fallback when audiomentations is absent."""

    def __init__(
        self,
        noise_min_snr_db: float = 5.0,
        noise_max_snr_db: float = 20.0,
        time_shift_max_ms: float = 100.0,
        speed_min: float = 0.9,
        speed_max: float = 1.1,
        p: float = 0.5,
    ) -> None:
        self.noise_min_snr_db = noise_min_snr_db
        self.noise_max_snr_db = noise_max_snr_db
        self.time_shift_max_samples = int(time_shift_max_ms * SAMPLE_RATE / 1000)
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.p = p

    def __call__(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        audio = samples.copy()
        n = len(audio)

        # Additive Gaussian noise
        if random.random() < self.p:
            snr_db = random.uniform(self.noise_min_snr_db, self.noise_max_snr_db)
            signal_power = float(np.mean(audio ** 2)) + 1e-12
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.randn(n).astype(np.float32) * np.sqrt(noise_power)
            audio = audio + noise

        # Time shift
        if random.random() < self.p:
            shift = random.randint(-self.time_shift_max_samples, self.time_shift_max_samples)
            audio = np.roll(audio, shift)
            if shift > 0:
                audio[:shift] = 0.0
            elif shift < 0:
                audio[shift:] = 0.0

        # Clip to [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(np.float32)
