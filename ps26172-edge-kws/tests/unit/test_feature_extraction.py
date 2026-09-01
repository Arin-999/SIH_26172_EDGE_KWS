"""
Unit tests for MFCC feature extraction.
"""

import numpy as np
import pytest

from ml.preprocessing.feature_extraction import (
    extract_mfcc,
    audio_to_features,
    _fix_frame_count,
    _normalize,
    NUM_FRAMES,
    NUM_MFCC,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def silence() -> np.ndarray:
    """1 second of silence."""
    return np.zeros(16_000, dtype=np.float32)


@pytest.fixture
def sine_440() -> np.ndarray:
    """1 second of 440 Hz sine wave."""
    t = np.linspace(0, 1.0, 16_000, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def white_noise() -> np.ndarray:
    """1 second of white noise."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(16_000).astype(np.float32) * 0.1


# ---------------------------------------------------------------------------
# Tests: extract_mfcc
# ---------------------------------------------------------------------------


class TestExtractMFCC:
    def test_output_shape(self, sine_440: np.ndarray) -> None:
        feat = extract_mfcc(sine_440)
        assert feat.shape == (NUM_FRAMES, NUM_MFCC, 1)

    def test_output_dtype(self, sine_440: np.ndarray) -> None:
        feat = extract_mfcc(sine_440)
        assert feat.dtype == np.float32

    def test_silence_is_finite(self, silence: np.ndarray) -> None:
        feat = extract_mfcc(silence)
        assert np.all(np.isfinite(feat))

    def test_sine_is_finite(self, sine_440: np.ndarray) -> None:
        feat = extract_mfcc(sine_440)
        assert np.all(np.isfinite(feat))

    def test_noise_is_finite(self, white_noise: np.ndarray) -> None:
        feat = extract_mfcc(white_noise)
        assert np.all(np.isfinite(feat))

    def test_silence_and_sine_differ(
        self, silence: np.ndarray, sine_440: np.ndarray
    ) -> None:
        """Different audio should produce different MFCC features."""
        feat_silence = extract_mfcc(silence)
        feat_sine = extract_mfcc(sine_440)
        # Features should differ by more than numerical noise
        assert float(np.abs(feat_silence - feat_sine).mean()) > 0.01


# ---------------------------------------------------------------------------
# Tests: audio_to_features
# ---------------------------------------------------------------------------


class TestAudioToFeatures:
    def test_same_as_extract_mfcc(self, sine_440: np.ndarray) -> None:
        a = extract_mfcc(sine_440)
        b = audio_to_features(sine_440)
        np.testing.assert_array_equal(a, b)

    def test_shape(self, white_noise: np.ndarray) -> None:
        feat = audio_to_features(white_noise)
        assert feat.shape == (NUM_FRAMES, NUM_MFCC, 1)


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_fix_frame_count_trim(self) -> None:
        mfcc = np.ones((100, NUM_MFCC), dtype=np.float32)
        result = _fix_frame_count(mfcc, NUM_FRAMES)
        assert result.shape == (NUM_FRAMES, NUM_MFCC)

    def test_fix_frame_count_pad(self) -> None:
        mfcc = np.ones((10, NUM_MFCC), dtype=np.float32)
        result = _fix_frame_count(mfcc, NUM_FRAMES)
        assert result.shape == (NUM_FRAMES, NUM_MFCC)
        # Padding rows should be zeros
        assert float(result[10:].sum()) == 0.0

    def test_normalize_mean_zero(self) -> None:
        mfcc = np.random.randn(NUM_FRAMES, NUM_MFCC).astype(np.float32)
        result = _normalize(mfcc)
        assert float(np.abs(result.mean())) < 0.1  # Approximate

    def test_normalize_finite(self) -> None:
        mfcc = np.ones((NUM_FRAMES, NUM_MFCC), dtype=np.float32)
        result = _normalize(mfcc)
        assert np.all(np.isfinite(result))
