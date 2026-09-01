"""
Unit tests for audio preprocessing pipeline.
"""

import numpy as np
import pytest
import tempfile
import soundfile as sf
from pathlib import Path

from ml.preprocessing.audio_preprocessing import (
    load_audio,
    resample,
    normalize,
    trim_or_pad,
    preprocess,
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sine_wav(tmp_path: Path) -> str:
    """Create a 1-second 440 Hz sine wave WAV file."""
    sr = 16_000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = str(tmp_path / "sine.wav")
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def short_wav(tmp_path: Path) -> str:
    """Create a 0.3-second sine wave WAV file (shorter than target)."""
    sr = 16_000
    audio = 0.3 * np.ones(int(sr * 0.3), dtype=np.float32)
    path = str(tmp_path / "short.wav")
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def stereo_wav(tmp_path: Path) -> str:
    """Create a stereo (2-channel) WAV file."""
    sr = 16_000
    audio = np.random.randn(sr, 2).astype(np.float32) * 0.1
    path = str(tmp_path / "stereo.wav")
    sf.write(path, audio, sr)
    return path


# ---------------------------------------------------------------------------
# Tests: load_audio
# ---------------------------------------------------------------------------


class TestLoadAudio:
    def test_loads_wav(self, sine_wav: str) -> None:
        audio, sr = load_audio(sine_wav)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 16_000

    def test_converts_stereo_to_mono(self, stereo_wav: str) -> None:
        audio, sr = load_audio(stereo_wav)
        assert audio.ndim == 1

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(Exception):
            load_audio("/nonexistent/path/audio.wav")


# ---------------------------------------------------------------------------
# Tests: resample
# ---------------------------------------------------------------------------


class TestResample:
    def test_no_op_when_same_rate(self) -> None:
        audio = np.ones(16_000, dtype=np.float32)
        result = resample(audio, from_sr=16_000, to_sr=16_000)
        np.testing.assert_array_equal(result, audio)

    def test_downsample(self) -> None:
        audio = np.random.randn(48_000).astype(np.float32)
        result = resample(audio, from_sr=48_000, to_sr=16_000)
        assert len(result) == pytest.approx(16_000, abs=50)

    def test_upsample(self) -> None:
        audio = np.random.randn(8_000).astype(np.float32)
        result = resample(audio, from_sr=8_000, to_sr=16_000)
        assert len(result) == pytest.approx(16_000, abs=50)


# ---------------------------------------------------------------------------
# Tests: normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_peak_is_one(self) -> None:
        audio = np.array([0.2, -0.4, 0.1, 0.3], dtype=np.float32)
        result = normalize(audio)
        assert float(np.abs(result).max()) == pytest.approx(1.0, abs=1e-6)

    def test_silent_audio_unchanged(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = normalize(audio)
        np.testing.assert_array_equal(result, audio)

    def test_preserves_sign(self) -> None:
        audio = np.array([-0.5, 0.5], dtype=np.float32)
        result = normalize(audio)
        assert result[0] < 0
        assert result[1] > 0


# ---------------------------------------------------------------------------
# Tests: trim_or_pad
# ---------------------------------------------------------------------------


class TestTrimOrPad:
    def test_pads_short_audio(self) -> None:
        audio = np.ones(8_000, dtype=np.float32)
        result = trim_or_pad(audio, target_samples=16_000)
        assert len(result) == 16_000
        assert float(np.sum(result[8_000:])) == 0.0  # Padding is zeros

    def test_trims_long_audio(self) -> None:
        audio = np.ones(20_000, dtype=np.float32)
        result = trim_or_pad(audio, target_samples=16_000)
        assert len(result) == 16_000

    def test_exact_length_unchanged(self) -> None:
        audio = np.random.randn(16_000).astype(np.float32)
        result = trim_or_pad(audio, target_samples=16_000)
        np.testing.assert_array_equal(result, audio)


# ---------------------------------------------------------------------------
# Tests: preprocess (integration)
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_output_shape(self, sine_wav: str) -> None:
        result = preprocess(sine_wav)
        assert result.shape == (TARGET_SAMPLES,)

    def test_output_dtype(self, sine_wav: str) -> None:
        result = preprocess(sine_wav)
        assert result.dtype == np.float32

    def test_output_range(self, sine_wav: str) -> None:
        result = preprocess(sine_wav)
        assert float(np.abs(result).max()) <= 1.0 + 1e-6

    def test_short_audio_padded(self, short_wav: str) -> None:
        result = preprocess(short_wav)
        assert result.shape == (TARGET_SAMPLES,)
