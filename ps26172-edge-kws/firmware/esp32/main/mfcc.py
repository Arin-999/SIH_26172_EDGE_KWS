"""
MFCC feature extraction for KWS.

C-equivalent Python reference implementation of the MCU-side MFCC pipeline.
Used for validation and the Python ESP32 simulator.

Matches: firmware/esp32/main/mfcc.c
"""

from __future__ import annotations

import numpy as np

# Must match config.yaml and firmware menuconfig
SAMPLE_RATE: int = 16_000
FRAME_LENGTH_SAMPLES: int = 480   # 30 ms
FRAME_STRIDE_SAMPLES: int = 160   # 10 ms
NUM_FRAMES: int = 49
NUM_MFCC: int = 40
N_FFT: int = 512
N_MELS: int = 40
PRE_EMPHASIS: float = 0.97


def compute_mfcc(audio: np.ndarray) -> np.ndarray:
    """Compute MFCC features from 1 second of 16 kHz audio.

    Matches the MCU-side fixed-point MFCC implementation closely enough
    for validation purposes. Uses float32 throughout.

    Args:
        audio: 1-D float32 array of shape (16000,).

    Returns:
        MFCC array of shape (NUM_FRAMES, NUM_MFCC), float32.
    """
    # Pre-emphasis filter: y[n] = x[n] - a * x[n-1]
    audio = _pre_emphasis(audio, PRE_EMPHASIS)

    # Hamming window
    window = np.hamming(FRAME_LENGTH_SAMPLES).astype(np.float32)

    # Frame the audio
    frames = _frame(audio, FRAME_LENGTH_SAMPLES, FRAME_STRIDE_SAMPLES)

    # Apply window
    frames = frames * window[np.newaxis, :]

    # FFT magnitude spectrum
    mag_spec = np.abs(np.fft.rfft(frames, n=N_FFT)).astype(np.float32)

    # Mel filterbank
    mel_fb = _mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE)
    mel_energy = np.maximum(mag_spec @ mel_fb.T, 1e-10)
    log_mel = np.log(mel_energy).astype(np.float32)

    # DCT-II to get MFCCs (only first NUM_MFCC coefficients)
    from scipy.fftpack import dct
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :NUM_MFCC]

    # Trim or pad to NUM_FRAMES
    n = mfcc.shape[0]
    if n >= NUM_FRAMES:
        mfcc = mfcc[:NUM_FRAMES]
    else:
        mfcc = np.pad(mfcc, ((0, NUM_FRAMES - n), (0, 0)))

    # Per-utterance mean-variance normalization
    mean = mfcc.mean()
    std = mfcc.std() + 1e-6
    mfcc = ((mfcc - mean) / std).astype(np.float32)

    return mfcc


def _pre_emphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return np.append(audio[0], audio[1:] - coeff * audio[:-1]).astype(np.float32)


def _frame(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Split audio into overlapping frames."""
    n_frames = 1 + (len(audio) - frame_len) // hop_len
    indices = (
        np.tile(np.arange(frame_len), (n_frames, 1)) +
        np.tile(np.arange(0, n_frames * hop_len, hop_len), (frame_len, 1)).T
    )
    return audio[indices].astype(np.float32)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Create a Mel filterbank matrix."""
    from librosa.filters import mel as librosa_mel
    return librosa_mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=20, fmax=8000).astype(np.float32)
