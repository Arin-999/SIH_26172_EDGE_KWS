#!/usr/bin/env python3
"""
Basic KWS detection example — no hardware required.

Demonstrates the complete wake-word detection pipeline using a synthetic
audio signal. Replace `synthetic_audio()` with a real WAV file load to
test on actual speech.

Usage:
    python examples/basic-kws/detect.py
    python examples/basic-kws/detect.py --audio path/to/audio.wav
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.preprocessing.audio_preprocessing import preprocess, normalize, trim_or_pad, resample
from ml.preprocessing.feature_extraction import audio_to_features
from ml.personalization.matcher import match, DebounceMatcher


# ── Synthetic prototype (stand-in for an enrolled keyword) ───────────────────
rng = np.random.default_rng(42)
PROTOTYPE = rng.standard_normal(64).astype(np.float32)
PROTOTYPE /= np.linalg.norm(PROTOTYPE)


def synthetic_audio(duration_s: float = 3.0, sr: int = 16_000) -> np.ndarray:
    """Generate a simple synthetic audio signal for demo purposes."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)        # 440 Hz tone
    signal += 0.05 * rng.standard_normal(len(signal))  # add noise
    return signal.astype(np.float32)


def fake_embedding(audio_chunk: np.ndarray) -> np.ndarray:
    """
    Stand-in for TFLite inference. Returns a noisy version of the prototype
    so the matcher occasionally fires. Replace with extract_embedding() once
    you have a trained model.
    """
    noise = rng.standard_normal(64).astype(np.float32) * 0.3
    emb = PROTOTYPE + noise
    return emb / np.linalg.norm(emb)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Basic KWS detection example.")
    parser.add_argument("--audio", help="Path to WAV file (optional)")
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()

    print("=== Basic KWS Detection Example ===")
    print(f"Threshold: {args.threshold}")

    if args.audio:
        print(f"Loading: {args.audio}")
        audio = preprocess(args.audio)
    else:
        print("Using synthetic 3-second audio (440 Hz tone + noise)")
        audio = synthetic_audio(3.0)

    CHUNK = 4_000  # 250 ms @ 16 kHz
    matcher = DebounceMatcher(PROTOTYPE, threshold=args.threshold, hits_required=2, window_size=3)

    print(f"\nProcessing {len(audio)/16000:.2f}s in {CHUNK}-sample chunks ...\n")
    wake_count = 0

    for i in range(0, len(audio), CHUNK):
        chunk = audio[i: i + CHUNK]
        if len(chunk) < CHUNK:
            chunk = np.pad(chunk, (0, CHUNK - len(chunk)))

        emb = fake_embedding(chunk)
        wake, score = matcher.update(emb)
        t = i / 16000

        marker = "  *** WAKE DETECTED ***" if wake else ""
        print(f"  t={t:5.2f}s  score={score:.4f}{marker}")

        if wake:
            wake_count += 1
            matcher.reset()

    print(f"\n[done] Wake events detected: {wake_count}")
    print("Replace fake_embedding() with ml.personalization.embedding.extract_embedding()")
    print("and load a real prototype with ml.personalization.enrollment.load_prototype().")


if __name__ == "__main__":
    main()
