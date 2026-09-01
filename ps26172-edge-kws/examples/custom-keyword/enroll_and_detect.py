#!/usr/bin/env python3
"""
Custom keyword enrollment + detection example.

Demonstrates:
  1. Enrolling a custom wake word from WAV files
  2. Finding the optimal detection threshold
  3. Running detection on a test audio file

Usage:
    # Enroll from 5 WAV files, then test detection:
    python examples/custom-keyword/enroll_and_detect.py \
        --enroll utt1.wav utt2.wav utt3.wav utt4.wav utt5.wav \
        --test   test.wav \
        --name   my_keyword \
        --model  ml/models/int8/model.tflite

Requirements:
    - Trained TFLite model at ml/models/int8/model.tflite
    - 5+ WAV files of your chosen wake word (each ~1 second, 16 kHz)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll + detect a custom wake word.")
    parser.add_argument("--enroll", nargs="+", required=True,
                        help="WAV files for enrollment (5+ recommended)")
    parser.add_argument("--test", help="WAV file to run detection on")
    parser.add_argument("--name", default="custom_keyword", help="Profile name")
    parser.add_argument("--model", default="ml/models/int8/model.tflite")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Detection threshold (default 0.75)")
    args = parser.parse_args()

    model_path = str(ROOT / args.model)

    # ── 1. Enrollment ────────────────────────────────────────────────────────
    print(f"\n=== Step 1: Enrollment ===")
    print(f"Profile name : {args.name}")
    print(f"Model        : {model_path}")
    print(f"Utterances   : {len(args.enroll)}")

    from ml.personalization.enrollment import enroll_from_files
    prototype = enroll_from_files(
        audio_paths=args.enroll,
        model_path=model_path,
        profile_name=args.name,
        min_utterances=3,
    )
    print(f"Prototype shape: {prototype.shape}, norm: {np.linalg.norm(prototype):.4f}")

    # ── 2. Test detection ────────────────────────────────────────────────────
    if args.test:
        print(f"\n=== Step 2: Detection ===")
        print(f"Test file  : {args.test}")
        print(f"Threshold  : {args.threshold}")

        from ml.preprocessing.audio_preprocessing import preprocess
        from ml.personalization.embedding import extract_embedding
        from ml.personalization.matcher import DebounceMatcher

        audio = preprocess(args.test)
        CHUNK = 4_000  # 250 ms
        matcher = DebounceMatcher(prototype, threshold=args.threshold,
                                  hits_required=2, window_size=3)

        print(f"\nProcessing {len(audio)/16000:.2f}s ...")
        wake_count = 0

        for i in range(0, len(audio), CHUNK):
            chunk = audio[i: i + CHUNK]
            if len(chunk) < CHUNK:
                chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
            emb = extract_embedding(chunk, model_path=model_path)
            wake, score = matcher.update(emb)
            print(f"  t={i/16000:5.2f}s  score={score:.4f}{'  *** WAKE ***' if wake else ''}")
            if wake:
                wake_count += 1
                matcher.reset()

        print(f"\n[result] Wake events: {wake_count}")
    else:
        print("\n[info] No --test file provided. Enrollment complete.")
        print(f"Profile saved to: ml/personalization/profiles/{args.name}.npy")


if __name__ == "__main__":
    main()
