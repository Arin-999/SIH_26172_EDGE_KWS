"""
Demo: enrollment from recorded WAV files + live matching.

Usage:
    # Step 1: Record 5 utterances of your wake word as WAV files
    python examples/enrollment_demo.py \
        --audio examples/demo/utterance_{1..5}.wav \
        --name my_keyword

    # Step 2: Test matching against a query file
    python examples/enrollment_demo.py \
        --audio examples/demo/utterance_{1..5}.wav \
        --test-audio examples/demo/test_query.wav \
        --name my_keyword
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.preprocessing.audio_preprocessing import preprocess
from ml.personalization.enrollment import enroll_from_files, load_prototype
from ml.personalization.matcher import match, DebounceMatcher


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Enrollment + matching demo.")
    parser.add_argument(
        "--audio", nargs="+", required=True, help="WAV files for enrollment"
    )
    parser.add_argument("--name", default="my_keyword", help="Profile name")
    parser.add_argument(
        "--model", default="ml/models/int8/model.tflite", help="TFLite model path"
    )
    parser.add_argument("--test-audio", help="Optional: test query WAV file to match")
    parser.add_argument(
        "--threshold", type=float, default=0.75, help="Cosine similarity threshold"
    )
    args = parser.parse_args()

    # ---- Enrollment ----
    print("\n=== Enrollment ===")
    prototype = enroll_from_files(
        audio_paths=args.audio,
        model_path=args.model,
        profile_name=args.name,
    )
    print(f"Prototype norm: {np.linalg.norm(prototype):.6f} (should be ~1.0)")

    # ---- Verify against enrollment set ----
    print("\n=== Self-verification ===")
    from ml.personalization.embedding import extract_embedding_from_file

    for path in args.audio:
        emb = extract_embedding_from_file(path, model_path=args.model)
        wake, score = match(emb, prototype, threshold=args.threshold)
        status = "ACCEPT" if wake else "reject"
        print(f"  {Path(path).name:30s}  score={score:.4f}  [{status}]")

    # ---- Test query matching ----
    if args.test_audio:
        print(f"\n=== Query Test: {args.test_audio} ===")
        emb = extract_embedding_from_file(args.test_audio, model_path=args.model)
        wake, score = match(emb, prototype, threshold=args.threshold)
        print(f"  Score: {score:.4f}  (threshold={args.threshold})")
        print(f"  Decision: {'✓ WAKE DETECTED' if wake else '✗ No wake'}")


if __name__ == "__main__":
    main()
