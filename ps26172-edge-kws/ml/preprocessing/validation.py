"""
Dataset validation utilities.

Walks the dataset directory and validates that all audio files meet the
pipeline requirements (sample rate, duration, channels, amplitude).
Prints a per-class summary and flags outliers.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Expected values
# ---------------------------------------------------------------------------

EXPECTED_SR: int = 16_000
EXPECTED_DURATION_S: float = 1.0
EXPECTED_CHANNELS: int = 1
DURATION_TOLERANCE_S: float = 0.05  # ±50 ms


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ValidationResult(NamedTuple):
    path: str
    ok: bool
    issues: list[str]
    sr: int
    duration_s: float
    channels: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_file(path: str) -> ValidationResult:
    """Validate a single audio file.

    Checks:
    - Can be opened by soundfile
    - Sample rate == 16 000 Hz
    - Duration within [0.95, 1.05] seconds
    - Mono (1 channel)
    - Not silent (peak amplitude > 1e-6)

    Args:
        path: Path to WAV/FLAC file.

    Returns:
        ValidationResult with pass/fail and list of issue descriptions.
    """
    issues: list[str] = []
    try:
        info = sf.info(path)
        sr = info.samplerate
        duration_s = info.duration
        channels = info.channels

        if sr != EXPECTED_SR:
            issues.append(f"sample_rate={sr} (expected {EXPECTED_SR})")
        if abs(duration_s - EXPECTED_DURATION_S) > DURATION_TOLERANCE_S:
            issues.append(f"duration={duration_s:.3f}s (expected ~{EXPECTED_DURATION_S}s)")
        if channels != EXPECTED_CHANNELS:
            issues.append(f"channels={channels} (expected {EXPECTED_CHANNELS})")

        # Check amplitude (silence detection)
        audio, _ = sf.read(path, dtype="float32", always_2d=False)
        peak = float(np.abs(audio).max())
        if peak < 1e-6:
            issues.append("silent (peak amplitude < 1e-6)")

    except Exception as exc:
        issues.append(f"cannot read: {exc}")
        return ValidationResult(path=path, ok=False, issues=issues, sr=0, duration_s=0.0, channels=0)

    return ValidationResult(
        path=path,
        ok=len(issues) == 0,
        issues=issues,
        sr=sr,
        duration_s=duration_s,
        channels=channels,
    )


def validate_dataset(dataset_root: str, extensions: tuple[str, ...] = (".wav", ".flac")) -> dict:
    """Validate all audio files under `dataset_root`.

    Args:
        dataset_root: Path to directory containing class subdirectories.
        extensions: File extensions to include.

    Returns:
        Summary dict with per-class counts and list of failed files.
    """
    root = Path(dataset_root)
    if not root.exists():
        print(f"[ERROR] Dataset root does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    class_counts: dict[str, int] = defaultdict(int)
    class_errors: dict[str, int] = defaultdict(int)
    failed: list[ValidationResult] = []

    all_files = [
        p for p in root.rglob("*")
        if p.suffix.lower() in extensions and p.is_file()
    ]

    print(f"[validate] Scanning {len(all_files)} files under {root} ...")

    for path in tqdm(all_files, desc="Validating", unit="file"):
        class_name = path.parent.name
        result = validate_file(str(path))
        class_counts[class_name] += 1
        if not result.ok:
            class_errors[class_name] += 1
            failed.append(result)

    # Print summary
    print("\n=== Dataset Validation Summary ===")
    print(f"{'Class':<20} {'Files':>7} {'Errors':>7}")
    print("-" * 38)
    total_files = 0
    total_errors = 0
    for cls in sorted(class_counts):
        n = class_counts[cls]
        e = class_errors[cls]
        total_files += n
        total_errors += e
        status = "OK" if e == 0 else f"WARN ({e} errors)"
        print(f"{cls:<20} {n:>7} {status:>7}")
    print("-" * 38)
    print(f"{'TOTAL':<20} {total_files:>7} {total_errors:>7}\n")

    if failed:
        print(f"[WARN] {len(failed)} files failed validation:")
        for r in failed[:20]:
            print(f"  {r.path}: {', '.join(r.issues)}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
    else:
        print("[OK] All files passed validation.")

    return {
        "total_files": total_files,
        "total_errors": total_errors,
        "class_counts": dict(class_counts),
        "class_errors": dict(class_errors),
        "failed": [r._asdict() for r in failed],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate the KWS dataset.")
    parser.add_argument("dataset_root", help="Path to dataset root directory")
    args = parser.parse_args()

    result = validate_dataset(args.dataset_root)
    sys.exit(0 if result["total_errors"] == 0 else 1)
