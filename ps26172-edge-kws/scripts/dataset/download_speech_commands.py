"""
Download and organize the Google Speech Commands v2 dataset.

Downloads from the official TensorFlow URL, extracts to ml/datasets/raw/,
and preprocesses all WAV files into MFCC .npy arrays in ml/datasets/processed/.

Usage:
    python scripts/dataset/download_speech_commands.py
    python scripts/dataset/download_speech_commands.py --skip-download  # if already downloaded
"""

from __future__ import annotations

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.preprocessing.audio_preprocessing import preprocess
from ml.preprocessing.feature_extraction import audio_to_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPEECH_COMMANDS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "speech_commands_v0.02.tar.gz"
)
ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "ml" / "training" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_dataset(raw_dir: str) -> str:
    """Download Speech Commands v2 archive if not already present.

    Args:
        raw_dir: Directory to download and extract into.

    Returns:
        Path to the extracted dataset directory.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    archive_path = raw_path / ARCHIVE_NAME

    if not archive_path.exists():
        print(f"[download] Downloading Speech Commands v2 (~2.4 GB)...")
        print(f"  URL: {SPEECH_COMMANDS_URL}")

        def _progress(block_count, block_size, total_size):
            downloaded = block_count * block_size
            pct = min(100.0, downloaded / total_size * 100) if total_size > 0 else 0
            print(f"\r  {pct:.1f}%  ({downloaded / 1e6:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(SPEECH_COMMANDS_URL, str(archive_path), reporthook=_progress)
        print()
        print(f"[download] Saved to {archive_path}")
    else:
        print(f"[download] Archive already exists: {archive_path}")

    # Extract
    print("[download] Extracting ...")
    with tarfile.open(str(archive_path), "r:gz") as tar:
        tar.extractall(str(raw_path))
    print(f"[download] Extracted to {raw_path}")

    return str(raw_path)


# ---------------------------------------------------------------------------
# Split files
# ---------------------------------------------------------------------------


def load_split_files(raw_dir: str) -> tuple[set[str], set[str]]:
    """Load the official validation and test file lists.

    Args:
        raw_dir: Extracted dataset directory.

    Returns:
        Tuple of (validation_files, test_files) as sets of relative paths.
    """
    raw = Path(raw_dir)
    val_files: set[str] = set()
    test_files: set[str] = set()

    val_list = raw / "validation_list.txt"
    test_list = raw / "testing_list.txt"

    if val_list.exists():
        val_files = set(val_list.read_text().strip().splitlines())
    if test_list.exists():
        test_files = set(test_list.read_text().strip().splitlines())

    return val_files, test_files


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_dataset(
    raw_dir: str,
    processed_dir: str,
    max_per_class: int | None = None,
) -> None:
    """Preprocess all WAV files to MFCC .npy arrays.

    Organizes files into train/val/test splits following official lists.

    Args:
        raw_dir: Path to extracted Speech Commands v2 directory.
        processed_dir: Output directory for .npy features.
        max_per_class: Optional limit on files per class (for quick mode).
    """
    raw = Path(raw_dir)
    processed = Path(processed_dir)
    val_files, test_files = load_split_files(raw_dir)

    # Find all class directories (exclude system dirs)
    skip_dirs = {"_background_noise_", ".git", "__pycache__"}
    class_dirs = [
        d for d in sorted(raw.iterdir())
        if d.is_dir() and d.name not in skip_dirs
    ]

    print(f"[preprocess] Found {len(class_dirs)} classes in {raw_dir}")
    total_files = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        wav_files = sorted(class_dir.glob("*.wav"))

        if max_per_class:
            wav_files = wav_files[:max_per_class]

        for wav_path in tqdm(wav_files, desc=f"  {class_name:<15}", leave=False):
            rel_path = f"{class_name}/{wav_path.name}"

            if rel_path in test_files:
                split = "test"
            elif rel_path in val_files:
                split = "val"
            else:
                split = "train"

            out_dir = processed / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (wav_path.stem + ".npy")

            if out_path.exists():
                continue

            try:
                audio = preprocess(str(wav_path))
                feat = audio_to_features(audio)
                np.save(str(out_path), feat)
                total_files += 1
            except Exception as exc:
                print(f"  [WARN] {wav_path}: {exc}")

    print(f"[preprocess] Done. {total_files} new .npy files created in {processed_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and preprocess Speech Commands v2.")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if already done")
    parser.add_argument("--raw-dir", default="ml/datasets/raw")
    parser.add_argument("--processed-dir", default="ml/datasets/processed")
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    if not args.skip_download:
        download_dataset(args.raw_dir)

    preprocess_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        max_per_class=args.max_per_class,
    )
