#!/usr/bin/env python3
"""
Download and prepare the Google Speech Commands v2 dataset.

Downloads the dataset, extracts it, and organises it into:
    ml/datasets/raw/<class>/<file>.wav

Usage:
    python scripts/dataset/download_google_speech.py [--output ml/datasets/raw]
"""
from __future__ import annotations
import argparse
import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

DATASET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "data/speech_commands_v0.02.tar.gz"
)
EXPECTED_MD5 = "af14739ee7dc311471de98f5f9d2da1e"
ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"

# KWS-relevant classes to keep (others discarded)
KEEP_CLASSES = {
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go", "_background_noise_",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] Already downloaded: {dest.name}")
        return
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _progress(count: int, block_size: int, total: int) -> None:
    pct = min(count * block_size / total * 100, 100)
    print(f"\r  {pct:.1f}%", end="", flush=True)


def extract(archive: Path, output: Path) -> None:
    print(f"[extract] {archive.name} → {output}")
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers()
                   if any(m.name.startswith(cls + "/") for cls in KEEP_CLASSES)]
        tar.extractall(path=output, members=members)
    print(f"[OK] Extracted {len(members)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Google Speech Commands v2.")
    parser.add_argument("--output", default="ml/datasets/raw",
                        help="Output directory for raw dataset")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output = root / args.output
    cache_dir = root / "ml" / "datasets" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive = cache_dir / ARCHIVE_NAME
    download(DATASET_URL, archive)

    print(f"[verify] Checking MD5 ... ", end="", flush=True)
    actual = md5(archive)
    if actual != EXPECTED_MD5:
        print(f"MISMATCH! Expected {EXPECTED_MD5}, got {actual}")
    else:
        print("OK")

    extract(archive, output)
    print(f"\n[done] Dataset at: {output}")
    print(f"Classes: {sorted(KEEP_CLASSES)}")


if __name__ == "__main__":
    main()
