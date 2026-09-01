#!/usr/bin/env python3
"""
Setup script: install all Python dependencies for the KWS Edge project.

Usage:
    python scripts/setup/install_deps.py [--server-only] [--ml-only]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install project dependencies.")
    parser.add_argument("--server-only", action="store_true", help="Install server deps only")
    parser.add_argument("--ml-only", action="store_true", help="Install ML deps only")
    args = parser.parse_args()

    print("=== KWS Edge — Dependency Setup ===")
    print(f"Python: {sys.version}")

    if args.server_only:
        run([sys.executable, "-m", "pip", "install", "-r", "server/requirements.txt"])
    elif args.ml_only:
        run([sys.executable, "-m", "pip", "install",
             "numpy>=1.24", "librosa>=0.10.0", "soundfile>=0.12.1",
             "tensorflow>=2.15,<2.17", "scikit-learn>=1.3.0",
             "scipy>=1.11.0", "audiomentations>=0.30.0",
             "tqdm>=4.65.0", "matplotlib>=3.7.0"])
    else:
        run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])

    print("\n[OK] All dependencies installed.")
    print("Run tests with: python -m pytest tests/ -m 'not hardware and not model'")


if __name__ == "__main__":
    main()
