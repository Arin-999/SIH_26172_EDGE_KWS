"""
Demo: end-to-end system demonstration using the Python firmware simulator.

Starts the FastAPI server in a background thread, runs the firmware simulator
against it, and prints the complete transcript pipeline output.

Usage:
    python examples/demo/run_demo.py
    python examples/demo/run_demo.py --audio examples/demo/demo.wav
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEMO_WAV = str(Path(__file__).parent / "demo.wav")
SERVER_URI = "ws://localhost:8766/v1/stream"
SECRET = b"dev-secret-do-not-use-in-production"


def start_demo_server() -> None:
    """Start the FastAPI server in a background thread."""
    import os
    import uvicorn
    from server.receiver.main import app

    os.environ["KWS_SECRET"] = SECRET.decode()
    os.environ["KWS_ASR_MODEL"] = "base.en"
    os.environ["KWS_ASR_DEVICE"] = "cpu"

    config = uvicorn.Config(app, host="127.0.0.1", port=8766, log_level="error")
    server = uvicorn.Server(config)
    server.run()


async def run_simulator(audio: np.ndarray) -> None:
    """Run the firmware simulator."""
    from firmware.esp32.main.simulator import simulate
    await simulate(audio, server_uri=SERVER_URI)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Full end-to-end KWS demo.")
    parser.add_argument("--audio", default=None, help="WAV file to use as input")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of synthetic audio")
    args = parser.parse_args()

    print("=" * 60)
    print("  SIH PS26172 — Edge KWS End-to-End Demo")
    print("=" * 60)

    # Load or generate audio
    if args.audio and Path(args.audio).exists():
        from ml.preprocessing.audio_preprocessing import preprocess
        audio = preprocess(args.audio)
        print(f"[demo] Audio loaded: {args.audio}")
    else:
        sr = 16_000
        n = int(sr * args.duration)
        print(f"[demo] Using {args.duration:.1f}s of synthetic audio (no WAV provided)")
        audio = np.random.randn(n).astype(np.float32) * 0.05

    # Start server in background
    print("[demo] Starting server on port 8766 ...")
    server_thread = threading.Thread(target=start_demo_server, daemon=True)
    server_thread.start()
    time.sleep(3.0)  # Wait for server to initialize
    print("[demo] Server ready.\n")

    # Run simulator
    print("[demo] Running firmware simulator ...")
    asyncio.run(run_simulator(audio))

    print("\n[demo] Done.")


if __name__ == "__main__":
    main()
