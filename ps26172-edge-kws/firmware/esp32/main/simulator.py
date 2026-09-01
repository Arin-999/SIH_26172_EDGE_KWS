"""
Python firmware simulator for local testing without physical hardware.

Simulates the ESP32 firmware behavior:
  - Reads audio from a WAV file or microphone
  - Runs MFCC extraction and KWS inference via TFLite interpreter
  - Detects wake word with debounce
  - Streams audio over WebSocket to the local server
  - Prints transcript responses

Usage:
    python firmware/esp32/main/simulator.py --audio examples/demo/demo.wav
    python firmware/esp32/main/simulator.py --mic  # use system microphone
"""

from __future__ import annotations

import asyncio
import struct
import sys
import time
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ml.preprocessing.audio_preprocessing import preprocess
from ml.preprocessing.feature_extraction import audio_to_features
from ml.personalization.enrollment import load_prototype
from ml.personalization.embedding import extract_embedding
from firmware.esp32.main.matcher import FirmwareMatcher
from server.security.auth import generate_token

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 4_000          # 250 ms chunks
PREROLL_SAMPLES = 8_000        # 500 ms pre-roll
SERVER_URI = "ws://localhost:8765/v1/stream"
SECRET = b"dev-secret-do-not-use-in-production"
MODEL_PATH = "ml/models/int8/model.tflite"
PROFILE_DIR = "ml/personalization/profiles"

# End-of-utterance marker
EOT = b"\xff"


# ---------------------------------------------------------------------------
# Packet builder
# ---------------------------------------------------------------------------


def build_packet(
    payload: bytes,
    seq: int,
    flags: int = 0x00,
    session_id_low: int = 0,
) -> bytes:
    """Build a binary audio chunk packet per packet-format.md."""
    version = 1
    payload_len = len(payload)
    timestamp_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF
    header = struct.pack(
        ">BBHHIH",
        version,
        flags,
        seq & 0xFFFF,
        payload_len,
        session_id_low & 0xFFFF,
        timestamp_ms & 0xFFFF,
    )
    # Pad header to 12 bytes (struct above is 10, add 2 padding)
    header = header[:4] + struct.pack(">H", payload_len) + struct.pack(">H", session_id_low) + struct.pack(">I", timestamp_ms)
    return header[:12] + payload + b"\x00\x00\x00\x00"  # dummy hmac_tag


# ---------------------------------------------------------------------------
# Simulator main
# ---------------------------------------------------------------------------


async def simulate(
    audio: np.ndarray,
    profile_name: str = "keyword",
    server_uri: str = SERVER_URI,
) -> None:
    """Run the firmware simulation loop against a local server.

    Args:
        audio: Full audio array to simulate (as if from microphone stream).
        profile_name: Enrolled wake-word profile name.
        server_uri: WebSocket server URI.
    """
    # Load prototype
    try:
        prototype = load_prototype(profile_name, PROFILE_DIR)
    except FileNotFoundError:
        print(f"[sim] No profile '{profile_name}' found. Using random prototype.")
        prototype = np.random.randn(64).astype(np.float32)
        prototype /= np.linalg.norm(prototype)

    matcher = FirmwareMatcher(prototype, threshold=0.75)

    # Connect to server
    token = generate_token(SECRET)
    headers = {"X-KWS-Token": token}

    print(f"[sim] Connecting to {server_uri} ...")
    async with websockets.connect(server_uri, extra_headers=headers) as ws:
        ack = await ws.recv()
        print(f"[sim] Server ACK: {ack}")

        # Pre-roll buffer
        preroll = np.zeros(PREROLL_SAMPLES, dtype=np.float32)
        n_total = len(audio)
        pos = 0
        seq = 0
        streaming = False
        stream_buffer: bytearray = bytearray()

        print(f"[sim] Simulating {n_total/SAMPLE_RATE:.2f}s of audio ...")

        while pos < n_total:
            chunk = audio[pos:pos + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            pos += CHUNK_SAMPLES

            # Update pre-roll ring buffer
            preroll = np.roll(preroll, -len(chunk))
            preroll[-len(chunk):] = chunk

            if not streaming:
                # Run KWS inference on this chunk
                if len(chunk) < SAMPLE_RATE:
                    # Use last 1 second of audio for inference
                    inference_audio = np.concatenate([preroll[-PREROLL_SAMPLES:], chunk])[-SAMPLE_RATE:]
                else:
                    inference_audio = chunk[:SAMPLE_RATE]

                emb = extract_embedding(inference_audio, model_path=MODEL_PATH)
                wake, score = matcher.push(emb)

                print(f"  pos={pos/SAMPLE_RATE:.2f}s  score={score:.3f}  {'WAKE!' if wake else ''}")

                if wake:
                    print("[sim] WAKE DETECTED — starting stream")
                    streaming = True
                    stream_buffer.clear()
                    seq = 0

                    # Send pre-roll
                    preroll_bytes = (preroll * 32767).astype(np.int16).tobytes()
                    pkt = preroll_bytes  # raw PCM (dev mode)
                    await ws.send(b"\x01\x80" + struct.pack(">H", seq) +
                                  struct.pack(">H", len(preroll_bytes)) +
                                  b"\x00\x00\x00\x00\x00\x00\x00\x00" +
                                  preroll_bytes +
                                  b"\x00\x00\x00\x00")
                    seq += 1

            else:
                # Stream this chunk
                chunk_bytes = (chunk * 32767).astype(np.int16).tobytes()
                stream_buffer.extend(chunk_bytes)

                flags = 0x00
                pkt = (b"\x01" + bytes([flags]) +
                       struct.pack(">H", seq) +
                       struct.pack(">H", len(chunk_bytes)) +
                       b"\x00\x00\x00\x00\x00\x00\x00\x00" +
                       chunk_bytes +
                       b"\x00\x00\x00\x00")
                await ws.send(pkt)
                seq += 1

                # Simulate end of utterance after 2 seconds of streaming
                if len(stream_buffer) >= SAMPLE_RATE * 2 * 2:
                    print("[sim] End of utterance — sending EOT")
                    await ws.send(EOT)
                    streaming = False

                    # Wait for transcript
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        print(f"[sim] Server: {response}")
                    except asyncio.TimeoutError:
                        print("[sim] Timeout waiting for transcript.")

        print("[sim] Simulation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESP32 firmware simulator.")
    parser.add_argument("--audio", help="Path to WAV file to simulate")
    parser.add_argument("--profile", default="keyword")
    parser.add_argument("--server", default=SERVER_URI)
    args = parser.parse_args()

    if args.audio:
        audio = preprocess(args.audio)
    else:
        print("[sim] No audio file specified. Using 3 seconds of noise.")
        audio = np.random.randn(SAMPLE_RATE * 3).astype(np.float32) * 0.1

    asyncio.run(simulate(audio, profile_name=args.profile, server_uri=args.server))
