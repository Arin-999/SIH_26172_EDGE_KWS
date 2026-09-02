# SIH 26172 — Ultra-Low-Latency Edge KWS (ISRO)

Hybrid edge-cloud voice activator for Smart India Hackathon 2026 (Problem PS26172).

- **Edge:** ESP32-S3 runs INT8 DS-CNN keyword spotting with runtime wake-word enrollment
- **Cloud:** FastAPI + faster-whisper ASR over persistent WebSocket
- **Constraints:** ≤256 KB RAM, <10% idle CPU, <200 ms wake-to-cloud latency

## Repository

All implementation lives in [`ps26172-edge-kws/`](ps26172-edge-kws/).

```text
ps26172-edge-kws/
├── ml/           # Training, quantization, evaluation
├── firmware/     # ESP-IDF ESP32-S3 application
├── server/       # Cloud ASR WebSocket server
├── tests/        # Integration and benchmark harness
└── docs/         # Architecture and protocol specs
```

## Quick Start

### 1. Train and export KWS model

```bash
cd ps26172-edge-kws
python -m venv .venv && .venv\Scripts\activate
pip install -r ml/requirements.txt
python scripts/dataset/synthetic_data.py --quick
python ml/training/train.py
python ml/quantization/export_tflite.py
```

### 2. Run cloud ASR server

```bash
pip install -r server/requirements.txt
uvicorn server.receiver.main:app --host 0.0.0.0 --port 8765
```

### 3. Flash ESP32-S3 firmware

```bash
cd firmware/esp32
idf.py set-target esp32s3
idf.py build flash monitor
```

## License

MIT — see [LICENSE](ps26172-edge-kws/LICENSE).
