# ps26172-edge-kws

**SIH 2026 · Problem Statement 26172 — Edge Keyword Spotting System**

A low-power, privacy-preserving voice command system for edge deployment on ESP32-S3. Detects a user-defined wake word entirely on-device using an INT8-quantized DS-CNN model, then streams the captured command over WiFi to a cloud ASR + intent server.

```
ESP32-S3 (INMP441 mic)
  └─ MFCC → DS-CNN (INT8 TFLite, ~50 ms) → cosine debounce
       └─ WAKE ─► stream PCM over WebSocket (HMAC-auth)
                     └─ faster-whisper ASR → intent handler → response
```

---

## Quick Start

### 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.10+ required.** TensorFlow is needed only for enrollment and training; it is not required to run the server.

### 2 — Start the ASR server

```bash
uvicorn server.receiver.main:app --host 0.0.0.0 --port 8765
```

The server loads `faster-whisper base.en` on startup (downloads ~150 MB on first run).

### 3 — Run the firmware simulator

No ESP32 hardware required — the simulator runs the full firmware loop on your PC:

```bash
# Simulate with a WAV file
python firmware/esp32/main/simulator.py --audio examples/demo/demo.wav

# Simulate with 3 s of noise (no audio file)
python firmware/esp32/main/simulator.py
```

### 4 — Enroll a custom wake word (optional)

```bash
# Record 5+ utterances of your wake word as WAV files, then:
python -m ml.personalization.enrollment \
    utt1.wav utt2.wav utt3.wav utt4.wav utt5.wav \
    --name my_keyword \
    --model ml/models/int8/model.tflite
```

The prototype is saved to `ml/personalization/profiles/my_keyword.npy`.

---

## Repository Layout

```
ps26172-edge-kws/
├── firmware/esp32/         ESP32-S3 firmware (C++ stubs + Python simulator)
├── hardware/               Wiring diagram, GPIO table, BOM, power budget
├── ml/                     DS-CNN training, MFCC preprocessing, enrollment, evaluation
├── server/                 FastAPI WebSocket server (ASR + intent)
├── protocol/               Binary packet spec, JSON message types, error codes
├── tests/                  68-test suite (unit, integration, security, network)
├── scripts/                Dataset, training, conversion, flashing utilities
├── examples/               Usage examples and demo scripts
├── docs/                   Requirements, security design, testing plan
├── INFRASTRUCTURE.md       Full architecture + module reference (start here)
├── requirements.txt        All Python dependencies
└── pytest.ini              Test configuration
```

> **Detailed documentation:** See [INFRASTRUCTURE.md](INFRASTRUCTURE.md) for the complete module reference, data-flow diagrams, protocol specification, and environment configuration.

---

## Running Tests

```bash
# All software tests (no hardware or model artifacts required)
python -m pytest tests/ -v -m "not hardware and not model"

# With coverage report
python -m pytest tests/ --cov=ml --cov=server --cov-report=term-missing \
    -m "not hardware and not model"
```

Current status: **68/68 passed**.

| Category | Tests |
|---|---|
| Unit — audio preprocessing | 16 |
| Unit — MFCC feature extraction | 12 |
| Unit — cosine matcher + debounce | 13 |
| Integration — enrollment → EER pipeline | 4 |
| Network — assembler + gap detector | 14 |
| Security — HMAC token auth | 9 |

---

## Server Configuration

All options are read from environment variables:

| Variable | Default | Description |
|---|---|---|
| `KWS_SECRET` | `dev-secret-do-not-use-in-production` | HMAC shared secret — **change in production** |
| `KWS_ASR_MODEL` | `base.en` | faster-whisper model (`tiny`, `base`, `small`, `medium`) |
| `KWS_ASR_DEVICE` | `cpu` | `cpu` or `cuda` |
| `KWS_MAX_SESSIONS` | `10` | Max concurrent WebSocket sessions |
| `KWS_AUTH_WINDOW_S` | `30` | Token timestamp validity window (seconds) |

```bash
# Production example
export KWS_SECRET="$(openssl rand -hex 32)"
export KWS_ASR_MODEL="small.en"
export KWS_ASR_DEVICE="cuda"
uvicorn server.receiver.main:app --host 0.0.0.0 --port 8765 --workers 1
```

---

## Hardware

**Target:** ESP32-S3-DevKitC-1 + INMP441 MEMS I2S microphone

| Signal | GPIO |
|---|---|
| I2S Data (SD) | GPIO 4 |
| Word Select (WS) | GPIO 5 |
| Bit Clock (SCK) | GPIO 6 |

Flash the firmware:

```bash
idf.py -p COM3 flash monitor          # Windows
idf.py -p /dev/ttyACM0 flash monitor  # Linux / macOS
```

See [hardware/README.md](hardware/README.md) for full wiring, power budget, and BOM.

---

## ML Pipeline Overview

```
WAV file → load_audio → resample (16 kHz) → normalize → trim/pad (1 s)
         → MFCC (49 × 40 × 1) → DS-CNN TFLite INT8 → 64-D embedding
         → cosine similarity → 2-of-3 debounce → WAKE
```

Training, quantization, and evaluation scripts live in `ml/`. See [INFRASTRUCTURE.md §5](INFRASTRUCTURE.md) for the full pipeline reference.

---

## License

See [LICENSE](LICENSE).
