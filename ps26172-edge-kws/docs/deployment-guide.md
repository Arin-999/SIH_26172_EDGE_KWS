# Deployment Guide

Step-by-step instructions to go from a fresh clone to a working end-to-end system.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.10 | [python.org](https://python.org) |
| ESP-IDF | v5.2 | See Step 3 |
| Git | any | [git-scm.com](https://git-scm.com) |
| A WiFi AP | — | Home/lab router |
| ESP32-S3 DevKit | — | With INMP441 I2S mic wired |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/<org>/ps26172-edge-kws.git
cd ps26172-edge-kws
```

---

## Step 2 — Train the KWS Model

### 2a. Create Python virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2b. Install ML dependencies

```bash
pip install -r ml/requirements.txt
```

### 2c. Download dataset

```bash
python scripts/dataset/download_speech_commands.py
```

This downloads the Google Speech Commands v2 dataset (~2.4 GB) into `ml/datasets/raw/`.

For a quick smoke-test with synthetic data (no download):

```bash
python scripts/dataset/synthetic_data.py --quick
```

### 2d. Train the model

```bash
python ml/training/train.py
```

Training logs appear in the terminal. Checkpoints saved to `ml/models/checkpoints/`.
Expected time: ~30 min on CPU, ~5 min on GPU.

### 2e. Export INT8 TFLite model

```bash
python ml/quantization/export_tflite.py
```

Output:
```
ml/models/int8/model.tflite        (≤ 60 KB)
ml/models/int8/model_metadata.json
```

### 2f. Convert model to C array for firmware

```bash
python scripts/conversion/model_to_c_array.py \
    ml/models/int8/model.tflite \
    firmware/esp32/main/model_data.h
```

---

## Step 3 — Set Up ESP-IDF

### Windows

```bat
scripts\setup\install_esp_idf.bat
```

### Linux / macOS

```bash
bash scripts/setup/install_esp_idf.sh
```

This clones ESP-IDF v5.2 to `~/esp/esp-idf` and runs the install script.

Then activate the environment:

```bash
# Windows:
%USERPROFILE%\esp\esp-idf\export.bat
# Linux/macOS:
source ~/esp/esp-idf/export.sh
```

---

## Step 4 — Configure Firmware

Edit WiFi credentials and server address in `menuconfig`:

```bash
cd firmware/esp32
idf.py menuconfig
```

Navigate to:
- `Example Connection Configuration` → set `WiFi SSID` and `WiFi Password`
- `KWS Configuration` → set `Server Host` (IP of your server) and `KWS Shared Secret`

Or set them directly in `sdkconfig.defaults` before building.

---

## Step 5 — Build and Flash Firmware

```bash
cd firmware/esp32
idf.py set-target esp32s3
idf.py build
idf.py -p COM3 flash monitor    # replace COM3 with your port
```

On Linux: use `/dev/ttyUSB0` or `/dev/ttyACM0`.

Monitor output shows boot log and system state.

---

## Step 6 — Start the Cloud Server

```bash
cd ps26172-edge-kws
pip install -r server/requirements.txt

# Set the shared secret (must match firmware menuconfig)
export KWS_SECRET="your-256-bit-hex-secret"

# Start server
uvicorn server.receiver.main:app --host 0.0.0.0 --port 8765
```

Verify the server is running:

```bash
curl http://localhost:8765/health
# {"status":"ok","version":"1.0"}
```

---

## Step 7 — Enroll a Wake Word

With firmware running, open a serial monitor and type:

```
enroll
```

The device will prompt you 10 times. Speak your wake word each time.

```
[KWS] Enrollment mode.
[KWS] Ready [1/10]. Speak now.
[KWS] Got utterance 1. Score: 0.93
...
[KWS] Enrollment complete. Prototype stored in NVS.
[KWS] Listening...
```

---

## Step 8 — Test the System

1. Say your wake word
2. Immediately say a command (e.g., "turn on the lights")
3. Watch the serial monitor:

```
[KWS] WAKE DETECTED (score: 0.89)
[STREAM] Sending pre-roll 500ms...
[STREAM] Streaming audio...
[STREAM] End-of-utterance sent.
[STREAM] Transcript: {"type":"transcript","text":"turn on the lights","latency_ms":147}
[AGENT] Intent: light_control, Action: on
```

---

## Step 9 — Run Evaluation

```bash
python ml/evaluation/evaluate.py
```

Results written to `ml/benchmarks/`:
- `far_frr.csv`
- `model_comparison.csv`
- `robustness.csv`
- `esp32_results.csv` (populated from HIL tests)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Firmware won't connect to WiFi | Check SSID/password in menuconfig. Verify 2.4 GHz band. |
| WebSocket connection refused | Check server IP in menuconfig. Verify `uvicorn` is running. |
| HMAC auth failure (403) | `KWS_SECRET` env var on server must match firmware config. |
| Poor wake-word accuracy | Re-enroll in quieter environment. Lower similarity threshold. |
| `model.tflite` not found | Run training and export steps first. |
| Flash too large | Check model size: `ls -lh ml/models/int8/model.tflite` |
