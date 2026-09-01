# Infrastructure Reference — ps26172-edge-kws

> **SIH 2026 · Problem Statement 26172 — Edge Keyword Spotting System**  
> Last updated: 2026-09-01 · Test status: **68/68 passed**

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Hardware Layer](#3-hardware-layer)
4. [Firmware Layer](#4-firmware-layer)
5. [ML Pipeline](#5-ml-pipeline)
6. [Server Layer](#6-server-layer)
7. [Protocol](#7-protocol)
8. [Data Flow — End to End](#8-data-flow--end-to-end)
9. [Environment & Configuration](#9-environment--configuration)
10. [Dependencies](#10-dependencies)
11. [Test Suite](#11-test-suite)
12. [Scripts & Utilities](#12-scripts--utilities)
13. [Known Limitations](#13-known-limitations)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EDGE DEVICE (ESP32-S3)                       │
│                                                                     │
│  INMP441 Mic ──► I2S DMA ──► MFCC ──► DS-CNN TFLite ──► Matcher     │
│                                          (INT8, ~50 ms)    (cosine) │
│                                                │ WAKE               │
│                                                ▼                    │
│                          Pre-roll + Live PCM stream                 │
│                          Binary WebSocket frames (8016 B each)      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  WiFi 802.11 b/g/n
                               │  ws://host:8765/v1/stream
                               │  X-KWS-Token: <HMAC-SHA256>
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SERVER (FastAPI + uvicorn)                   │
│                                                                     │
│  Auth (HMAC) ──► Gap Detector ──► Assembler ──► faster-whisper      │
│                                                       │             │
│                                                  Transcript         │
│                                                       │             │
│                                              Intent Handler         │
│                                              (rule-based / LLM)     │
│                                                       │             │
│                                              JSON response ──► ESP32│
└─────────────────────────────────────────────────────────────────────┘
```

**Key design constraints:**

| Constraint | Value |
|---|---|
| Wake detection latency | ≤ 500 ms end-to-end on device |
| ASR latency budget | ≤ 2 000 ms (server-side) |
| Audio format | 16 kHz, mono, int16 PCM |
| Model format | INT8 quantized TFLite |
| Authentication | HMAC-SHA256 session token (30 s window) |
| Max concurrent sessions | 10 |
| Max utterance length | 20 s |

---

## 2. Directory Structure

```
ps26172-edge-kws/
│
├── firmware/
│   └── esp32/
│       ├── CMakeLists.txt          ESP-IDF project root
│       ├── sdkconfig.defaults      ESP-IDF build config
│       └── main/
│           ├── main.cpp            Firmware entry point (stub)
│           ├── app_state.h         State machine header (stub)
│           ├── matcher.py          Python-mirror of C++ matcher logic
│           ├── mfcc.py             Python-mirror of C++ MFCC extraction
│           └── simulator.py        Full firmware simulator (no hardware needed)
│
├── hardware/
│   ├── README.md                   Wiring, GPIO table, power budget
│   ├── bom.md                      Bill of Materials
│   └── esp32-s3/                   KiCad schematic / PCB files
│
├── ml/
│   ├── preprocessing/
│   │   ├── audio_preprocessing.py  load → resample → normalize → trim/pad
│   │   ├── feature_extraction.py   MFCC (49×40×1) for DS-CNN input
│   │   └── validation.py           Audio quality gate checks
│   │
│   ├── personalization/
│   │   ├── embedding.py            TFLite inference → 64-D embedding
│   │   ├── enrollment.py           Enrollment pipeline (files or arrays)
│   │   ├── matcher.py              Cosine match + 2-of-N debounce
│   │   └── threshold.py            Adaptive threshold utilities
│   │
│   ├── evaluation/
│   │   ├── far_frr.py              FAR/FRR/EER curve computation
│   │   ├── metrics.py              Accuracy, precision, recall, F1, AUC
│   │   ├── evaluate.py             Full evaluation runner
│   │   ├── latency.py              Latency profiling utilities
│   │   └── robustness.py           Noise-robustness evaluation
│   │
│   ├── training/                   DS-CNN training scripts + config.yaml
│   ├── quantization/               INT8 post-training quantization + TFLite export
│   ├── augmentation/               Audio augmentation pipeline (audiomentations)
│   ├── models/
│   │   ├── fp32/                   Full-precision Keras model checkpoints
│   │   ├── int8/                   Quantized TFLite artifacts (model.tflite)
│   │   ├── checkpoints/            Training checkpoints
│   │   └── exported/               Final exported models
│   ├── datasets/
│   │   ├── raw/                    Unprocessed audio (Google Speech Commands, etc.)
│   │   ├── processed/              Preprocessed .npy feature arrays
│   │   └── metadata/               Dataset manifests and split files
│   ├── benchmarks/                 Benchmark output CSV files
│   └── notebooks/                  Jupyter exploration notebooks
│
├── server/
│   ├── receiver/
│   │   └── main.py                 FastAPI app + WebSocket /v1/stream endpoint
│   ├── security/
│   │   └── auth.py                 HMAC-SHA256 token generate/verify (replay-safe)
│   ├── packet_reassembly/
│   │   └── assembler.py            Per-session PCM buffer (max 20 s)
│   ├── fec/
│   │   └── decoder.py              uint16 sequence-number gap detector
│   ├── asr/
│   │   └── whisper.py              faster-whisper wrapper (VAD, int8/float16)
│   ├── audio/
│   │   └── reconstruction.py       PCM ↔ numpy conversion + quality validation
│   ├── agent/
│   │   └── handler.py              Rule-based intent handler (LLM swap-in point)
│   └── tts/
│       └── synthesizer.py          pyttsx3 TTS stub (optional response audio)
│
├── protocol/
│   ├── README.md                   Protocol overview
│   ├── packet-format.md            Binary frame layout (12B header + PCM + HMAC)
│   ├── message-types.md            JSON message type catalogue
│   └── error-codes.md              E1xx–E5xx error code reference
│
├── tests/
│   ├── conftest.py                 Shared fixtures (rng, unit_embedding)
│   ├── unit/                       Unit tests (audio preprocessing, MFCC, matcher)
│   ├── integration/                E2E pipeline tests (enrollment→matching→EER)
│   ├── network/                    WebSocket assembler + gap detector tests
│   ├── security/                   HMAC token auth tests
│   ├── audio/                      (placeholder)
│   ├── kws/                        (placeholder)
│   └── end-to-end/                 (placeholder)
│
├── scripts/
│   ├── benchmarking/               Latency + accuracy benchmarking scripts
│   ├── conversion/                 Model format conversion utilities
│   ├── dataset/                    Dataset download and preparation scripts
│   ├── flashing/                   ESP32 flash helpers
│   ├── setup/                      Environment setup scripts
│   └── training/                   Training launch scripts
│
├── examples/
│   ├── basic-kws/                  Minimal wake-word detection example
│   ├── custom-keyword/             Custom keyword enrollment + detection
│   ├── demo/                       Hackathon demo scripts
│   ├── end-to-end/                 Full pipeline demo
│   └── enrollment_demo.py          Interactive enrollment walkthrough
│
├── docs/
│   ├── requirements.md             System requirements specification
│   ├── testing-plan.md             Test strategy and coverage plan
│   └── security-design.md          Security architecture document
│
├── pytest.ini                      Test runner configuration
├── requirements.txt                Unified dependency list (ML + server + test)
└── INFRASTRUCTURE.md               This file
```

---

## 3. Hardware Layer

**Target platform:** ESP32-S3-DevKitC-1

| Component | Specification |
|---|---|
| SoC | ESP32-S3, Xtensa LX7 dual-core @ 240 MHz |
| RAM | 512 KB internal SRAM + 8 MB PSRAM (QSPI) |
| Flash | 4 MB internal |
| Wireless | 802.11 b/g/n 2.4 GHz + BLE 5.0 |
| Microphone | INMP441 MEMS I2S (SNR: 61 dB, 60 Hz–15 kHz) |
| Interface | Native USB-C (Serial/JTAG, no adapter needed) |

**I2S GPIO wiring:**

```
INMP441 pin   →   ESP32-S3 GPIO
  SD (data)   →   GPIO 4  (I2S_DIN)
  WS          →   GPIO 5  (I2S_WS)
  SCK         →   GPIO 6  (I2S_CLK)
  L/R         →   GND     (left channel / mono)
```

**Power budget:**

| State | Current | Power (3.3 V) |
|---|---|---|
| Active streaming (WiFi TX + CPU) | ~236 mA | ~1.2 W |
| Idle / listening (modem sleep) | ~12 mA | ~60 mW |

---

## 4. Firmware Layer

### State Machine

```
BOOT → IDLE (listening) → KWS_INFERENCE → [score < threshold] → IDLE
                                        → [score ≥ threshold, debounce fires]
                                             ↓
                                         STREAMING
                                         (send pre-roll + live PCM)
                                             ↓
                                         EOT_MARKER (0xFF)
                                             ↓
                                         WAIT_TRANSCRIPT → IDLE
```

### Firmware source files

| File | Language | Purpose |
|---|---|---|
| `firmware/esp32/main/main.cpp` | C++ | ESP-IDF app entry, task init (stub) |
| `firmware/esp32/main/app_state.h` | C++ | State enum + transition API (stub) |
| `firmware/esp32/main/mfcc.py` | Python | Python mirror of C++ MFCC (for simulator) |
| `firmware/esp32/main/matcher.py` | Python | Python mirror of C++ debounce matcher |
| `firmware/esp32/main/simulator.py` | Python | Full end-to-end firmware simulator |

### Firmware simulator

Runs the complete firmware flow on a PC (no ESP32 hardware required):

```bash
# Simulate wake detection from a WAV file
python firmware/esp32/main/simulator.py --audio examples/demo/demo.wav

# Simulate with live microphone
python firmware/esp32/main/simulator.py --mic

# Custom server and profile
python firmware/esp32/main/simulator.py \
    --audio examples/demo/demo.wav \
    --profile my_keyword \
    --server ws://192.168.1.10:8765/v1/stream
```

**Simulator flow:**
1. Loads enrolled prototype from `ml/personalization/profiles/<profile>.npy`
2. Processes audio in 250 ms chunks (4 000 samples)
3. Runs MFCC → TFLite inference → cosine match on each chunk
4. On wake detection: sends 500 ms pre-roll + live PCM over WebSocket
5. Sends `0xFF` end-of-utterance marker after 2 seconds of streaming
6. Prints server transcript response

### Flashing real hardware

```bash
# Install ESP-IDF first (v5.x recommended)
idf.py -p COM3 flash monitor          # Windows
idf.py -p /dev/ttyACM0 flash monitor  # Linux/macOS
```

---

## 5. ML Pipeline

### Audio Preprocessing (`ml/preprocessing/`)

```
WAV/FLAC file
    │
    ▼  load_audio()          soundfile (primary) → librosa (fallback)
    │                        stereo → mono (mean channels)
    ▼  resample()            librosa.resample → 16 000 Hz
    │
    ▼  trim_or_pad()         trim front / zero-pad end → exactly 16 000 samples (1 s)
    │
    ▼  normalize()           peak normalize to [-1, 1]  (silent audio unchanged)
    │
    └─► float32 array (16 000,)
```

**Constants:** `TARGET_SAMPLE_RATE = 16_000`, `TARGET_SAMPLES = 16_000`

### Feature Extraction (`ml/preprocessing/feature_extraction.py`)

```
float32 audio (16 000,)
    │
    ▼  librosa.feature.mfcc()
    │    n_mfcc=40, n_fft=512, win=480 samples (30 ms), hop=160 samples (10 ms)
    │    window="hamming", center=True
    │
    ▼  _fix_frame_count()    trim/pad time axis → exactly 49 frames
    │
    ▼  _normalize()          per-utterance mean-variance normalization
    │
    └─► float32 tensor (49, 40, 1)   ← DS-CNN model input shape
```

**Output shape:** `(NUM_FRAMES=49, NUM_MFCC=40, 1)` — matches `config.yaml` training parameters.

### Embedding Extraction (`ml/personalization/embedding.py`)

```
float32 audio (16 000,)
    │
    ▼  audio_to_features()   → (49, 40, 1)
    │
    ▼  TFLite INT8 inference
    │    Input quantization:  float32 → int8 (per scale/zero_point)
    │    Model: DS-CNN (Depthwise Separable CNN)
    │    Output: 64-D embedding (int8 → dequantize → float32)
    │
    ▼  L2 normalize
    │
    └─► float32 embedding (64,)  ∥e∥₂ = 1.0
```

The TFLite interpreter is cached via `@lru_cache(maxsize=1)` — loaded once per process.

### Personalization / Enrollment (`ml/personalization/enrollment.py`)

```bash
# Enroll from WAV files (requires trained TFLite model)
python -m ml.personalization.enrollment \
    utterance1.wav utterance2.wav utterance3.wav \
    --name my_keyword \
    --model ml/models/int8/model.tflite
```

**Enrollment pipeline:**
1. Preprocess each utterance → float32 (16 000,)
2. Extract embedding via TFLite → (64,)
3. Compute mean embedding → re-normalize → prototype (64,)
4. Validate intra-class consistency (avg pairwise cosine ≥ 0.60 recommended)
5. Save to `ml/personalization/profiles/<name>.npy`

### Matching (`ml/personalization/matcher.py`)

**Single-shot match:**
```python
wake, score = match(embedding, prototype, threshold=0.75)
# score = cosine_similarity(L2_norm(embedding), L2_norm(prototype))
# wake  = score >= threshold
```

**Debounce matcher (stateful, 2-of-N sliding window):**
```python
matcher = DebounceMatcher(prototype, threshold=0.75, hits_required=2, window_size=3)
for chunk_embedding in stream:
    wake, score = matcher.update(chunk_embedding)
    if wake:
        matcher.reset()   # one-shot, prevent double-fire
        trigger_streaming()
```

Mirrors the exact debounce logic in `firmware/esp32/main/matcher.py` (C++ equivalent).

### Evaluation (`ml/evaluation/`)

| Module | Purpose |
|---|---|
| `far_frr.py` | FAR/FRR curves, EER, operating point at max FAR constraint |
| `metrics.py` | Accuracy, precision, recall, F1, ROC-AUC (sklearn-backed) |
| `evaluate.py` | Full evaluation runner with dataset loading |
| `latency.py` | Inference latency profiling (µs per call) |
| `robustness.py` | Noise-robustness sweep (SNR from −5 to +30 dB) |

---

## 6. Server Layer

### Entry Point — `server/receiver/main.py`

**Start the server:**
```bash
uvicorn server.receiver.main:app --host 0.0.0.0 --port 8765
```

**HTTP endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check. Returns `{status, version, active_sessions, asr_ready}` |
| `GET` | `/sessions` | Lists active session IDs with age and utterance count (debug) |
| `WS` | `/v1/stream` | Audio streaming endpoint (binary + JSON frames) |

**WebSocket session lifecycle:**

```
Client                                    Server
  │                                          │
  │── WS upgrade + X-KWS-Token header ──────►│ verify_token()
  │                                          │   └─ HMAC check
  │                                          │   └─ timestamp window (30 s)
  │                                          │   └─ replay detection
  │◄── {"type":"ack", "session_id":...} ─────│
  │                                          │
  │── binary frame (audio chunk) ───────────►│ GapDetector.update(seq)
  │                                          │ SessionAssembler.add_chunk(payload)
  │── binary frame (audio chunk) ───────────►│
  │── binary 0xFF (end-of-utterance) ────────►│ assembler.flush()
  │                                          │ whisper.transcribe(pcm_bytes)
  │◄── {"type":"transcript","text":...} ─────│ IntentHandler.process(text)
  │◄── {"type":"agent_response",...} ────────│
  │                                          │
  │── WS close ──────────────────────────────►│ session cleanup
```

### Security — `server/security/auth.py`

**Token format** (32 raw bytes → base64 encoded):

```
[session_id: 16 bytes] [timestamp_unix: 8 bytes BE] [hmac_tag: 8 bytes]

hmac_tag = HMAC-SHA256(session_id + timestamp, secret)[0:8]
```

| Feature | Implementation |
|---|---|
| Algorithm | HMAC-SHA256 truncated to 8 bytes |
| Replay protection | In-memory token cache; each token usable once |
| Expiry | Configurable window (default 30 s), checked against `time.time()` |
| Comparison | `hmac.compare_digest()` (constant-time, timing-safe) |

**Generate a token (Python):**
```python
from server.security.auth import generate_token
token = generate_token(b"your-secret-key")
# Use as: X-KWS-Token: <token>
```

### Packet Reassembly — `server/packet_reassembly/assembler.py`

- Accumulates raw PCM payload bytes per session in a `bytearray`
- Buffer hard-cap: `MAX_BUFFER_BYTES = 640_000` (20 s @ 16 kHz int16)
- Overflow drops silently with a warning log — connection stays open
- `flush()` returns accumulated bytes and resets the buffer
- `buffered_duration_s` property: `len(buffer) / (16_000 × 2)`

### FEC Gap Detector — `server/fec/decoder.py`

- Tracks uint16 sequence numbers per session
- Detects gaps between consecutive chunks (uint16 wrap-around at 65 535 → 0 handled)
- Returns list of missing sequence numbers on each `update(seq_num)` call
- Server sends `{"type":"retransmit_request","missing_seq":[...]}` to client
- Safety valve: reports at most 100 missing seqs per gap event

### ASR — `server/asr/whisper.py`

Uses **faster-whisper** (CTranslate2 backend):

| Config | Default | Env override |
|---|---|---|
| Model size | `base.en` | `KWS_ASR_MODEL` |
| Device | `cpu` | `KWS_ASR_DEVICE` |
| Compute type | `int8` (CPU) / `float16` (GPU) | auto-detected |
| Beam size | 5 | hardcoded |
| VAD | enabled, 300 ms min silence | hardcoded |

Input: raw 16-bit PCM bytes → `np.int16 / 32768.0` → float32 array → Whisper inference.

### Intent Handler — `server/agent/handler.py`

Rule-based regex intent classifier. Recognized intents:

| Pattern | Intent | Action | Target |
|---|---|---|---|
| "turn on/off light" | `lights_on/off` | on/off | lights |
| "increase/decrease temp" | `temperature_increase/decrease` | increase/decrease | temperature |
| "status/report" | `system_query` | query | system |
| "emergency/abort/halt" | `mission_abort` | abort | mission |
| "go to/navigate to" | `target_navigate` | navigate | target |
| "camera/capture/photo" | `camera_capture` | capture | camera |
| "power on/off/shutdown" | `system_power_on/off` | power_on/off | system |

**Swap point:** Replace `IntentHandler.process()` body with an LLM API call (Ollama, OpenAI, Gemini) when ready. The method signature and return schema are stable.

### TTS — `server/tts/synthesizer.py`

`pyttsx3`-based text-to-speech stub. Returns synthesized audio bytes. Optional — not currently wired into the main streaming path.

---

## 7. Protocol

### Binary Frame Layout (v1)

```
Byte offset  Field             Size   Endian   Value
──────────   ─────             ────   ──────   ─────
0            version           1 B    —        0x01
1            flags             1 B    —        see below
2–3          sequence_number   2 B    BE       0–65535, wraps
4–5          payload_length    2 B    BE       typically 8000
6–7          session_id        2 B    BE       low 2 bytes of session ID
8–11         timestamp_ms      4 B    BE       device uptime in ms
12 … 12+N-1  PCM payload       N B    LE int16 16 kHz mono
12+N … +3    hmac_tag          4 B    —        HMAC-SHA256(header+payload)[0:4]
```

**Standard chunk:** 12 B header + 8 000 B PCM (250 ms) + 4 B HMAC = **8 016 bytes**

**Flags byte:**

| Bit | Name | Set when |
|---|---|---|
| 7 | `PREROLL` | First chunk (500 ms pre-roll buffer) |
| 6 | `LAST` | Final chunk before EOT marker |
| 5 | `FEC_REQUEST` | Server → client retransmit request only |
| 4–0 | Reserved | Must be 0 |

**End-of-utterance marker:** single byte `0xFF`

### JSON Message Types

| Direction | `type` | Key fields |
|---|---|---|
| Server → Client | `ack` | `session_id`, `server_time` |
| Server → Client | `transcript` | `text`, `language`, `latency_ms`, `session_id` |
| Server → Client | `agent_response` | `intent`, `action`, `target`, `response`, `confidence` |
| Server → Client | `error` | `code`, `message`, `fatal` |
| Server → Client | `retransmit_request` | `missing_seq`, `session_id` |

### Error Codes

| Range | Category |
|---|---|
| E1xx | Authentication (missing token, bad HMAC, expired, replay, session limit) |
| E2xx | Audio / Protocol (frame too large, version mismatch, HMAC mismatch, gap, too short, clipped) |
| E3xx | ASR (timeout, decode failed, empty transcript, unavailable) |
| E5xx | Server (internal error, rate limited, shutdown) |

Non-fatal errors (E2xx, E3xx): connection stays open, utterance discarded.  
Fatal errors (E1xx, E5xx): server closes WebSocket; client must reconnect with fresh token.

---

## 8. Data Flow — End to End

```
1. ENROLLMENT (one-time, offline or on-device)
   ─────────────────────────────────────────────
   5–10 WAV utterances
       → audio_preprocessing.preprocess()       # 16 kHz float32
       → feature_extraction.audio_to_features() # (49, 40, 1)
       → embedding.extract_embedding()           # TFLite INT8 → (64,)
       → enrollment._compute_prototype()         # mean + L2-norm
       → save ml/personalization/profiles/<name>.npy

2. DETECTION LOOP (continuous, on-device / simulator)
   ─────────────────────────────────────────────────
   I2S DMA → 250 ms PCM chunk (4 000 int16 samples)
       → MFCC extraction (30 ms frame, 10 ms stride) → (49, 40, 1)
       → DS-CNN TFLite INT8 inference             # ~50 ms on ESP32-S3
       → L2-normalize                             # (64,) unit vector
       → DebounceMatcher.update()                 # 2-of-3 window, cosine ≥ 0.75
       → WAKE DETECTED

3. STREAMING (on wake, ~1–3 s)
   ────────────────────────────
   Pre-roll ring buffer (500 ms, flag=PREROLL)
       + live PCM chunks (250 ms each, seq=1,2,3…)
       + HMAC-SHA256 tag per packet
       → WebSocket binary frames → server

4. SERVER PROCESSING
   ──────────────────
   Auth: verify X-KWS-Token (HMAC, timestamp, replay check)
       → GapDetector: detect missing sequence numbers
       → SessionAssembler: accumulate PCM bytes
       → 0xFF received: assembler.flush() → PCM bytes
       → audio quality gate (≥ 200 ms, not silent)
       → WhisperASR.transcribe() [run_in_executor]  # non-blocking
       → JSON {"type":"transcript","text":"..."}    → client
       → IntentHandler.process(text)
       → JSON {"type":"agent_response",...}          → client
```

---

## 9. Environment & Configuration

All server config is read from environment variables at startup:

| Variable | Default | Description |
|---|---|---|
| `KWS_SECRET` | `dev-secret-do-not-use-in-production` | HMAC shared secret (must match firmware) |
| `KWS_ASR_MODEL` | `base.en` | faster-whisper model identifier |
| `KWS_ASR_DEVICE` | `cpu` | `cpu` or `cuda` |
| `KWS_MAX_SESSIONS` | `10` | Max concurrent WebSocket sessions |
| `KWS_AUTH_WINDOW_S` | `30` | Token timestamp validity window in seconds |

**Production deployment example:**
```bash
export KWS_SECRET="$(openssl rand -hex 32)"
export KWS_ASR_MODEL="small.en"
export KWS_ASR_DEVICE="cuda"
uvicorn server.receiver.main:app --host 0.0.0.0 --port 8765 --workers 1
```

> **Note:** Use `--workers 1` — the ASR model and session dict are global state and not safe for multi-process use without a shared cache (Redis, etc.).

---

## 10. Dependencies

### Unified (`requirements.txt`)

| Package | Version | Purpose |
|---|---|---|
| `tensorflow` | ≥2.15, <2.17 | TFLite interpreter (embedding extraction, training) |
| `tensorflow-model-optimization` | ≥0.7.5 | INT8 quantization |
| `numpy` | ≥1.24 | Numerical core |
| `librosa` | ≥0.10.0 | MFCC, resampling |
| `soundfile` | ≥0.12.1 | WAV/FLAC I/O |
| `audiomentations` | ≥0.30.0 | Training-time audio augmentation |
| `scikit-learn` | ≥1.3.0 | Evaluation metrics |
| `scipy` | ≥1.11.0 | Signal processing utilities |
| `fastapi` | ≥0.110.0 | WebSocket server framework |
| `uvicorn[standard]` | ≥0.29.0 | ASGI server |
| `websockets` | ≥12.0 | WebSocket client (simulator) |
| `faster-whisper` | ≥1.0.0 | CTranslate2 Whisper ASR |
| `pytest` | ≥7.4.0 | Test runner |
| `pytest-asyncio` | ≥0.23.0 | Async test support |
| `pytest-cov` | ≥4.1.0 | Coverage reporting |

### Server-only (`server/requirements.txt`)

Same as above minus `tensorflow`, `audiomentations`, `scikit-learn`, `scipy`. Plus:
- `pyttsx3` ≥2.90 — TTS (Windows/Linux)

---

## 11. Test Suite

Run all software tests (excludes hardware and model-artifact tests):

```bash
python -m pytest tests/ -v -m "not hardware and not model"
```

**Current status: 68/68 PASSED**

| Module | Tests | Coverage |
|---|---|---|
| `tests/unit/test_audio_preprocessing.py` | 16 | load, resample, normalize, trim/pad, full pipeline |
| `tests/unit/test_feature_extraction.py` | 12 | MFCC shape/dtype/finiteness, helpers |
| `tests/unit/test_kws_model.py` | 13 | cosine match, debounce window, batch sequence |
| `tests/integration/test_pipeline.py` | 4 | enrollment→matching→EER, FAR/FRR, debounce |
| `tests/network/test_websocket.py` | 14 | SessionAssembler (7), GapDetector (7) |
| `tests/security/test_auth.py` | 9 | token round-trip, wrong secret, replay, expiry, format |

**Pytest markers:**

| Marker | Skip with | Requires |
|---|---|---|
| `hardware` | `-m "not hardware"` | Physical ESP32-S3 connected |
| `model` | `-m "not model"` | `ml/models/int8/model.tflite` artifact |
| `slow` | `-m "not slow"` | Long-running evaluations |

---

## 12. Scripts & Utilities

| Path | Purpose |
|---|---|
| `scripts/dataset/` | Download Google Speech Commands, preprocess into `.npy` arrays |
| `scripts/training/` | Launch DS-CNN training with `config.yaml` |
| `scripts/conversion/` | Export Keras → TFLite INT8 with representative dataset |
| `scripts/benchmarking/` | Measure inference latency, FAR/FRR on test split |
| `scripts/flashing/` | Flash firmware to ESP32-S3 over USB |
| `scripts/setup/` | One-shot environment setup (pip install + model download) |
| `examples/enrollment_demo.py` | Interactive enrollment walkthrough with 5 utterances |
| `firmware/esp32/main/simulator.py` | Firmware simulator (see §4) |

---

## 13. Known Limitations

| Area | Limitation | Mitigation |
|---|---|---|
| Firmware C++ | `main.cpp` and `app_state.h` are stubs | Use `simulator.py` for full firmware behavior |
| TFLite model | `ml/models/int8/model.tflite` not committed (binary artifact) | Run training + quantization pipeline; tests use mock embeddings |
| ASR server | `--workers 1` only — session dict is in-process | Add Redis session store for multi-worker deployment |
| Token replay cache | In-memory only — lost on server restart | Use Redis with TTL for production |
| HMAC chunk tag | Only 4 bytes (32-bit) per packet | Adequate for LAN; increase to 8 bytes for untrusted networks |
| TTS | `pyttsx3` stub — no audio routed back to device | Implement WebSocket audio-back channel |
| Intent handler | Regex rules only — limited vocabulary | Swap `IntentHandler.process()` for LLM API call |
| numpy 2.x | `server/requirements.txt` pins `numpy<2.0` (faster-whisper constraint) | Project root `requirements.txt` updated to numpy 2.x (librosa requires it) |
