# Requirements

Problem Statement: SIH 2026 — PS26172
System: Ultra-Low-Latency Edge Keyword Spotting for Space/Defence Command Applications

---

## 1. Functional Requirements

### 1.1 Keyword Spotting (Edge)

| ID | Requirement |
|----|-------------|
| FR-01 | The device shall detect a user-configured wake word from continuous audio without cloud connectivity. |
| FR-02 | The system shall support runtime enrollment of arbitrary wake words (no retraining). |
| FR-03 | Enrollment shall require exactly 10 spoken utterances of the wake word. |
| FR-04 | The KWS model shall produce a 64-dimensional embedding per 1-second audio frame. |
| FR-05 | Wake detection shall use cosine similarity with a 2-of-3 sliding-window debounce to suppress false alarms. |
| FR-06 | The device shall maintain a 500 ms pre-roll audio buffer so the start of the command is not lost. |

### 1.2 Audio Pipeline (Edge)

| ID | Requirement |
|----|-------------|
| FR-07 | Audio shall be captured at 16 000 Hz, mono, 16-bit signed PCM via I2S interface. |
| FR-08 | An energy-based Voice Activity Detector (VAD) shall gate the KWS inference to reduce idle CPU load. |
| FR-09 | After wake detection the device shall stream the pre-roll followed by live audio to the cloud server. |
| FR-10 | Audio shall be transmitted in 250 ms chunks (8 000 bytes). |
| FR-11 | An end-of-utterance marker (`0xFF` byte) shall be sent when silence is detected after the command. |

### 1.3 Communication

| ID | Requirement |
|----|-------------|
| FR-12 | The device shall communicate with the cloud server over a persistent WebSocket connection. |
| FR-13 | The WebSocket connection shall use TLS (`wss://`) in production. |
| FR-14 | Each audio chunk shall be tagged with a 2-byte sequence number. |
| FR-15 | The client shall include an HMAC-SHA256 token in the WebSocket handshake header for session authentication. |
| FR-16 | The server shall detect and log missing sequence numbers. |

### 1.4 Cloud Server (ASR + Agent)

| ID | Requirement |
|----|-------------|
| FR-17 | The server shall receive binary PCM audio via WebSocket and reassemble per-utterance audio. |
| FR-18 | The server shall transcribe speech using faster-whisper (`base.en` model). |
| FR-19 | The server shall return a JSON transcript frame: `{type, text, latency_ms}`. |
| FR-20 | The server shall expose a `/health` HTTP endpoint returning `{status: "ok"}`. |
| FR-21 | An intent-handler agent shall process transcript text and return a structured response. |
| FR-22 | The server shall support at least 10 simultaneous WebSocket sessions. |

### 1.5 Personalization

| ID | Requirement |
|----|-------------|
| FR-23 | Enrolled wake-word prototypes shall be stored in ESP32 NVS flash and survive power cycles. |
| FR-24 | The user shall be able to delete and re-enroll a wake word via serial command. |

---

## 2. Non-Functional Requirements

### 2.1 Latency

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-01 | Wake-word detection latency (mic → wake decision) | ≤ 100 ms |
| NFR-02 | Wake-to-cloud-stream latency (wake decision → first byte sent) | ≤ 50 ms |
| NFR-03 | End-to-end latency (wake → ASR transcript received) | ≤ 200 ms |
| NFR-04 | TFLite INT8 inference time per frame on ESP32-S3 | ≤ 30 ms |

### 2.2 Resource Budget

| ID | Requirement | Budget |
|----|-------------|--------|
| NFR-05 | TFLite INT8 model size in flash | ≤ 60 KB |
| NFR-06 | TFLite tensor arena (SRAM) | ≤ 80 KB |
| NFR-07 | Audio ring buffer | 32 KB |
| NFR-08 | Pre-roll buffer | 16 KB |
| NFR-09 | Total peak SRAM usage | ≤ 256 KB |
| NFR-10 | Idle CPU utilization on ESP32-S3 | < 10% |

### 2.3 Accuracy

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-11 | False Accept Rate (FAR) on unseen speakers | < 5% |
| NFR-12 | False Reject Rate (FRR) on enrolled speaker | < 10% |
| NFR-13 | Keyword detection accuracy (clean audio) | ≥ 90% |
| NFR-14 | Robustness at 10 dB SNR | ≥ 80% accuracy |

### 2.4 Security

| ID | Requirement |
|----|-------------|
| NFR-15 | All cloud communication shall use TLS 1.2 or higher. |
| NFR-16 | Session tokens shall be HMAC-SHA256 with a 256-bit shared secret. |
| NFR-17 | NVS partition storing wake-word prototypes shall use ESP32 NVS encryption. |

### 2.5 Reliability

| ID | Requirement |
|----|-------------|
| NFR-18 | The firmware shall automatically reconnect to WiFi and WebSocket server after disconnect. |
| NFR-19 | The system shall remain operational after 72 hours of continuous use without restart. |

### 2.6 Portability / Maintainability

| ID | Requirement |
|----|-------------|
| NFR-20 | The ML pipeline shall be fully reproducible from a single `config.yaml`. |
| NFR-21 | All public Python functions shall have type annotations and docstrings. |
| NFR-22 | The firmware shall compile without modification for ESP32-S3 using ESP-IDF v5.2. |
