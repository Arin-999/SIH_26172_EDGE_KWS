# Project Proposal — SIH 2026

**Problem:** PS26172
**Team:** SIH-26172-EDGE-KWS
**Institution:** [Your Institution Name]

---

## 1. Executive Summary

We propose an ultra-low-latency hybrid edge-cloud voice activation system for ISRO ground-station operators. The system runs a custom INT8 neural keyword spotter on an ESP32-S3 microcontroller, enabling operators to configure arbitrary wake words at runtime — no retraining required. On wake detection, audio is streamed over a secure WebSocket to a cloud faster-whisper ASR server for full command transcription and intent routing.

The system satisfies all PS26172 constraints: ≤256 KB SRAM, <10% idle CPU, <200 ms end-to-end latency, FAR < 5%, FRR < 10%.

---

## 2. Team

| Role | Responsibility |
|------|---------------|
| Member 1 — AI/ML | KWS model design, training, quantization, evaluation, personalization |
| Member 2 — Software/Systems | ESP32 firmware, server, communication, testing, CI |

---

## 3. Technical Approach

### 3.1 Edge KWS Model

We train a **Depthwise Separable CNN (DS-CNN)** as an embedding network using prototypical few-shot learning on Google Speech Commands v2 (105 000 utterances, 35 classes).

The model outputs a **64-dimensional L2-normalized embedding** per 1-second audio frame (40 log-Mel MFCC coefficients, 49 frames). At enrollment time, 10 user utterances produce a mean prototype embedding stored in NVS flash. At runtime, cosine similarity between the live embedding and the prototype triggers wake detection.

INT8 post-training quantization reduces model size to ≤60 KB with <2% accuracy degradation.

### 3.2 Firmware Architecture

The ESP32-S3 firmware (ESP-IDF v5.2) runs four FreeRTOS tasks:
- **Audio task:** I2S DMA capture → 32 KB ring buffer + 16 KB pre-roll
- **KWS task:** VAD gate → MFCC → TFLite Micro inference → cosine matcher
- **Stream task:** WebSocket client → pre-roll + live PCM → end-of-utterance marker
- **Serial task:** Enrollment commands, debug output

### 3.3 Cloud Server

FastAPI WebSocket server (`ws://:8765/v1/stream`) receives binary PCM, reassembles utterances via sequence numbers, and pipes audio to faster-whisper (`base.en`). An intent handler routes the transcript to a response. End-to-end server latency target: <150 ms.

### 3.4 Security

- **Transport:** TLS (`wss://`) via ESP-IDF mbedTLS
- **Authentication:** HMAC-SHA256 session token in WebSocket upgrade header, 30-second replay window
- **Storage:** NVS partition encryption using AES-256 key in ESP32-S3 eFuse

---

## 4. Novelty

1. **Runtime custom wake-word enrollment** using prototypical few-shot learning — no retraining, no cloud round-trip for enrollment.
2. **Deterministic MCU inference** using TFLite Micro with statically allocated tensor arena — no heap allocation after boot.
3. **500 ms pre-roll buffer** ensures zero command truncation despite KWS processing latency.

---

## 5. Timeline

| Week | Milestone |
|------|-----------|
| 1 | Dataset pipeline, MFCC extraction, model training |
| 2 | INT8 quantization, TFLite export, evaluation (FAR/FRR) |
| 3 | ESP32 firmware: audio, VAD, MFCC, KWS inference |
| 4 | Firmware: enrollment, WebSocket streaming, WiFi |
| 5 | Cloud server: receiver, ASR, agent, security |
| 6 | Integration, end-to-end testing, benchmarking |
| 7 | Documentation, final demo |

---

## 6. Expected Outcomes

| Metric | Target | Justification |
|--------|--------|--------------|
| Model flash size | ≤ 60 KB INT8 | DS-CNN architecture benchmark |
| SRAM usage | ≤ 256 KB | Resource budget analysis |
| Idle CPU | < 10% | VAD gating eliminates continuous inference |
| FAR | < 5% | Prototypical network + cosine threshold |
| FRR | < 10% | 2-of-3 debounce reduces false rejects |
| End-to-end latency | < 200 ms | WiFi LAN + faster-whisper base.en benchmarks |
| Enrollment utterances | 10 | Validated in personalization experiments |

---

## 7. Hardware Bill of Materials

| Component | Qty | Approx. Cost |
|-----------|-----|-------------|
| ESP32-S3-DevKitC-1 | 1 | ₹800 |
| INMP441 I2S MEMS Microphone | 1 | ₹200 |
| USB-C cable | 1 | ₹100 |
| Jumper wires | 1 set | ₹50 |
| Cloud server (local PC or VM) | 1 | — |
| **Total** | | **≈ ₹1150** |
