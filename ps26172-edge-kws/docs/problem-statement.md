# Problem Statement — PS26172

**Smart India Hackathon 2026**
**Organization:** Indian Space Research Organisation (ISRO)
**Theme:** Smart Automation
**Category:** Software

---

## Statement

> Design and develop an ultra-low-latency keyword spotting (KWS) system for edge devices that enables voice-activated command execution in resource-constrained environments. The system should support custom wake-word enrollment without retraining, operate within strict memory and power budgets, and interface with a cloud ASR pipeline for full command transcription.

---

## Context

ISRO ground stations and satellite control systems increasingly incorporate voice-command interfaces for operator efficiency. These systems must function in noisy environments (server rooms, launch pads), handle operator-specific wake words, and maintain strict security and latency guarantees.

Existing commercial solutions (Amazon Alexa, Google Assistant) are unsuitable due to:
- Cloud dependency for wake-word detection (unacceptable latency and privacy concerns)
- No support for custom wake words without expensive retraining
- Large memory footprints incompatible with embedded controllers

---

## Constraints

| Constraint | Value |
|-----------|-------|
| Target platform | ESP32-S3 (Xtensa LX7, 240 MHz, 512 KB SRAM + 8 MB PSRAM) |
| Maximum SRAM for KWS subsystem | 256 KB |
| Maximum model flash size | 60 KB (INT8 TFLite) |
| Idle CPU utilization | < 10% |
| Wake-to-cloud latency | < 200 ms |
| False Accept Rate | < 5% |
| False Reject Rate | < 10% |
| Wake-word enrollment | Runtime, no retraining, ≤ 10 utterances |
| Communication | WiFi (802.11 b/g/n), WebSocket, TLS |
| Audio | 16 kHz, mono, 16-bit PCM |

---

## Deliverables

1. Trained INT8 TFLite KWS model (DS-CNN + prototypical embedding)
2. ESP32-S3 firmware (ESP-IDF) with I2S audio, inference, enrollment, and streaming
3. Cloud FastAPI server with faster-whisper ASR and intent agent
4. Evaluation benchmarks (FAR, FRR, latency, RAM)
5. Complete source code, documentation, and deployment guide

---

## Success Criteria

The system is considered successful when the following flow works reliably:

1. User configures a custom wake word (enrollment, ≤ 10 utterances)
2. Device enters low-power listening (< 10% CPU idle)
3. User speaks the wake word → device detects it within 100 ms
4. Device captures the command with 500 ms pre-roll
5. Audio is streamed securely to the cloud server
6. ASR transcribes the command → response received within 200 ms
7. Device returns to listening
