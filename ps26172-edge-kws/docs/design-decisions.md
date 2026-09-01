# Design Decisions

Architecture Decision Records (ADRs) for SIH PS26172 — Edge KWS System.

---

## ADR-01: DS-CNN over MobileNetV2 for KWS

**Status:** Accepted

**Context:** The KWS model must fit in ≤60 KB of flash and use ≤80 KB SRAM for the tensor arena. MobileNetV2 at its smallest configuration is ~160 KB INT8. We need a model purpose-built for audio keyword spotting on MCUs.

**Decision:** Use a Depthwise Separable CNN (DS-CNN) with 4 DS blocks, as described in _"Hello Edge: Keyword Spotting on Microcontrollers"_ (Zhang et al., 2018). This architecture is specifically designed for 16 kHz MFCC input and achieves <60 KB INT8.

**Consequences:** Model capacity is intentionally limited. Compensate with prototypical few-shot learning instead of large classification head.

---

## ADR-02: Prototypical / Embedding-based KWS over Softmax Classification

**Status:** Accepted

**Context:** A softmax classifier requires retraining when the wake word changes. The system requirement (FR-02) mandates runtime enrollment without retraining.

**Decision:** Train the DS-CNN as an **embedding network** using prototypical loss (few-shot learning). The 64-dim L2-normalized embedding is compared to a stored prototype via cosine similarity at runtime.

**Consequences:**
- Wake word can be changed by re-enrolling (recording 10 utterances) without reflashing.
- The threshold is a runtime parameter, tunable per environment.
- Requires a representative embedding space, which means training on a diverse dataset (Google Speech Commands v2).

---

## ADR-03: Raw PCM WebSocket over Opus / MQTT

**Status:** Accepted

**Context:** The architecture supports compression and packetization. Multiple transport options were considered: raw PCM WebSocket, Opus-compressed WebSocket, MQTT binary.

**Decision:** Use raw PCM over WebSocket with sequence-numbered 250 ms chunks.

**Rationale:**
- ESP-IDF has a mature `esp_websocket_client` component.
- Opus encoding on the MCU is possible but adds 64 KB+ of flash overhead and ~15 ms encoding latency.
- At 16 kHz mono int16, raw PCM is 32 kbps — well within a typical WiFi link.
- MQTT is designed for small messages, not streaming audio; adds broker dependency.

**Consequences:** Higher bandwidth than Opus. Acceptable for WiFi/LAN. Not suitable for cellular/IoT narrow-band networks (future work).

---

## ADR-04: Sequence-Number Gap Detection over True Reed-Solomon FEC

**Status:** Accepted

**Context:** README specifies FEC. True Reed-Solomon on an ESP32-S3 is computationally feasible but adds code complexity and memory overhead.

**Decision:** Use **sequence-numbered chunks** (2-byte seq in each chunk header) with server-side gap detection and optional retransmit request. Full RS FEC deferred to future work.

**Rationale:**
- WebSocket over TCP already provides reliable in-order delivery on a LAN.
- RS FEC is primarily useful over lossy links (RF, UDP). TCP makes it redundant on WiFi LAN.
- If UDP transport is needed in future, the chunk header already reserves a `flags` byte for FEC type.

---

## ADR-05: HMAC-SHA256 Session Auth + wss:// TLS

**Status:** Accepted

**Context:** All communication must be authenticated and encrypted (NFR-15, NFR-16).

**Decision:**
1. **Transport encryption:** `wss://` (WebSocket over TLS 1.2+). ESP-IDF mbedTLS handles this.
2. **Session authentication:** HMAC-SHA256 token in the HTTP upgrade header (`X-KWS-Token`). Token = HMAC(session_id + timestamp, shared_secret). Replay window: 30 seconds.
3. **NVS encryption:** ESP32-S3 NVS encryption partition using 256-bit AES key burned to eFuse.

**Consequences:** Shared secret must be provisioned to the device at flash time via `menuconfig`. Key rotation requires reflash (acceptable for hackathon prototype).

---

## ADR-06: 500 ms Pre-roll Buffer

**Status:** Accepted

**Context:** The user begins speaking the command immediately after the wake word. Without pre-roll, the first 100–300 ms of the command is lost while the system transitions from IDLE → STREAM.

**Decision:** Maintain a circular ring buffer of 500 ms (= 16 000 bytes at 16 kHz int16) that continuously captures audio. On wake detection, this buffer is the first payload sent to the server before live streaming begins.

**Consequences:** 16 KB SRAM permanently allocated. Included in the ≤256 KB total budget.

---

## ADR-07: TFLite Micro for Edge Inference

**Status:** Accepted

**Context:** The INT8 quantized model must run on the ESP32-S3 Xtensa LX7 (no FPU for int8 SIMD).

**Decision:** Use **TensorFlow Lite Micro** (TFLM) with the ESP32 optimized kernels (`esp-tflite-micro` component). TFLM has no dynamic memory allocation after initialization, which is critical for deterministic latency.

**Consequences:** Model must be stored as a C byte array in flash. Tensor arena is stack-allocated (80 KB). TFLM v1 API is stable.

---

## ADR-08: Google Speech Commands v2 as Training Dataset

**Status:** Accepted

**Context:** Training requires a large labeled audio dataset. Collecting custom data is out of scope for the hackathon timeline.

**Decision:** Use the **Google Speech Commands v2** dataset (~105 000 1-second WAV clips, 35 classes). Train the embedding network on all 35 classes. At enrollment time, any new utterance (custom wake word) is embedded into the same space.

**Consequences:** Model generalizes to arbitrary short words. Performance on very long or non-English wake words may degrade (acceptable; document as limitation).
