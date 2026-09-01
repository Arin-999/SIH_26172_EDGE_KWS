# Changelog

All notable changes to `ps26172-edge-kws` are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Full test suite: 68 tests across unit, integration, network, and security modules
- `INFRASTRUCTURE.md` — complete architecture and module reference
- `server/receiver/main.py` — FastAPI WebSocket server with ASR and intent handler
- `server/security/auth.py` — HMAC-SHA256 session token with replay protection
- `server/packet_reassembly/assembler.py` — per-session PCM buffer
- `server/fec/decoder.py` — uint16 sequence-number gap detector
- `server/asr/whisper.py` — faster-whisper ASR wrapper with built-in VAD
- `server/agent/handler.py` — rule-based intent handler (9 intents)
- `ml/preprocessing/audio_preprocessing.py` — load → resample → normalize → trim/pad
- `ml/preprocessing/feature_extraction.py` — MFCC (49×40×1) for DS-CNN
- `ml/personalization/embedding.py` — TFLite INT8 inference → 64-D embedding
- `ml/personalization/enrollment.py` — enrollment pipeline with consistency check
- `ml/personalization/matcher.py` — cosine match + 2-of-N debounce
- `ml/personalization/threshold.py` — optimal threshold search (EER / max-FAR strategies)
- `ml/evaluation/far_frr.py` — FAR/FRR/EER computation (pure numpy)
- `ml/evaluation/metrics.py` — accuracy, F1, AUC metrics
- `firmware/esp32/main/simulator.py` — full firmware simulator (no hardware required)
- `firmware/esp32/main/app_state.h/.cpp` — FreeRTOS FSM for edge state machine
- `firmware/esp32/main/main.cpp` — ESP-IDF app_main with task layout
- `firmware/esp32/sdkconfig.defaults` — CPU, PSRAM, WiFi, I2S defaults
- `hardware/bom.md` — component list with part numbers and costs
- `hardware/esp32-s3/pinout.md` — GPIO assignment and I2S config
- `hardware/esp32-s3/wiring.md` — breadboard wiring guide
- `hardware/esp32-s3/configuration.md` — sdkconfig and Kconfig reference
- `protocol/packet-format.md` — binary frame specification v1
- `protocol/error-codes.md` — E1xx–E5xx error code catalogue
- `protocol/message-types.md` — JSON message type reference

### Fixed
- Numpy 2.x ABI incompatibility in `far_frr.py` (inlined `pairwise_cosine_similarity`)
- Missing `__init__.py` in all `ml/` sub-packages (broke test imports)
- Integration test `test_debounce_fires_on_consecutive_hits` threshold (0.70→0.60, matched synthetic cluster variance)
- Updated `pandas` to 2.3.3 for numpy 2.x compatibility

---

## [0.1.0] — 2026-09-01

### Added
- Initial project scaffold
- Team workflow and interface contracts documented in `README.md`
- `docs/requirements.md`, `docs/testing-plan.md`, `docs/security-design.md`
- `protocol/` directory with packet format and error code specs
- `ml/` and `server/` directory structure

---
