# Testing Plan

---

## 1. Strategy

Testing is organized in four layers, executed bottom-up:

```
Unit tests       → fast, no hardware, no network
KWS tests        → requires trained TFLite model
Integration      → requires running server
End-to-end       → full pipeline (PC simulation of ESP32)
```

Hardware-in-the-loop (HIL) testing with a real ESP32-S3 is performed manually during the integration milestone.

---

## 2. Unit Tests (`tests/unit/`)

### 2.1 Preprocessing
File: `tests/unit/test_preprocessing.py`

| Test | Assertion |
|------|-----------|
| `test_load_and_normalize` | Output is `float32`, range `[-1, 1]`, shape `(16000,)` |
| `test_trim_to_1s` | Clips longer than 1s are trimmed correctly |
| `test_pad_to_1s` | Clips shorter than 1s are zero-padded correctly |
| `test_mfcc_shape` | `feature_extraction.py` output shape is `(49, 40, 1)` |
| `test_mfcc_dtype` | Output dtype is `float32` |
| `test_mfcc_normalization` | Per-utterance mean ≈ 0, std ≈ 1 |

### 2.2 Augmentation
File: `tests/unit/test_augmentation.py`

| Test | Assertion |
|------|-----------|
| `test_noise_shape` | Augmented audio has same shape as input |
| `test_time_shift_shape` | Time-shifted audio has same shape |
| `test_speed_perturb_range` | Speed factor stays in [0.9, 1.1] |
| `test_augmented_is_different` | Augmented != original (augmentation applied) |

### 2.3 Personalization
File: `tests/unit/test_personalization.py`

| Test | Assertion |
|------|-----------|
| `test_cosine_identical` | Same embedding → similarity = 1.0 |
| `test_cosine_orthogonal` | Orthogonal embeddings → similarity = 0.0 |
| `test_matcher_true` | High-similarity embedding returns `wake=True` |
| `test_matcher_false` | Low-similarity embedding returns `wake=False` |
| `test_debounce_2_of_3` | 2 hits out of 3 frames → wake confirmed |
| `test_debounce_1_of_3` | Only 1 hit → no wake |

---

## 3. KWS Model Tests (`tests/kws/`)

### 3.1 Model Inference
File: `tests/kws/test_model.py`

Prerequisite: `ml/models/int8/model.tflite` must exist (run training pipeline first).

| Test | Assertion |
|------|-----------|
| `test_model_loads` | TFLite model loads without error |
| `test_model_size` | `model.tflite` file size ≤ 61 440 bytes (60 KB) |
| `test_output_shape` | Inference output shape is `(1, 64)` |
| `test_output_normalized` | L2 norm of output ≈ 1.0 (unit-norm embedding) |
| `test_inference_latency` | 100 inferences, p95 latency < 50 ms (on CPU) |

### 3.2 FAR / FRR
File: `tests/kws/test_far_frr.py`

Prerequisite: `ml/benchmarks/far_frr.csv` must exist (run `ml/evaluation/evaluate.py` first).

| Test | Assertion |
|------|-----------|
| `test_far_below_5pct` | FAR at optimal threshold < 0.05 |
| `test_frr_below_10pct` | FRR at optimal threshold < 0.10 |
| `test_eer_exists` | EER row exists in CSV |

---

## 4. Audio Tests (`tests/audio/`)

File: `tests/audio/test_reconstruction.py`

| Test | Assertion |
|------|-----------|
| `test_pcm_roundtrip` | `bytes → numpy → bytes` is bit-identical |
| `test_validate_short_audio` | Audio < 0.1s flagged as invalid |
| `test_validate_clipping` | Amplitude > 1.0 after normalization flagged |

---

## 5. Network Tests (`tests/network/`)

File: `tests/network/test_websocket.py`

Prerequisite: FastAPI test server started via `pytest` fixture.

| Test | Assertion |
|------|-----------|
| `test_health_endpoint` | `GET /health` returns 200 + `{"status": "ok"}` |
| `test_ws_connect` | WebSocket handshake succeeds with valid token |
| `test_ws_reject_no_token` | Handshake rejected (403) without token |
| `test_ws_audio_transcript` | Send 5 PCM chunks + `0xFF` → JSON transcript received |
| `test_ws_latency` | Transcript received within 300 ms of `0xFF` on loopback |

---

## 6. Security Tests (`tests/security/`)

File: `tests/security/test_auth.py`

| Test | Assertion |
|------|-----------|
| `test_valid_token_accepted` | Valid HMAC token → 101 Upgrade |
| `test_invalid_token_rejected` | Tampered token → 403 |
| `test_expired_token_rejected` | Token with timestamp > 30s ago → 403 |
| `test_replay_attack_rejected` | Same token used twice within window → 403 |

---

## 7. Integration Tests (`tests/integration/`)

File: `tests/integration/test_pipeline.py`

| Test | Assertion |
|------|-----------|
| `test_full_server_pipeline` | PCM bytes → assembler → ASR → agent → JSON response, <500ms |
| `test_gap_detection` | Skip seq number → gap logged, remaining audio transcribed |
| `test_multi_session` | 5 concurrent WebSocket connections each receive their transcript |

---

## 8. End-to-End Tests (`tests/end-to-end/`)

File: `tests/end-to-end/test_e2e.py`

Simulates ESP32 behaviour from Python. Requires server running.

| Test | Assertion |
|------|-----------|
| `test_e2e_preroll_plus_stream` | Send 500ms pre-roll + 2s speech + `0xFF` → transcript contains words |
| `test_e2e_latency` | Wake detection → transcript ≤ 250 ms on local loopback |
| `test_e2e_reconnect` | Kill server mid-stream → client reconnects → next utterance succeeds |

---

## 9. Hardware-in-the-Loop (Manual)

To be performed with physical ESP32-S3 + INMP441 microphone:

1. Flash firmware: `idf.py flash monitor`
2. Serial enroll: type `enroll`, speak wake word 10 times
3. Verify: `"Enrollment complete"` printed on serial
4. Speak wake word → verify `"WAKE DETECTED"` printed
5. Speak a command → verify JSON transcript printed on serial
6. Measure: capture serial timestamps, assert wake-to-transcript ≤ 200 ms
7. Disconnect WiFi AP → verify device reconnects within 10 s
8. Power cycle → verify prototype survives (NVS persistent)

---

## 10. Running Tests

```bash
cd ps26172-edge-kws
pip install -r ml/requirements-dev.txt

# Unit + KWS + audio + network + security
pytest tests/unit/ tests/kws/ tests/audio/ tests/network/ tests/security/ -v

# Integration (starts server automatically via fixture)
pytest tests/integration/ -v

# End-to-end (start server manually first)
uvicorn server.receiver.main:app --port 8765 &
pytest tests/end-to-end/ -v

# All tests with coverage
pytest --cov=ml --cov=server tests/ -v
```
