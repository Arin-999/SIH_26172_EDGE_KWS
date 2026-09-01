# Message Types

All WebSocket message types exchanged between the ESP32-S3 client and the FastAPI server.

---

## Client → Server (Binary Frames)

### `audio_chunk`

Binary frame following the format in [`packet-format.md`](packet-format.md).

Sent continuously during a streaming session after wake detection.

### `end_utterance`

Single byte `0xFF`. No header.

Signals the server to flush the accumulated audio buffer and begin ASR transcription.

---

## Server → Client (JSON Text Frames)

All server → client messages are UTF-8 JSON text frames.

### `transcript`

Sent after ASR completes for the current utterance.

```json
{
  "type": "transcript",
  "text": "turn on the lights",
  "language": "en",
  "latency_ms": 142,
  "session_id": "a3f2b1c0d4e5f6a7"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"transcript"` |
| `text` | string | Transcribed text from ASR |
| `language` | string | Detected language code (ISO 639-1) |
| `latency_ms` | integer | Time from `0xFF` received to transcript sent (ms) |
| `session_id` | string | Echo of session identifier |

---

### `agent_response`

Sent after the intent agent processes the transcript.

```json
{
  "type": "agent_response",
  "intent": "light_control",
  "action": "on",
  "target": "lights",
  "response": "Lights turned on.",
  "confidence": 0.94
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"agent_response"` |
| `intent` | string | Classified intent name |
| `action` | string | Action to perform |
| `target` | string | Target entity |
| `response` | string | Human-readable response text |
| `confidence` | float | Agent confidence [0.0–1.0] |

---

### `ack`

Sent by the server to confirm receipt of the WebSocket connection and authenticate session.

```json
{
  "type": "ack",
  "session_id": "a3f2b1c0d4e5f6a7",
  "server_time": 1722430800
}
```

---

### `retransmit_request`

Sent by the server when a sequence number gap is detected.

```json
{
  "type": "retransmit_request",
  "missing_seq": [3, 4],
  "session_id": "a3f2b1c0d4e5f6a7"
}
```

The client should resend the indicated sequence numbers if still in its transmit buffer. If unavailable, the client should ignore (the server will proceed with the gap).

---

### `error`

Sent when a recoverable or unrecoverable error occurs server-side.

```json
{
  "type": "error",
  "code": "E002",
  "message": "ASR decode failed: audio too short",
  "fatal": false
}
```

If `fatal` is `true`, the server will close the WebSocket connection immediately after sending this message.

See [`error-codes.md`](error-codes.md) for all error codes.

---

### `ping` / `pong`

Standard WebSocket protocol-level keep-alive. The server sends a WebSocket `ping` every 30 seconds. The client must respond with `pong`. If no pong is received within 10 seconds, the server closes the connection.

ESP-IDF's `esp_websocket_client` handles ping/pong automatically.
