# Error Codes

All error codes returned by the server in `{"type": "error", "code": "...", ...}` frames.

---

## Authentication Errors (E1xx)

| Code | Name | Description | Fatal | Client Action |
|------|------|-------------|-------|---------------|
| E101 | `AUTH_MISSING` | No `X-KWS-Token` header in WebSocket upgrade request. | Yes | Include token in next connection attempt. |
| E102 | `AUTH_INVALID` | Token HMAC verification failed. | Yes | Recompute token with correct shared secret. |
| E103 | `AUTH_EXPIRED` | Token timestamp is older than 30 seconds. | Yes | Generate a new token with current timestamp. |
| E104 | `AUTH_REPLAY` | Token has already been used in this replay window. | Yes | Generate a new token. |
| E105 | `SESSION_LIMIT` | Maximum concurrent sessions (10) reached. | Yes | Retry after another session disconnects. |

---

## Audio / Protocol Errors (E2xx)

| Code | Name | Description | Fatal | Client Action |
|------|------|-------------|-------|---------------|
| E201 | `FRAME_TOO_LARGE` | Binary frame exceeds 32 KB. | No | Reduce chunk size to ≤8016 bytes. |
| E202 | `FRAME_VERSION_MISMATCH` | `version` byte in chunk header is not `0x01`. | No | Update firmware to use protocol v1. |
| E203 | `HMAC_MISMATCH` | Chunk HMAC tag verification failed. | No | Check shared secret configuration. |
| E204 | `SEQUENCE_GAP` | Missing sequence numbers detected (informational). | No | Resend if available, otherwise continue. |
| E205 | `AUDIO_TOO_SHORT` | Assembled utterance is shorter than 200 ms. | No | Ensure wake word is fully captured before streaming. |
| E206 | `AUDIO_CLIPPED` | Audio amplitude exceeds int16 range (overdriven mic). | No | Reduce microphone gain or move further from source. |

---

## ASR Errors (E3xx)

| Code | Name | Description | Fatal | Client Action |
|------|------|-------------|-------|---------------|
| E301 | `ASR_TIMEOUT` | ASR did not return within 2000 ms. | No | Retry by sending another utterance. |
| E302 | `ASR_DECODE_FAILED` | faster-whisper raised an exception during decode. | No | Check server logs for details. |
| E303 | `ASR_EMPTY_TRANSCRIPT` | ASR returned an empty or whitespace-only transcript. | No | Speak more clearly or increase utterance duration. |
| E304 | `ASR_UNAVAILABLE` | ASR model not loaded (server still initializing). | No | Retry after 5 seconds. |

---

## Server Errors (E5xx)

| Code | Name | Description | Fatal | Client Action |
|------|------|-------------|-------|---------------|
| E501 | `INTERNAL_ERROR` | Unhandled exception on the server. | Yes | Reconnect and retry. Report to server administrator. |
| E502 | `RATE_LIMITED` | Client IP exceeded connection rate limit. | Yes | Wait 60 seconds before reconnecting. |
| E503 | `SERVER_SHUTDOWN` | Server is shutting down gracefully. | Yes | Reconnect with exponential backoff. |

---

## Client Handling

### Non-fatal errors

The WebSocket connection remains open. The server discards the current utterance and is ready for the next one. The client should:

1. Log the error code and message.
2. Reset the streaming state (discard current utterance buffer).
3. Return to listening mode.

### Fatal errors

The server closes the WebSocket connection with an appropriate close code after sending the error frame. The client should:

1. Log the error.
2. Wait for the exponential backoff period (1 s → 2 s → 4 s → ... → 60 s max).
3. Reconnect with a fresh session token.
