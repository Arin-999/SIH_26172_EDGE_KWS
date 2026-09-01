# WebSocket Streaming Protocol

## Endpoint

`WS /v1/stream`

## Client → Server

| Message | Format |
|---------|--------|
| Audio chunk | Raw PCM int16 LE, 16 kHz mono |
| End utterance | Single byte `0xFF` |

Chunk size: 8000 bytes (250 ms @ 16 kHz).

## Server → Client

JSON text frames:

```json
{"type": "transcript", "text": "turn on lights", "latency_ms": 142}
{"type": "error", "message": "decode failed"}
```

## Pre-roll

Edge device sends 500 ms (16000 bytes) of buffered audio immediately on wake detection before live stream continues.
