# Packet Format

Binary wire format for audio chunks sent from the ESP32-S3 edge device to the cloud server.

---

## Audio Chunk Frame

Each binary WebSocket frame carries one 250 ms audio chunk using this layout:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
├───────────────────────────────────────────────────────────────────┤
│  version (1B) │  flags (1B)   │     sequence_number (2B, BE)      │
├───────────────────────────────────────────────────────────────────┤
│             payload_length (2B, BE)    │    session_id (2B, BE)   │
├───────────────────────────────────────────────────────────────────┤
│              timestamp_ms (4B, BE)                                │
├───────────────────────────────────────────────────────────────────┤
│              PCM payload (payload_length bytes)                   │
│              16 kHz, mono, int16 little-endian                    │
│              (8000 bytes = 250 ms per standard chunk)             │
├───────────────────────────────────────────────────────────────────┤
│              hmac_tag (4B) — first 4 bytes of HMAC-SHA256         │
└───────────────────────────────────────────────────────────────────┘
```

### Field Definitions

| Field | Size | Endian | Description |
|-------|------|--------|-------------|
| `version` | 1 byte | — | Protocol version. Always `0x01` for v1. |
| `flags` | 1 byte | — | Bit flags (see below) |
| `sequence_number` | 2 bytes | Big-endian | Monotonically increasing per session, wraps at 65535 |
| `payload_length` | 2 bytes | Big-endian | Length of PCM payload in bytes (typically 8000) |
| `session_id` | 2 bytes | Big-endian | Lower 2 bytes of the 16-byte session ID |
| `timestamp_ms` | 4 bytes | Big-endian | Device uptime in milliseconds at capture time |
| `PCM payload` | `payload_length` bytes | Little-endian int16 | Raw 16-bit signed PCM audio samples |
| `hmac_tag` | 4 bytes | — | First 4 bytes of `HMAC-SHA256(header[0:12] + payload, secret)` |

**Total header size:** 12 bytes  
**Total frame size (standard chunk):** 12 + 8000 + 4 = **8016 bytes**

---

### Flags Byte

| Bit | Name | Description |
|-----|------|-------------|
| 7 | `PREROLL` | Set if this chunk is from the pre-roll buffer (sent before live stream) |
| 6 | `LAST` | Set on the last chunk before end-of-utterance marker |
| 5 | `FEC_REQUEST` | Server sets this in retransmit requests (not used client→server) |
| 4–0 | Reserved | Must be `0` |

---

## End-of-Utterance Marker

A single-byte binary WebSocket frame containing `0xFF` signals the end of an utterance.

```
┌────────┐
│  0xFF  │
└────────┘
```

The server flushes the assembled utterance and begins ASR transcription upon receiving this marker.

---

## Pre-Roll Chunk

The first binary frame after WebSocket connection (or after a wake event) contains the 500 ms pre-roll buffer:

- `flags` bit 7 (`PREROLL`) is set
- `payload_length` = 16000 (500 ms × 16 000 Hz × 2 bytes = 16 000 bytes)
- `sequence_number` = 0

---

## Example: Full Utterance Transmission

```
Frame 1:  version=1, flags=0x80 (PREROLL), seq=0,  len=16000  [500 ms pre-roll]
Frame 2:  version=1, flags=0x00,           seq=1,  len=8000   [250 ms live]
Frame 3:  version=1, flags=0x00,           seq=2,  len=8000   [250 ms live]
Frame 4:  version=1, flags=0x00,           seq=3,  len=8000   [250 ms live]
Frame 5:  version=1, flags=0x40 (LAST),    seq=4,  len=4800   [150 ms live, final]
Frame 6:  0xFF                                                 [end-of-utterance]
```

Total audio: 500 + 250×3 + 150 = **1400 ms** transmitted, ~1000 ms actual command.
