# User Flow

Sequence diagrams and state descriptions for the three primary user interactions.

---

## Flow 1 — Initial Setup & Enrollment

```mermaid
sequenceDiagram
    actor User
    participant ESP32 as ESP32-S3
    participant NVS as NVS Flash
    participant Serial as Serial Monitor

    User->>Serial: idf.py monitor (open serial)
    Serial->>ESP32: boot
    ESP32->>ESP32: Initialize WiFi, I2S, KWS
    ESP32->>Serial: "Ready. No wake word enrolled."
    ESP32->>Serial: "Send 'enroll' to begin enrollment."

    User->>Serial: enroll
    ESP32->>Serial: "Enrollment mode. Say your wake word 10 times."
    loop 10 utterances
        ESP32->>User: "Ready [N/10]. Speak now."
        User->>ESP32: speaks wake word
        ESP32->>ESP32: capture audio → MFCC → embedding
        ESP32->>Serial: "Got utterance N. Score: 0.94"
    end
    ESP32->>ESP32: compute mean prototype embedding
    ESP32->>NVS: store prototype (64 × float32 = 256 bytes)
    ESP32->>Serial: "Enrollment complete. Threshold: 0.75"
    ESP32->>Serial: "Entering listening mode."
```

---

## Flow 2 — Normal Wake-Word Detection → ASR

```mermaid
sequenceDiagram
    actor User
    participant Mic as Microphone
    participant ESP32 as ESP32-S3
    participant WS as WebSocket Server
    participant Whisper as faster-whisper ASR
    participant Agent as Intent Agent

    loop Always-on listening
        Mic->>ESP32: PCM audio (16 kHz mono int16)
        ESP32->>ESP32: VAD energy gate
        alt Voice activity detected
            ESP32->>ESP32: MFCC extraction (49×40)
            ESP32->>ESP32: TFLite inference → 64-dim embedding
            ESP32->>ESP32: cosine_sim(embedding, prototype)
            alt similarity > threshold (2-of-3 hits)
                ESP32->>ESP32: STATE → STREAM
                ESP32->>WS: send 500ms pre-roll PCM
                ESP32->>WS: stream live PCM (250ms chunks)
                ESP32->>ESP32: detect end-of-speech silence
                ESP32->>WS: send 0xFF (end-utterance)
                ESP32->>ESP32: STATE → IDLE
            end
        end
    end

    WS->>WS: reassemble utterance from chunks
    WS->>Whisper: transcribe(pcm_audio)
    Whisper->>WS: {"text": "turn on lights", "latency_ms": 118}
    WS->>Agent: process("turn on lights")
    Agent->>WS: {"intent": "light_control", "action": "on", "response": "Lights on."}
    WS->>ESP32: {"type": "transcript", "text": "turn on lights", "latency_ms": 142}
    ESP32->>User: (optional TTS / LED feedback)
```

---

## Flow 3 — Error Recovery

```mermaid
sequenceDiagram
    participant ESP32 as ESP32-S3
    participant WiFi as WiFi AP
    participant WS as WebSocket Server

    ESP32->>WiFi: connect (stored credentials)
    WiFi-->>ESP32: connected, IP assigned
    ESP32->>WS: WebSocket connect + HMAC token
    WS-->>ESP32: 101 Switching Protocols

    note over ESP32,WS: Normal operation...

    WS--xESP32: connection dropped
    ESP32->>ESP32: STATE → IDLE, discard stream buffer
    ESP32->>ESP32: wait 2s backoff
    ESP32->>WS: reconnect attempt
    alt reconnect succeeds
        WS-->>ESP32: 101 Switching Protocols
        ESP32->>ESP32: resume listening
    else WiFi also down
        ESP32->>WiFi: reconnect WiFi
        WiFi-->>ESP32: connected
        ESP32->>WS: reconnect WebSocket
    end
```

---

## Device State Machine

```text
                        ┌──────────┐
            boot        │          │
          ──────────►   │   IDLE   │ ◄─────────────────┐
                        │          │                   │
                        └────┬─────┘                   │
                             │                         │
                    2-of-3 cosine hits                 │
                             │                         │
                        ┌────▼─────┐                   │
                        │          │                   │
                        │   WAKE   │                   │
                        │          │                   │
                        └────┬─────┘                   │
                             │                         │
                    WS connected                       │
                             │                         │
                        ┌────▼─────┐                   │
                        │          │    silence / 0xFF │
                        │  STREAM  │───────────────────┘
                        │          │
                        └────┬─────┘
                             │
                    "enroll" serial command
                             │
                        ┌────▼─────┐
                        │          │
                        │  ENROLL  │
                        │          │
                        └──────────┘
```
