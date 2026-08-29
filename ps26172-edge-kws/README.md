# Development Workflow

## Team Roles

### Member 1 — System Design + AI/ML

**Primary responsibility:** System architecture, KWS/AI pipeline, ML research, model development, optimization, and ML validation.

Owns:

```text
docs/
├── system-architecture.md
├── user-flow.md
├── requirements.md
├── design-decisions.md
└── testing-plan.md

ml/
├── datasets/
├── preprocessing/
├── augmentation/
├── training/
├── evaluation/
├── quantization/
├── personalization/
├── models/
└── benchmarks/
```

Responsibilities:

* Define complete system architecture.
* Define interfaces between edge, communication and server.
* Define audio and ML requirements.
* Design the KWS pipeline.
* Build and validate the dataset pipeline.
* Implement preprocessing and feature extraction.
* Train baseline KWS models.
* Optimize the KWS model for edge deployment.
* Implement INT8 quantization.
* Implement custom wake-word personalization.
* Evaluate FAR, FRR, accuracy and latency.
* Measure model size and RAM requirements.
* Prepare the model for ESP32 deployment.
* Define ML-related test cases.
* Maintain ML experiment records and research documentation.

---

### Member 2 — Full Software / Systems Engineer

**Primary responsibility:** Firmware, communication, server, integration, testing infrastructure and software deployment.

Owns:

```text
firmware/
server/
protocol/
tests/
scripts/
examples/
```

Responsibilities:

* Develop ESP32-S3 firmware.
* Integrate microphone and audio acquisition.
* Implement DMA/audio buffering.
* Implement edge state machine.
* Integrate the KWS model provided by the ML member.
* Implement compression.
* Implement packetization.
* Implement FEC.
* Implement authentication and encryption.
* Implement wireless communication.
* Develop server receiver.
* Implement packet reassembly.
* Integrate ASR.
* Integrate AI/agent services.
* Implement optional TTS.
* Build end-to-end testing.
* Maintain build and deployment scripts.
* Maintain CI/testing workflow.

---

# Development Principle

Both members work **in parallel**.

Do not wait for the entire project to be completed by one person.

The project is divided into interfaces:

```text
                    SYSTEM
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
       AI / ML                SOFTWARE
          │                       │
          │                       │
          └──────────┬────────────┘
                     │
                  INTERFACE
                     │
                     ▼
                INTEGRATED
                 PROTOTYPE
```

The AI/ML member develops the intelligence.

The Software member develops the system that runs and communicates with that intelligence.

---

# Shared System Architecture

The final system should follow:

```text
                         USER
                           │
                           ▼
                      MICROPHONE
                           │
                           ▼
                 ┌──────────────────┐
                 │ Audio Acquisition│
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Preprocessing    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Acoustic Gate     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Tiny KWS Model   │
                 │    [AI/ML]       │
                 └────────┬─────────┘
                          │
                   Wake detected
                          │
                          ▼
                 ┌──────────────────┐
                 │ Personalization  │
                 │    [AI/ML]       │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Voice Capture    │
                 │    [Software]    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Compression      │
                 │    [Software]    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Packet + FEC     │
                 │    [Software]    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Encryption       │
                 │    [Software]    │
                 └────────┬─────────┘
                          │
                          ▼
                       NETWORK
                          │
                          ▼
                 ┌──────────────────┐
                 │ Server Receiver  │
                 │    [Software]    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │       ASR        │
                 │    [Software]    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ AI / Agent       │
                 │    [Software]    │
                 └────────┬─────────┘
                          │
                          ▼
                       RESPONSE
```

---

# Interface Contract Between Both Members

The most important rule is:

> **AI/ML and Software must communicate through clearly defined interfaces.**

Do not directly depend on each other's internal implementation.

---

## Interface 1 — Audio Input

Software provides audio to ML in:

```text
Format:
PCM

Sample rate:
16,000 Hz

Channels:
1

Sample format:
16-bit signed integer
```

Conceptually:

```text
Software
   │
   │ PCM audio frames
   ▼
ML/KWS
```

The ML member must document the exact frame size, hop size and preprocessing requirements.

---

# Interface 2 — KWS Output

The ML system should expose a simple interface:

```text
KWS(audio_frame)
        ↓
{
    keyword_probability,
    class,
    timestamp
}
```

Example:

```text
{
    "class": "keyword",
    "probability": 0.94,
    "timestamp": 12840
}
```

The Software member should not need to know the internal neural-network architecture.

---

# Interface 3 — Personalization

ML exposes:

```text
enroll_keyword(audio_samples)
```

and:

```text
verify_keyword(audio_samples)
```

Conceptually:

```text
User
 ↓
Software enrollment interface
 ↓
ML personalization module
 ↓
Keyword profile
```

Runtime:

```text
Audio
 ↓
KWS
 ↓
Personalization
 ↓
WAKE / IGNORE
```

---

# Interface 4 — Model Artifact

The ML member provides:

```text
model.tflite
```

along with:

```text
model_metadata.json
```

The metadata should specify:

```text
model version
input shape
input datatype
output shape
output datatype
sample rate
feature parameters
quantization parameters
class labels
expected preprocessing
```

Example:

```text
models/
└── kws_v1/
    ├── model.tflite
    └── model_metadata.json
```

The Software member integrates this artifact into the ESP32 firmware.

---

# Interface 5 — Packet Protocol

The Software member owns the communication protocol.

The protocol must be documented before integration.

Example:

```text
Packet
├── version
├── session_id
├── sequence_number
├── timestamp
├── flags
├── payload_length
├── payload
└── authentication_tag
```

Documentation:

```text
protocol/
└── packet-format.md
```

The ML member does not need to implement the transport layer.

---

# Interface 6 — Server API

The Software member owns the server interface.

Example:

```text
ESP32
  ↓
Secure packet stream
  ↓
Server
  ↓
Audio
  ↓
ASR
  ↓
Text
  ↓
AI Agent
  ↓
Response
```

The exact API/protocol should be documented in:

```text
docs/protocol-specification.md
```

---

# Git Branch Ownership

Use:

```text
main
│
└── develop
     │
     ├── feature/ml-kws
     ├── feature/ml-personalization
     ├── feature/esp32
     ├── feature/audio-pipeline
     ├── feature/network
     ├── feature/server
     └── feature/integration
```

### AI/ML member

Primarily works on:

```text
feature/ml-kws
feature/ml-personalization
```

### Software member

Primarily works on:

```text
feature/esp32
feature/audio-pipeline
feature/network
feature/server
```

Integration work uses:

```text
feature/integration
```

---

# GitHub Working Rules

## Rule 1 — Never directly work on `main`

Use:

```text
feature branch
      ↓
Pull Request
      ↓
Review
      ↓
develop
      ↓
Integration
      ↓
main
```

---

## Rule 2 — One task = one branch

Example:

```text
feature/log-mel-extraction
```

not:

```text
feature/all-ml-work
```

Keep branches focused.

---

## Rule 3 — Commit meaningful changes

Good:

```text
Add log-Mel feature extraction
Implement KWS baseline
Add INT8 model export
Add ESP32 microphone driver
Implement packet reassembly
Add FEC decoder
```

Bad:

```text
update
changes
final
test
new
```

---

# Integration Strategy

The project should be integrated in layers.

## Integration 1 — Audio

```text
Microphone
    ↓
ESP32
    ↓
PCM
    ↓
PC/Python
```

Goal:

> Verify that the audio captured by ESP32 matches the ML pipeline assumptions.

---

## Integration 2 — KWS

```text
ESP32 microphone
      ↓
Preprocessing
      ↓
KWS model
      ↓
Wake / Ignore
```

Goal:

> Run the trained model on real microphone input.

---

## Integration 3 — Personalization

```text
User
 ↓
Keyword enrollment
 ↓
Personalized profile
 ↓
ESP32 KWS
 ↓
Wake / Ignore
```

Goal:

> Verify that a new user can configure and use a custom wake word.

---

## Integration 4 — Capture

```text
Wake word
    ↓
Pre-roll buffer
    ↓
Command capture
    ↓
Audio packet
```

Goal:

> Ensure the command is captured without losing its beginning.

---

## Integration 5 — Communication

```text
ESP32
 ↓
Compression
 ↓
Packetization
 ↓
FEC
 ↓
Encryption
 ↓
Network
 ↓
Server
```

Goal:

> Verify reliable and secure transmission.

---

## Integration 6 — ASR

```text
ESP32
 ↓
Server
 ↓
Audio reconstruction
 ↓
ASR
 ↓
Text
```

Goal:

> Verify speech reaches ASR correctly.

---

## Integration 7 — Complete System

```text
USER
 ↓
CUSTOM WAKE WORD
 ↓
LOCAL KWS
 ↓
PERSONALIZATION
 ↓
COMMAND
 ↓
SECURE TRANSMISSION
 ↓
ASR
 ↓
AI AGENT
 ↓
RESPONSE
```

This is the final prototype.

---

# Definition of Final Prototype

The project is considered complete only when the following flow works repeatedly:

```text
1. User configures a custom wake word
                ↓
2. Device enters low-power listening
                ↓
3. User speaks the wake word
                ↓
4. Local KWS detects it
                ↓
5. Personalization verifies it
                ↓
6. Device captures the command
                ↓
7. Audio is compressed
                ↓
8. Audio is packetized
                ↓
9. FEC is applied
                ↓
10. Audio is authenticated/encrypted
                ↓
11. Audio is transmitted
                ↓
12. Server reconstructs audio
                ↓
13. ASR converts speech to text
                ↓
14. AI agent processes the request
                ↓
15. Response is generated
                ↓
16. User receives response
                ↓
17. Device returns to listening
```

---

# Shared Definition of Done

A feature is **not complete** until:

```text
[ ] Code implemented
[ ] Unit test written
[ ] Test passed
[ ] Documentation updated
[ ] Interface documented
[ ] Git commit created
[ ] Pull request created
[ ] Code reviewed
[ ] Integrated with develop
```

For ML:

```text
[ ] Dataset validated
[ ] Training reproducible
[ ] Evaluation completed
[ ] FAR measured
[ ] FRR measured
[ ] Model size measured
[ ] RAM measured
[ ] Latency measured
[ ] INT8 model validated
[ ] ESP32 inference tested
```

For Software:

```text
[ ] Firmware builds
[ ] Microphone works
[ ] Buffering works
[ ] Packetization works
[ ] FEC works
[ ] Encryption works
[ ] Network works
[ ] Server receives packets
[ ] Audio reconstructed
[ ] ASR works
[ ] End-to-end integration works
```

---

# Final Principle

The two members should think of the project as:

```text
             AI / ML
                │
                │ defined interface
                ▼
          EDGE SOFTWARE
                │
                │ defined protocol
                ▼
          COMMUNICATION
                │
                ▼
             SERVER
                │
                ▼
              ASR
                │
                ▼
             AI AGENT
```

**AI/ML owns the intelligence.**

**Software owns the execution, communication and integration.**

**Both members jointly own the interfaces and final prototype.**

Never wait for the other member to finish the entire subsystem. Build with **mock inputs, test interfaces early, integrate frequently, and replace mocks with the real implementation as each subsystem becomes ready.**
