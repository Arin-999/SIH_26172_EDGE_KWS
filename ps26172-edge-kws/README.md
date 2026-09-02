<div align="center">
  
# 🎙️ PS26172 Edge KWS Architecture

**SIH 2026 · Problem Statement 26172 — Edge Keyword Spotting System**

*An ultra-lightweight, hybrid edge-cloud voice command system.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32--S3-orange.svg)](https://www.espressif.com/)
[![Framework](https://img.shields.io/badge/ML-TensorFlow_Lite-orange.svg)](https://www.tensorflow.org/lite/microcontrollers)
[![Server](https://img.shields.io/badge/Server-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

</div>

---

## 📖 Overview

As voice-controlled IoT proliferates, processing everything in the cloud is too costly, privacy-invasive, and slow. 

This repository provides a **hybrid architecture** solution:
1. **The Edge (<256KB RAM):** An ESP32-S3 continuously listens for a custom wake word using an INT8-quantized Depthwise-Separable CNN (DS-CNN).
2. **The Cloud (Heavy Lifting):** Upon detecting the keyword, the ESP32 instantly streams the subsequent audio over WebSockets to a FastAPI server running `faster-whisper` for Automatic Speech Recognition (ASR).

---

## 🏗️ Project Architecture

```mermaid
graph LR
    subgraph Edge [Edge Device - ESP32-S3]
        Mic[INMP441 Mic] -->|I2S| TFLite[TFLite Micro DS-CNN]
        TFLite -->|Wake Word Detected| WS_Client[WebSocket Client]
    end

    subgraph Server [Local/Cloud Server]
        WS_Server[FastAPI WS] --> Whisper[Faster-Whisper ASR]
        Whisper --> Intent[Intent Handler]
    end

    WS_Client -.->|Raw PCM Stream| WS_Server
```

---

## 🚀 Quick Start Guide

### 1. Train the ML Model (PC)
Train the keyword spotting model and export it to a C-header file for the ESP32.

```bash
# Install dependencies
pip install -r ml/requirements.txt

# Train the model and quantize to INT8
python ml/train_kws.py

# Export the model to firmware/ps26172_firmware/model_data.h
python ml/export_tflite.py
```

### 2. Start the ASR Server (PC)
Start the WebSocket server to receive audio streams from the ESP32 and transcribe them.

```bash
# Install dependencies
pip install -r server/requirements.txt

# Run the server
python server/main.py
```
> **Note:** Take note of your PC's local IP address (e.g., `192.168.x.x`). You will need it for the Arduino sketch.

### 3. Flash the ESP32-S3 Firmware
We use the **Arduino IDE** for simple and accessible flashing.

1. Install the **WebSockets** library by Markus Sattler via the Arduino Library Manager.
2. Open `firmware/ps26172_firmware/ps26172_firmware.ino` in the Arduino IDE.
3. Update the Wi-Fi credentials and the WebSocket server IP at the top of the file:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* websocket_server = "192.168.31.74"; // <--- Update this
   ```
4. Select your **ESP32-S3** board, ensure **PSRAM is Enabled**, and click **Upload**.

---

## 🔌 Hardware Wiring

| INMP441 Microphone | ESP32-S3 Pin | Description |
| :---: | :---: | :--- |
| **VDD** | `3V3` | Power supply |
| **GND** | `GND` | Ground |
| **L/R** | `GND` | Left Channel selection |
| **SD** | `GPIO 4` | I2S Data |
| **WS** | `GPIO 5` | I2S Word Select |
| **SCK**| `GPIO 6` | I2S Bit Clock |

> [!TIP]
> Keep the I2S signal wires short (<10 cm) to prevent clock jitter and signal degradation.

---

## ⚖️ License
This project is open-source and available under the MIT License.
