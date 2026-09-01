# ESP32-S3 Configuration Reference

## ESP-IDF Build Configuration (`sdkconfig.defaults`)

| Key | Value | Reason |
|---|---|---|
| `CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240` | `y` | Maximum CPU for TFLite inference |
| `CONFIG_SPIRAM` | `y` | Enable external PSRAM (required for TFLite arena) |
| `CONFIG_SPIRAM_MODE_OCT` | `y` | Octal SPI mode for max PSRAM bandwidth |
| `CONFIG_SPIRAM_SPEED_80M` | `y` | 80 MHz PSRAM clock |
| `CONFIG_SPIRAM_USE_MALLOC` | `y` | Allow `malloc()` to use PSRAM |
| `CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL` | `16384` | Keep small allocs (<16 KB) in SRAM |
| `CONFIG_ESPTOOLPY_FLASHSIZE_4MB` | `y` | 4 MB internal flash |
| `CONFIG_FREERTOS_HZ` | `1000` | 1 ms tick for precise audio timing |
| `CONFIG_LOG_DEFAULT_LEVEL_INFO` | `y` | Info-level logging |

## WiFi Configuration

| Setting | Value |
|---|---|
| SSID | Set via `CONFIG_KWS_WIFI_SSID` in `menuconfig` or Kconfig |
| Password | Set via `CONFIG_KWS_WIFI_PASSWORD` |
| Auth mode | WPA2-PSK |
| Power save | `WIFI_PS_MIN_MODEM` during streaming, `WIFI_PS_MAX_MODEM` during KWS idle |

## WebSocket Client

| Parameter | Value |
|---|---|
| Server URI | `ws://<KWS_SERVER_HOST>:8765/v1/stream` |
| Auth header | `X-KWS-Token: <HMAC-SHA256 token>` |
| Shared secret | Must match `KWS_SECRET` env var on server |
| Frame size | 8 016 bytes (standard audio chunk) |
| Timeout (ASR response) | 5 000 ms |

## TFLite Model Configuration

| Parameter | Value |
|---|---|
| Model file | `model.tflite` (embedded in firmware flash via `EMBED_FILES`) |
| Arena size | 256 KB (allocated from PSRAM) |
| Input shape | `(1, 49, 40, 1)` INT8 |
| Output shape | `(1, 64)` INT8 |
| Inference time (target) | ≤ 50 ms @ 240 MHz |

## Kconfig Options (`main/Kconfig.projbuild`)

```
KWS_WIFI_SSID         — WiFi network name
KWS_WIFI_PASSWORD     — WiFi password
KWS_SERVER_HOST       — WebSocket server hostname/IP
KWS_SERVER_PORT       — WebSocket server port (default 8765)
KWS_WAKE_THRESHOLD    — Cosine similarity threshold (default 75, meaning 0.75)
KWS_PREROLL_MS        — Pre-roll buffer size in ms (default 500)
KWS_STREAM_MAX_S      — Maximum streaming duration in seconds (default 20)
```

Configure via `idf.py menuconfig → KWS Configuration`.
