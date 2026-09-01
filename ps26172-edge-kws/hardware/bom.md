# Bill of Materials — KWS Edge Node (Prototype)

Hackathon prototype BOM. All components available from LCSC, Robu.in, or Amazon India.

---

| # | Component | Part Number | Qty | Unit Cost (₹) | Supplier | Notes |
|---|---|---|---|---|---|---|
| 1 | ESP32-S3-DevKitC-1 (N8R8) | ESP32-S3-DevKitC-1-N8R8 | 1 | 1,200 | Robu.in | 8 MB flash, 8 MB PSRAM |
| 2 | INMP441 MEMS I2S Microphone module | INMP441 breakout | 1 | 180 | Amazon | I2S, 60 Hz–15 kHz, 61 dB SNR |
| 3 | Breadboard (400-point) | — | 1 | 60 | Local | For prototype wiring |
| 4 | Jumper wires (M-M, 10 cm) | — | 10 | 20 | Local | INMP441 ↔ ESP32 connections |
| 5 | USB-C cable (data-capable) | — | 1 | 80 | Local | Power + flashing |
| 6 | 5 V / 2 A USB charger | — | 1 | 150 | Local | Power supply |
| 7 | 100 nF decoupling capacitor | C0402 100nF | 2 | 5 | LCSC | VDD bypass on INMP441 |

**Total estimated cost: ₹1,695**

---

## Notes

- The ESP32-S3-DevKitC-1 has a built-in USB-JTAG/Serial bridge — no external USB-UART adapter required.
- PSRAM (`R8` variant) is mandatory for TFLite arena allocation (>200 KB required).
- For a custom PCB revision, replace the breakout board with a direct LGA-6 INMP441 footprint and remove the breadboard.

---

## PCB revision (future)

| Additional Component | Purpose |
|---|---|
| LDO 3.3 V (e.g., AMS1117-3.3) | Standalone power from Li-Po |
| Li-Po 3.7 V 1000 mAh | Battery operation |
| TP4056 charger IC | USB charging |
| Status LED (red + green) | Power / wake indicator |
| Reset button | User reset |
