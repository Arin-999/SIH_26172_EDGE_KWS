# Wiring Guide — INMP441 ↔ ESP32-S3-DevKitC-1

## Physical Wiring (Breadboard Prototype)

```
INMP441 Breakout       Jumper Wire       ESP32-S3-DevKitC-1
┌─────────────┐                          ┌──────────────────────┐
│  VDD  ●─────┼──── RED   ──────────────►│ 3V3                  │
│  GND  ●─────┼──── BLACK ──────────────►│ GND                  │
│  SD   ●─────┼──── YELLOW─────────────►│ GPIO 4  (I2S DIN)    │
│  WS   ●─────┼──── GREEN ──────────────►│ GPIO 5  (I2S WS)     │
│  SCK  ●─────┼──── BLUE  ──────────────►│ GPIO 6  (I2S CLK)    │
│  L/R  ●─────┼──── BLACK ──────────────►│ GND     (left ch.)   │
└─────────────┘                          └──────────────────────┘
```

> **Important:** The L/R pin selects the I2S channel address.
> - Connect to **GND** for left channel (address 0).
> - Connect to **3V3** for right channel (address 1).
> The firmware reads only the left channel.

## Decoupling Capacitor

Place a **100 nF ceramic capacitor** as close as possible to the INMP441 VDD pin between VDD and GND. This suppresses power-supply noise that would appear as a DC offset or hum in the captured audio.

## Wire Length

Keep all I2S signal wires under **10 cm** to avoid capacitive loading on the clock line. Longer wires at 16 kHz × 32-bit = 512 kHz SCK can cause intermittent bit errors.

## Verification

After wiring, verify correct connections before powering:

1. Use a multimeter in continuity mode.
2. Check VDD → 3V3 continuity.
3. Check GND → GND continuity.
4. Check SD → GPIO 4, WS → GPIO 5, SCK → GPIO 6.
5. Confirm L/R → GND (or 3V3 depending on channel choice).

Then power on and check with an oscilloscope or logic analyser that SCK toggles at ~512 kHz and WS toggles at ~16 kHz.
