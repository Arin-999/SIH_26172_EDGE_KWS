#!/usr/bin/env python3
"""
Flash the KWS firmware to an ESP32-S3 device.

Wraps `idf.py flash monitor` with the correct port and build directory.

Usage:
    python scripts/flashing/flash.py --port COM3
    python scripts/flashing/flash.py --port /dev/ttyACM0 --monitor
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

FIRMWARE_DIR = Path(__file__).resolve().parents[2] / "firmware" / "esp32"


def main() -> None:
    parser = argparse.ArgumentParser(description="Flash KWS firmware to ESP32-S3.")
    parser.add_argument("--port", required=True, help="Serial port (e.g. COM3 or /dev/ttyACM0)")
    parser.add_argument("--monitor", action="store_true",
                        help="Open serial monitor after flashing")
    parser.add_argument("--baud", default="921600", help="Flash baud rate (default 921600)")
    parser.add_argument("--erase", action="store_true",
                        help="Erase flash before programming (full erase)")
    args = parser.parse_args()

    if not FIRMWARE_DIR.exists():
        print(f"[ERROR] Firmware directory not found: {FIRMWARE_DIR}", file=sys.stderr)
        sys.exit(1)

    cmd = ["idf.py", "-p", args.port, "-b", args.baud]

    if args.erase:
        cmd.append("erase-flash")

    cmd.append("flash")

    if args.monitor:
        cmd.append("monitor")

    print(f"[flash] Running: {' '.join(cmd)}")
    print(f"[flash] Working directory: {FIRMWARE_DIR}")

    result = subprocess.run(cmd, cwd=FIRMWARE_DIR)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
