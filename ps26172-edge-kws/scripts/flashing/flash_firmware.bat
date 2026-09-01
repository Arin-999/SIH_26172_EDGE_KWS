@echo off
REM Flash firmware to ESP32-S3 and open serial monitor
REM Usage: flash_firmware.bat [PORT]
REM Example: flash_firmware.bat COM3

setlocal

set PORT=%1
if "%PORT%"=="" set PORT=COM3

set FIRMWARE_DIR=%~dp0..\..\firmware\esp32

echo [flash] Firmware directory: %FIRMWARE_DIR%
echo [flash] Port: %PORT%
echo.

REM Activate ESP-IDF if not already active
if "%IDF_PATH%"=="" (
    echo [flash] Activating ESP-IDF ...
    call "%USERPROFILE%\esp\esp-idf\export.bat"
)

REM Build + flash + monitor
cd /d "%FIRMWARE_DIR%"
idf.py set-target esp32s3
idf.py -p %PORT% flash monitor

endlocal
