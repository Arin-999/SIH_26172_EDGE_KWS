@echo off
REM Install ESP-IDF v5.2 for Windows
REM Run this script once before building the firmware

setlocal

set IDF_PATH=%USERPROFILE%\esp\esp-idf
set IDF_VERSION=v5.2.3

echo ============================================
echo  ESP-IDF v5.2 Setup for Windows
echo ============================================

REM Check if git is available
where git >nul 2>&1 || (
    echo [ERROR] git is not in PATH. Install Git for Windows first.
    echo   https://git-scm.com/download/win
    exit /b 1
)

REM Clone ESP-IDF if not already present
if not exist "%IDF_PATH%" (
    echo [setup] Cloning ESP-IDF %IDF_VERSION% to %IDF_PATH% ...
    git clone -b %IDF_VERSION% --depth 1 https://github.com/espressif/esp-idf.git "%IDF_PATH%"
) else (
    echo [setup] ESP-IDF already exists at %IDF_PATH%
)

REM Run install script
echo [setup] Running ESP-IDF install.bat ...
call "%IDF_PATH%\install.bat" esp32s3

echo.
echo ============================================
echo  Setup complete!
echo  Activate ESP-IDF with:
echo    %IDF_PATH%\export.bat
echo ============================================
endlocal
