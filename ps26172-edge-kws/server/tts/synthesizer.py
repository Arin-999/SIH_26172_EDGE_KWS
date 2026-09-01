"""
Optional TTS response synthesis.

Converts text responses to audio bytes using pyttsx3 (offline) or
gTTS (online). Used to send spoken responses back to the ESP32 for
playback on a connected speaker.

Default: disabled (returns empty bytes). Enable by setting KWS_TTS_ENABLED=1.
"""

from __future__ import annotations

import io
import logging
import os

logger = logging.getLogger("kws.tts")

TTS_ENABLED = os.environ.get("KWS_TTS_ENABLED", "0") == "1"
TTS_BACKEND = os.environ.get("KWS_TTS_BACKEND", "pyttsx3")  # 'pyttsx3' or 'gtts'


def synthesize(text: str) -> bytes:
    """Convert text to speech audio bytes (WAV format).

    Args:
        text: Text to synthesize.

    Returns:
        WAV audio bytes (16 kHz, mono, 16-bit PCM).
        Empty bytes if TTS is disabled or synthesis fails.
    """
    if not TTS_ENABLED:
        return b""

    if not text.strip():
        return b""

    try:
        if TTS_BACKEND == "gtts":
            return _synthesize_gtts(text)
        else:
            return _synthesize_pyttsx3(text)
    except Exception as exc:
        logger.warning(f"TTS synthesis failed: {exc}")
        return b""


def _synthesize_pyttsx3(text: str) -> bytes:
    """Offline TTS using pyttsx3."""
    import pyttsx3
    import tempfile
    import wave

    engine = pyttsx3.init()
    engine.setProperty("rate", 160)  # words per minute

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def _synthesize_gtts(text: str) -> bytes:
    """Online TTS using Google Text-to-Speech."""
    from gtts import gTTS

    buf = io.BytesIO()
    tts = gTTS(text=text, lang="en", slow=False)
    tts.write_to_fp(buf)
    return buf.getvalue()
