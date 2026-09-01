"""
faster-whisper ASR wrapper for the KWS server.
"""

from __future__ import annotations

import io
import logging
import time

import numpy as np

logger = logging.getLogger("kws.asr")

SAMPLE_RATE = 16_000


class WhisperASR:
    """Wrapper around faster-whisper for PCM audio transcription.

    Args:
        model_size: faster-whisper model identifier (e.g., 'base.en', 'small').
        device: 'cpu' or 'cuda'.
        compute_type: 'int8' (CPU) or 'float16' (GPU).
        beam_size: Beam search width. Lower = faster. Default 5.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str | None = None,
        beam_size: int = 5,
    ) -> None:
        from faster_whisper import WhisperModel

        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"

        logger.info(f"Loading WhisperModel('{model_size}', device='{device}', compute='{compute_type}') ...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.beam_size = beam_size
        logger.info("WhisperModel loaded.")

    def transcribe(
        self,
        pcm_bytes: bytes,
        sample_rate: int = SAMPLE_RATE,
        language: str | None = None,
    ) -> dict:
        """Transcribe raw PCM audio bytes.

        Args:
            pcm_bytes: Raw 16-bit signed PCM bytes (little-endian, mono, 16 kHz).
            sample_rate: Sample rate of the audio. Must be 16 000 Hz for Whisper.
            language: Optional language hint. None = auto-detect.

        Returns:
            Dict with keys: text (str), language (str), latency_ms (int).
        """
        t0 = time.perf_counter()

        # Convert PCM bytes → float32 numpy array
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample if needed (Whisper always expects 16 kHz)
        if sample_rate != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

        # Run inference
        segments, info = self.model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=language,
            vad_filter=True,  # Built-in VAD to filter silence
            vad_parameters={"min_silence_duration_ms": 300},
        )

        # Collect all segment text
        text = " ".join(seg.text.strip() for seg in segments).strip()
        detected_language = info.language if hasattr(info, "language") else "en"

        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug(f"Transcribed {len(audio)/SAMPLE_RATE:.2f}s audio in {latency_ms}ms: '{text}'")

        return {
            "text": text,
            "language": detected_language,
            "latency_ms": latency_ms,
            "audio_duration_s": round(len(audio) / SAMPLE_RATE, 3),
        }
