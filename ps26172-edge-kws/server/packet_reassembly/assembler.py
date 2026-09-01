"""
Per-session audio buffer assembler.

Accumulates binary PCM audio chunks for a session and assembles them
into a complete utterance buffer when flush() is called (triggered by
the 0xFF end-of-utterance marker).
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("kws.assembler")

# Maximum accumulated audio per session (20 seconds @ 16 kHz int16)
MAX_BUFFER_BYTES: int = 16_000 * 2 * 20


class SessionAssembler:
    """Accumulates audio chunks for a single WebSocket session.

    Thread-safety: not thread-safe. Each session is handled in a single
    asyncio task, so no locking is needed.

    Args:
        session_id: Session identifier for logging.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id[:8]
        self._buffer = bytearray()
        self._chunk_count: int = 0
        self._start_time: float | None = None

    def add_chunk(self, payload: bytes) -> None:
        """Append a PCM payload to the utterance buffer.

        Drops audio beyond MAX_BUFFER_BYTES to prevent memory exhaustion.

        Args:
            payload: Raw PCM bytes from a single audio chunk.
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()

        remaining = MAX_BUFFER_BYTES - len(self._buffer)
        if remaining <= 0:
            logger.warning(
                f"[{self.session_id}] Buffer overflow — dropping {len(payload)} bytes"
            )
            return

        self._buffer.extend(payload[:remaining])
        self._chunk_count += 1

    def flush(self) -> bytes:
        """Flush and return the accumulated audio bytes, then reset.

        Returns:
            Accumulated PCM bytes for the completed utterance.
            Empty bytes if no audio was received.
        """
        audio = bytes(self._buffer)
        elapsed = (
            time.perf_counter() - self._start_time
            if self._start_time is not None
            else 0.0
        )
        logger.debug(
            f"[{self.session_id}] Flush: {len(audio)} bytes, "
            f"{self._chunk_count} chunks, {elapsed:.2f}s"
        )
        self.reset()
        return audio

    def reset(self) -> None:
        """Clear the buffer without returning data."""
        self._buffer.clear()
        self._chunk_count = 0
        self._start_time = None

    @property
    def buffered_bytes(self) -> int:
        """Current number of bytes in the buffer."""
        return len(self._buffer)

    @property
    def buffered_duration_s(self) -> float:
        """Estimated duration of buffered audio in seconds."""
        return len(self._buffer) / (16_000 * 2)  # 16 kHz, 16-bit
