"""
Sequence-number gap detection for the KWS streaming protocol.

Tracks the expected sequence number for each session and detects missing
chunks (dropped frames). Reports gaps so the server can request retransmit
or proceed gracefully with the partial audio.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("kws.fec")

SEQ_WRAP = 65536  # uint16 wrap-around


class GapDetector:
    """Per-session sequence number gap detector.

    Tracks incoming chunk sequence numbers and identifies missing chunks.
    Sequence numbers are 16-bit unsigned (0–65535) and wrap around.

    Args:
        session_id: Session identifier for logging.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id[:8]
        self._expected_seq: int | None = None  # None = waiting for first chunk
        self._total_chunks: int = 0
        self._total_gaps: int = 0

    def update(self, seq_num: int) -> list[int]:
        """Process a new chunk sequence number and detect gaps.

        Args:
            seq_num: 16-bit sequence number from the incoming chunk header.

        Returns:
            List of missing sequence numbers detected. Empty if no gap.
        """
        self._total_chunks += 1
        gaps: list[int] = []

        if self._expected_seq is None:
            # First chunk — initialize expected seq
            self._expected_seq = (seq_num + 1) % SEQ_WRAP
            return gaps

        if seq_num == self._expected_seq:
            # In-order delivery
            self._expected_seq = (seq_num + 1) % SEQ_WRAP
            return gaps

        # Gap detected: enumerate missing sequence numbers
        missing_seq = seq_num
        expected = self._expected_seq
        while expected != missing_seq:
            gaps.append(expected)
            expected = (expected + 1) % SEQ_WRAP
            if len(gaps) > 100:
                # Safety valve: don't report more than 100 missing seqs
                break

        self._total_gaps += len(gaps)
        logger.warning(
            f"[{self.session_id}] Gap detected: "
            f"expected seq={self._expected_seq}, got seq={seq_num}. "
            f"Missing: {gaps[:5]}{'...' if len(gaps) > 5 else ''}"
        )

        self._expected_seq = (seq_num + 1) % SEQ_WRAP
        return gaps

    def reset(self) -> None:
        """Reset gap detector (call at start of each new utterance)."""
        self._expected_seq = None

    @property
    def stats(self) -> dict:
        """Return gap detection statistics."""
        return {
            "total_chunks": self._total_chunks,
            "total_gaps": self._total_gaps,
            "gap_rate": (
                self._total_gaps / self._total_chunks
                if self._total_chunks > 0
                else 0.0
            ),
        }
