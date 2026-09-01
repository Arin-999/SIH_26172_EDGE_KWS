"""
Network / WebSocket protocol tests.

Tests the packet assembler, gap detector, and WebSocket server
using an in-process test client.
"""

import struct
import time

import numpy as np
import pytest

from server.packet_reassembly.assembler import SessionAssembler, MAX_BUFFER_BYTES
from server.fec.decoder import GapDetector


# ---------------------------------------------------------------------------
# Tests: SessionAssembler
# ---------------------------------------------------------------------------


class TestSessionAssembler:
    def test_flush_returns_accumulated_bytes(self) -> None:
        asm = SessionAssembler("test-session")
        chunk = b"\x00\x01" * 100
        asm.add_chunk(chunk)
        result = asm.flush()
        assert result == chunk

    def test_multiple_chunks_concatenated(self) -> None:
        asm = SessionAssembler("test-session")
        asm.add_chunk(b"hello")
        asm.add_chunk(b"world")
        result = asm.flush()
        assert result == b"helloworld"

    def test_flush_resets_buffer(self) -> None:
        asm = SessionAssembler("test-session")
        asm.add_chunk(b"data")
        asm.flush()
        # Second flush should return empty
        assert asm.flush() == b""

    def test_empty_flush_returns_empty_bytes(self) -> None:
        asm = SessionAssembler("test-session")
        assert asm.flush() == b""

    def test_overflow_protection(self) -> None:
        asm = SessionAssembler("test-session")
        large_chunk = b"\x00" * (MAX_BUFFER_BYTES + 1024)
        asm.add_chunk(large_chunk)
        result = asm.flush()
        assert len(result) <= MAX_BUFFER_BYTES

    def test_buffered_bytes_property(self) -> None:
        asm = SessionAssembler("test-session")
        asm.add_chunk(b"\x00" * 500)
        assert asm.buffered_bytes == 500

    def test_buffered_duration_s(self) -> None:
        asm = SessionAssembler("test-session")
        # 16000 samples * 2 bytes = 32000 bytes = 1 second
        asm.add_chunk(b"\x00" * 32_000)
        assert asm.buffered_duration_s == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: GapDetector
# ---------------------------------------------------------------------------


class TestGapDetector:
    def test_no_gaps_sequential(self) -> None:
        gd = GapDetector("test")
        for seq in range(10):
            gaps = gd.update(seq)
            assert gaps == [], f"Unexpected gaps at seq={seq}: {gaps}"

    def test_detects_single_gap(self) -> None:
        gd = GapDetector("test")
        gd.update(0)
        gd.update(1)
        gaps = gd.update(3)  # seq=2 is missing
        assert 2 in gaps

    def test_detects_multiple_gaps(self) -> None:
        gd = GapDetector("test")
        gd.update(0)
        gaps = gd.update(5)  # seqs 1,2,3,4 missing
        assert set(gaps) == {1, 2, 3, 4}

    def test_first_chunk_no_gap(self) -> None:
        gd = GapDetector("test")
        gaps = gd.update(42)  # Any first seq is OK
        assert gaps == []

    def test_wraparound(self) -> None:
        gd = GapDetector("test")
        gd.update(65534)
        gd.update(65535)
        gaps = gd.update(0)  # Wrap-around, no gap
        assert gaps == []

    def test_stats(self) -> None:
        gd = GapDetector("test")
        gd.update(0)
        gd.update(2)  # Gap at 1
        stats = gd.stats
        assert stats["total_chunks"] == 2
        assert stats["total_gaps"] == 1

    def test_reset_reinitializes(self) -> None:
        gd = GapDetector("test")
        gd.update(5)
        gd.reset()
        # After reset, seq=0 is fine (new utterance)
        gaps = gd.update(0)
        assert gaps == []
