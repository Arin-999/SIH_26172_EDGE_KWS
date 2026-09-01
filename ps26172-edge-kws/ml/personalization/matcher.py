"""
Cosine similarity matcher with 2-of-3 sliding window debounce.

Mirrors the firmware matcher logic in `firmware/esp32/main/matcher.cpp`.
Used by the personalization pipeline and examples on the host PC.
"""

from __future__ import annotations

from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Single-shot matcher
# ---------------------------------------------------------------------------


def match(
    embedding: np.ndarray,
    prototype: np.ndarray,
    threshold: float = 0.75,
) -> tuple[bool, float]:
    """Compare an embedding to a prototype using cosine similarity.

    Args:
        embedding: Query embedding, shape (64,). Need not be normalized.
        prototype: Stored wake-word prototype, shape (64,). Need not be normalized.
        threshold: Cosine similarity threshold for acceptance.

    Returns:
        Tuple of (accepted: bool, score: float) where score ∈ [-1, 1].
    """
    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
    proto_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
    score = float(np.dot(emb_norm, proto_norm))
    return score >= threshold, score


# ---------------------------------------------------------------------------
# Debounce matcher (stateful)
# ---------------------------------------------------------------------------


class DebounceMatcher:
    """Stateful 2-of-N sliding window keyword matcher.

    Implements the same debounce logic as the ESP32 firmware.
    A wake event is reported when `hits_required` out of the last
    `window_size` frames exceed the cosine similarity threshold.

    Example::
        matcher = DebounceMatcher(prototype, threshold=0.75, hits=2, window=3)
        for audio_frame in stream:
            embedding = extract_embedding(audio_frame)
            wake, score = matcher.update(embedding)
            if wake:
                print("WAKE DETECTED")
    """

    def __init__(
        self,
        prototype: np.ndarray,
        threshold: float = 0.75,
        hits_required: int = 2,
        window_size: int = 3,
    ) -> None:
        """Initialize the debounce matcher.

        Args:
            prototype: Wake-word prototype embedding, shape (64,).
            threshold: Cosine similarity threshold.
            hits_required: Number of frames above threshold needed to fire.
            window_size: Size of the sliding window.
        """
        self.prototype = prototype / (np.linalg.norm(prototype) + 1e-8)
        self.threshold = threshold
        self.hits_required = hits_required
        self.window: deque[bool] = deque(maxlen=window_size)
        self.window_size = window_size
        self._last_score: float = 0.0

    def update(self, embedding: np.ndarray) -> tuple[bool, float]:
        """Process a new embedding and check for wake event.

        Args:
            embedding: Query embedding from the KWS model, shape (64,).

        Returns:
            Tuple of (wake_detected: bool, score: float).
            `wake_detected` is True only on the exact frame when the
            wake condition is newly satisfied (one-shot event).
        """
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        score = float(np.dot(emb_norm, self.prototype))
        self._last_score = score

        hit = score >= self.threshold
        self.window.append(hit)

        # Check if hits_required out of the last window_size frames are hits
        wake = sum(self.window) >= self.hits_required
        return wake, score

    def reset(self) -> None:
        """Clear the sliding window (call after a wake event is handled)."""
        self.window.clear()

    @property
    def last_score(self) -> float:
        """Most recent cosine similarity score."""
        return self._last_score


# ---------------------------------------------------------------------------
# Batch matching (for evaluation)
# ---------------------------------------------------------------------------


def match_sequence(
    embeddings: np.ndarray,
    prototype: np.ndarray,
    threshold: float = 0.75,
    hits_required: int = 2,
    window_size: int = 3,
) -> list[tuple[int, float]]:
    """Run debounce matching over a sequence of embeddings.

    Args:
        embeddings: Array of shape (T, 64) containing T consecutive embeddings.
        prototype: Wake-word prototype, shape (64,).
        threshold: Cosine similarity threshold.
        hits_required: Hits needed within window to fire.
        window_size: Sliding window size.

    Returns:
        List of (frame_index, score) tuples for each wake event detected.
    """
    matcher = DebounceMatcher(prototype, threshold, hits_required, window_size)
    events: list[tuple[int, float]] = []

    for i, emb in enumerate(embeddings):
        wake, score = matcher.update(emb)
        if wake:
            events.append((i, score))
            matcher.reset()  # Prevent consecutive fires

    return events
