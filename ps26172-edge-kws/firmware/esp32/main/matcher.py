"""
Python reference implementation of the ESP32 KWS cosine similarity matcher.

Matches: firmware/esp32/main/matcher.c
Used by the firmware simulator and unit tests to validate firmware logic.
"""

from __future__ import annotations

from collections import deque

import numpy as np

# Must match firmware menuconfig / config.yaml
SIMILARITY_THRESHOLD: float = 0.75
DEBOUNCE_HITS: int = 2
DEBOUNCE_WINDOW: int = 3
EMBEDDING_DIM: int = 64


def cosine_similarity_int8(
    embedding: np.ndarray,  # int8 (64,)
    prototype: np.ndarray,  # int8 (64,)
    out_scale: float = 0.007874016,
    out_zero_point: int = 0,
) -> float:
    """Compute cosine similarity between two INT8 embeddings.

    Dequantizes to float32 before computing similarity, matching the
    firmware fixed-point-to-float conversion step.

    Args:
        embedding: INT8 query embedding from TFLite inference.
        prototype: INT8 stored prototype from enrollment.
        out_scale: Output quantization scale from model_metadata.json.
        out_zero_point: Output quantization zero point.

    Returns:
        Cosine similarity in [-1, 1].
    """
    # Dequantize
    emb_f = (embedding.astype(np.float32) - out_zero_point) * out_scale
    proto_f = (prototype.astype(np.float32) - out_zero_point) * out_scale

    # L2 normalize
    emb_norm = np.linalg.norm(emb_f)
    proto_norm = np.linalg.norm(proto_f)
    if emb_norm < 1e-8 or proto_norm < 1e-8:
        return 0.0

    return float(np.dot(emb_f / emb_norm, proto_f / proto_norm))


class FirmwareMatcher:
    """Python reference of the firmware's debounce matcher state machine.

    Exactly mirrors the C implementation in matcher.c.

    States:
        IDLE → KWS inference running every 250 ms
        WAKE → 2-of-3 hits detected → trigger streaming
    """

    def __init__(
        self,
        prototype: np.ndarray,
        threshold: float = SIMILARITY_THRESHOLD,
        hits: int = DEBOUNCE_HITS,
        window: int = DEBOUNCE_WINDOW,
    ) -> None:
        self.prototype = prototype
        self.threshold = threshold
        self.hits_required = hits
        self.window: deque[bool] = deque(maxlen=window)
        self._scores: list[float] = []

    def push(self, embedding: np.ndarray) -> tuple[bool, float]:
        """Process one embedding. Returns (wake_detected, score)."""
        if embedding.dtype == np.int8:
            score = cosine_similarity_int8(embedding, self.prototype.astype(np.int8))
        else:
            emb_n = embedding / (np.linalg.norm(embedding) + 1e-8)
            proto_n = self.prototype / (np.linalg.norm(self.prototype) + 1e-8)
            score = float(np.dot(emb_n, proto_n))

        self._scores.append(score)
        hit = score >= self.threshold
        self.window.append(hit)
        wake = sum(self.window) >= self.hits_required
        if wake:
            self.reset()
        return wake, score

    def reset(self) -> None:
        self.window.clear()

    @property
    def recent_scores(self) -> list[float]:
        return self._scores[-10:]
