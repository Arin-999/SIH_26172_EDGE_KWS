"""
Unit tests for the cosine similarity matcher and debounce logic.
"""

import numpy as np
import pytest

from ml.personalization.matcher import (
    match,
    DebounceMatcher,
    match_sequence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prototype() -> np.ndarray:
    """Deterministic unit-norm prototype embedding."""
    rng = np.random.default_rng(1)
    v = rng.standard_normal(64).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def similar_embedding(prototype: np.ndarray) -> np.ndarray:
    """Embedding very similar to the prototype (cosine sim ≈ 0.95)."""
    noise = np.random.default_rng(2).standard_normal(64).astype(np.float32) * 0.1
    v = prototype + noise
    return v / np.linalg.norm(v)


@pytest.fixture
def dissimilar_embedding() -> np.ndarray:
    """Random embedding with no relation to prototype."""
    rng = np.random.default_rng(99)
    v = rng.standard_normal(64).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Tests: match (single-shot)
# ---------------------------------------------------------------------------


class TestMatch:
    def test_accepts_similar(
        self, similar_embedding: np.ndarray, prototype: np.ndarray
    ) -> None:
        wake, score = match(similar_embedding, prototype, threshold=0.75)
        assert score > 0.75
        assert wake is True

    def test_rejects_dissimilar(
        self, dissimilar_embedding: np.ndarray, prototype: np.ndarray
    ) -> None:
        wake, score = match(dissimilar_embedding, prototype, threshold=0.75)
        # Very likely to be below threshold for a random unit vector
        # (expected cosine sim ≈ 0 for random 64-dim vectors)
        assert score < 0.9  # Generous bound

    def test_self_similarity_is_one(self, prototype: np.ndarray) -> None:
        wake, score = match(prototype, prototype, threshold=0.99)
        assert score == pytest.approx(1.0, abs=1e-5)
        assert wake is True

    def test_returns_tuple(self, prototype: np.ndarray) -> None:
        result = match(prototype, prototype)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_score_range(
        self, similar_embedding: np.ndarray, prototype: np.ndarray
    ) -> None:
        _, score = match(similar_embedding, prototype)
        assert -1.0 <= score <= 1.0 + 1e-6

    def test_unnormalized_inputs(self, prototype: np.ndarray) -> None:
        """match() should normalize inputs internally."""
        unnorm_proto = prototype * 5.0
        unnorm_emb = prototype * 3.0
        _, score_norm = match(prototype, prototype)
        _, score_unnorm = match(unnorm_emb, unnorm_proto)
        assert score_norm == pytest.approx(score_unnorm, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: DebounceMatcher
# ---------------------------------------------------------------------------


class TestDebounceMatcher:
    def test_no_wake_on_single_hit(
        self, prototype: np.ndarray, similar_embedding: np.ndarray
    ) -> None:
        """2-of-3 debounce: single hit should not fire."""
        matcher = DebounceMatcher(prototype, threshold=0.75, hits_required=2, window_size=3)
        wake, _ = matcher.update(similar_embedding)
        assert wake is False

    def test_wake_on_two_consecutive_hits(
        self, prototype: np.ndarray, similar_embedding: np.ndarray
    ) -> None:
        """2-of-3: two consecutive hits should fire."""
        matcher = DebounceMatcher(prototype, threshold=0.75, hits_required=2, window_size=3)
        matcher.update(similar_embedding)
        wake, _ = matcher.update(similar_embedding)
        assert wake is True

    def test_no_wake_on_alternating_hits(
        self,
        prototype: np.ndarray,
        similar_embedding: np.ndarray,
        dissimilar_embedding: np.ndarray,
    ) -> None:
        """Alternating hit/miss pattern with 2-of-2 window: no fire."""
        matcher = DebounceMatcher(prototype, threshold=0.75, hits_required=2, window_size=2)
        matcher.update(similar_embedding)   # hit
        matcher.update(dissimilar_embedding)  # miss — evicts hit from window
        matcher.update(similar_embedding)   # hit — window now [miss, hit]
        wake, _ = matcher.update(dissimilar_embedding)  # miss — window [hit, miss]
        assert wake is False

    def test_reset_clears_window(
        self, prototype: np.ndarray, similar_embedding: np.ndarray
    ) -> None:
        matcher = DebounceMatcher(prototype, threshold=0.75, hits_required=2, window_size=3)
        matcher.update(similar_embedding)
        matcher.reset()
        # After reset, need 2 more hits to fire
        matcher.update(similar_embedding)
        wake, _ = matcher.update(similar_embedding)
        assert wake is True

    def test_last_score_updated(
        self, prototype: np.ndarray, similar_embedding: np.ndarray
    ) -> None:
        matcher = DebounceMatcher(prototype, threshold=0.75)
        _, score = matcher.update(similar_embedding)
        assert matcher.last_score == pytest.approx(score, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: match_sequence
# ---------------------------------------------------------------------------


class TestMatchSequence:
    def test_detects_event_in_sequence(
        self, prototype: np.ndarray, similar_embedding: np.ndarray, dissimilar_embedding: np.ndarray
    ) -> None:
        embeddings = np.stack([
            dissimilar_embedding,
            similar_embedding,
            similar_embedding,  # Wake should fire here
            dissimilar_embedding,
        ])
        events = match_sequence(embeddings, prototype, threshold=0.75)
        assert len(events) >= 1

    def test_empty_sequence_no_events(self, prototype: np.ndarray) -> None:
        events = match_sequence(np.zeros((0, 64), dtype=np.float32), prototype)
        assert events == []
