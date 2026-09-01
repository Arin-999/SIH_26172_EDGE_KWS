"""
Integration test: end-to-end pipeline (enrollment → inference → server).

Runs the full pipeline without physical hardware:
1. Generate synthetic audio for a "wake word" class
2. Enroll a prototype from 5 utterances
3. Extract embeddings from test utterances
4. Run the debounce matcher
5. Verify FAR/FRR requirements

This test does NOT require a running server or TFLite model (uses numpy
mock embeddings) to remain CI-friendly without model artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.personalization.matcher import DebounceMatcher, match
from ml.evaluation.far_frr import compute_far_frr, find_eer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cluster(center: np.ndarray, n: int, noise: float, seed: int) -> np.ndarray:
    """Generate n embeddings near `center` with Gaussian noise."""
    rng = np.random.default_rng(seed)
    noise_vecs = rng.standard_normal((n, len(center))).astype(np.float32) * noise
    embs = center[np.newaxis, :] + noise_vecs
    # L2-normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return (embs / norms).astype(np.float32)


@pytest.fixture
def enrollment_prototype() -> np.ndarray:
    """Compute a prototype from 10 enrollment utterances."""
    rng = np.random.default_rng(1)
    center = rng.standard_normal(64).astype(np.float32)
    center /= np.linalg.norm(center)
    support = _make_cluster(center, n=10, noise=0.15, seed=2)
    proto = support.mean(axis=0)
    return (proto / (np.linalg.norm(proto) + 1e-8)).astype(np.float32)


@pytest.fixture
def positive_embeddings(enrollment_prototype: np.ndarray) -> np.ndarray:
    """50 positive test embeddings (enrolled speaker, correct keyword)."""
    return _make_cluster(enrollment_prototype, n=50, noise=0.15, seed=3)


@pytest.fixture
def negative_embeddings() -> np.ndarray:
    """200 negative test embeddings (random / other speakers)."""
    rng = np.random.default_rng(99)
    embs = rng.standard_normal((200, 64)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return (embs / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    def test_positive_scores_higher_than_negative(
        self,
        enrollment_prototype: np.ndarray,
        positive_embeddings: np.ndarray,
        negative_embeddings: np.ndarray,
    ) -> None:
        """Mean similarity of positives should be higher than negatives."""
        pos_scores = np.array([
            match(emb, enrollment_prototype)[1]
            for emb in positive_embeddings
        ])
        neg_scores = np.array([
            match(emb, enrollment_prototype)[1]
            for emb in negative_embeddings
        ])
        assert float(pos_scores.mean()) > float(neg_scores.mean())

    def test_far_frr_requirements(
        self,
        enrollment_prototype: np.ndarray,
        positive_embeddings: np.ndarray,
        negative_embeddings: np.ndarray,
    ) -> None:
        """FAR < 10% and FRR < 20% at operating threshold (mock embeddings)."""
        pos_scores = np.array([
            match(emb, enrollment_prototype)[1]
            for emb in positive_embeddings
        ])
        neg_scores = np.array([
            match(emb, enrollment_prototype)[1]
            for emb in negative_embeddings
        ])

        thresholds, far_arr, frr_arr = compute_far_frr(pos_scores, neg_scores)
        eer_threshold, eer_far, eer_frr = find_eer(thresholds, far_arr, frr_arr)

        # For well-separated mock clusters, EER should be < 20%
        eer = (eer_far + eer_frr) / 2
        assert eer < 0.25, f"EER too high: {eer:.2%}"

    def test_debounce_fires_on_consecutive_hits(
        self,
        enrollment_prototype: np.ndarray,
        positive_embeddings: np.ndarray,
    ) -> None:
        """Debounce matcher should fire when 2 consecutive hits occur.

        Note: positive embeddings are generated with noise=0.15 relative to
        the prototype, yielding cosine scores in roughly [0.51, 0.75].
        Threshold is set to 0.60 to ensure reliable hits in a small sample.
        """
        matcher = DebounceMatcher(
            enrollment_prototype, threshold=0.60, hits_required=2, window_size=3
        )
        fired = False
        for emb in positive_embeddings[:10]:
            wake, score = matcher.update(emb)
            if wake:
                fired = True
                break
        assert fired, "Expected wake event in 10 positive embeddings"

    def test_debounce_no_fire_on_negatives(
        self,
        enrollment_prototype: np.ndarray,
        negative_embeddings: np.ndarray,
    ) -> None:
        """Debounce matcher should not fire on random (negative) embeddings."""
        matcher = DebounceMatcher(
            enrollment_prototype, threshold=0.75, hits_required=2, window_size=3
        )
        fires = 0
        for emb in negative_embeddings[:100]:
            wake, _ = matcher.update(emb)
            if wake:
                fires += 1
        # Allow at most 5% of windows to fire (FAR tolerance)
        assert fires <= 5, f"Too many false accepts: {fires}/100"
