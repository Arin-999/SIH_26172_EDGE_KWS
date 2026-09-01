"""
Threshold optimization for the cosine similarity matcher.

Finds the optimal similarity threshold for a given user's enrollment
by sweeping thresholds over their positive (enrolled keyword) and
negative (background / other words) audio samples.

The optimal threshold maximizes the F1 score (or minimizes EER, configurable).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.evaluation.far_frr import compute_far_frr, find_eer, find_operating_point
from ml.personalization.embedding import extract_embedding
from ml.preprocessing.audio_preprocessing import preprocess


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_optimal_threshold(
    positive_audio_paths: list[str],
    negative_audio_paths: list[str],
    prototype: np.ndarray,
    model_path: str = "ml/models/int8/model.tflite",
    strategy: str = "max_far_5pct",
    n_thresholds: int = 200,
) -> dict:
    """Find the optimal cosine similarity threshold for a specific user.

    Args:
        positive_audio_paths: WAV files of the enrolled keyword (positive class).
        negative_audio_paths: WAV files of other words / background (negative class).
        prototype: Enrolled prototype embedding, shape (64,).
        model_path: Path to INT8 TFLite model.
        strategy: Threshold selection strategy:
            - 'eer': Minimize Equal Error Rate
            - 'max_far_5pct': Best FRR with FAR ≤ 5% (recommended)
            - 'max_far_10pct': Best FRR with FAR ≤ 10% (permissive)
        n_thresholds: Number of threshold candidate values.

    Returns:
        Dict with optimal threshold and corresponding FAR/FRR/EER.
    """
    proto_norm = prototype / (np.linalg.norm(prototype) + 1e-8)

    # Compute cosine similarity scores
    positive_scores = _compute_scores(positive_audio_paths, proto_norm, model_path)
    negative_scores = _compute_scores(negative_audio_paths, proto_norm, model_path)

    print(f"[threshold] {len(positive_scores)} positive samples, "
          f"{len(negative_scores)} negative samples")
    print(f"[threshold] Positive scores: mean={np.mean(positive_scores):.3f}, "
          f"std={np.std(positive_scores):.3f}")
    print(f"[threshold] Negative scores: mean={np.mean(negative_scores):.3f}, "
          f"std={np.std(negative_scores):.3f}")

    thresholds, far_array, frr_array = compute_far_frr(
        positive_scores, negative_scores, n_thresholds=n_thresholds
    )

    eer_threshold, eer_far, eer_frr = find_eer(thresholds, far_array, frr_array)

    if strategy == "eer":
        chosen_threshold, chosen_far, chosen_frr = eer_threshold, eer_far, eer_frr
    elif strategy == "max_far_10pct":
        chosen_threshold, chosen_far, chosen_frr = find_operating_point(
            thresholds, far_array, frr_array, max_far=0.10
        )
    else:  # default: max_far_5pct
        chosen_threshold, chosen_far, chosen_frr = find_operating_point(
            thresholds, far_array, frr_array, max_far=0.05
        )

    result = {
        "strategy": strategy,
        "optimal_threshold": round(chosen_threshold, 4),
        "far_at_threshold": round(chosen_far, 4),
        "frr_at_threshold": round(chosen_frr, 4),
        "eer_threshold": round(eer_threshold, 4),
        "eer_value": round((eer_far + eer_frr) / 2, 4),
        "n_positive": len(positive_scores),
        "n_negative": len(negative_scores),
    }

    print(f"\n[threshold] Optimal threshold ({strategy}): {chosen_threshold:.4f}")
    print(f"  FAR: {chosen_far*100:.2f}%  FRR: {chosen_frr*100:.2f}%")
    print(f"  EER: {result['eer_value']*100:.2f}% @ {eer_threshold:.4f}")

    return result


def adjust_sensitivity(
    current_threshold: float,
    direction: str,
    step: float = 0.02,
    min_threshold: float = 0.50,
    max_threshold: float = 0.95,
) -> float:
    """Manually adjust the similarity threshold by a fixed step.

    Used when a user wants to make the system more or less sensitive
    after deployment.

    Args:
        current_threshold: The current threshold value.
        direction: 'more_sensitive' (lower threshold) or 'less_sensitive' (higher).
        step: Step size to adjust by.
        min_threshold: Minimum allowed threshold.
        max_threshold: Maximum allowed threshold.

    Returns:
        New threshold value.
    """
    if direction == "more_sensitive":
        new_threshold = max(min_threshold, current_threshold - step)
    elif direction == "less_sensitive":
        new_threshold = min(max_threshold, current_threshold + step)
    else:
        raise ValueError(f"direction must be 'more_sensitive' or 'less_sensitive', got {direction!r}")

    print(f"[threshold] Adjusted: {current_threshold:.4f} → {new_threshold:.4f}")
    return new_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_scores(
    audio_paths: list[str],
    prototype_norm: np.ndarray,
    model_path: str,
) -> np.ndarray:
    """Extract embeddings and compute cosine similarity against prototype."""
    scores = []
    for path in audio_paths:
        try:
            audio = preprocess(path)
            emb = extract_embedding(audio, model_path=model_path)
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            scores.append(float(np.dot(emb_norm, prototype_norm)))
        except Exception as exc:
            print(f"  [WARN] Skipping {path}: {exc}")
    return np.array(scores, dtype=np.float32)
