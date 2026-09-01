"""
False Accept Rate (FAR) and False Reject Rate (FRR) computation.

Sweeps cosine similarity threshold over a test set of positive (enrolled
speaker speaking their keyword) and negative (other speakers / other words)
pairs to generate a DET curve and find the Equal Error Rate (EER).

Output: ml/benchmarks/far_frr.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def pairwise_cosine_similarity(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
) -> np.ndarray:
    """Pairwise cosine similarity matrix (pure numpy, no sklearn dependency)."""
    a_norm = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-8)
    b_norm = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_far_frr(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    thresholds: np.ndarray | None = None,
    n_thresholds: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FAR and FRR curves over a range of thresholds.

    FAR (False Accept Rate): fraction of negative samples accepted as keyword.
    FRR (False Reject Rate): fraction of positive samples rejected as keyword.

    A sample is "accepted" when its similarity score >= threshold.

    Args:
        positive_scores: Cosine similarity scores for true keyword pairs.
        negative_scores: Cosine similarity scores for impostor pairs.
        thresholds: Optional array of thresholds to evaluate. If None,
            linspace from min(all_scores) to max(all_scores).
        n_thresholds: Number of threshold points if `thresholds` is None.

    Returns:
        Tuple of (thresholds, far_array, frr_array).
    """
    all_scores = np.concatenate([positive_scores, negative_scores])
    if thresholds is None:
        thresholds = np.linspace(all_scores.min() - 0.01, all_scores.max() + 0.01, n_thresholds)

    far_array = np.array([
        float(np.mean(negative_scores >= t)) for t in thresholds
    ])
    frr_array = np.array([
        float(np.mean(positive_scores < t)) for t in thresholds
    ])

    return thresholds, far_array, frr_array


def find_eer(
    thresholds: np.ndarray,
    far_array: np.ndarray,
    frr_array: np.ndarray,
) -> tuple[float, float, float]:
    """Find the Equal Error Rate (EER) point.

    The EER is the threshold where FAR ≈ FRR. Lower is better.

    Args:
        thresholds: Array of threshold values.
        far_array: FAR at each threshold.
        frr_array: FRR at each threshold.

    Returns:
        Tuple of (eer_threshold, eer_far, eer_frr).
    """
    # Find the crossing point by minimizing |FAR - FRR|
    diff = np.abs(far_array - frr_array)
    idx = int(np.argmin(diff))
    eer_threshold = float(thresholds[idx])
    eer_far = float(far_array[idx])
    eer_frr = float(frr_array[idx])
    return eer_threshold, eer_far, eer_frr


def find_operating_point(
    thresholds: np.ndarray,
    far_array: np.ndarray,
    frr_array: np.ndarray,
    max_far: float = 0.05,
) -> tuple[float, float, float]:
    """Find the best operating threshold satisfying a maximum FAR constraint.

    Selects the threshold with FAR ≤ max_far that minimizes FRR.

    Args:
        thresholds: Array of threshold values.
        far_array: FAR at each threshold.
        frr_array: FRR at each threshold.
        max_far: Maximum acceptable FAR (e.g. 0.05 = 5%).

    Returns:
        Tuple of (threshold, far, frr) at the operating point.
        Returns EER point if no threshold satisfies the constraint.
    """
    valid = np.where(far_array <= max_far)[0]
    if len(valid) == 0:
        return find_eer(thresholds, far_array, frr_array)

    best_idx = valid[np.argmin(frr_array[valid])]
    return float(thresholds[best_idx]), float(far_array[best_idx]), float(frr_array[best_idx])


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def save_far_frr_csv(
    thresholds: np.ndarray,
    far_array: np.ndarray,
    frr_array: np.ndarray,
    output_path: str,
) -> None:
    """Save FAR/FRR curve to CSV.

    Args:
        thresholds: Threshold values.
        far_array: FAR at each threshold.
        frr_array: FRR at each threshold.
        output_path: Output CSV file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "far", "frr"])
        for t, far, frr in zip(thresholds, far_array, frr_array):
            writer.writerow([f"{t:.6f}", f"{far:.6f}", f"{frr:.6f}"])
    print(f"[far_frr] Saved to {output_path}")


# ---------------------------------------------------------------------------
# Plot (optional)
# ---------------------------------------------------------------------------


def plot_det_curve(
    thresholds: np.ndarray,
    far_array: np.ndarray,
    frr_array: np.ndarray,
    eer_threshold: float,
    output_path: str | None = None,
) -> None:
    """Plot DET (Detection Error Tradeoff) curve.

    Args:
        thresholds: Threshold values.
        far_array: FAR array.
        frr_array: FRR array.
        eer_threshold: EER threshold to annotate on the plot.
        output_path: If provided, save the figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[far_frr] matplotlib not available. Skipping plot.")
        return

    plt.figure(figsize=(7, 5))
    plt.plot(far_array * 100, frr_array * 100, "b-", linewidth=2, label="DET curve")

    # EER point
    eer_idx = int(np.argmin(np.abs(thresholds - eer_threshold)))
    plt.scatter(
        far_array[eer_idx] * 100,
        frr_array[eer_idx] * 100,
        color="red",
        zorder=5,
        label=f"EER = {far_array[eer_idx]*100:.1f}% @ threshold={eer_threshold:.3f}",
    )

    plt.xlabel("FAR (%)")
    plt.ylabel("FRR (%)")
    plt.title("Detection Error Tradeoff (DET) Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"[far_frr] DET curve saved to {output_path}")
    else:
        plt.show()
    plt.close()
