"""
Evaluation metrics for the KWS embedding model.

Provides accuracy, precision, recall, F1, cosine similarity distribution,
and ROC-AUC computation for embedding-based KWS evaluation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Accuracy in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def compute_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """Compute precision (macro-averaged by default).

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        average: Averaging strategy ('macro', 'micro', 'weighted').

    Returns:
        Precision score.
    """
    return float(precision_score(y_true, y_pred, average=average, zero_division=0))


def compute_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """Compute recall (macro-averaged by default)."""
    return float(recall_score(y_true, y_pred, average=average, zero_division=0))


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """Compute F1 score (macro-averaged by default)."""
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


# ---------------------------------------------------------------------------
# Embedding / similarity metrics
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D embedding vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pairwise_cosine_similarity(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings_a: Array of shape (m, dim).
        embeddings_b: Array of shape (n, dim).

    Returns:
        Similarity matrix of shape (m, n).
    """
    a_norm = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-8)
    b_norm = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def embedding_similarity_stats(
    positive_pairs: list[tuple[np.ndarray, np.ndarray]],
    negative_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Compute cosine similarity statistics for positive and negative pairs.

    Args:
        positive_pairs: List of (embedding_a, embedding_b) from the same class.
        negative_pairs: List of (embedding_a, embedding_b) from different classes.

    Returns:
        Dict with mean/std similarities for positive and negative pairs.
    """
    pos_sims = [cosine_similarity(a, b) for a, b in positive_pairs]
    neg_sims = [cosine_similarity(a, b) for a, b in negative_pairs]

    return {
        "positive_mean": float(np.mean(pos_sims)) if pos_sims else 0.0,
        "positive_std": float(np.std(pos_sims)) if pos_sims else 0.0,
        "negative_mean": float(np.mean(neg_sims)) if neg_sims else 0.0,
        "negative_std": float(np.std(neg_sims)) if neg_sims else 0.0,
        "n_positive_pairs": len(pos_sims),
        "n_negative_pairs": len(neg_sims),
    }


# ---------------------------------------------------------------------------
# ROC / AUC
# ---------------------------------------------------------------------------


def compute_roc_auc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve and AUC for binary wake/no-wake classification.

    Args:
        scores: Cosine similarity scores (higher = more likely keyword).
        labels: Binary labels (1 = keyword, 0 = non-keyword).

    Returns:
        Tuple of (auc, fpr_array, tpr_array, thresholds_array).
    """
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return auc, fpr, tpr, thresholds


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_metrics_summary(metrics: dict) -> None:
    """Print a formatted metrics summary table."""
    print("\n" + "=" * 50)
    print("  KWS Evaluation Metrics")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<30} {value:.4f}")
        else:
            print(f"  {key:<30} {value}")
    print("=" * 50 + "\n")
