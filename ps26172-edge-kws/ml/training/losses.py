"""
Loss functions for prototypical few-shot KWS training.

Implements:
  1. Prototypical loss — core few-shot metric learning loss
  2. Classification cross-entropy — for pre-training backbone on Speech Commands
  3. Combined loss — weighted sum used in practice
"""

from __future__ import annotations

import tensorflow as tf


# ---------------------------------------------------------------------------
# Prototypical loss
# ---------------------------------------------------------------------------


def prototypical_loss(
    support_embeddings: tf.Tensor,
    query_embeddings: tf.Tensor,
    support_labels: tf.Tensor,
    query_labels: tf.Tensor,
    n_way: int,
    n_shot: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute prototypical network loss for a single N-way K-shot episode.

    For each class, the prototype is the mean of its K support embeddings.
    Distances from each query embedding to every prototype are computed via
    negative squared Euclidean distance. Cross-entropy over these distances
    gives the episode loss.

    Args:
        support_embeddings: Tensor of shape (n_way * n_shot, embedding_dim).
        query_embeddings: Tensor of shape (n_queries, embedding_dim).
        support_labels: Integer class labels for support set (n_way * n_shot,).
        query_labels: Integer class labels for query set (n_queries,).
        n_way: Number of classes per episode.
        n_shot: Number of support examples per class.

    Returns:
        Tuple of (loss scalar, accuracy scalar).
    """
    # Compute class prototypes: shape (n_way, embedding_dim)
    prototypes = _compute_prototypes(support_embeddings, support_labels, n_way)

    # Pairwise squared Euclidean distance: (n_queries, n_way)
    dists = _pairwise_squared_distance(query_embeddings, prototypes)

    # Logits = negative distance (closer = higher score)
    logits = -dists

    # Remap query labels to episode-local indices [0, n_way)
    unique_classes = tf.unique(support_labels).y
    local_query_labels = _remap_labels(query_labels, unique_classes)

    # Cross-entropy loss
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=local_query_labels,
            logits=logits,
        )
    )

    # Accuracy
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, local_query_labels), tf.float32)
    )

    return loss, accuracy


def _compute_prototypes(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    n_way: int,
) -> tf.Tensor:
    """Compute class prototypes as mean of support embeddings.

    Args:
        embeddings: (n_way * n_shot, dim) support embeddings.
        labels: (n_way * n_shot,) integer class labels.
        n_way: Number of classes.

    Returns:
        (n_way, dim) prototype tensor.
    """
    unique_classes, _ = tf.unique(labels)
    # Sort to ensure deterministic order
    unique_classes = tf.sort(unique_classes)

    prototypes = tf.stack([
        tf.reduce_mean(
            tf.gather(embeddings, tf.where(tf.equal(labels, c))[:, 0]),
            axis=0,
        )
        for c in tf.unstack(unique_classes)
    ])  # (n_way, dim)

    return prototypes


def _pairwise_squared_distance(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """Compute pairwise squared Euclidean distances.

    Args:
        a: Tensor of shape (m, dim).
        b: Tensor of shape (n, dim).

    Returns:
        Distance tensor of shape (m, n).
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b^T
    a_sq = tf.reduce_sum(a ** 2, axis=1, keepdims=True)  # (m, 1)
    b_sq = tf.reduce_sum(b ** 2, axis=1, keepdims=True)  # (n, 1)
    ab = tf.matmul(a, b, transpose_b=True)               # (m, n)
    dist = a_sq + tf.transpose(b_sq) - 2 * ab
    return tf.maximum(dist, 0.0)  # Clamp to avoid numerical negatives


def _remap_labels(labels: tf.Tensor, unique_classes: tf.Tensor) -> tf.Tensor:
    """Remap arbitrary integer labels to episode-local indices [0, n_way).

    Args:
        labels: Query labels with values from `unique_classes`.
        unique_classes: Sorted tensor of unique class values in this episode.

    Returns:
        Remapped labels in [0, n_way).
    """
    remapped = tf.zeros_like(labels)
    for i, cls in enumerate(tf.unstack(unique_classes)):
        mask = tf.cast(tf.equal(labels, cls), tf.int32)
        remapped = remapped + mask * i
    return remapped


# ---------------------------------------------------------------------------
# Angular margin softmax (ArcFace-lite)
# ---------------------------------------------------------------------------


def arcface_loss(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    weight_matrix: tf.Variable,
    num_classes: int,
    margin: float = 0.3,
    scale: float = 30.0,
) -> tf.Tensor:
    """Additive Angular Margin (ArcFace) loss for metric learning pre-training.

    Encourages tight, well-separated clusters in embedding space.

    Args:
        embeddings: L2-normalized embeddings, shape (batch, embedding_dim).
        labels: Integer class labels, shape (batch,).
        weight_matrix: Learnable class weight matrix, shape (num_classes, embedding_dim).
        num_classes: Total number of classes.
        margin: Additive angular margin in radians.
        scale: Logit scale factor.

    Returns:
        Scalar loss tensor.
    """
    # Normalize weight matrix (class centres)
    w = tf.math.l2_normalize(weight_matrix, axis=1)  # (num_classes, dim)

    # Cosine similarities: (batch, num_classes)
    cosine = tf.matmul(embeddings, w, transpose_b=True)
    cosine = tf.clip_by_value(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

    # theta + m for the ground-truth class
    theta = tf.acos(cosine)
    one_hot = tf.one_hot(labels, num_classes)
    theta_m = theta + margin * one_hot
    cosine_m = tf.cos(theta_m)

    logits = scale * cosine_m
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    )
    return loss


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------


def combined_loss(
    proto_loss: tf.Tensor,
    arc_loss: tf.Tensor,
    alpha: float = 0.7,
) -> tf.Tensor:
    """Weighted combination of prototypical and ArcFace losses.

    Args:
        proto_loss: Scalar prototypical loss.
        arc_loss: Scalar ArcFace loss.
        alpha: Weight for prototypical loss. ArcFace weight = (1 - alpha).

    Returns:
        Combined scalar loss.
    """
    return alpha * proto_loss + (1.0 - alpha) * arc_loss
