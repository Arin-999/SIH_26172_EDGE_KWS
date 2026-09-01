"""
DS-CNN embedding model for keyword spotting.

Architecture:
  Input: (batch, 49, 40, 1) MFCC frames
  Backbone: 4× Depthwise Separable CNN blocks
  Head: GlobalAveragePooling2D → Dense(64) → L2 normalization
  Output: (batch, 64) unit-norm embedding

Reference:
  Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers" (2018)
  https://arxiv.org/abs/1711.07128

Target: ≤60 KB INT8 TFLite, ≤80 KB tensor arena on ESP32-S3.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# DS-CNN block
# ---------------------------------------------------------------------------


def _ds_cnn_block(
    x: tf.Tensor,
    filters: int,
    stride: int = 1,
    dropout_rate: float = 0.1,
) -> tf.Tensor:
    """Depthwise Separable CNN block with BN and ReLU.

    Structure: DepthwiseConv2D → BN → ReLU → Conv2D(1×1) → BN → ReLU

    Args:
        x: Input tensor.
        filters: Number of output filters in the pointwise convolution.
        stride: Stride for the depthwise convolution.
        dropout_rate: Dropout rate applied after the block.

    Returns:
        Output tensor.
    """
    # Depthwise convolution
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Pointwise convolution
    x = layers.Conv2D(filters, kernel_size=(1, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    return x


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_dscnn_model(
    input_shape: tuple[int, int, int] = (49, 40, 1),
    embedding_dim: int = 64,
    dropout: float = 0.1,
) -> keras.Model:
    """Build the DS-CNN embedding model.

    Args:
        input_shape: Shape of the MFCC input (frames, coeffs, channels).
        embedding_dim: Dimensionality of the output embedding vector.
        dropout: Dropout rate applied in DS-CNN blocks and before the head.

    Returns:
        Keras Model with:
          - Input: (batch, 49, 40, 1) float32
          - Output: (batch, embedding_dim) L2-normalized float32 embedding
    """
    inputs = keras.Input(shape=input_shape, name="mfcc_input")

    # Initial standard convolution
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 4 DS-CNN blocks with progressive downsampling
    x = _ds_cnn_block(x, filters=64, stride=2, dropout_rate=dropout)   # 25 × 20 × 64
    x = _ds_cnn_block(x, filters=128, stride=2, dropout_rate=dropout)  # 13 × 10 × 128
    x = _ds_cnn_block(x, filters=128, stride=1, dropout_rate=dropout)  # 13 × 10 × 128
    x = _ds_cnn_block(x, filters=128, stride=1, dropout_rate=dropout)  # 13 × 10 × 128

    # Global average pooling → compact representation
    x = layers.GlobalAveragePooling2D()(x)  # 128-dim

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    # Projection to embedding space
    x = layers.Dense(embedding_dim, use_bias=False, name="embedding_dense")(x)

    # L2 normalization → unit-norm embedding on the hypersphere
    outputs = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=-1),
        name="l2_norm",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="dscnn_kws")
    return model


def build_classifier_model(
    backbone: keras.Model,
    num_classes: int,
) -> keras.Model:
    """Attach a softmax classification head to the embedding backbone.

    Used only during pre-training on Speech Commands v2. Discarded at
    export time (only the backbone/embedding is used for deployment).

    Args:
        backbone: DS-CNN embedding model (output: L2-normalized embedding).
        num_classes: Number of classification classes.

    Returns:
        Keras Model with softmax output for standard cross-entropy training.
    """
    inputs = backbone.input
    embeddings = backbone.output

    # Stop gradient through L2 norm for the classification head
    logits = layers.Dense(num_classes, name="class_logits")(embeddings)
    outputs = layers.Softmax(name="class_probs")(logits)

    return keras.Model(inputs=inputs, outputs=outputs, name="dscnn_classifier")


# ---------------------------------------------------------------------------
# Model summary helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_dscnn_model()
    model.summary()
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Estimated INT8 model size: ~{total_params / 1024:.1f} KB")
