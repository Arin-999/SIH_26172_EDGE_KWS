"""
Quantization-Aware Training (QAT) for the DS-CNN KWS model.

Fine-tunes the FP32 model with fake quantization nodes inserted, so the
model is aware of quantization effects during training. This typically
recovers 1–2% accuracy lost during post-training quantization.

Usage:
    python ml/quantization/qat.py --config ml/training/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.training.model import build_dscnn_model
from ml.training.losses import prototypical_loss
from ml.training.train import load_class_files, sample_episode, set_seed


def apply_qat(model: tf.keras.Model) -> tf.keras.Model:
    """Wrap model with QAT fake-quantization nodes.

    Args:
        model: FP32 Keras model.

    Returns:
        QAT-annotated Keras model.
    """
    qat_model = tfmot.quantization.keras.quantize_model(model)
    return qat_model


def fine_tune_qat(
    config: dict,
    fp32_weights_path: str,
    output_dir: str,
    n_epochs: int = 5,
) -> tf.keras.Model:
    """Fine-tune the model with QAT for a few epochs.

    Args:
        config: Parsed config.yaml.
        fp32_weights_path: Path to best FP32 checkpoint weights (.h5).
        output_dir: Directory to save QAT model.
        n_epochs: Number of fine-tuning epochs.

    Returns:
        Fine-tuned QAT-aware Keras model.
    """
    set_seed(config["training"]["seed"])

    # Build base model and load FP32 weights
    base_model = build_dscnn_model(
        input_shape=(config["mfcc"]["num_frames"], config["mfcc"]["num_coeffs"], 1),
        embedding_dim=config["model"]["embedding_dim"],
        dropout=0.0,  # No dropout during QAT fine-tune
    )
    base_model.load_weights(fp32_weights_path)

    # Apply QAT
    qat_model = apply_qat(base_model)
    qat_model.summary()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["training"]["learning_rate"] * 0.1  # Lower LR for fine-tune
    )

    # Load training data
    train_files = load_class_files(config["paths"]["processed_dir"], split="train")
    val_files = load_class_files(config["paths"]["processed_dir"], split="val")

    n_way = config["training"]["n_way"]
    n_shot = config["training"]["n_shot"]
    n_query = max(1, n_shot // 2)
    n_episodes = config["training"]["n_episodes"] // 5  # Fewer episodes for fine-tune

    best_val_acc = 0.0
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        train_losses, train_accs = [], []

        for _ in range(n_episodes):
            s_feat, s_lbl, q_feat, q_lbl = sample_episode(
                train_files, n_way=n_way, n_shot=n_shot, n_query=n_query
            )
            with tf.GradientTape() as tape:
                s_emb = qat_model(s_feat, training=True)
                q_emb = qat_model(q_feat, training=True)
                loss, acc = prototypical_loss(s_emb, q_emb, s_lbl, q_lbl, n_way, n_shot)
            grads = tape.gradient(loss, qat_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, qat_model.trainable_variables))
            train_losses.append(float(loss))
            train_accs.append(float(acc))

        # Validate
        val_accs = []
        for _ in range(30):
            s_feat, s_lbl, q_feat, q_lbl = sample_episode(
                val_files, n_way=n_way, n_shot=n_shot, n_query=n_query
            )
            _, acc = prototypical_loss(
                qat_model(s_feat, training=False),
                qat_model(q_feat, training=False),
                s_lbl, q_lbl, n_way, n_shot,
            )
            val_accs.append(float(acc))

        val_acc = float(np.mean(val_accs))
        print(
            f"QAT Epoch {epoch}/{n_epochs} | "
            f"loss={np.mean(train_losses):.4f} acc={np.mean(train_accs):.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            qat_model.save_weights(str(out_path / "qat_best.weights.h5"))
            print(f"  [*] QAT best model saved (val_acc={val_acc:.4f})")

    return qat_model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QAT fine-tuning for KWS model.")
    parser.add_argument("--config", default="ml/training/config.yaml")
    parser.add_argument(
        "--fp32-weights",
        default="ml/models/checkpoints/best.weights.h5",
        help="Path to FP32 checkpoint weights",
    )
    parser.add_argument(
        "--output-dir",
        default="ml/models/exported",
        help="Output directory for QAT model",
    )
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    fine_tune_qat(
        config=config,
        fp32_weights_path=args.fp32_weights,
        output_dir=args.output_dir,
        n_epochs=args.epochs,
    )
