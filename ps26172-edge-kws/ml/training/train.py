"""
Training script for DS-CNN KWS embedding model.

Trains the model using episodic few-shot learning (prototypical loss) on the
Google Speech Commands v2 dataset. Saves checkpoints and the final FP32
SavedModel.

Usage:
    python ml/training/train.py
    python ml/training/train.py --config ml/training/config.yaml
    python ml/training/train.py --epochs 10 --quick  # Smoke-test
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.preprocessing.audio_preprocessing import preprocess
from ml.preprocessing.feature_extraction import audio_to_features
from ml.augmentation.audio_augmentation import build_augmenter, augment
from ml.training.model import build_dscnn_model
from ml.training.losses import prototypical_loss


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_class_files(processed_dir: str, split: str = "train") -> dict[str, list[str]]:
    """Load file paths grouped by class from the processed dataset directory.

    Args:
        processed_dir: Path to `ml/datasets/processed/`.
        split: 'train', 'val', or 'test'.

    Returns:
        Dict mapping class_name -> list of .npy file paths.
    """
    split_dir = Path(processed_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {split_dir}. "
            "Run: python scripts/dataset/download_speech_commands.py"
        )

    class_files: dict[str, list[str]] = {}
    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():
            files = sorted(class_dir.glob("*.npy"))
            if files:
                class_files[class_dir.name] = [str(f) for f in files]

    if not class_files:
        raise RuntimeError(f"No processed .npy files found in {split_dir}.")

    return class_files


def load_features(path: str) -> np.ndarray:
    """Load a pre-processed MFCC feature array from a .npy file."""
    return np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# Episode sampler
# ---------------------------------------------------------------------------


def sample_episode(
    class_files: dict[str, list[str]],
    n_way: int,
    n_shot: int,
    n_query: int,
    augmenter=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a single N-way K-shot episode.

    Args:
        class_files: Dict of class_name -> list of feature file paths.
        n_way: Number of classes per episode.
        n_shot: Number of support samples per class.
        n_query: Number of query samples per class.
        augmenter: Optional augmenter applied to support samples.

    Returns:
        Tuple (support_feats, support_labels, query_feats, query_labels)
        where all shapes are (n_way * n_{shot,query}, 49, 40, 1).
    """
    classes = random.sample(list(class_files.keys()), k=n_way)

    support_feats, support_labels = [], []
    query_feats, query_labels = [], []

    for label_idx, cls in enumerate(classes):
        files = class_files[cls]
        chosen = random.sample(files, k=min(n_shot + n_query, len(files)))
        support_paths = chosen[:n_shot]
        query_paths = chosen[n_shot:n_shot + n_query]

        for path in support_paths:
            feat = load_features(path)
            if augmenter is not None:
                # Reconstruct raw audio from .npy is not stored; augment is
                # applied at preprocess time if using raw audio pipeline.
                # For pre-extracted features, skip augmentation.
                pass
            support_feats.append(feat)
            support_labels.append(label_idx)

        for path in query_paths:
            query_feats.append(load_features(path))
            query_labels.append(label_idx)

    return (
        np.array(support_feats, dtype=np.float32),
        np.array(support_labels, dtype=np.int32),
        np.array(query_feats, dtype=np.float32),
        np.array(query_labels, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(config: dict, quick: bool = False) -> keras.Model:
    """Full training loop.

    Args:
        config: Parsed config.yaml as dict.
        quick: If True, run a quick 2-epoch smoke test.

    Returns:
        Trained Keras embedding model.
    """
    import tensorflow.keras as keras

    set_seed(config["training"]["seed"])

    # Build model
    model = build_dscnn_model(
        input_shape=(
            config["mfcc"]["num_frames"],
            config["mfcc"]["num_coeffs"],
            1,
        ),
        embedding_dim=config["model"]["embedding_dim"],
        dropout=config["model"]["dropout"],
    )
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"])

    # Paths
    processed_dir = config["paths"]["processed_dir"]
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("[train] Loading training files...")
    train_files = load_class_files(processed_dir, split="train")
    val_files = load_class_files(processed_dir, split="val")
    print(f"[train] {len(train_files)} classes, training set.")

    n_way = config["training"]["n_way"]
    n_shot = config["training"]["n_shot"]
    n_query = max(1, n_shot // 2)
    n_episodes = 10 if quick else config["training"]["n_episodes"]
    epochs = 2 if quick else config["training"]["epochs"]

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.trainable = True
        train_losses, train_accs = [], []

        pbar = tqdm(range(n_episodes), desc=f"Epoch {epoch}/{epochs} [train]")
        for _ in pbar:
            s_feat, s_lbl, q_feat, q_lbl = sample_episode(
                train_files, n_way=n_way, n_shot=n_shot, n_query=n_query
            )
            with tf.GradientTape() as tape:
                s_emb = model(s_feat, training=True)
                q_emb = model(q_feat, training=True)
                loss, acc = prototypical_loss(s_emb, q_emb, s_lbl, q_lbl, n_way, n_shot)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_losses.append(float(loss))
            train_accs.append(float(acc))
            pbar.set_postfix(loss=f"{np.mean(train_losses):.4f}", acc=f"{np.mean(train_accs):.4f}")

        # --- Validation ---
        val_losses, val_accs = [], []
        for _ in range(min(50, n_episodes // 5)):
            s_feat, s_lbl, q_feat, q_lbl = sample_episode(
                val_files, n_way=n_way, n_shot=n_shot, n_query=n_query
            )
            s_emb = model(s_feat, training=False)
            q_emb = model(q_feat, training=False)
            loss, acc = prototypical_loss(s_emb, q_emb, s_lbl, q_lbl, n_way, n_shot)
            val_losses.append(float(loss))
            val_accs.append(float(acc))

        val_acc = float(np.mean(val_accs))
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={np.mean(train_losses):.4f}  train_acc={np.mean(train_accs):.4f} | "
            f"val_loss={np.mean(val_losses):.4f}  val_acc={val_acc:.4f}"
        )

        # Save checkpoint
        ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}_acc{val_acc:.4f}.weights.h5"
        model.save_weights(str(ckpt_path))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = checkpoint_dir / "best.weights.h5"
            model.save_weights(str(best_path))
            print(f"  [*] New best saved: {best_path}")

    print(f"\n[train] Training complete. Best val_acc={best_val_acc:.4f}")
    return model


# ---------------------------------------------------------------------------
# Save FP32 SavedModel
# ---------------------------------------------------------------------------


def save_fp32_model(model, output_dir: str) -> None:
    """Save the trained model as a TensorFlow SavedModel (FP32)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.export(str(out))
    print(f"[train] FP32 SavedModel saved to {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DS-CNN KWS embedding model.")
    parser.add_argument(
        "--config",
        default="ml/training/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick 2-epoch smoke test",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    model = train(config, quick=args.quick)
    save_fp32_model(model, config["paths"]["checkpoint_dir"].replace("checkpoints", "fp32"))
