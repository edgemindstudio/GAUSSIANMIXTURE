# common/data.py

"""
Common dataset utilities for all model families (GMM / Diffusion / VAE / GAN).

What you get
------------
- load_dataset_npy(...)     -> load train/val/test numpy arrays into [0,1] HWC with one-hot labels
- load_synth_dataset(...)   -> load synthetic data saved by pipelines in several supported layouts
- save_synth_per_class(...) -> persist per-class dumps (contract used by evaluators)
- sanitize_images(...)      -> drop non-finite, clamp to [0,1]
- one_hot(...), to_01_hwc(...) helpers

File expectations
-----------------
DATA_DIR/
    train_data.npy, train_labels.npy
    test_data.npy,  test_labels.npy

SYNTH_DIR/
    (any of these layouts; function auto-detects)
      1) x_synth.npy, y_synth.npy
      2) gen_class_{k}.npy + labels_class_{k}.npy  for k in [0..K-1]
      3) class_{k}/sample_*.npy  (images only; labels implied by folder)

Conventions
-----------
- Images are channels-last (H, W, C), float32 in [0, 1].
- Labels returned as one-hot (float32) where requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tensorflow as tf


# =============================================================================
# Core helpers
# =============================================================================
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Return one-hot labels (float32). If already one-hot, pass through."""
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y.astype("float32", copy=False)
    return tf.keras.utils.to_categorical(y.astype(int), num_classes=num_classes).astype("float32")


def to_01_hwc(x: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Ensure float32 images in [0,1] with channels-last shape (N, H, W, C).
    Accepts (N, H*W*C) or (N, H, W, C). If inputs are 0..255, scales to 0..1.
    """
    H, W, C = img_shape
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == H * W * C:
        x = x.reshape((-1, H, W, C))
    elif x.ndim == 3 and C == 1:
        # Accept (N, H, W) for grayscale
        x = x[..., None]
    elif x.ndim != 4 or x.shape[1:] != (H, W, C):
        raise ValueError(f"Input images have shape {x.shape}, expected (N,{H},{W},{C}) or flattened.")

    x = x.astype("float32", copy=False)
    # Heuristic: scale if looks like 0..255
    if x.max(initial=0.0) > 1.5:
        x = x / 255.0
    # Final clamp for safety
    return np.clip(x, 0.0, 1.0)


def sanitize_images(x: np.ndarray) -> np.ndarray:
    """
    Drop samples with any non-finite values and clamp to [0,1].
    Returns possibly smaller array. If all invalid, returns an empty array with same HWC.
    """
    if x.size == 0:
        return x
    mask = np.isfinite(x).all(axis=(1, 2, 3))
    if not mask.any():
        # preserve dtype/shape header
        H, W, C = x.shape[1:]
        return np.empty((0, H, W, C), dtype="float32")
    x = x[mask]
    return np.clip(x.astype("float32", copy=False), 0.0, 1.0)


# =============================================================================
# Dataset loaders
# =============================================================================
def load_dataset_npy(
    data_dir: Path | str,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from four .npy files in `data_dir`:
      train_data.npy, train_labels.npy, test_data.npy, test_labels.npy

    Returns
    -------
    (x_train, y_train, x_val, y_val, x_test, y_test)
      x_* : float32 in [0,1], shape (N, H, W, C)
      y_* : float32 one-hot, shape (N, K)
    """
    data_dir = Path(data_dir)
    H, W, C = img_shape

    x_train = np.load(data_dir / "train_data.npy")
    y_train = np.load(data_dir / "train_labels.npy")
    x_test = np.load(data_dir / "test_data.npy")
    y_test = np.load(data_dir / "test_labels.npy")

    x_train01 = to_01_hwc(x_train, img_shape)
    x_test01 = to_01_hwc(x_test, img_shape)
    y_train1h = one_hot(y_train, num_classes)
    y_test1h = one_hot(y_test, num_classes)

    # Split provided test -> (val, test)
    n_val = int(len(x_test01) * float(val_fraction))
    x_val01, y_val1h = x_test01[:n_val], y_test1h[:n_val]
    x_test01, y_test1h = x_test01[n_val:], y_test1h[n_val:]

    return x_train01, y_train1h, x_val01, y_val1h, x_test01, y_test1h


def load_synth_dataset(
    synth_dir: Path | str,
    num_classes: int,
    img_shape: Tuple[int, int, int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load synthetic dataset from `synth_dir` supporting multiple layouts:

    1) Combined:
         x_synth.npy, y_synth.npy
    2) Per-class:
         gen_class_{k}.npy, labels_class_{k}.npy
    3) Folder-of-npys:
         class_{k}/sample_*.npy  (labels implied by k)

    Returns
    -------
    (x_s, y_s) or (None, None) if nothing found
      x_s : float32 in [0,1], shape (N, H, W, C)
      y_s : float32 one-hot, shape (N, K)
    """
    synth_dir = Path(synth_dir)
    H, W, C = img_shape

    # --- Layout 1: combined dumps ---
    x_all = synth_dir / "x_synth.npy"
    y_all = synth_dir / "y_synth.npy"
    if x_all.exists() and y_all.exists():
        x = to_01_hwc(np.load(x_all), img_shape)
        y = np.asarray(np.load(y_all))
        # Accept one-hot or integer labels
        if y.ndim == 1:
            y = one_hot(y, num_classes)
        elif y.ndim == 2 and y.shape[1] == num_classes:
            y = y.astype("float32", copy=False)
        else:
            raise ValueError(f"Unexpected y_synth.npy shape {y.shape}.")
        x = sanitize_images(x)
        if x.size == 0:
            return None, None
        if len(x) != len(y):
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
        return x, y

    # --- Layout 2: per-class dumps ---
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    found_any = False
    for k in range(num_classes):
        x_path = synth_dir / f"gen_class_{k}.npy"
        y_path = synth_dir / f"labels_class_{k}.npy"
        if x_path.exists() and y_path.exists():
            found_any = True
            xk = to_01_hwc(np.load(x_path), img_shape)
            yk = np.asarray(np.load(y_path)).ravel().astype(int)
            yk = one_hot(yk, num_classes)
            xs.append(xk)
            ys.append(yk)
    if found_any:
        x = sanitize_images(np.concatenate(xs, axis=0))
        if x.size == 0:
            return None, None
        y = np.concatenate(ys, axis=0).astype("float32", copy=False)
        # align lengths if sanitation dropped rows
        n = min(len(x), len(y))
        return x[:n], y[:n]

    # --- Layout 3: folder-of-npys ---
    xs, ys = [], []
    found_any = False
    for k in range(num_classes):
        cls_dir = synth_dir / f"class_{k}"
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.glob("*.npy"))
        if not files:
            continue
        found_any = True
        imgs = [np.load(f) for f in files]
        xk = to_01_hwc(np.stack(imgs, axis=0), img_shape)
        yk = one_hot(np.full(len(xk), k, dtype=int), num_classes)
        xs.append(xk)
        ys.append(yk)
    if found_any:
        x = sanitize_images(np.concatenate(xs, axis=0))
        if x.size == 0:
            return None, None
        y = np.concatenate(ys, axis=0).astype("float32", copy=False)
        n = min(len(x), len(y))
        return x[:n], y[:n]

    # Nothing found
    return None, None


# =============================================================================
# Writers
# =============================================================================
def save_synth_per_class(
    x_s: np.ndarray,
    y_s_onehot: np.ndarray,
    out_dir: Path | str,
    *,
    overwrite: bool = True,
) -> None:
    """
    Persist per-class dumps expected by evaluators and other tools.

    Writes:
      gen_class_{k}.npy  (images)
      labels_class_{k}.npy (integer labels)

    Also writes combined:
      x_synth.npy, y_synth.npy
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_s = to_01_hwc(x_s, (x_s.shape[1], x_s.shape[2], x_s.shape[3]))
    y_s_onehot = y_s_onehot.astype("float32", copy=False)
    labels = np.argmax(y_s_onehot, axis=1).astype(np.int32)

    # Per-class
    K = y_s_onehot.shape[1]
    for k in range(K):
        mask = labels == k
        cls_imgs = x_s[mask]
        cls_labels = labels[mask]
        np.save(out_dir / f"gen_class_{k}.npy", cls_imgs)
        np.save(out_dir / f"labels_class_{k}.npy", cls_labels)

    # Combined
    if overwrite or not (out_dir / "x_synth.npy").exists():
        np.save(out_dir / "x_synth.npy", x_s)
        np.save(out_dir / "y_synth.npy", y_s_onehot)


# =============================================================================
# Lightweight summaries (handy for logs)
# =============================================================================
def dataset_summary(x: np.ndarray, y_onehot: Optional[np.ndarray] = None) -> str:
    """Return a small human-readable summary string for logging."""
    if x is None or x.size == 0:
        return "empty"
    H, W, C = x.shape[1:]
    parts = [f"N={len(x)} shape=({H},{W},{C}) range=[{float(x.min()):.3f},{float(x.max()):.3f}]"]
    if y_onehot is not None and y_onehot.ndim == 2:
        counts = np.sum(y_onehot, axis=0).astype(int)
        parts.append(f"counts={counts.tolist()}")
    return " | ".join(parts)


__all__ = [
    "one_hot",
    "to_01_hwc",
    "sanitize_images",
    "load_dataset_npy",
    "load_synth_dataset",
    "save_synth_per_class",
    "dataset_summary",
]
