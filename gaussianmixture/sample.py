# gaussianmixture/sample.py

"""
Sampling helpers for the GaussianMixture baseline.

What this provides
------------------
- load_gmms_from_dir(...): load per-class GMM checkpoints (+ optional global fallback)
- sample_balanced_from_models(...): draw a class-balanced synthetic set
- save_per_class_npy(...): write evaluator-friendly per-class .npy dumps
- save_grid_from_dir(...): quick preview grid (PNG) sampled from checkpoints

Conventions
-----------
- Images are channels-last (H, W, C) with values in [0, 1].
- Checkpoints are saved one per class as: GMM_class_{k}.joblib
  An optional "GMM_global_fallback.joblib" may exist for classes with no model.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import joblib
from sklearn.mixture import GaussianMixture

from gaussianmixture.models import reshape_to_images
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
# --- PNG helpers (top-level once) ---
from PIL import Image

# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_gmms_from_dir(
    ckpt_dir: Path | str,
    num_classes: int,
) -> tuple[list[Optional[GaussianMixture]], Optional[GaussianMixture]]:
    """
    Load per-class GMMs from a checkpoint directory.

    Returns
    -------
    (models, global_fallback)
        models: list length K; each item a GaussianMixture or None if missing
        global_fallback: a GaussianMixture or None if not present
    """
    ckpt_dir = Path(ckpt_dir)
    models: List[Optional[GaussianMixture]] = [None] * int(num_classes)

    for k in range(num_classes):
        path = ckpt_dir / f"GMM_class_{k}.joblib"
        if path.exists():
            models[k] = joblib.load(path)

    fb_path = ckpt_dir / "GMM_global_fallback.joblib"
    fallback = joblib.load(fb_path) if fb_path.exists() else None
    return models, fallback


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------
def _clip_and_sanitize(x: np.ndarray) -> np.ndarray:
    """Clamp to [0,1] and replace non-finite values with 0 (rare, but safe)."""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    mask = np.isfinite(x)
    if not mask.all():
        x = np.where(mask, x, 0.0)
    return x


def sample_balanced_from_models(
    models: list[Optional[GaussianMixture]],
    *,
    img_shape: Tuple[int, int, int],
    samples_per_class: int,
    global_fallback: Optional[GaussianMixture] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw a class-balanced batch from per-class GMMs.

    Args
    ----
    models            : list of length K with class GMMs (some entries may be None)
    img_shape         : (H, W, C)
    samples_per_class : number of samples to draw for each class
    global_fallback   : optional GMM to use if a class model is missing

    Returns
    -------
    x_synth : float32 (N, H, W, C) in [0,1]
    y_onehot: float32 (N, K)        one-hot labels
    """
    H, W, C = img_shape
    K = len(models)
    per_class = int(samples_per_class)

    xs, ys = [], []
    for k in range(K):
        gmm = models[k] if models[k] is not None else global_fallback
        if gmm is None:
            raise FileNotFoundError(
                f"No GMM available for class {k} and no global fallback provided."
            )
        flat, _ = gmm.sample(per_class)  # shape: (per_class, D)
        flat = _clip_and_sanitize(flat)
        imgs = reshape_to_images(flat, (H, W, C), clip=True)  # (per_class, H, W, C)

        y1h = np.zeros((per_class, K), dtype=np.float32)
        y1h[:, k] = 1.0

        xs.append(imgs)
        ys.append(y1h)

    x_synth = np.concatenate(xs, axis=0).astype(np.float32)
    y_onehot = np.concatenate(ys, axis=0).astype(np.float32)
    return x_synth, y_onehot


# ---------------------------------------------------------------------
# Persistence for evaluator
# ---------------------------------------------------------------------
def save_per_class_npy(
    x_synth: np.ndarray,
    y_onehot: np.ndarray,
    synth_dir: Path | str,
) -> None:
    """
    Write per-class dumps and combined convenience files.

    Layout
    ------
    synth_dir/
      gen_class_{k}.npy
      labels_class_{k}.npy
      x_synth.npy
      y_synth.npy
    """
    synth_dir = Path(synth_dir)
    synth_dir.mkdir(parents=True, exist_ok=True)

    labels_int = np.argmax(y_onehot, axis=1).astype(int)
    K = y_onehot.shape[1]

    for k in range(K):
        cls_mask = labels_int == k
        cls_imgs = x_synth[cls_mask]
        np.save(synth_dir / f"gen_class_{k}.npy", cls_imgs)
        np.save(synth_dir / f"labels_class_{k}.npy", np.full((cls_imgs.shape[0],), k, dtype=np.int32))

    np.save(synth_dir / "x_synth.npy", x_synth)
    np.save(synth_dir / "y_synth.npy", y_onehot)


# ---------------------------------------------------------------------
# Preview grid
# ---------------------------------------------------------------------
def save_grid_from_dir(
    ckpt_dir: Path | str,
    *,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    path: Path | str,
    per_class: int = 1,
) -> None:
    """
    Render a compact preview grid sampled from saved checkpoints.

    Grid layout
    -----------
    rows = per_class
    cols = num_classes
    """
    import matplotlib.pyplot as plt

    models, fallback = load_gmms_from_dir(ckpt_dir, num_classes)
    x_s, y = sample_balanced_from_models(
        models,
        img_shape=img_shape,
        samples_per_class=per_class,
        global_fallback=fallback,
    )

    H, W, C = img_shape
    K = int(num_classes)
    rows, cols = per_class, K

    # Arrange samples as [row i = i-th sample per class, left->right = class 0..K-1]
    # x_s is currently stacked by classes; reshape to (K, per_class, H, W, C)
    x_class_major = x_s.reshape(K, per_class, H, W, C)

    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.6 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)  # make 2D

    for j in range(cols):
        for i in range(rows):
            ax = axes[i, j]
            img = x_class_major[j, i]
            if C == 1:
                ax.imshow(img[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(np.clip(img, 0.0, 1.0))
            ax.set_axis_off()
            if i == 0:
                ax.set_title(f"C{j}", fontsize=9)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)

def _save_png(img01: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = img01
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
        mode = "L"
    elif x.ndim == 3 and x.shape[-1] == 3:
        mode = "RGB"
    else:
        x = x.squeeze()
        mode = "L"
    Image.fromarray(_to_uint8(x), mode=mode).save(out_path)

# --- tiny config getter ---
def _cfg_get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

# --- PUBLIC: synth() for the unified CLI ---
def synth(cfg: dict, output_root: str, seed: int = 42) -> dict:
    """
    Generate S PNGs/class into {output_root}/{class}/{seed} and return a manifest.
    Uses the GMM checkpoints trained by gaussianmixture/train.py.
    """
    rng = np.random.default_rng(int(seed))

    # Resolve shape/classes/count from either NEW or LEGACY config keys
    H, W, C = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", [40, 40, 1])))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    S = int(_cfg_get(cfg, "SAMPLES_PER_CLASS", _cfg_get(cfg, "samples_per_class", 25)))

    # Where checkpoints live
    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    ckpt_dir = Path(_cfg_get(cfg, "ARTIFACTS.gaussianmixture_checkpoints",
                             artifacts_root / "gaussianmixture" / "checkpoints"))

    # Load models and draw a balanced batch (your existing helpers)
    models, fallback = load_gmms_from_dir(ckpt_dir, K)
    x_synth, y_onehot = sample_balanced_from_models(
        models,
        img_shape=(H, W, C),
        samples_per_class=S,
        global_fallback=fallback,
    )

    # Save PNGs in the layout the CLI expects and build manifest
    out_root = Path(output_root)
    labels_int = np.argmax(y_onehot, axis=1).astype(int)
    per_class_counts: dict[str, int] = {str(k): 0 for k in range(K)}
    paths: list[dict] = []

    # group indices by class to have deterministic filenames per class
    for k in range(K):
        idxs = np.where(labels_int == k)[0]
        if idxs.size == 0:
            continue
        cls_dir = out_root / str(k) / str(seed)
        cls_dir.mkdir(parents=True, exist_ok=True)
        for j, i in enumerate(idxs):
            out_path = cls_dir / f"gmm_{j:05d}.png"
            _save_png(x_synth[i], out_path)
            paths.append({"path": str(out_path), "label": int(k)})
        per_class_counts[str(k)] = int(idxs.size)

    manifest = {
        "dataset": _cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "USTC-TFC2016_40x40_gray")),
        "seed": int(seed),
        "per_class_counts": per_class_counts,
        "paths": paths,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    return manifest


__all__ = [
    "load_gmms_from_dir",
    "sample_balanced_from_models",
    "save_per_class_npy",
    "save_grid_from_dir",
    "synth",
]
