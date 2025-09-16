# gaussianmixture/train.py

"""
Trainer for the GaussianMixture baseline (per-class GMMs).

What this script does
---------------------
- Loads config.yaml (or a path you pass via --config)
- Loads dataset (images in [0,1], channels-last (H, W, C); labels int or one-hot)
- Trains one GMM per class, with a cap on components if a class is small
- Optionally trains a global fallback GMM (used if a class model is missing)
- Saves checkpoints as:
    ARTIFACTS/gaussianmixture/checkpoints/
      GMM_class_{k}.joblib
      GMM_global_fallback.joblib  (optional)
- Writes a small preview grid PNG sampled from the saved checkpoints.

Conventions
-----------
- Uses scikit-learn GaussianMixture (EM). No GPUs needed.
- Images are flattened for GMM fitting; sampling reshapes back to (H,W,C).
- All synthetic samples are clipped to [0,1] downstream (in sampler).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import joblib
from sklearn.mixture import GaussianMixture

# Local modules
try:
    from common.data import load_dataset_npy  # shared loader (preferred)
except Exception:
    load_dataset_npy = None  # fallback path below

from gaussianmixture.models import (
    reshape_to_images,
    flatten_images,
    create_gmm,  # convenience factory; falls back to sklearn if needed
)
from gaussianmixture.sample import save_grid_from_dir


# =============================================================================
# Small utilities
# =============================================================================
def _to_int_labels(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Accept (N,) ints or (N,K) one-hot -> return (N,) ints."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return np.argmax(y, axis=1).astype(int)
    return y.astype(int)


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _default_artifact_paths(cfg: Dict) -> Tuple[Path, Path, Path]:
    """
    Choose artifact roots for checkpoints/synthetic/summaries, with safe defaults
    under 'artifacts/gaussianmixture/...'. You can override in config.yaml:

    ARTIFACTS:
      gaussianmixture_checkpoints: artifacts/gaussianmixture/checkpoints
      gaussianmixture_synthetic:   artifacts/gaussianmixture/synthetic
      gaussianmixture_summaries:   artifacts/gaussianmixture/summaries
    """
    arts = cfg.get("ARTIFACTS", {})
    ckpt = Path(arts.get("gaussianmixture_checkpoints", "artifacts/gaussianmixture/checkpoints"))
    synth = Path(arts.get("gaussianmixture_synthetic", "artifacts/gaussianmixture/synthetic"))
    sums = Path(arts.get("gaussianmixture_summaries", "artifacts/gaussianmixture/summaries"))
    return ckpt, synth, sums


# =============================================================================
# Core training
# =============================================================================
def train_per_class_gmms(
    *,
    x_train: np.ndarray,          # (N,H,W,C) in [0,1]
    y_train: np.ndarray,          # (N,) ints or (N,K) one-hot
    img_shape: Tuple[int, int, int],
    num_classes: int,
    ckpt_dir: Path,
    n_components: int = 10,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    max_iter: int = 200,
    random_state: int = 42,
    train_global_fallback: bool = True,
    verbose: bool = True,
) -> None:
    """
    Fit one GaussianMixture per class and save to ckpt_dir.

    Notes
    -----
    - Caps components per class to <= #samples in that class.
    - If a class has < 2 samples, we skip and rely on the global fallback GMM.
    """
    H, W, C = img_shape
    D = H * W * C

    _ensure_dirs(ckpt_dir)

    y_int = _to_int_labels(y_train, num_classes)
    X_flat = flatten_images(x_train) if x_train.ndim == 4 else x_train.reshape((-1, D))

    # Optional: global fallback GMM trained on all data
    fallback = None
    if train_global_fallback:
        if verbose:
            print(f"[GMM] Training global fallback on {X_flat.shape[0]} samples, D={D}...")
        fallback = create_gmm(
            n_components=min(n_components, max(1, X_flat.shape[0])),
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state,
            max_iter=max_iter,
            verbose=1 if verbose else 0,
        )
        fallback.fit(X_flat)
        joblib.dump(fallback, ckpt_dir / "GMM_global_fallback.joblib")

    # Per-class models
    for k in range(num_classes):
        idx = (y_int == k)
        count = int(idx.sum())
        if verbose:
            print(f"[GMM] Class {k}: {count} samples")

        if count < 2:
            print(f"[warn] Class {k} too small for GMM (n={count}); will use fallback at sampling.")
            continue

        Xk = X_flat[idx]
        k_components = min(n_components, count)  # cap components to available samples

        gmm = create_gmm(
            n_components=k_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state,
            max_iter=max_iter,
            verbose=1 if verbose else 0,
        )
        gmm.fit(Xk)
        joblib.dump(gmm, ckpt_dir / f"GMM_class_{k}.joblib")

    if verbose:
        print(f"[GMM] Saved checkpoints to {ckpt_dir}")


# =============================================================================
# Orchestration
# =============================================================================
def run_train(cfg: Dict) -> None:
    """
    High-level runner:
      - sets defaults
      - loads dataset (shared loader if available)
      - trains per-class GMMs (+ optional global fallback)
      - writes a preview grid PNG
    """
    # -------- Defaults --------
    seed = int(cfg.get("SEED", 42))
    img_shape = tuple(cfg.get("IMG_SHAPE", (40, 40, 1)))
    num_classes = int(cfg.get("NUM_CLASSES", 10))
    data_dir = Path(cfg.get("DATA_DIR", "data"))

    # GMM knobs
    n_components = int(cfg.get("GMM_COMPONENTS", 10))
    covariance_type = cfg.get("GMM_COVARIANCE", "full")
    reg_covar = float(cfg.get("GMM_REG_COVAR", 1e-6))
    max_iter = int(cfg.get("GMM_MAX_ITER", 200))
    train_global_fallback = bool(cfg.get("GMM_TRAIN_GLOBAL", True))

    ckpt_dir, synth_dir, sums_dir = _default_artifact_paths(cfg)
    _ensure_dirs(ckpt_dir, synth_dir, sums_dir)

    # -------- Reproducibility --------
    np.random.seed(seed)

    # -------- Load dataset --------
    if load_dataset_npy is not None:
        # Uses the same loader your Diffusion pipeline relies on
        x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_npy(
            data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
        )
    else:
        # Minimal fallback loader (expects 4 files under DATA_DIR)
        x_train = np.load(data_dir / "train_data.npy").astype("float32")
        y_train = np.load(data_dir / "train_labels.npy")
        x_test = np.load(data_dir / "test_data.npy").astype("float32")
        y_test = np.load(data_dir / "test_labels.npy")

        # Normalize & reshape
        if x_train.max() > 1.5:
            x_train = x_train / 255.0
            x_test = x_test / 255.0
        H, W, C = img_shape
        x_train = x_train.reshape((-1, H, W, C))
        x_test = x_test.reshape((-1, H, W, C))

        # Split test -> (val, test)
        n_val = int(len(x_test) * float(cfg.get("VAL_FRACTION", 0.5)))
        x_val, y_val = x_test[:n_val], y_test[:n_val]
        x_test, y_test = x_test[n_val:], y_test[n_val:]

    # -------- Train per-class GMMs --------
    print(f"[config] GMM components={n_components}, cov='{covariance_type}', reg_covar={reg_covar}, max_iter={max_iter}")
    print(f"[paths]  ckpts={ckpt_dir.resolve()} | synthetic={synth_dir.resolve()} | summaries={sums_dir.resolve()}")

    train_per_class_gmms(
        x_train=x_train,
        y_train=y_train,
        img_shape=img_shape,
        num_classes=num_classes,
        ckpt_dir=ckpt_dir,
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        max_iter=max_iter,
        random_state=seed,
        train_global_fallback=train_global_fallback,
        verbose=True,
    )

    # -------- Preview grid (1 sample per class) --------
    preview_path = sums_dir / "gmm_train_preview.png"
    try:
        save_grid_from_dir(
            ckpt_dir=ckpt_dir,
            img_shape=img_shape,
            num_classes=num_classes,
            path=preview_path,
            per_class=1,
        )
        print(f"[preview] Saved GMM preview grid to {preview_path}")
    except Exception as e:
        print(f"[warn] Could not save preview grid: {e}")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-class GaussianMixture models")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = _load_yaml(Path(args.config))

    # Sensible defaults if not present
    cfg.setdefault("SEED", 42)
    cfg.setdefault("VAL_FRACTION", 0.5)
    cfg.setdefault("GMM_COMPONENTS", 10)
    cfg.setdefault("GMM_COVARIANCE", "full")
    cfg.setdefault("GMM_REG_COVAR", 1e-6)
    cfg.setdefault("GMM_MAX_ITER", 200)
    cfg.setdefault("GMM_TRAIN_GLOBAL", True)
    cfg.setdefault("ARTIFACTS", {})

    run_train(cfg)

# ---- Adapter for unified CLI (keep your file's current content) ----
def train(cfg: dict):
    """
    Adapter for the GenCyberSynth unified CLI.
    The CLI passes a parsed YAML dict; delegate to the repo's runner.
    """
    return run_train(cfg)

if __name__ == "__main__":
    main()
