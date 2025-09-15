# gaussianmixture/models.py

"""
Model builders and utilities for the Gaussian Mixture (GMM) baseline.

What this module provides
-------------------------
- build_gmm_model(...): returns a configured (unfitted) sklearn GaussianMixture
- flatten_images(...): robustly flattens HWC images -> (N, D) and normalizes to [0,1]
- reshape_to_images(...): reshapes flattened samples back to (N, H, W, C)
- sample_gmm_images(...): convenience helper to sample from a fitted GMM and
  return images in [0,1] with the desired HWC shape

Design notes
------------
- Training and artifact I/O should be handled in the pipeline (see
  `gaussianmixture.pipeline`). This module is deliberately stateless.
- We avoid importing heavy DL libs here. Only NumPy and scikit-learn are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------
# Public config for GaussianMixture (nice for IDEs / defaults)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class GMMConfig:
    """
    Configuration for building a scikit-learn GaussianMixture.

    Parameters mirror sklearn.mixture.GaussianMixture with sensible defaults.
    """
    n_components: int = 10
    covariance_type: str = "full"      # {"full","tied","diag","spherical"}
    tol: float = 1e-3
    reg_covar: float = 1e-6
    max_iter: int = 300
    n_init: int = 1
    init_params: str = "kmeans"        # {"kmeans","random"}
    random_state: int | None = 42
    warm_start: bool = False
    verbose: int = 0


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------
def build_gmm_model(cfg: GMMConfig | None = None, **overrides) -> GaussianMixture:
    """
    Construct an (unfitted) GaussianMixture instance.

    You can pass a `GMMConfig` or override individual kwargs directly:
        build_gmm_model(n_components=20, covariance_type="diag")

    Returns
    -------
    sklearn.mixture.GaussianMixture (UNFITTED)
    """
    cfg = cfg or GMMConfig()
    params = dict(
        n_components=cfg.n_components,
        covariance_type=cfg.covariance_type,
        tol=cfg.tol,
        reg_covar=cfg.reg_covar,
        max_iter=cfg.max_iter,
        n_init=cfg.n_init,
        init_params=cfg.init_params,
        random_state=cfg.random_state,
        warm_start=cfg.warm_start,
        verbose=cfg.verbose,
    )
    params.update(overrides or {})
    return GaussianMixture(**params)


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------
def flatten_images(
    x: np.ndarray,
    img_shape: Tuple[int, int, int] | None = None,
    assume_01: bool = True,
    clip: bool = True,
) -> np.ndarray:
    """
    Robustly convert HWC images to a flat 2D array (N, D).

    - Ensures float32 dtype.
    - If `assume_01` is True, data are expected in [0,1]. If max(x) > 1.5,
      we scale by 255 to map 0..255 -> 0..1 (helpful for legacy arrays).
    - Optionally clips to [0,1].

    Parameters
    ----------
    x : (N, H, W, C) or (N, D)
    img_shape : optional target (H, W, C) to validate/reshape if x is not already 4D
    assume_01 : treat inputs as [0,1] unless values > 1.5 are found
    clip : clamp to [0,1] after normalization

    Returns
    -------
    flat : np.ndarray shape (N, D), float32 in [0,1]
    """
    x = np.asarray(x)
    if x.ndim == 2:
        flat = x.astype(np.float32, copy=False)
        if clip:
            flat = np.clip(flat, 0.0, 1.0)
        return flat

    if x.ndim != 4:
        raise ValueError(f"Expected 4D HWC images or 2D already-flattened; got shape {x.shape}")

    if img_shape is not None and tuple(x.shape[1:]) != tuple(img_shape):
        try:
            x = x.reshape((-1, *img_shape))
        except Exception as e:
            raise ValueError(
                f"Could not reshape input to (-1, {img_shape}); original shape: {x.shape}"
            ) from e

    x = x.astype(np.float32, copy=False)
    if assume_01 and float(np.nanmax(x)) > 1.5:
        x = x / 255.0
    if clip:
        x = np.clip(x, 0.0, 1.0)

    n = x.shape[0]
    flat = x.reshape(n, -1)
    return flat


def reshape_to_images(
    flat: np.ndarray,
    img_shape: Tuple[int, int, int],
    clip: bool = True,
) -> np.ndarray:
    """
    Reshape flattened samples back to HWC images.

    Parameters
    ----------
    flat : (N, D) array
    img_shape : (H, W, C) target shape
    clip : clamp to [0,1]

    Returns
    -------
    imgs : (N, H, W, C) float32
    """
    flat = np.asarray(flat, dtype=np.float32)
    H, W, C = img_shape
    expected = H * W * C
    if flat.ndim != 2 or flat.shape[1] != expected:
        raise ValueError(f"Flat array must have shape (N, {expected}); got {flat.shape}")
    imgs = flat.reshape(-1, H, W, C)
    if clip:
        imgs = np.clip(imgs, 0.0, 1.0)
    return imgs


# ---------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------
def sample_gmm_images(
    gmm: GaussianMixture,
    n: int,
    img_shape: Tuple[int, int, int],
    clip: bool = True,
) -> np.ndarray:
    """
    Draw `n` samples from a FITTED GMM and return images in [0,1].

    Notes
    -----
    - Assumes the GMM was trained on flattened images normalized to [0,1].
    - Clipping is applied as a safety guard to keep results in-range.

    Returns
    -------
    imgs : (n, H, W, C) float32
    """
    if not hasattr(gmm, "weights_"):
        raise ValueError("The provided GMM appears to be unfitted. Call `gmm.fit(...)` first.")

    flat, _ = gmm.sample(n)  # shape (n, D)
    # Some covariance settings can slightly overshoot bounds; clamp back to [0,1].
    if clip:
        flat = np.clip(flat, 0.0, 1.0).astype(np.float32, copy=False)
    return reshape_to_images(flat.astype(np.float32, copy=False), img_shape, clip=clip)


__all__ = [
    "GMMConfig",
    "build_gmm_model",
    "flatten_images",
    "reshape_to_images",
    "sample_gmm_images",
]
