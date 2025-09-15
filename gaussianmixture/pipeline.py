# gaussianmixture/pipeline.py

"""
Training + synthesis pipeline for the Gaussian Mixture (GMM) baseline.

What this provides
------------------
• Trains *class-conditional* GMMs (one GMM per label).
• Optional PCA front-end (fit once, then train GMMs in PCA space).
• Robust fitting with retries (handles ill-conditioned covariances cleanly).
• Synthesis per class with evaluator-friendly artifacts.
• Per-class overrides for both sample count and component count.

Artifacts written by `.synthesize(...)`
---------------------------------------
ARTIFACTS/gaussianmixture/
  checkpoints/
    GMM_class_{k}.joblib          # one per class
    GMM_global_fallback.joblib    # optional, if some classes were empty
    PCA.joblib                    # optional, when PCA is enabled
    GMM_LAST_OK                   # small marker file
  synthetic/
    gen_class_{k}.npy             # float32 images in [0, 1], shape (Nk, H, W, C)
    labels_class_{k}.npy          # int labels (k), shape (Nk,)
    x_synth.npy, y_synth.npy      # convenience concatenations

Expected config keys (with safe defaults)
-----------------------------------------
IMG_SHAPE: [H, W, C]              # channels-last; C=1 for grayscale
NUM_CLASSES: 9
GMM_COMPONENTS: 10                # default per-class components
GMM_COMPONENTS_BY_CLASS: {4: 16}  # optional per-class override
COVARIANCE_TYPE: "full"           # {"full","tied","diag","spherical"}
REG_COVAR: 1e-6                   # numeric stabilizer; retries will increase if needed
MAX_ITER: 300
N_INIT: 1
INIT_PARAMS: "kmeans"
RANDOM_STATE: 42
VERBOSE: 0

# PCA (optional)
USE_PCA: false
PCA_DIM: 128                      # if USE_PCA is true and not set, defaults to min(128, D)
PCA_WHITEN: true
PCA_SVDSOLVER: "auto"             # {"auto","full","randomized","arpack"}

# Synthesis
SAMPLES_PER_CLASS: 1000
SAMPLES_PER_CLASS_BY_CLASS: {4: 400, 7: 400}  # optional per-class sampling override

ARTIFACTS:
  checkpoints: artifacts/gaussianmixture/checkpoints
  synthetic:   artifacts/gaussianmixture/synthetic
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import warnings
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from gaussianmixture.models import (
    GMMConfig,
    build_gmm_model,
    flatten_images,
    reshape_to_images,
)

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _labels_to_int(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Accept one-hot (N,K) or int (N,) → int labels (N,)."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return np.argmax(y, axis=1).astype(np.int64)
    if y.ndim == 1:
        return y.astype(np.int64)
    raise ValueError(f"Labels must be (N,) ints or (N,{num_classes}) one-hot; got {y.shape}")

def _is_fitted_gmm(gmm: GaussianMixture) -> bool:
    return hasattr(gmm, "weights_") and gmm.weights_ is not None

# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
@dataclass
class GMMPipelineConfig:
    # Data / shapes
    IMG_SHAPE: Tuple[int, int, int] = (40, 40, 1)
    NUM_CLASSES: int = 9

    # GMM hyperparameters (per class)
    GMM_COMPONENTS: int = 10
    COVARIANCE_TYPE: str = "full"   # safer defaults handled by retries
    TOL: float = 1e-3
    REG_COVAR: float = 1e-6
    MAX_ITER: int = 300
    N_INIT: int = 1
    INIT_PARAMS: str = "kmeans"
    RANDOM_STATE: Optional[int] = 42
    VERBOSE: int = 0

    # PCA front-end (optional)
    USE_PCA: bool = False
    PCA_DIM: Optional[int] = None
    PCA_WHITEN: bool = True
    PCA_SVDSOLVER: str = "auto"     # {"auto","full","randomized","arpack"}

    # Training
    PATIENCE: int = 0               # (present for symmetry with other models)

    # Synthesis
    SAMPLES_PER_CLASS: int = 1000

    # Artifacts
    ARTIFACTS: Dict[str, str] = None  # filled in __init__ of pipeline


class GaussianMixturePipeline:
    """
    Orchestrates training and synthesis for class-conditional GMMs
    with optional PCA and robust fitting.
    """

    # -------------- init / config --------------
    def __init__(self, cfg: Dict):
        self.cfg = GMMPipelineConfig(
            IMG_SHAPE=tuple(cfg.get("IMG_SHAPE", (40, 40, 1))),
            NUM_CLASSES=int(cfg.get("NUM_CLASSES", 9)),
            GMM_COMPONENTS=int(cfg.get("GMM_COMPONENTS", 10)),
            COVARIANCE_TYPE=str(cfg.get("COVARIANCE_TYPE", "full")),
            TOL=float(cfg.get("TOL", 1e-3)),
            REG_COVAR=float(cfg.get("REG_COVAR", 1e-6)),
            MAX_ITER=int(cfg.get("MAX_ITER", 300)),
            N_INIT=int(cfg.get("N_INIT", 1)),
            INIT_PARAMS=str(cfg.get("INIT_PARAMS", "kmeans")),
            RANDOM_STATE=cfg.get("RANDOM_STATE", 42),
            VERBOSE=int(cfg.get("VERBOSE", 0)),
            USE_PCA=bool(cfg.get("USE_PCA", False)),
            PCA_DIM=cfg.get("PCA_DIM", None),
            PCA_WHITEN=bool(cfg.get("PCA_WHITEN", True)),
            PCA_SVDSOLVER=str(cfg.get("PCA_SVDSOLVER", "auto")),
            SAMPLES_PER_CLASS=int(cfg.get("SAMPLES_PER_CLASS", 1000)),
            ARTIFACTS=cfg.get("ARTIFACTS", {}),
        )

        arts = self.cfg.ARTIFACTS or {}
        self.ckpt_dir = Path(arts.get("checkpoints", "artifacts/gaussianmixture/checkpoints"))
        self.synth_dir = Path(arts.get("synthetic", "artifacts/gaussianmixture/synthetic"))
        _ensure_dir(self.ckpt_dir)
        _ensure_dir(self.synth_dir)

        # Optional callbacks / overrides
        self.log_cb = cfg.get("LOG_CB", None)
        self.samples_per_class_by_class: Dict[int, int] = cfg.get("SAMPLES_PER_CLASS_BY_CLASS", {}) or {}
        self.components_by_class: Dict[int, int] = cfg.get("GMM_COMPONENTS_BY_CLASS", {}) or {}

        # In-memory state
        self.models: List[Optional[GaussianMixture]] = [None] * self.cfg.NUM_CLASSES
        self.global_fallback_: Optional[GaussianMixture] = None
        self.pca_: Optional[PCA] = None   # fitted PCA (if any)
        self.trained_in_pca_: bool = False

    # -------------- logging --------------
    def _log(self, stage: str, msg: str) -> None:
        if self.log_cb:
            try:
                self.log_cb(stage, msg)
                return
            except Exception:
                pass
        print(f"[{stage}] {msg}")

    # -------------- PCA utilities --------------
    def _fit_pca(self, X64: np.ndarray) -> Tuple[PCA, np.ndarray]:
        """
        Fit PCA on float64 data and return (pca, Z).
        If PCA_DIM is None, picks min(128, D).
        """
        D = X64.shape[1]
        n_comp = int(self.cfg.PCA_DIM if self.cfg.PCA_DIM is not None else min(128, D))
        self._log("train", f"Fitting PCA: dim={n_comp}, whiten={self.cfg.PCA_WHITEN}, svd={self.cfg.PCA_SVDSOLVER}")
        pca = PCA(
            n_components=n_comp,
            whiten=self.cfg.PCA_WHITEN,
            svd_solver=self.cfg.PCA_SVDSOLVER,
            random_state=self.cfg.RANDOM_STATE,
        )
        Z = pca.fit_transform(X64)
        joblib.dump(pca, self.ckpt_dir / "PCA.joblib")
        return pca, Z

    def _load_pca(self) -> Optional[PCA]:
        p = self.ckpt_dir / "PCA.joblib"
        if p.exists():
            if self.pca_ is None:
                self.pca_ = joblib.load(p)
            return self.pca_
        return None

    # -------------- robust GMM fitting --------------
    def _fit_gmm_with_retries(
        self,
        X64: np.ndarray,
        *,
        base_cfg: GMMConfig,
        class_id: int,
        n_components: int,
    ) -> GaussianMixture:
        """
        Try several safer settings if the initial fit fails. Always uses float64.
        """
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        attempts = [
            # 1) User settings (as-is, but guard reg_covar)
            dict(covariance_type=base_cfg.covariance_type, reg_covar=max(base_cfg.reg_covar, 1e-6), n_components=n_components),
            # 2) More regularization
            dict(covariance_type=base_cfg.covariance_type, reg_covar=max(base_cfg.reg_covar, 1e-4), n_components=n_components),
            # 3) Even more regularization
            dict(covariance_type=base_cfg.covariance_type, reg_covar=max(base_cfg.reg_covar, 5e-4), n_components=n_components),
            # 4) Switch to diag
            dict(covariance_type="diag", reg_covar=max(base_cfg.reg_covar, 1e-4), n_components=n_components),
            # 5) diag + fewer components
            dict(covariance_type="diag", reg_covar=max(base_cfg.reg_covar, 1e-3), n_components=max(1, n_components // 2)),
            # 6) spherical last resort
            dict(covariance_type="spherical", reg_covar=max(base_cfg.reg_covar, 1e-3), n_components=max(1, n_components // 2)),
        ]

        for i, params in enumerate(attempts, 1):
            cfg_try = GMMConfig(
                n_components=int(params["n_components"]),
                covariance_type=params["covariance_type"],
                tol=base_cfg.tol,
                reg_covar=float(params["reg_covar"]),
                max_iter=base_cfg.max_iter,
                n_init=base_cfg.n_init,
                init_params=base_cfg.init_params,
                random_state=base_cfg.random_state,
                verbose=base_cfg.verbose,
            )

            self._log(
                "train",
                f"[class {class_id}] attempt {i}: GMM(n={cfg_try.n_components}, cov='{cfg_try.covariance_type}', reg={cfg_try.reg_covar:g})",
            )

            gmm = build_gmm_model(cfg_try)
            try:
                gmm.fit(X64)
                return gmm
            except Exception as e:
                self._log("warn", f"[class {class_id}] attempt {i} failed: {type(e).__name__}: {e}")

        raise RuntimeError(
            f"GMM training failed for class {class_id} after {len(attempts)} attempts. "
            f"Consider lowering GMM_COMPONENTS, using COVARIANCE_TYPE='diag', or increasing REG_COVAR."
        )

    # -------------- training --------------
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        save_checkpoints: bool = True,
    ) -> List[GaussianMixture]:
        """
        Fit one GMM per class on flattened [0,1] images.
        If USE_PCA is true, PCA is fit on *all* data and GMMs are trained in PCA space.
        """
        H, W, C = self.cfg.IMG_SHAPE
        K = self.cfg.NUM_CLASSES

        y_ids = _labels_to_int(y_train, K)
        X = flatten_images(x_train, img_shape=(H, W, C), assume_01=True, clip=True)  # (N, D)
        X64 = np.asarray(X, dtype=np.float64, order="C")
        N, D = X64.shape

        self._log("train", f"Training class-conditional GMMs on {N} samples, dim={D}, classes={K}")

        # --- PCA (optional) ---
        Z64 = X64
        if self.cfg.USE_PCA:
            pca, Z64 = self._fit_pca(X64)
            self.pca_ = pca
            self.trained_in_pca_ = True
        else:
            # if an old PCA is lying around, remove confusion by not loading it here
            self.trained_in_pca_ = False

        # --- Fit per class ---
        for k in range(K):
            idx = (y_ids == k)
            Zk = Z64[idx]
            nk, Dz = Zk.shape[0], Z64.shape[1]
            if nk == 0:
                self._log("warn", f"Class {k} has no training samples; skipping (will fall back to global).")
                self.models[k] = None
                continue

            # components: per-class override > default; never exceed sample count
            n_comp_cfg = int(self.components_by_class.get(k, self.cfg.GMM_COMPONENTS))
            n_comp = int(min(max(1, n_comp_cfg), nk))

            base_cfg = GMMConfig(
                n_components=n_comp,
                covariance_type=self.cfg.COVARIANCE_TYPE,
                tol=self.cfg.TOL,
                reg_covar=self.cfg.REG_COVAR,
                max_iter=self.cfg.MAX_ITER,
                n_init=self.cfg.N_INIT,
                init_params=self.cfg.INIT_PARAMS,
                random_state=self.cfg.RANDOM_STATE,
                verbose=self.cfg.VERBOSE,
            )

            gmm = self._fit_gmm_with_retries(Zk, base_cfg=base_cfg, class_id=k, n_components=n_comp)
            self.models[k] = gmm

            if save_checkpoints:
                path = self.ckpt_dir / f"GMM_class_{k}.joblib"
                joblib.dump(gmm, path)
                self._log("ckpt", f"Saved {path.name}")

        # --- Global fallback if any class missing ---
        if any(m is None for m in self.models):
            self._log("train", "Training global fallback GMM (for classes with no samples).")
            n_comp_global = int(min(max(1, self.cfg.GMM_COMPONENTS), max(1, N // 10)))
            gmm_global = self._fit_gmm_with_retries(
                Z64,
                base_cfg=GMMConfig(
                    n_components=n_comp_global,
                    covariance_type=self.cfg.COVARIANCE_TYPE,
                    tol=self.cfg.TOL,
                    reg_covar=self.cfg.REG_COVAR,
                    max_iter=self.cfg.MAX_ITER,
                    n_init=self.cfg.N_INIT,
                    init_params=self.cfg.INIT_PARAMS,
                    random_state=self.cfg.RANDOM_STATE,
                    verbose=self.cfg.VERBOSE,
                ),
                class_id=-1,
                n_components=n_comp_global,
            )
            self.global_fallback_ = gmm_global
            if save_checkpoints:
                joblib.dump(gmm_global, self.ckpt_dir / "GMM_global_fallback.joblib")
        else:
            self.global_fallback_ = None

        (self.ckpt_dir / "GMM_LAST_OK").write_text("ok", encoding="utf-8")
        return [m for m in self.models if m is not None]

    # -------------- checkpoint I/O --------------
    def _load_model_for_class(self, k: int) -> Optional[GaussianMixture]:
        if 0 <= k < len(self.models) and self.models[k] is not None and _is_fitted_gmm(self.models[k]):
            return self.models[k]
        path = self.ckpt_dir / f"GMM_class_{k}.joblib"
        if path.exists():
            gmm = joblib.load(path)
            self.models[k] = gmm
            return gmm
        return None

    def _load_global_fallback(self) -> Optional[GaussianMixture]:
        if self.global_fallback_ is not None and _is_fitted_gmm(self.global_fallback_):
            return self.global_fallback_
        path = self.ckpt_dir / "GMM_global_fallback.joblib"
        if path.exists():
            self.global_fallback_ = joblib.load(path)
            return self.global_fallback_
        return None

    # -------------- synthesis --------------
    def synthesize(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a class-balanced synthetic dataset (float32 in [0,1]).
        If a PCA checkpoint exists, sampling/inverse-transform happen accordingly.
        """
        H, W, C = self.cfg.IMG_SHAPE
        K = self.cfg.NUM_CLASSES

        # Decide synthesis space based on presence of PCA checkpoint
        pca = self._load_pca()
        using_pca = pca is not None

        xs, ys = [], []
        _ensure_dir(self.synth_dir)

        for k in range(K):
            gmm = self._load_model_for_class(k)
            if gmm is None:
                gmm = self._load_global_fallback()
                if gmm is None:
                    raise FileNotFoundError(
                        f"No GMM checkpoint for class {k} and no global fallback found in {self.ckpt_dir}"
                    )
                self._log("warn", f"[class {k}] using global fallback GMM for sampling.")

            # Per-class count override -> default
            per_class = int(self.samples_per_class_by_class.get(k, self.cfg.SAMPLES_PER_CLASS))

            # Draw samples in the *training space*
            Z_flat, _ = gmm.sample(per_class)           # PCA space (if trained with PCA) OR original space
            Z_flat = np.asarray(Z_flat, dtype=np.float64)

            # If PCA was used, map back to original pixel space
            if using_pca:
                X_flat = pca.inverse_transform(Z_flat)
            else:
                X_flat = Z_flat

            # Clip to [0,1] and reshape
            X_flat = np.clip(X_flat, 0.0, 1.0)
            imgs = reshape_to_images(X_flat.astype(np.float32, copy=False), (H, W, C), clip=True)

            # Persist per-class dumps
            np.save(self.synth_dir / f"gen_class_{k}.npy", imgs)
            np.save(self.synth_dir / f"labels_class_{k}.npy", np.full((per_class,), k, dtype=np.int32))

            xs.append(imgs)
            y1h = np.zeros((per_class, K), dtype=np.float32)
            y1h[:, k] = 1.0
            ys.append(y1h)

        x_synth = np.concatenate(xs, axis=0).astype(np.float32)
        y_synth = np.concatenate(ys, axis=0).astype(np.float32)

        # Sanity: drop any non-finite rows (rare, but safe)
        mask = np.isfinite(x_synth).all(axis=(1, 2, 3))
        if not mask.all():
            dropped = int((~mask).sum())
            self._log("warn", f"Dropping {dropped} non-finite synthetic samples.")
            x_synth = x_synth[mask]
            y_synth = y_synth[mask]

        # Combined convenience dumps
        np.save(self.synth_dir / "x_synth.npy", x_synth)
        np.save(self.synth_dir / "y_synth.npy", y_synth)

        total = int(x_synth.shape[0])
        per_default = self.cfg.SAMPLES_PER_CLASS
        self._log("synthesize", f"{total} samples (~{per_default} per class; overrides may apply) -> {self.synth_dir}")
        return x_synth, y_synth


# Back-compat alias so `from gaussianmixture.pipeline import GMMPipeline` works.
GMMPipeline = GaussianMixturePipeline

__all__ = ["GaussianMixturePipeline", "GMMPipeline", "GMMPipelineConfig"]
