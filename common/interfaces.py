# common/interfaces.py

"""
Lightweight interfaces and shared type helpers for all pipelines.

Goals
-----
- Provide stable Protocols (typing-only contracts) that pipelines can implement:
    • LogCallback          -> epoch-wise logger
    • Trainable            -> .train(...)
    • Synthesizer          -> .synthesize(...)
    • GenerativePipeline   -> union of Trainable + Synthesizer + common attrs
- Offer tiny utilities that are safe to import anywhere (no heavy deps).
- Keep runtime dependencies minimal; TensorFlow is optional for type hints.

Why Protocols?
--------------
We don’t want hard inheritance requirements across very different model
families (Diffusion / GMM / GAN / VAE / AR). Using typing.Protocol lets us
express the “shape” of a pipeline while keeping implementations free.

Typical usage
-------------
from common.interfaces import GenerativePipeline, LogCallback

def make_log_cb(dir: Path | None) -> LogCallback: ...
class DiffusionPipeline(GenerativePipeline):   # satisfies the protocol
    ...

The Protocols are “duck-typed”: any class with the right methods/attrs
conforms, no explicit inheritance required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Protocol, runtime_checkable, Callable, Any, Iterable

# Optional TF import strictly for type annotations.
# Pipelines that don't use TF (e.g., pure sklearn GMM) won’t break.
try:  # pragma: no cover - best-effort typing only
    import tensorflow as tf
    TFModel = tf.keras.Model
except Exception:  # pragma: no cover
    TFModel = Any  # fallback for environments without TF


# =============================================================================
# Type aliases
# =============================================================================
Shape3 = Tuple[int, int, int]           # (H, W, C)
ArrayLike = Any                          # avoid strict numpy dependency here
ImageBatch = Any                         # expected (N, H, W, C) float32 in [0,1]


# =============================================================================
# Logging callback (epoch-wise)
# =============================================================================
class LogCallback(Protocol):
    """
    Signature:
        cb(epoch: int, train_loss: float, val_loss: Optional[float]) -> None
    """
    def __call__(self, epoch: int, train_loss: float, val_loss: Optional[float]) -> None: ...


# =============================================================================
# Core pipeline Protocols
# =============================================================================
@runtime_checkable
class Trainable(Protocol):
    """
    Minimal training surface.

    Returns the fitted/updated underlying model object (e.g., tf.keras.Model).
    """
    def train(
        self,
        x_train: ImageBatch,
        y_train: ArrayLike,
        x_val: Optional[ImageBatch] = None,
        y_val: Optional[ArrayLike] = None,
    ) -> TFModel: ...


@runtime_checkable
class Synthesizer(Protocol):
    """
    Minimal synthesis surface.

    Returns:
        x_synth: (N, H, W, C) float32 in [0,1]
        y_synth: (N, K) one-hot float32 (or compatible)
    """
    def synthesize(self, model: Optional[TFModel] = None) -> tuple[ImageBatch, ArrayLike]: ...


@runtime_checkable
class Checkpointing(Protocol):
    """
    Optional helper for pipelines that manage checkpoints.
    Implementers often expose this to allow external tools to load weights.
    """
    def _latest_checkpoint(self) -> Optional[Path]: ...


@runtime_checkable
class GenerativePipeline(Trainable, Synthesizer, Protocol):
    """
    Composite protocol implemented by “generative” pipelines.

    Required attrs (read-only in practice):
        img_shape    : (H, W, C)
        num_classes  : int
        ckpt_dir     : Path (may be a dummy for non-weighted models)
        synth_dir    : Path where .synthesize() writes outputs
    """
    # Attributes
    img_shape: Shape3
    num_classes: int
    ckpt_dir: Path
    synth_dir: Path

    # From Trainable + Synthesizer
    def train(
        self,
        x_train: ImageBatch,
        y_train: ArrayLike,
        x_val: Optional[ImageBatch] = None,
        y_val: Optional[ArrayLike] = None,
    ) -> TFModel: ...
    def synthesize(self, model: Optional[TFModel] = None) -> tuple[ImageBatch, ArrayLike]: ...


# =============================================================================
# Small, reusable utilities
# =============================================================================
def pick_checkpoint(
    ckpt_dir: Path,
    *,
    stem_prefix: str = "DIF",           # e.g., "DIF", "GAN", "VAE"
    ext: str = ".weights.h5",          # Keras 3-style weights; legacy ".h5" also supported
) -> Optional[Path]:
    """
    Choose a checkpoint in priority order:
       best -> last -> newest epoch -> newest legacy
    Returns None if nothing found.

    Examples
    --------
    pick_checkpoint(Path(".../checkpoints"), stem_prefix="DIF")
    -> prefers: DIF_best.weights.h5, DIF_last.weights.h5, DIF_epoch_xxxx.weights.h5
    """
    ckpt_dir = Path(ckpt_dir)
    best = ckpt_dir / f"{stem_prefix}_best{ext}"
    last = ckpt_dir / f"{stem_prefix}_last{ext}"
    if best.exists():
        return best
    if last.exists():
        return last

    epoch_ckpts = sorted(ckpt_dir.glob(f"{stem_prefix}_epoch_*{ext}"))
    if epoch_ckpts:
        # newest by mtime
        return max(epoch_ckpts, key=lambda p: p.stat().st_mtime)

    # Legacy (*.h5 without the .weights suffix)
    legacy = sorted(ckpt_dir.glob(f"{stem_prefix}_epoch_*.h5"))
    return max(legacy, key=lambda p: p.stat().st_mtime) if legacy else None


def ensure_dirs(paths: Iterable[Path | str]) -> None:
    """Create directories if they don’t exist (idempotent)."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    """
    Set common RNG seeds (Python, NumPy, TensorFlow if present).
    Safe to call even if TF is not installed.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover
        import tensorflow as _tf
        _tf.random.set_seed(seed)
    except Exception:
        pass


__all__ = [
    # Protocols
    "LogCallback",
    "Trainable",
    "Synthesizer",
    "Checkpointing",
    "GenerativePipeline",
    # Types
    "Shape3",
    "ArrayLike",
    "ImageBatch",
    "TFModel",
    # Utils
    "pick_checkpoint",
    "ensure_dirs",
    "seed_everything",
]
