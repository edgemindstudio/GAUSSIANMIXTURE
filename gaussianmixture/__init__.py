# gaussianmixture/__init__.py

"""
Gaussian Mixture (GMM) baseline package.

Exports
-------
- GMMPipeline / GaussianMixturePipeline : the pipeline class
- make_pipeline(cfg)                    : convenience constructor
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Type

__version__ = "0.1.0"
__all__ = ["GMMPipeline", "GaussianMixturePipeline", "make_pipeline", "__version__"]

# Capture the first import error (if any) so we can surface a helpful message later.
_IMPORT_ERR: Exception | None = None
_PIPELINE_CLS: Type | None = None


def _load_pipeline_class() -> Type:
    """
    Lazily import the pipeline class from gaussianmixture.pipeline.

    Tries `GMMPipeline` first (our current name), then `GaussianMixturePipeline`
    for compatibility with alternative codebases.
    """
    global _IMPORT_ERR
    mod = import_module(".pipeline", __name__)
    for name in ("GMMPipeline", "GaussianMixturePipeline"):
        if hasattr(mod, name):
            return getattr(mod, name)
    # If we got here, the module loaded but the class wasn't found.
    raise ImportError(
        "Expected 'GMMPipeline' (or 'GaussianMixturePipeline') in "
        "'gaussianmixture/pipeline.py' but neither was found."
    )


def make_pipeline(cfg: Dict[str, Any]):
    """Return a pipeline instance from a config dict."""
    cls = _get_pipeline_cls()
    return cls(cfg)


def _get_pipeline_cls() -> Type:
    global _PIPELINE_CLS, _IMPORT_ERR
    if _PIPELINE_CLS is not None:
        return _PIPELINE_CLS
    try:
        _PIPELINE_CLS = _load_pipeline_class()
        return _PIPELINE_CLS
    except Exception as e:  # store and re-raise on attribute access
        _IMPORT_ERR = e
        raise


# Try eager bind (optional). If it fails, we’ll resolve on first attribute access.
try:
    _PIPELINE_CLS = _load_pipeline_class()
    GMMPipeline = _PIPELINE_CLS                   # type: ignore
    GaussianMixturePipeline = _PIPELINE_CLS       # type: ignore
except Exception as e:
    _IMPORT_ERR = e
    GMMPipeline = None                            # type: ignore
    GaussianMixturePipeline = None                # type: ignore


def __getattr__(name: str):
    """
    Late-bind attributes so imports like
        `from gaussianmixture import GMMPipeline`
    still work even if the first eager import failed.
    """
    global GMMPipeline, GaussianMixturePipeline
    if name in ("GMMPipeline", "GaussianMixturePipeline"):
        if _PIPELINE_CLS is None:
            # Try one more time to load; if it fails, raise a helpful error.
            try:
                cls = _get_pipeline_cls()
            except Exception as e:
                raise ImportError(
                    "Failed to import the GMM pipeline. Ensure "
                    "'gaussianmixture/pipeline.py' exists and dependencies "
                    "(e.g., scikit-learn, joblib) are installed."
                ) from e
            GMMPipeline = cls                # bind for future accesses
            GaussianMixturePipeline = cls
            return cls
        return _PIPELINE_CLS
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
