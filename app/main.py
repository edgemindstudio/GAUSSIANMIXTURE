# app/main.py
# =============================================================================
# Gaussian Mixture (per-class) synthesizer — production pipeline entry point
#
# Commands
# --------
#   python -m app.main train         # fit per-class GMMs, save checkpoints
#   python -m app.main synth         # load checkpoints and synthesize per-class samples
#   python -m app.main eval          # standardized evaluation with/without synthetic
#   python -m app.main all           # train -> synth -> eval
#
# Phase-2 evaluation
# ------------------
# Uses gcs_core.val_common.compute_all_metrics(...) and *tries* to use
# write_summary_with_gcs_core(...). If that writer is incompatible with your
# gcs_core version, we fall back to a local writer that emits:
#   - runs/console.txt
#   - runs/summary.jsonl (append-only)
#   - artifacts/gaussianmixture/summaries/GaussianMixture_eval_summary_seed{SEED}.json
#
# Conventions
# -----------
# - Images are float32, NHWC in [0,1]; labels are one-hot (N, K) float32.
# - File structure mirrors the other repos (Diffusion, DCGAN, VAE, AR, RBM).
# =============================================================================

from __future__ import annotations

# --- Make top-level packages importable (gaussianmixture/, eval/, common/) ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------------

import argparse
import json
from typing import Dict, Tuple, Optional

import importlib
import numpy as np
import tensorflow as tf
import yaml

# ------------------------------------------------------------------------------
# Frozen core helpers (shared & versioned)
# ------------------------------------------------------------------------------
from gcs_core.synth_loader import resolve_synth_dir, load_synth_any
from gcs_core.val_common import compute_all_metrics, write_summary_with_gcs_core


# =============================================================================
# GPU niceties (safe on CPU-only machines)
# =============================================================================
def _enable_gpu_mem_growth() -> None:
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


# =============================================================================
# Pipeline loader (modern/legacy) — small, local and drift-proof
# =============================================================================
def _make_pipeline(cfg):
    """
    Return a Gaussian Mixture pipeline instance, trying modern names first and
    falling back to the legacy module if needed.
    """
    # Modern package: prefer factory, then class
    try:
        gm_mod = importlib.import_module("gaussianmixture")
        if hasattr(gm_mod, "make_pipeline"):
            return gm_mod.make_pipeline(cfg)
        if hasattr(gm_mod, "GaussianMixturePipeline"):
            return gm_mod.GaussianMixturePipeline(cfg)
    except Exception:
        pass  # fall through to legacy

    # Legacy module
    try:
        legacy_mod = importlib.import_module("cGaussianMixture")
        return getattr(legacy_mod, "GMMPipeline")(cfg)
    except Exception as e:
        raise ImportError(
            "Could not import a GaussianMixture pipeline "
            "(tried 'gaussianmixture' and 'cGaussianMixture')."
        ) from e


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int = 42) -> None:
    """Deterministic NumPy + TF RNG."""
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def load_yaml(path: Path) -> Dict:
    """Parse YAML at `path`."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: Dict) -> None:
    """Create artifact directories present in cfg (idempotent)."""
    arts = cfg.get("ARTIFACTS", {})
    for key in ("checkpoints", "synthetic", "summaries", "tensorboard"):
        p = arts.get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Return one-hot (N, K) float32; pass-through when already one-hot."""
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y.astype("float32")
    return tf.keras.utils.to_categorical(y.astype(int), num_classes=num_classes).astype("float32")


def load_dataset_npy(
    data_dir: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expected files in DATA_DIR:
      train_data.npy, train_labels.npy, test_data.npy, test_labels.npy

    Returns images in [0,1] float32 shaped (N,H,W,C) and labels as one-hot.
    Splits provided test -> (val, test) using val_fraction.
    """
    H, W, C = img_shape
    x_train = np.load(data_dir / "train_data.npy")
    y_train = np.load(data_dir / "train_labels.npy")
    x_test  = np.load(data_dir / "test_data.npy")
    y_test  = np.load(data_dir / "test_labels.npy")

    def to_01_hwc(x: np.ndarray) -> np.ndarray:
        x = x.astype("float32")
        if x.max() > 1.5:  # handle 0..255 inputs
            x = x / 255.0
        x = x.reshape((-1, H, W, C))
        return np.clip(x, 0.0, 1.0)

    x_train01 = to_01_hwc(x_train)
    x_test01  = to_01_hwc(x_test)

    y_train1h = one_hot(y_train, num_classes)
    y_test1h  = one_hot(y_test,  num_classes)

    n_val = int(len(x_test01) * val_fraction)
    x_val01, y_val1h = x_test01[:n_val], y_test1h[:n_val]
    x_test01, y_test1h = x_test01[n_val:], y_test1h[n_val:]

    return x_train01, y_train1h, x_val01, y_val1h, x_test01, y_test1h


def _save_preview_grid_png(arr: np.ndarray, path: Path) -> None:
    """Save a horizontal grid (one image per class) as PNG."""
    import matplotlib.pyplot as plt  # lazy import
    x = arr.copy()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    k = x.shape[0]
    fig, axes = plt.subplots(1, k, figsize=(1.4 * k, 1.6))
    if k == 1:
        axes = [axes]
    for i in range(k):
        axes[i].imshow(x[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[i].set_axis_off()
        axes[i].set_title(f"C{i}", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _sanitize_synth(
    x_s: Optional[np.ndarray], y_s: Optional[np.ndarray], img_shape: Tuple[int, int, int]
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Ensure NHWC shape and float32 dtype; leave (None, None) untouched.
    val_common handles intensity normalization internally.
    """
    if x_s is None or y_s is None:
        return None, None
    H, W, C = img_shape
    x_s = np.asarray(x_s, dtype=np.float32)
    y_s = np.asarray(y_s)
    if x_s.ndim == 3:
        x_s = x_s.reshape((-1, H, W, C))
    if x_s.size == 0 or y_s.size == 0:
        return None, None
    return x_s, y_s


# =============================================================================
# Local writer shim (for gcs_core incompatibilities)
# =============================================================================
def _map_util_names(util_block: Optional[Dict]) -> Dict:
    """Normalize utility metric names to a stable set."""
    if not util_block:
        return {}
    bal = util_block.get("balanced_accuracy", util_block.get("bal_acc"))
    return {
        "accuracy":               util_block.get("accuracy"),
        "macro_f1":               util_block.get("macro_f1"),
        "balanced_accuracy":      bal,
        "macro_auprc":            util_block.get("macro_auprc"),
        "recall_at_1pct_fpr":     util_block.get("recall_at_1pct_fpr"),
        "ece":                    util_block.get("ece"),
        "brier":                  util_block.get("brier"),
        "per_class":              util_block.get("per_class"),
    }


def _build_phase2_record(model_name: str, seed: int, images_counts: Dict, metrics: Dict) -> Dict:
    """Create a Phase-2 schema dict from metrics."""
    util_R  = _map_util_names(metrics.get("real_only"))
    util_RS = _map_util_names(metrics.get("real_plus_synth"))

    def _delta(k: str) -> Optional[float]:
        a, b = util_RS.get(k), util_R.get(k)
        return None if (a is None or b is None) else float(a - b)

    deltas = {
        "accuracy":           _delta("accuracy"),
        "macro_f1":           _delta("macro_f1"),
        "balanced_accuracy":  _delta("balanced_accuracy"),
        "macro_auprc":        _delta("macro_auprc"),
        "recall_at_1pct_fpr": _delta("recall_at_1pct_fpr"),
        "ece":                _delta("ece"),
        "brier":              _delta("brier"),
    }

    generative = {
        "fid":          metrics.get("fid_macro"),
        "fid_macro":    metrics.get("fid_macro"),
        "cfid_macro":   metrics.get("cfid_macro"),
        "js":           metrics.get("js"),
        "kl":           metrics.get("kl"),
        "diversity":    metrics.get("diversity"),
        "cfid_per_class": metrics.get("cfid_per_class"),
        "fid_domain":     metrics.get("fid_domain"),
    }

    return {
        "model": model_name,
        "seed":  seed,
        "images": images_counts,
        "generative": generative,
        "utility_real_only": util_R,
        "utility_real_plus_synth": util_RS,
        "deltas_RS_minus_R": deltas,
    }


from typing import Dict

def _write_console_block(record: Dict) -> str:
    """Format a concise console block and return it."""
    gen = record.get("generative", {}) or {}
    util_R  = record.get("utility_real_only", {}) or {}
    util_RS = record.get("utility_real_plus_synth", {}) or {}
    imgs = record.get("images", {}) or {}

    lines = [
        f"Model: {record.get('model')}   Seed: {record.get('seed')}",
        f"Counts → train:{imgs.get('train_real')}  "
        f"val:{imgs.get('val_real')}  "
        f"test:{imgs.get('test_real')}  "
        f"synth:{imgs.get('synthetic')}",
        f"Generative → FID(macro): {gen.get('fid_macro')}  "
        f"cFID(macro): {gen.get('cfid_macro')}  "
        f"JS: {gen.get('js')}  KL: {gen.get('kl')}  "
        f"Div: {gen.get('diversity')}",
        f"Utility R → acc: {util_R.get('accuracy')}  "
        f"bal_acc: {util_R.get('balanced_accuracy')}  "
        f"macro_f1: {util_R.get('macro_f1')}",
        f"Utility R+S → acc: {util_RS.get('accuracy')}  "
        f"bal_acc: {util_RS.get('balanced_accuracy')}  "
        f"macro_f1: {util_RS.get('macro_f1')}",
    ]
    return "\n".join(lines) + "\n"



def _safe_write_summary(
    model_name: str,
    seed: int,
    data_dir: Path,
    synth_dir_str: str,
    fid_cap: int,
    metrics: Dict,
    images_counts: Dict,
) -> Dict:
    """
    Try modern gcs_core writer first (with images_counts). If any TypeError occurs
    (e.g., older gcs_core where its internal evaluate_model_suite() signature
    doesn't accept 'real_dirs'), fall back to a local writer that emits the same
    set of files.
    """
    try:
        # Preferred path (newer gcs_core)
        return write_summary_with_gcs_core(
            model_name=model_name,
            seed=seed,
            real_dirs={
                "train": str(data_dir / "train_data.npy"),
                "val":   f"{data_dir}/(split of test_data.npy)",
                "test":  f"{data_dir}/(split of test_data.npy)",
            },
            synth_dir=synth_dir_str,
            fid_cap_per_class=fid_cap,
            output_json="runs/summary.jsonl",
            output_console="runs/console.txt",
            metrics=metrics,
            notes="phase2-real",
            images_counts=images_counts,
        )
    except TypeError as e:
        # Local fallback: build Phase-2 record + write console/jsonl ourselves
        record = _build_phase2_record(model_name, seed, images_counts, metrics)

        # Ensure runs dir exists
        Path("runs").mkdir(parents=True, exist_ok=True)

        # Console
        console = _write_console_block(record)
        (Path("runs") / "console.txt").write_text(console)

        # JSONL append
        with (Path("runs") / "summary.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

        return record


# =============================================================================
# Orchestration
# =============================================================================
def run_train(cfg: Dict) -> None:
    """Fit per-class GMMs and save a small 1×K preview grid."""
    set_seed(cfg.get("SEED", 42))
    ensure_dirs(cfg)

    data_dir    = Path(cfg["DATA_DIR"])
    img_shape   = tuple(cfg["IMG_SHAPE"])
    num_classes = int(cfg["NUM_CLASSES"])

    x_train01, y_train, x_val01, y_val, _, _ = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
    )

    pipe = _make_pipeline(cfg)
    # Some pipelines expose .fit, others .train — support both.
    if hasattr(pipe, "fit"):
        pipe.fit(x_train=x_train01, y_train=y_train)
    else:
        pipe.train(x_train=x_train01, y_train=y_train)

    # Optional preview: one sample per class
    preview_path = Path(cfg["ARTIFACTS"]["summaries"]) / "train_preview.png"
    try:
        old = getattr(pipe, "samples_per_class", 1)
        setattr(pipe, "samples_per_class", 1)
        grid, _ = pipe.synthesize()  # expected to return (images, labels)
        setattr(pipe, "samples_per_class", old)
        if isinstance(grid, np.ndarray) and grid.size:
            _save_preview_grid_png(grid[:num_classes], preview_path)
            print(f"Saved preview grid to {preview_path}")
    except Exception:
        # Non-fatal: training completed even if preview failed.
        pass


def run_synth(cfg: Dict) -> None:
    """
    Synthesize per-class datasets and also persist a monolithic pair
    (x_synth.npy, y_synth.npy) so downstream tools can find a single file set.
    """
    set_seed(cfg.get("SEED", 42))
    ensure_dirs(cfg)

    synth_dir = Path(cfg["ARTIFACTS"]["synthetic"])
    pipe = _make_pipeline(cfg)
    x_s, y_s = pipe.synthesize()  # pipeline handles checkpoint selection & sampling

    if isinstance(x_s, np.ndarray) and isinstance(y_s, np.ndarray) and x_s.size and y_s.size:
        synth_dir.mkdir(parents=True, exist_ok=True)
        np.save(synth_dir / "x_synth.npy", x_s.astype(np.float32))
        np.save(synth_dir / "y_synth.npy", y_s)
        print(f"Synthesized: {x_s.shape[0]} samples (saved under {synth_dir}).")
    else:
        print(f"[warn] Pipeline did not return arrays; relying on per-class files under {synth_dir}.")


def run_eval(cfg: Dict, include_synth: bool) -> None:
    """
    Phase-2 standardized evaluation:
      • Generative quality (FID/cFID/JS/KL/Diversity) on VAL vs SYNTH.
      • Downstream utility on REAL test with the fixed small CNN.
      • Writes:
          - runs/console.txt
          - runs/summary.jsonl
          - ARTIFACTS/summaries/GaussianMixture_eval_summary_seed{SEED}.json
    """
    set_seed(cfg.get("SEED", 42))
    ensure_dirs(cfg)
    Path("runs").mkdir(exist_ok=True)

    data_dir    = Path(cfg["DATA_DIR"])
    img_shape   = tuple(cfg["IMG_SHAPE"])
    num_classes = int(cfg["NUM_CLASSES"])
    fid_cap     = int(cfg.get("FID_CAP", 200))
    seed        = int(cfg.get("SEED", 42))
    eval_epochs = int(cfg.get("EVAL_EPOCHS", 20))

    # --- Load REAL data (standardized to [0,1]) ---
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
    )

    # --- Optional SYNTH via gcs_core ---
    x_s, y_s = (None, None)
    synth_dir_str = ""
    if include_synth:
        repo_root = Path(__file__).resolve().parents[1]
        synth_dir = resolve_synth_dir(cfg, repo_root)
        synth_dir_str = str(synth_dir)
        try:
            x_s, y_s = load_synth_any(synth_dir, num_classes)
            x_s, y_s = _sanitize_synth(x_s, y_s, img_shape)
            if x_s is not None:
                print(f"[eval] Using synthetic from {synth_dir} (N={len(x_s)})")
            else:
                print(f"[eval] WARN: no usable synthetic under {synth_dir}; proceeding REAL-only.")
        except Exception as e:
            print(f"[eval] WARN: could not load synthetic -> {e}. Proceeding REAL-only.")
            x_s, y_s = None, None
    else:
        synth_dir_str = str(Path(cfg.get("ARTIFACTS", {}).get("synthetic", "")))

    # --- Compute metrics (FID/cFID handled gracefully inside val_common) ---
    metrics = compute_all_metrics(
        img_shape=img_shape,
        x_train_real=x_tr, y_train_real=y_tr,
        x_val_real=x_va,   y_val_real=y_va,
        x_test_real=x_te,  y_test_real=y_te,
        x_synth=x_s,       y_synth=y_s,
        fid_cap_per_class=fid_cap,
        seed=seed,
        domain_embed_fn=None,
        epochs=eval_epochs,
    )

    # --- Emit standardized console + JSONL via gcs_core or local shim ---
    images_counts = {
        "train_real": int(x_tr.shape[0]),
        "val_real":   int(x_va.shape[0]),
        "test_real":  int(x_te.shape[0]),
        "synthetic":  (int(x_s.shape[0]) if isinstance(x_s, np.ndarray) else None),
    }

    record = _safe_write_summary(
        model_name="GaussianMixture",
        seed=seed,
        data_dir=data_dir,
        synth_dir_str=synth_dir_str or str(Path(cfg["ARTIFACTS"]["synthetic"])),
        fid_cap=fid_cap,
        metrics=metrics,
        images_counts=images_counts,
    )

    # --- Pretty JSON copy under ARTIFACTS/summaries (authoritative for aggregator) ---
    out = Path(cfg["ARTIFACTS"]["summaries"]) / f"GaussianMixture_eval_summary_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved evaluation summary to {out}")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gaussian Mixture (per-class) pipeline runner")
    p.add_argument("command", choices=["train", "synth", "eval", "all"], help="Which step to run")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    p.add_argument("--no-synth", action="store_true", help="(for eval/all) skip synthetic data in evaluation")
    return p.parse_args()


def main() -> None:
    _enable_gpu_mem_growth()

    args = parse_args()
    cfg = load_yaml(Path(args.config))

    # Sensible defaults (non-destructive; parity with other repos)
    cfg.setdefault("SEED", 42)
    cfg.setdefault("VAL_FRACTION", 0.5)
    cfg.setdefault("FID_CAP", 200)
    cfg.setdefault("EVAL_EPOCHS", 20)
    cfg.setdefault("IMG_SHAPE", [40, 40, 1])
    cfg.setdefault("NUM_CLASSES", 9)
    cfg.setdefault("SAMPLES_PER_CLASS", 25)
    cfg.setdefault("ARTIFACTS", {})
    cfg["ARTIFACTS"].setdefault("checkpoints", "artifacts/gaussianmixture/checkpoints")
    cfg["ARTIFACTS"].setdefault("synthetic",   "artifacts/gaussianmixture/synthetic")
    cfg["ARTIFACTS"].setdefault("summaries",   "artifacts/gaussianmixture/summaries")
    cfg["ARTIFACTS"].setdefault("tensorboard", "artifacts/tensorboard")

    print(f"[config] Using {Path(args.config).resolve()}")
    print(f"Synth outputs -> {Path(cfg['ARTIFACTS']['synthetic']).resolve()}")

    if args.command == "train":
        run_train(cfg)
    elif args.command == "synth":
        run_synth(cfg)
    elif args.command == "eval":
        run_eval(cfg, include_synth=not args.no_synth)
    elif args.command == "all":
        run_train(cfg)
        run_synth(cfg)
        run_eval(cfg, include_synth=not args.no_synth)


if __name__ == "__main__":
    main()
