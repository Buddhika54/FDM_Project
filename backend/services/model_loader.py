"""Load preprocessor.pkl, final_model.pkl, and final_model_config.json once."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib

from utils.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)

_STATE = {
    "preprocessor": None,
    "model": None,
    "config": None,
    "loaded": False,
}


def _cfg(config, key):
    if hasattr(config, "get"):
        value = config.get(key)
        if value is not None:
            return value
    return getattr(config, key)


def _ensure_project_root_on_path(config) -> None:
    """Unpickling FeatureCleaner requires the `ml` package (project root)."""
    root = Path(_cfg(config, "PREPROCESSOR_PATH")).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import ml.preprocess  # noqa: F401  — register FeatureCleaner for joblib


def load_artifacts(config) -> None:
    pre_path = Path(_cfg(config, "PREPROCESSOR_PATH"))
    model_path = Path(_cfg(config, "FINAL_MODEL_PATH"))
    cfg_path = Path(_cfg(config, "FINAL_MODEL_CONFIG_PATH"))
    missing = [str(p) for p in (pre_path, model_path, cfg_path) if not p.exists()]
    if missing:
        raise RuntimeError(
            "Cannot start API — missing artifacts: "
            + ", ".join(missing)
            + ". Run Phases 2 and 7 first."
        )

    _ensure_project_root_on_path(config)
    try:
        preprocessor = joblib.load(pre_path)
        model = joblib.load(model_path)
        meta = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot start API — failed to load model artifacts: {exc}") from exc

    if "threshold" not in meta:
        raise RuntimeError("Cannot start API — final_model_config.json has no threshold.")

    _STATE["preprocessor"] = preprocessor
    _STATE["model"] = model
    _STATE["config"] = meta
    _STATE["loaded"] = True
    logger.info(
        "Loaded preprocessor + %s (threshold=%s)",
        meta.get("model"),
        meta.get("threshold"),
    )


def artifacts_loaded() -> bool:
    return bool(_STATE["loaded"])


def get_preprocessor():
    if not _STATE["loaded"]:
        raise ModelNotLoadedError("Preprocessor is not loaded.")
    return _STATE["preprocessor"]


def get_model():
    if not _STATE["loaded"]:
        raise ModelNotLoadedError("Model is not loaded.")
    return _STATE["model"]


def get_model_config() -> dict:
    if not _STATE["loaded"] or _STATE["config"] is None:
        raise ModelNotLoadedError("Model config is not loaded.")
    return _STATE["config"]
