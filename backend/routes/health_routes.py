"""GET /api/health."""

from flask import Blueprint, jsonify

from services.model_loader import artifacts_loaded, get_model_config
from utils.exceptions import ModelNotLoadedError

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health():
    loaded = artifacts_loaded()
    body = {
        "status": "ok" if loaded else "degraded",
        "service": "return-risk-prediction",
        "model_loaded": loaded,
        "preprocessor_loaded": loaded,
    }
    if loaded:
        try:
            meta = get_model_config()
            body["model"] = meta.get("model")
            body["threshold"] = meta.get("threshold")
            if meta.get("n_model_features") is not None:
                body["n_model_features"] = meta.get("n_model_features")
            body["feature_engineering"] = bool(meta.get("includes_feature_engineering"))
        except ModelNotLoadedError:
            body["status"] = "degraded"
    status = 200 if loaded else 503
    return jsonify(body), status
