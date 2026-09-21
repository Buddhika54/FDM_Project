"""POST /api/predict — HTTP only. No sklearn imports."""

from flask import Blueprint, jsonify, request

from services.prediction_service import predict_return_risk
from utils.exceptions import ModelNotLoadedError, ValidationError
from utils.validation import validate_prediction_payload

predict_bp = Blueprint("predict", __name__)


@predict_bp.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(
            {
                "error": "validation_error",
                "message": "JSON body required.",
                "details": {},
            }
        ), 400
    try:
        features = validate_prediction_payload(payload)
        result = predict_return_risk(features)
        return jsonify(result), 200
    except ValidationError as err:
        return jsonify(
            {
                "error": "validation_error",
                "message": str(err) or "Invalid or missing fields.",
                "details": err.details,
            }
        ), 400
    except ModelNotLoadedError:
        return jsonify(
            {
                "error": "model_unavailable",
                "message": "Model or preprocessor is not loaded. Train and save artifacts to models/.",
            }
        ), 503
