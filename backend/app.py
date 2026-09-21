"""
Flask application factory.

Run from the project root:
    python backend/app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from utils.exceptions import ModelNotLoadedError, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _cors_origins(config) -> list[str]:
    raw = config.get("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(
        app,
        resources={r"/api/*": {"origins": _cors_origins(app.config)}},
        allow_headers=["Content-Type"],
        methods=["GET", "POST", "OPTIONS"],
    )

    from routes.health_routes import health_bp
    from routes.predict import predict_bp
    from services.model_loader import load_artifacts

    load_artifacts(app.config)

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(predict_bp, url_prefix="/api")

    @app.errorhandler(ValidationError)
    def _validation(err: ValidationError):
        return jsonify(
            {
                "error": "validation_error",
                "message": str(err) or "Invalid or missing fields.",
                "details": err.details,
            }
        ), 400

    @app.errorhandler(ModelNotLoadedError)
    def _unavailable(_err: ModelNotLoadedError):
        return jsonify(
            {
                "error": "model_unavailable",
                "message": "Model or preprocessor is not loaded. Train and save artifacts to models/.",
            }
        ), 503

    @app.errorhandler(Exception)
    def _unexpected(err: Exception):
        if isinstance(err, HTTPException):
            return err
        logger.exception("Unhandled error")
        return jsonify(
            {
                "error": "internal_error",
                "message": "Prediction failed.",
            }
        ), 500

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(
        host=application.config.get("HOST", "127.0.0.1"),
        port=application.config.get("PORT", 5000),
        debug=application.config.get("DEBUG", True),
    )
