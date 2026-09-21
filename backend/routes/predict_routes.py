"""HTTP layer — request/response only. No sklearn here."""

from routes.predict import predict_bp

__all__ = ["predict_bp"]
