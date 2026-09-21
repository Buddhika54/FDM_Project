"""
Backend configuration.

Responsibilities:
- Paths to serialized ML artifacts (project-root/models)
- Server host/port
- CORS origins
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    PORT = int(os.getenv("FLASK_PORT", "5000"))
    CORS_ORIGINS = os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    )

    MODEL_DIR = PROJECT_ROOT / "models"
    FINAL_MODEL_PATH = MODEL_DIR / os.getenv("FINAL_MODEL_FILENAME", "final_model.pkl")
    PREPROCESSOR_PATH = MODEL_DIR / os.getenv("PREPROCESSOR_FILENAME", "preprocessor.pkl")
    FINAL_MODEL_CONFIG_PATH = MODEL_DIR / os.getenv(
        "FINAL_MODEL_CONFIG_FILENAME", "final_model_config.json"
    )
