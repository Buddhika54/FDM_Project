"""
ML pipeline paths and reproducibility settings.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATASET_PATH = RAW_DATA_DIR / "train.csv"
TARGET_COLUMN = "returned"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FINAL_MODEL_PATH = MODELS_DIR / "final_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
FINAL_MODEL_CONFIG_PATH = MODELS_DIR / "final_model_config.json"
