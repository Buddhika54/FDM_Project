"""
Phase 2 — cleaning and sklearn preprocessor.

Reads dataset/raw/train.csv (never modified). Writes:
  dataset/processed/train.csv, test.csv          cleaned tabular splits (+ target)
  dataset/processed/X_train.npy, X_test.npy      scaled + one-hot matrices
  dataset/processed/y_train.npy, y_test.npy
  dataset/processed/feature_names.json
  dataset/processed/preprocessing_log.json
  models/preprocessor.pkl                        fitted Pipeline (clean → encode/scale)

No model training.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import (
    MODELS_DIR,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATASET_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "backend" / "models" / "prediction_schema.py"
_spec = importlib.util.spec_from_file_location("prediction_schema", _SCHEMA_PATH)
_schema = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schema)

REQUIRED_FEATURES = list(_schema.REQUIRED_FEATURES)
NUMERIC_FEATURES = list(_schema.NUMERIC_FEATURES)
CATEGORICAL_FEATURES = list(_schema.CATEGORICAL_FEATURES)
BINARY_FEATURES = list(_schema.BINARY_FEATURES)
CATEGORICAL_VALUES = {k: list(v) for k, v in _schema.CATEGORICAL_VALUES.items()}
IDENTIFIER_COLUMNS = list(_schema.IDENTIFIER_COLUMNS)

# Invalid negatives → 0 (Phase 0). delivery_delay_days is NOT in this list.
CLIP_NEGATIVES_TO_ZERO = [
    "product_price",
    "discount_percent",
    "past_return_rate",
    "session_length_minutes",
    "num_product_views",
]

PRICE_CAP_QUANTILE = 0.99


def apply_deterministic_clips(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise clips only (no fitted statistics). Does not drop rows."""
    out = df.copy()
    for col in CLIP_NEGATIVES_TO_ZERO:
        out[col] = out[col].clip(lower=0)
    out["product_rating"] = out["product_rating"].clip(lower=1, upper=5)
    out["discount_percent"] = out["discount_percent"].clip(lower=0, upper=100)
    out["past_return_rate"] = out["past_return_rate"].clip(lower=0, upper=1)
    return out


def validate_categories(df: pd.DataFrame) -> None:
    """Fail loudly if any categorical value is outside the Phase 0 contract."""
    for col in CATEGORICAL_FEATURES:
        allowed = set(CATEGORICAL_VALUES[col])
        observed = set(df[col].dropna().unique())
        unknown = observed - allowed
        if unknown:
            raise ValueError(
                f"Unexpected {col} value(s): {sorted(unknown)}. "
                f"Allowed: {CATEGORICAL_VALUES[col]}"
            )


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(np.asarray(X), columns=REQUIRED_FEATURES)


class FeatureCleaner(BaseEstimator, TransformerMixin):
    """
    DataFrame in / DataFrame out.

    fit: store product_price cap = training-split quantile (after clips).
    transform: deterministic clips, cap price, validate categories, drop ids/target.
    """

    def __init__(self, price_quantile: float = PRICE_CAP_QUANTILE):
        self.price_quantile = price_quantile

    def fit(self, X, y=None):
        frame = _as_frame(X)
        _require_columns(frame)
        validate_categories(frame)
        clipped = apply_deterministic_clips(frame)
        self.price_cap_ = float(clipped["product_price"].quantile(self.price_quantile))
        self.n_features_in_ = len(REQUIRED_FEATURES)
        self.feature_names_in_ = np.asarray(REQUIRED_FEATURES, dtype=object)
        return self

    def transform(self, X):
        if not hasattr(self, "price_cap_"):
            raise RuntimeError("FeatureCleaner is not fitted.")
        frame = _as_frame(X)
        _require_columns(frame)
        validate_categories(frame)
        out = apply_deterministic_clips(frame)
        out["product_price"] = out["product_price"].clip(upper=self.price_cap_)
        return out.loc[:, REQUIRED_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(REQUIRED_FEATURES, dtype=object)


# So joblib always pickles this as ml.preprocess.FeatureCleaner (not __main__)
FeatureCleaner.__module__ = "ml.preprocess"


def _require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")


def build_preprocessor() -> Pipeline:
    """Unfitted Pipeline: FeatureCleaner → scale numerics / one-hot cats / passthrough binary."""
    ohe = OneHotEncoder(
        categories=[CATEGORICAL_VALUES[c] for c in CATEGORICAL_FEATURES],
        handle_unknown="error",
        sparse_output=False,
    )
    column_transform = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", ohe, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline(
        [
            ("clean", FeatureCleaner(price_quantile=PRICE_CAP_QUANTILE)),
            ("encode_scale", column_transform),
        ]
    )


def _clip_impact(before: pd.Series, after: pd.Series) -> dict:
    changed = int((before != after).sum())
    return {
        "n_changed": changed,
        "pct_changed": round(100.0 * changed / len(before), 4),
        "min_before": float(before.min()),
        "max_before": float(before.max()),
        "min_after": float(after.min()),
        "max_after": float(after.max()),
        "n_negative_before": int((before < 0).sum()),
        "n_negative_after": int((after < 0).sum()),
    }


def _build_log(raw: pd.DataFrame, cleaned_all: pd.DataFrame, price_cap: float, n_train: int, n_test: int) -> dict:
    cols = CLIP_NEGATIVES_TO_ZERO + ["product_rating"]
    impact = {}
    for col in cols:
        impact[col] = _clip_impact(raw[col], cleaned_all[col])
    n_capped = int((raw["product_price"].clip(lower=0) > price_cap).sum())
    return {
        "n_rows_in": int(len(raw)),
        "n_rows_out": int(len(cleaned_all)),
        "rows_dropped": 0,
        "train_size": n_train,
        "test_size": n_test,
        "test_size_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratified_on": TARGET_COLUMN,
        "product_price_cap_quantile": PRICE_CAP_QUANTILE,
        "product_price_cap_value": price_cap,
        "product_price_cap_fit_on": "X_train after deterministic clips (no test leakage)",
        "product_price_n_capped_on_full_table_vs_train_cap": n_capped,
        "delivery_delay_days": "kept signed; not clipped",
        "clip_impact": impact,
        "decision_product_price_tail": (
            "Winsorize/cap product_price at the training 99th percentile instead of "
            "dropping high-price rows. Cap is stored on FeatureCleaner.price_cap_ "
            "inside preprocessor.pkl so inference uses the same threshold."
        ),
    }


def run_preprocessing(
    raw_path: Path = RAW_DATASET_PATH,
    processed_dir: Path = PROCESSED_DATA_DIR,
    preprocessor_path: Path = PREPROCESSOR_PATH,
) -> dict:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    raw = pd.read_csv(raw_path)
    _require_columns(raw)
    if TARGET_COLUMN not in raw.columns:
        raise ValueError(f"Missing target column {TARGET_COLUMN}")

    validate_categories(raw)

    feature_frame = raw.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in raw.columns])
    y = feature_frame[TARGET_COLUMN].astype(int)
    X = feature_frame.drop(columns=[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train, y_train)

    cleaner: FeatureCleaner = preprocessor.named_steps["clean"]
    price_cap = float(cleaner.price_cap_)

    X_train_clean = cleaner.transform(X_train)
    X_test_clean = cleaner.transform(X_test)

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = preprocessor.named_steps["encode_scale"].get_feature_names_out().tolist()

    processed_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_out = X_train_clean.copy()
    train_out[TARGET_COLUMN] = y_train.to_numpy()
    test_out = X_test_clean.copy()
    test_out[TARGET_COLUMN] = y_test.to_numpy()
    train_out.to_csv(processed_dir / "train.csv", index=False)
    test_out.to_csv(processed_dir / "test.csv", index=False)

    np.save(processed_dir / "X_train.npy", X_train_proc)
    np.save(processed_dir / "X_test.npy", X_test_proc)
    np.save(processed_dir / "y_train.npy", y_train.to_numpy())
    np.save(processed_dir / "y_test.npy", y_test.to_numpy())

    cleaned_all = pd.concat([X_train_clean, X_test_clean], axis=0).sort_index()
    raw_aligned = raw.loc[cleaned_all.index]
    log = _build_log(raw_aligned, cleaned_all, price_cap, len(X_train), len(X_test))
    log["X_train_shape"] = list(X_train_proc.shape)
    log["X_test_shape"] = list(X_test_proc.shape)
    log["y_train_shape"] = list(np.asarray(y_train).shape)
    log["y_test_shape"] = list(np.asarray(y_test).shape)
    log["n_output_features"] = len(feature_names)
    log["y_train_positive_rate"] = float(np.mean(y_train))
    log["y_test_positive_rate"] = float(np.mean(y_test))

    (processed_dir / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )
    (processed_dir / "preprocessing_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )

    joblib.dump(preprocessor, preprocessor_path)

    print("Phase 2 complete.")
    print(f"  product_price cap (train p{int(PRICE_CAP_QUANTILE * 100)}): {price_cap:.6f}")
    print(f"  X_train {X_train_proc.shape}  X_test {X_test_proc.shape}")
    print(f"  y_train {y_train.shape}  y_test {y_test.shape}")
    print(f"  preprocessor -> {preprocessor_path}")
    print(f"  cleaned splits -> {processed_dir}")
    return {
        "preprocessor": preprocessor,
        "log": log,
        "price_cap": price_cap,
        "feature_names": feature_names,
        "X_train": X_train_proc,
        "X_test": X_test_proc,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "X_train_clean": X_train_clean,
        "X_test_clean": X_test_clean,
        "raw": raw,
    }


def main() -> None:
    run_preprocessing()


if __name__ == "__main__":
    main()
