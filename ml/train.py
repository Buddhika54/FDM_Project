"""
Train return-risk classifiers on Phase 2 arrays.

Phase 3: Logistic Regression + Decision Tree (sklearn defaults).
Phase 4: Random Forest + XGBoost (depth-constrained; no grid search).

Does not load or write models/preprocessor.pkl.
Does not pick a production model (Phase 7) or tune (Phase 6).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, REPORTS_DIR
from ml.evaluation.metrics import classification_metrics

LR_PATH = MODELS_DIR / "baseline_logistic_regression.pkl"
DT_PATH = MODELS_DIR / "baseline_decision_tree.pkl"
RF_PATH = MODELS_DIR / "random_forest.pkl"
XGB_PATH = MODELS_DIR / "xgboost.pkl"

METRICS_CSV = REPORTS_DIR / "baseline_metrics.csv"
METRICS_JSON = REPORTS_DIR / "baseline_metrics.json"
COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
COMPARISON_JSON = REPORTS_DIR / "model_comparison.json"
LR_COEF_CSV = REPORTS_DIR / "baseline_lr_coefficients.csv"
DT_IMP_CSV = REPORTS_DIR / "baseline_dt_importances.csv"
RF_IMP_CSV = REPORTS_DIR / "rf_importances.csv"
XGB_IMP_CSV = REPORTS_DIR / "xgboost_importances.csv"

LR_MAX_ITER = 2000
OVERFIT_GAP = 0.05

# Phase 3 LR signs that we check in ensemble importances (EDA category/coupon story).
PHASE3_SIGNAL_FEATURES = [
    "bin__used_coupon",
    "cat__product_category_clothing",
    "cat__shipping_method_express",
    "cat__payment_method_apple_pay",
    "cat__device_type_desktop",
]


def load_processed_arrays(processed_dir: Path = PROCESSED_DATA_DIR):
    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    feature_names = json.loads((processed_dir / "feature_names.json").read_text(encoding="utf-8"))
    if len(feature_names) != X_train.shape[1]:
        raise ValueError(
            f"feature_names.json length {len(feature_names)} != X_train columns {X_train.shape[1]}"
        )
    return X_train, X_test, y_train, y_test, feature_names


def train_logistic_regression(X_train, y_train, random_state: int = RANDOM_STATE):
    model = LogisticRegression(random_state=random_state, max_iter=LR_MAX_ITER)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, random_state: int = RANDOM_STATE):
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state: int = RANDOM_STATE):
    model = RandomForestClassifier(
        random_state=random_state,
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state: int = RANDOM_STATE):
    model = XGBClassifier(
        random_state=random_state,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model,
    X_test,
    y_test,
    model_name: str,
    X_train=None,
    y_train=None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, y_pred, y_proba)
    metrics["model"] = model_name
    if X_train is not None and y_train is not None:
        train_proba = model.predict_proba(X_train)[:, 1]
        metrics["train_roc_auc"] = float(roc_auc_score(y_train, train_proba))
        metrics["test_roc_auc"] = metrics["roc_auc"]
        metrics["roc_auc_gap"] = metrics["train_roc_auc"] - metrics["test_roc_auc"]
        metrics["overfit_flag"] = bool(metrics["roc_auc_gap"] > OVERFIT_GAP)
    return metrics, y_pred, y_proba


def logistic_coefficient_table(model: LogisticRegression, feature_names: list[str]) -> pd.DataFrame:
    coef = model.coef_.ravel()
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }
    )
    return table.sort_values("coefficient", ascending=False).reset_index(drop=True)


def importance_table(model, feature_names: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def decision_tree_importance_table(model, feature_names: list[str]) -> pd.DataFrame:
    return importance_table(model, feature_names)


def signal_features_in_top10(importance_df: pd.DataFrame) -> dict:
    top = set(importance_df.head(10)["feature"])
    return {name: name in top for name in PHASE3_SIGNAL_FEATURES}


def _metrics_row(metrics: dict) -> dict:
    row = {
        "model": metrics["model"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }
    if "train_roc_auc" in metrics:
        row["train_roc_auc"] = metrics["train_roc_auc"]
        row["roc_auc_gap"] = metrics["roc_auc_gap"]
        row["overfit_flag"] = metrics["overfit_flag"]
    return row


def _write_baseline_reports(lr, dt, lr_metrics, dt_metrics, lr_coef, dt_imp, reports_dir: Path):
    comparison = pd.DataFrame([_metrics_row(lr_metrics), _metrics_row(dt_metrics)])
    comparison.to_csv(reports_dir / "baseline_metrics.csv", index=False)
    (reports_dir / "baseline_metrics.json").write_text(
        json.dumps(
            {
                "logistic_regression": lr_metrics,
                "decision_tree": dt_metrics,
                "comparison": comparison.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lr_coef.to_csv(LR_COEF_CSV, index=False)
    dt_imp.to_csv(DT_IMP_CSV, index=False)
    return comparison


def run_training(
    processed_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    """Phase 3: LR + unpruned Decision Tree."""
    X_train, X_test, y_train, y_test, feature_names = load_processed_arrays(processed_dir)

    lr = train_logistic_regression(X_train, y_train)
    dt = train_decision_tree(X_train, y_train)

    lr_metrics, _, _ = evaluate_classifier(lr, X_test, y_test, "Logistic Regression", X_train, y_train)
    dt_metrics, _, _ = evaluate_classifier(dt, X_test, y_test, "Decision Tree", X_train, y_train)

    lr_metrics["n_iter"] = int(np.asarray(lr.n_iter_).ravel()[0])
    dt_metrics["tree_depth"] = int(dt.get_depth())
    dt_metrics["node_count"] = int(dt.tree_.node_count)
    dt_metrics["n_leaves"] = int(dt.get_n_leaves())

    lr_coef = logistic_coefficient_table(lr, feature_names)
    dt_imp = importance_table(dt, feature_names)

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, LR_PATH)
    joblib.dump(dt, DT_PATH)
    comparison = _write_baseline_reports(lr, dt, lr_metrics, dt_metrics, lr_coef, dt_imp, reports_dir)

    print("Phase 3 complete (baselines only; preprocessor.pkl not modified).")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  models -> {LR_PATH.name}, {DT_PATH.name}")
    print(f"  metrics -> {METRICS_CSV}")

    return {
        "logistic_regression": lr,
        "decision_tree": dt,
        "lr_metrics": lr_metrics,
        "dt_metrics": dt_metrics,
        "comparison": comparison,
        "lr_coefficients": lr_coef,
        "dt_importances": dt_imp,
        "feature_names": feature_names,
    }


def _load_or_train_baselines(X_train, y_train, X_test, y_test, feature_names):
    if LR_PATH.exists() and DT_PATH.exists():
        lr = joblib.load(LR_PATH)
        dt = joblib.load(DT_PATH)
    else:
        lr = train_logistic_regression(X_train, y_train)
        dt = train_decision_tree(X_train, y_train)
        joblib.dump(lr, LR_PATH)
        joblib.dump(dt, DT_PATH)

    lr_metrics, _, _ = evaluate_classifier(lr, X_test, y_test, "Logistic Regression", X_train, y_train)
    dt_metrics, _, _ = evaluate_classifier(dt, X_test, y_test, "Decision Tree", X_train, y_train)
    dt_metrics["tree_depth"] = int(dt.get_depth())
    dt_metrics["node_count"] = int(dt.tree_.node_count)
    lr_coef = logistic_coefficient_table(lr, feature_names)
    dt_imp = importance_table(dt, feature_names)
    return lr, dt, lr_metrics, dt_metrics, lr_coef, dt_imp


def run_ensembles(
    processed_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    """Phase 4: constrained RF + XGBoost; compare all four models."""
    X_train, X_test, y_train, y_test, feature_names = load_processed_arrays(processed_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    lr, dt, lr_metrics, dt_metrics, lr_coef, dt_imp = _load_or_train_baselines(
        X_train, y_train, X_test, y_test, feature_names
    )

    rf = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)
    joblib.dump(rf, RF_PATH)
    joblib.dump(xgb, XGB_PATH)

    rf_metrics, _, _ = evaluate_classifier(rf, X_test, y_test, "Random Forest", X_train, y_train)
    xgb_metrics, _, _ = evaluate_classifier(xgb, X_test, y_test, "XGBoost", X_train, y_train)

    rf_imp = importance_table(rf, feature_names)
    xgb_imp = importance_table(xgb, feature_names)
    rf_imp.to_csv(RF_IMP_CSV, index=False)
    xgb_imp.to_csv(XGB_IMP_CSV, index=False)

    comparison = pd.DataFrame(
        [
            _metrics_row(lr_metrics),
            _metrics_row(dt_metrics),
            _metrics_row(rf_metrics),
            _metrics_row(xgb_metrics),
        ]
    ).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    comparison.to_csv(COMPARISON_CSV, index=False)

    payload = {
        "logistic_regression": lr_metrics,
        "decision_tree": dt_metrics,
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
        "comparison": comparison.to_dict(orient="records"),
        "rf_signal_in_top10": signal_features_in_top10(rf_imp),
        "xgb_signal_in_top10": signal_features_in_top10(xgb_imp),
        "overfit_threshold": OVERFIT_GAP,
    }
    COMPARISON_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Phase 4 complete (ensembles; preprocessor.pkl not modified; no final-model pick).")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for name, m in [("Random Forest", rf_metrics), ("XGBoost", xgb_metrics)]:
        flag = "FLAG gap>0.05" if m["overfit_flag"] else "ok"
        print(
            f"  {name} train AUC={m['train_roc_auc']:.4f}  "
            f"test AUC={m['test_roc_auc']:.4f}  gap={m['roc_auc_gap']:.4f}  {flag}"
        )
        print(f"  {name} confusion matrix:", m["confusion_matrix"])
    print("RF top 10 importances:")
    print(rf_imp.head(10).to_string(index=False))
    print("XGB top 10 importances:")
    print(xgb_imp.head(10).to_string(index=False))
    print("Phase 3 LR signal features in RF top 10:", payload["rf_signal_in_top10"])
    print("Phase 3 LR signal features in XGB top 10:", payload["xgb_signal_in_top10"])
    print(f"  models -> {RF_PATH.name}, {XGB_PATH.name}")
    print(f"  comparison -> {COMPARISON_CSV}")

    return {
        "logistic_regression": lr,
        "decision_tree": dt,
        "random_forest": rf,
        "xgboost": xgb,
        "lr_metrics": lr_metrics,
        "dt_metrics": dt_metrics,
        "rf_metrics": rf_metrics,
        "xgb_metrics": xgb_metrics,
        "comparison": comparison,
        "lr_coefficients": lr_coef,
        "dt_importances": dt_imp,
        "rf_importances": rf_imp,
        "xgb_importances": xgb_imp,
        "feature_names": feature_names,
        "rf_signal_in_top10": payload["rf_signal_in_top10"],
        "xgb_signal_in_top10": payload["xgb_signal_in_top10"],
    }


def main() -> None:
    run_ensembles()


if __name__ == "__main__":
    main()
