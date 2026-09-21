"""
Phase 6 — GridSearchCV on the training split only.

Goal: shrink the Phase 4 train/test ROC-AUC gap. A ~0 lift in test AUC
is an acceptable finding given Phase 1–5 (models already plateau near 0.59).

Does not write final_model.pkl (Phase 7). Does not touch X_test during search.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, REPORTS_DIR
from ml.evaluation.metrics import classification_metrics
from ml.train import COMPARISON_CSV, OVERFIT_GAP, load_processed_arrays

CV = 5
SCORING = "roc_auc"

TUNED_PATHS = {
    "Logistic Regression": MODELS_DIR / "tuned_logistic_regression.pkl",
    "Decision Tree": MODELS_DIR / "tuned_decision_tree.pkl",
    "Random Forest": MODELS_DIR / "tuned_random_forest.pkl",
    "XGBoost": MODELS_DIR / "tuned_xgboost.pkl",
}
COMPARISON_OUT = REPORTS_DIR / "tuned_model_comparison.csv"
PARAMS_OUT = REPORTS_DIR / "best_params.json"
TUNED_JSON = REPORTS_DIR / "tuned_model_comparison.json"

GRIDS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(
            solver="liblinear",
            random_state=RANDOM_STATE,
            max_iter=2000,
        ),
        "param_grid": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
        },
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_leaf": [5, 20, 50],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [6, 10, 15],
            "min_samples_leaf": [5, 20, 50],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=1,
        ),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 8],
            "learning_rate": [0.05, 0.1, 0.2],
        },
    },
}


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if obj is None:
        return None
    return obj


def load_untuned_metrics(path: Path = COMPARISON_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Phase 4 comparison table required: {path}")
    table = pd.read_csv(path)
    return table.set_index("model")


def run_grid_search(estimator, param_grid, X_train, y_train, scoring=SCORING, cv=CV):
    search = GridSearchCV(
        estimator,
        param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def _evaluate(model, X_train, y_train, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, y_pred, y_proba)
    train_auc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    metrics["model"] = name
    metrics["train_roc_auc"] = train_auc
    metrics["test_roc_auc"] = metrics["roc_auc"]
    metrics["roc_auc_gap"] = train_auc - metrics["roc_auc"]
    metrics["overfit_flag"] = bool(metrics["roc_auc_gap"] > OVERFIT_GAP)
    return metrics


def tune_one(name: str, spec: dict, X_train, y_train, X_test, y_test, untuned: pd.Series) -> dict:
    n_combos = int(np.prod([len(v) for v in spec["param_grid"].values()]))
    print(f"\n=== {name}: GridSearchCV cv={CV} scoring={SCORING} ({n_combos} combos) ===")
    t0 = time.perf_counter()
    search = run_grid_search(spec["estimator"], spec["param_grid"], X_train, y_train)
    elapsed = time.perf_counter() - t0
    model = search.best_estimator_
    metrics = _evaluate(model, X_train, y_train, X_test, y_test, name)
    untuned_auc = float(untuned["roc_auc"])
    untuned_gap = float(untuned["roc_auc_gap"])
    metrics["best_params"] = search.best_params_
    metrics["cv_roc_auc"] = float(search.best_score_)
    metrics["untuned_test_roc_auc"] = untuned_auc
    metrics["delta_vs_untuned"] = metrics["test_roc_auc"] - untuned_auc
    metrics["untuned_gap"] = untuned_gap
    metrics["gap_delta"] = metrics["roc_auc_gap"] - untuned_gap
    metrics["search_seconds"] = elapsed
    print(f"  best_params_ = {search.best_params_}")
    print(f"  best CV ROC-AUC = {search.best_score_:.4f}  ({elapsed:.1f}s)")
    print(
        f"  test ROC-AUC = {metrics['test_roc_auc']:.4f}  "
        f"(untuned {untuned_auc:.4f}, delta {metrics['delta_vs_untuned']:+.4f})"
    )
    print(
        f"  train/test gap = {metrics['roc_auc_gap']:.4f}  "
        f"(untuned {untuned_gap:.4f}, gap change {metrics['gap_delta']:+.4f})  "
        f"{'FLAG' if metrics['overfit_flag'] else 'ok'}"
    )
    return {"model": model, "search": search, "metrics": metrics}


def run_tuning(
    processed_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    X_train, X_test, y_train, y_test, _ = load_processed_arrays(processed_dir)
    untuned_table = load_untuned_metrics()
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    rows = []
    best_params = {}
    for name, spec in GRIDS.items():
        packed = tune_one(name, spec, X_train, y_train, X_test, y_test, untuned_table.loc[name])
        results[name] = packed
        joblib.dump(packed["model"], TUNED_PATHS[name])
        m = packed["metrics"]
        best_params[name] = m["best_params"]
        rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["test_roc_auc"],
                "train_roc_auc": m["train_roc_auc"],
                "roc_auc_gap": m["roc_auc_gap"],
                "overfit_flag": m["overfit_flag"],
                "cv_roc_auc": m["cv_roc_auc"],
                "untuned_test_roc_auc": m["untuned_test_roc_auc"],
                "delta_vs_untuned": m["delta_vs_untuned"],
                "untuned_gap": m["untuned_gap"],
                "gap_delta": m["gap_delta"],
                "best_params": json.dumps(m["best_params"]),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    comparison.to_csv(COMPARISON_OUT, index=False)
    PARAMS_OUT.write_text(json.dumps(_json_ready(best_params), indent=2), encoding="utf-8")
    TUNED_JSON.write_text(
        json.dumps(
            {
                "scoring": SCORING,
                "cv": CV,
                "goal": "close train/test ROC-AUC gap; do not expect large test-AUC gains",
                "best_params": _json_ready(best_params),
                "comparison": comparison.to_dict(orient="records"),
                "note": "No final_model.pkl in this phase.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    max_abs_delta = float(comparison["delta_vs_untuned"].abs().max())
    print("\nPhase 6 complete (no final-model selection).")
    print(comparison.drop(columns=["best_params"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if max_abs_delta < 0.01:
        print(
            "Tuning did NOT meaningfully improve test ROC-AUC for any model "
            f"(max |delta|={max_abs_delta:.4f}). That is an expected finding, not a failure."
        )
    print(f"  table -> {COMPARISON_OUT}")
    print(f"  params -> {PARAMS_OUT}")
    return {"results": results, "comparison": comparison, "best_params": best_params}


def main() -> None:
    run_tuning()


if __name__ == "__main__":
    main()
