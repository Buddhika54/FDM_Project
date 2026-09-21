"""
Phase 5 — evaluation of the four fitted models.

Loads pickles + Phase 2 test arrays. Does not retrain, tune, or choose
a production artifact.

Writes reports/figures/07_roc_curves.png, 08_pr_curves.png, plus
calibration / threshold plots and CSV/JSON tables under reports/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, REPORTS_DIR
from ml.train import COMPARISON_CSV, DT_PATH, LR_PATH, RF_PATH, XGB_PATH

FIG_DIR = REPORTS_DIR / "figures"
ROC_PATH = FIG_DIR / "07_roc_curves.png"
PR_PATH = FIG_DIR / "08_pr_curves.png"
CAL_PATH = FIG_DIR / "13_calibration_curves.png"
THR_PATH = FIG_DIR / "14_threshold_precision_recall.png"

MODEL_FILES = {
    "Logistic Regression": LR_PATH,
    "Decision Tree": DT_PATH,
    "Random Forest": RF_PATH,
    "XGBoost": XGB_PATH,
}
PLOT_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
]
COLORS = {
    "Logistic Regression": "#2563eb",
    "Decision Tree": "#64748b",
    "Random Forest": "#c2410c",
    "XGBoost": "#15803d",
}
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
CAT_COLS = ["device_type", "product_category", "shipping_method", "payment_method", "used_coupon"]


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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_models(models_dir: Path = MODELS_DIR) -> dict:
    models = {}
    for name, path in MODEL_FILES.items():
        resolved = models_dir / path.name
        if not resolved.exists():
            raise FileNotFoundError(f"Missing fitted model: {resolved}")
        models[name] = joblib.load(resolved)
    return models


def load_test_set(processed_dir: Path = PROCESSED_DATA_DIR):
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    test_frame = pd.read_csv(processed_dir / "test.csv")
    if len(test_frame) != len(y_test):
        raise ValueError("test.csv and y_test.npy row counts differ")
    return X_test, y_test, test_frame


def predict_probas(models: dict, X_test: np.ndarray) -> dict[str, np.ndarray]:
    return {name: model.predict_proba(X_test)[:, 1] for name, model in models.items()}


def best_model_from_comparison(comparison_csv: Path = COMPARISON_CSV) -> str:
    if not comparison_csv.exists():
        raise FileNotFoundError(f"Need {comparison_csv} from Phase 4")
    table = pd.read_csv(comparison_csv)
    return str(table.sort_values("roc_auc", ascending=False).iloc[0]["model"])


def plot_roc(y_test, probas: dict, path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    aucs = {}
    for name in PLOT_ORDER:
        p = probas[name]
        fpr, tpr, _ = roc_curve(y_test, p)
        auc = float(roc_auc_score(y_test, p))
        aucs[name] = auc
        ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="#94a3b8", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (hold-out test)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return aucs


def plot_pr(y_test, probas: dict, path: Path) -> dict:
    baseline = float(np.mean(y_test))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    aps = {}
    for name in PLOT_ORDER:
        p = probas[name]
        prec, rec, _ = precision_recall_curve(y_test, p)
        ap = float(average_precision_score(y_test, p))
        aps[name] = ap
        ax.plot(rec, prec, color=COLORS[name], lw=2, label=f"{name} (AP={ap:.3f})")
    ax.axhline(baseline, ls="--", c="#94a3b8", lw=1, label=f"Prevalence={baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curves (hold-out test)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0.35, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return aps


def threshold_table(y_test, proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "predicted_positive_rate": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows)


def recommend_threshold(thr_df: pd.DataFrame) -> dict:
    """
    Recall-first operating point for pre-fulfillment intervention.

    Missing a true return (FN) is costlier than a false alarm (FP): the product
    exists to flag risky orders before they ship. Default 0.5 is not assumed.

    0.3 is discarded if it flags >90% of orders (no operational selectivity).
    Among remaining rows, pick the highest-recall threshold whose precision is
    at least the class prevalence floor (0.47) — typically 0.4 on this data.
    """
    selective = thr_df[thr_df["predicted_positive_rate"] <= 0.90].copy()
    if selective.empty:
        selective = thr_df.copy()
    floor = 0.47
    eligible = selective[selective["precision"] >= floor]
    pool = eligible if len(eligible) else selective
    chosen = pool.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0]
    return {
        "recommended_threshold": float(chosen["threshold"]),
        "precision": float(chosen["precision"]),
        "recall": float(chosen["recall"]),
        "f1": float(chosen["f1"]),
        "predicted_positive_rate": float(chosen["predicted_positive_rate"]),
        "rule": (
            "recall-first among thresholds that flag <=90% of orders and keep "
            "precision >= 0.47 (near prevalence); 0.5 is not the default"
        ),
        "business_framing": (
            "Recall is more valuable than precision: the system should catch orders "
            "that will be returned so operations can intervene before fulfillment. "
            "False alarms cost review effort; missed returns cost reverse logistics. "
            "Threshold 0.3 flags almost every order so it is not useful. "
            "0.4 roughly doubles recall vs 0.5 while precision stays near the base rate."
        ),
    }


def plot_threshold_curves(y_test, proba: np.ndarray, path: Path, recommended: float) -> None:
    grid = np.linspace(0.05, 0.95, 37)
    prec, rec = [], []
    for t in grid:
        pred = (proba >= t).astype(int)
        prec.append(precision_score(y_test, pred, zero_division=0))
        rec.append(recall_score(y_test, pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(grid, prec, color="#2563eb", lw=2, label="Precision")
    ax.plot(grid, rec, color="#c2410c", lw=2, label="Recall")
    ax.axvline(0.5, color="#94a3b8", ls="--", lw=1, label="default 0.5")
    ax.axvline(recommended, color="#15803d", ls=":", lw=2, label=f"recommended {recommended:.1f}")
    ax.set_xlabel("Threshold on P(returned=1)")
    ax.set_ylabel("Score")
    ax.set_title("Precision and recall vs threshold (best model by test ROC-AUC)")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def calibration_report(y_test, probas: dict, path: Path) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    summary = {}
    for ax, name in zip(axes, PLOT_ORDER):
        p = np.clip(probas[name], 1e-6, 1 - 1e-6)
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        brier = float(brier_score_loss(y_test, p))
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        # Overconfident if high-probability bins sit below the diagonal.
        high = mean_pred >= 0.55
        overconfident = bool(high.any() and np.mean(mean_pred[high] - frac_pos[high]) > 0.03)
        if ece < 0.03 and not overconfident:
            flag = "well_calibrated"
        elif overconfident:
            flag = "overconfident"
        else:
            flag = "miscalibrated"
        summary[name] = {
            "brier": brier,
            "ece_quantile_bins": ece,
            "flag": flag,
            "overconfident": overconfident,
        }
        ax.plot([0, 1], [0, 1], ls="--", c="#94a3b8", lw=1)
        ax.plot(mean_pred, frac_pos, marker="o", color=COLORS[name], lw=2)
        ax.set_title(f"{name}\nBrier={brier:.3f} ECE~{ece:.3f} ({flag})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[2].set_xlabel("Predicted probability")
    axes[3].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Actual return rate")
    axes[2].set_ylabel("Actual return rate")
    fig.suptitle("Calibration (quantile bins)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary


def error_analysis(y_test, proba: np.ndarray, test_frame: pd.DataFrame, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    y = np.asarray(y_test)
    fp_mask = (pred == 1) & (y == 0)
    fn_mask = (pred == 0) & (y == 1)
    rng = np.random.default_rng(RANDOM_STATE)

    def _sample(mask, n=10):
        idx = np.flatnonzero(mask)
        take = rng.choice(idx, size=min(n, len(idx)), replace=False)
        take.sort()
        out = test_frame.iloc[take].copy()
        out.insert(0, "test_row", take)
        out.insert(1, "pred_proba", proba[take])
        return out, idx

    fp_sample, fp_idx = _sample(fp_mask)
    fn_sample, fn_idx = _sample(fn_mask)

    def _share(frame, col):
        return frame[col].value_counts(normalize=True).round(4).to_dict()

    positives = test_frame[y == 1]
    negatives = test_frame[y == 0]
    fn_all = test_frame.iloc[fn_idx]
    fp_all = test_frame.iloc[fp_idx]

    patterns = {
        "n_false_positives": int(fp_mask.sum()),
        "n_false_negatives": int(fn_mask.sum()),
        "fn_vs_all_positives_category": {
            "false_negatives": _share(fn_all, "product_category"),
            "all_returns": _share(positives, "product_category"),
        },
        "fn_vs_all_positives_shipping": {
            "false_negatives": _share(fn_all, "shipping_method"),
            "all_returns": _share(positives, "shipping_method"),
        },
        "fn_mean_product_price": float(fn_all["product_price"].mean()),
        "all_returns_mean_product_price": float(positives["product_price"].mean()),
        "fp_vs_all_negatives_category": {
            "false_positives": _share(fp_all, "product_category"),
            "all_non_returns": _share(negatives, "product_category"),
        },
        "fp_coupon_rate": float(fp_all["used_coupon"].mean()),
        "non_return_coupon_rate": float(negatives["used_coupon"].mean()),
        "fn_coupon_rate": float(fn_all["used_coupon"].mean()),
        "return_coupon_rate": float(positives["used_coupon"].mean()),
    }
    return {
        "fp_sample": fp_sample,
        "fn_sample": fn_sample,
        "patterns": patterns,
    }


def run_evaluation(
    models_dir: Path = MODELS_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    models = load_models(models_dir)
    X_test, y_test, test_frame = load_test_set(processed_dir)
    probas = predict_probas(models, X_test)
    best_name = best_model_from_comparison()

    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    aucs = plot_roc(y_test, probas, fig_dir / ROC_PATH.name)
    aps = plot_pr(y_test, probas, fig_dir / PR_PATH.name)
    cal = calibration_report(y_test, probas, fig_dir / CAL_PATH.name)

    best_proba = probas[best_name]
    thr = threshold_table(y_test, best_proba)
    rec = recommend_threshold(thr)
    plot_threshold_curves(y_test, best_proba, fig_dir / THR_PATH.name, rec["recommended_threshold"])
    errors = error_analysis(y_test, best_proba, test_frame, threshold=0.5)

    thr.to_csv(reports_dir / "threshold_analysis.csv", index=False)
    errors["fp_sample"].to_csv(reports_dir / "error_analysis_fp.csv", index=False)
    errors["fn_sample"].to_csv(reports_dir / "error_analysis_fn.csv", index=False)

    summary = {
        "best_model_by_test_roc_auc": best_name,
        "test_roc_auc": aucs,
        "average_precision": aps,
        "calibration": cal,
        "threshold_analysis": thr.to_dict(orient="records"),
        "threshold_recommendation": rec,
        "error_patterns_at_0.5": errors["patterns"],
        "note": "Phase 5 only. No hyperparameter tuning (Phase 6) and no final_model.pkl (Phase 7).",
    }
    (reports_dir / "evaluation_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8"
    )

    print("Phase 5 complete (no retrain / no final-model pick).")
    print(f"  best by test ROC-AUC (from {COMPARISON_CSV.name}): {best_name}")
    print("  ROC-AUC", {k: round(v, 4) for k, v in aucs.items()})
    print("  AP     ", {k: round(v, 4) for k, v in aps.items()})
    print(thr.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("  recommend threshold", rec["recommended_threshold"], rec["rule"])
    print("  calibration flags", {k: v["flag"] for k, v in cal.items()})
    print(f"  figures -> {ROC_PATH.name}, {PR_PATH.name}, {CAL_PATH.name}, {THR_PATH.name}")
    return {
        "probas": probas,
        "y_test": y_test,
        "test_frame": test_frame,
        "aucs": aucs,
        "aps": aps,
        "calibration": cal,
        "threshold_table": thr,
        "recommendation": rec,
        "best_model": best_name,
        "errors": errors,
        "summary": summary,
    }


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
