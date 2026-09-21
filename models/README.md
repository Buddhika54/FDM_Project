# Serialized models

The Flask API (Phase 8) loads this pair (joblib) plus the threshold config:

| File | Produced by | Role |
|------|-------------|------|
| `preprocessor.pkl` | Phase 2 `ml/preprocess.py` | Clean + scale + one-hot |
| `final_model.pkl` | Phase 7 `ml/select_model.py` | Tuned XGBoost estimator |
| `final_model_config.json` | Phase 7 | `threshold`, metrics, params |

Do not refit either pickle. Label `returned` if `predict_proba[:, 1] >= threshold`.

Binaries are gitignored; keep this README in git.
