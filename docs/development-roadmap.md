# Development roadmap

Phased plan for turning this scaffold into a working mini project. Complete phases in order unless noted. Do not skip the feature contract (Phase 0) — frontend, backend, and training must share it.

---

## Phase 0 — Contracts and data (foundation)

**Outcome:** Everyone agrees on columns, JSON keys, and folder usage.

- [ ] Place dataset CSV in `dataset/raw/` and describe it in `dataset/README.md`.
- [ ] Fill `docs/data-dictionary.md` with types, ranges, and category values.
- [ ] Align `frontend/src/constants/featureFields.js` and `backend/models/prediction_schema.py` with the dictionary.
- [ ] Confirm target `returned` ∈ {0, 1}.

**Exit:** A sample row can be written as both a CSV record and a `/api/predict` JSON body.

---

## Phase 1 — EDA

**Outcome:** Understand data quality and class balance.

- [ ] Implement `ml/eda/exploratory_analysis.py`.
- [ ] Complete `notebooks/01_eda.ipynb` (plots belong in `reports/` or notebook outputs).
- [ ] Document missing values, outliers, leakage risks (e.g. post-return fields).

**Exit:** Short EDA notes in `reports/` (class ratio, key correlations).

---

## Phase 2 — Preprocessing and feature engineering

**Outcome:** A fitted sklearn preprocessor that can be pickled.

- [ ] `data_cleaner.py` — types, null strategy, duplicate handling.
- [ ] `feature_builder.py` — only features available **before fulfillment**.
- [ ] `preprocessor_pipeline.py` — numeric scale + categorical encode in one `Pipeline` / `ColumnTransformer`.
- [ ] Write processed tables to `dataset/processed/` (train/test split, fixed `random_state` in `ml/config.py`).
- [ ] Mirror experiments in `notebooks/02_preprocessing.ipynb` and `03_feature_engineering.ipynb`.

**Exit:** `preprocessor.fit(X_train)` works; transform shapes match.

---

## Phase 3 — Model training

**Outcome:** Four comparable estimators on the same split.

- [ ] Implement the four trainers under `ml/training/`.
- [ ] `train_all_models.py` fits all and stores candidates (memory or temp files).
- [ ] Use class imbalance strategy if needed (class_weight / scale_pos_weight) — document the choice.
- [ ] Notebook: `04_model_training.ipynb`.

**Exit:** Four fitted models and training logs.

---

## Phase 4 — Evaluation

**Outcome:** Justified model ranking.

- [ ] `metrics.py` — accuracy, precision, recall, F1, ROC-AUC.
- [ ] `evaluate_models.py` — confusion matrices, ROC curves → `reports/`.
- [ ] Fill `reports/templates/model_comparison.md`.
- [ ] Notebook: `05_model_evaluation.ipynb`.

**Exit:** Written comparison; preliminary “best” model named.

---

## Phase 5 — Hyperparameter tuning

**Outcome:** Best params via `GridSearchCV`.

- [ ] Define grids in `ml/tuning/grid_search.py` (keep grids small for class machines).
- [ ] Tune the top 1–2 models from Phase 4; scoring e.g. `f1` or `roc_auc`.
- [ ] Notebook: `06_hyperparameter_tuning.ipynb`.
- [ ] Record best params in `reports/`.

**Exit:** Final estimator selected.

---

## Phase 6 — Persist artifacts

**Outcome:** Backend can load files from `models/`.

- [ ] `saving/save_artifacts.py` dumps:
  - `models/preprocessor.pkl`
  - `models/final_model.pkl`
- [ ] Smoke-test: load both with joblib in a one-off script; `predict` on one row.
- [ ] Wire `ml/run_pipeline.py` if you want a single entry command.

**Exit:** Pickles exist and round-trip.

---

## Phase 7 — Flask API

**Outcome:** Valid JSON in → prediction JSON out.

- [ ] `model_loader.py` loads pickles at startup (fail clearly if missing).
- [ ] `validators.py` + `prediction_schema.py` reject bad payloads.
- [ ] `prediction_service.py`: DataFrame → transform → `predict` / `predict_proba`.
- [ ] Map probability to risk level (document thresholds, e.g. Low < 0.35, High ≥ 0.65).
- [ ] Map risk to a **business recommendation** string.
- [ ] `POST /api/predict` and `GET /api/health` per `api-specification.md`.
- [ ] CORS allows the Vite origin.

**Exit:** `curl`/Postman success against a sample payload.

---

## Phase 8 — React dashboard

**Outcome:** Responsive UI for overview, form, and results.

- [ ] Layout: `Navbar`, `Footer`, `Layout`.
- [ ] `HomePage` — problem, objectives, how it helps operations.
- [ ] `PredictionForm` — all features; client-side required checks.
- [ ] Axios: `apiClient` + `predictionService`.
- [ ] `usePrediction` — submit, errors, redirect to `/results`.
- [ ] `ResultsPage` — prediction, risk, probability %, recommendation.
- [ ] Tailwind: usable on mobile and desktop.

**Exit:** End-to-end demo: form → API → results.

---

## Phase 9 — Academic wrap-up

**Outcome:** Report and presentation.

- [ ] `reports/templates/final_report_outline.md` → actual report in `reports/`.
- [ ] Slides in `presentation/` (see presentation README).
- [ ] Demo script: one low-risk and one high-risk example.

**Exit:** Ready for viva / submission.

---

## Suggested order of coding (quick)

1. Schema + sample JSON  
2. ML pipeline through pickle  
3. `/predict`  
4. UI form + results  
5. Report  

## Out of scope for the scaffold

Full training loops, GridSearch grids, styled production UI, authentication, databases, and real-time streaming are **not** included on purpose. Add them in the phases above.
