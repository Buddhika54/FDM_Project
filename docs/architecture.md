# System architecture

This document describes the **clean architecture** of the E-Commerce Product Return Risk Prediction System. Implementation belongs in the named modules; this file defines boundaries and dependencies.

## Goals

- Keep HTTP, business logic, and ML artifacts independent.
- Train once in `ml/`; serve only in `backend/`.
- Keep the React UI ignorant of sklearn internals; it talks to a JSON API.
- Make each file’s job obvious (see `file-responsibilities.md`).

## High-level context

```
┌─────────────┐     JSON      ┌─────────────┐     joblib      ┌─────────────┐
│  frontend   │ ────────────► │   backend   │ ◄───────────── │   models/   │
│  (React)    │ ◄──────────── │   (Flask)   │                │  *.pkl      │
└─────────────┘               └─────────────┘                └─────────────┘
                                     ▲
                                     │ same feature contract
                                     │
                              ┌─────────────┐     CSV      ┌─────────────┐
                              │     ml/     │ ◄─────────── │  dataset/   │
                              │  training   │              └─────────────┘
                              └─────────────┘
```

## Frontend layers (`frontend/src`)

| Layer | Folder | May depend on | Must not |
|-------|--------|---------------|----------|
| Pages | `pages/` | components, hooks, router | Axios URLs, Flask details |
| Components | `components/` | other UI, props | API calls (prefer hooks/services) |
| Hooks | `hooks/` | services | JSX layout |
| Services | `services/` | Axios client, env base URL | React components |
| Assets | `assets/` | — | Logic |

**Routing:** `App.jsx` + React Router. Shared chrome in `components/layout/`.

**Data flow (prediction):**  
`PredictionPage` → form component → `usePrediction` → `predictionService` → `apiClient` → Flask → navigate to `ResultsPage` with result payload (location state or a small store).

## Backend layers (`backend/`)

Dependency rule: **outer layers depend inward**. Routes know services; services know loaders/validators; nothing imports Flask from `ml/`.

```
routes/          HTTP: parse request, map status codes, return JSON
    ↓
services/        Use-cases: predict_return_risk(), load artifacts once
    ↓
models/          Request/response schemas and field lists (not .pkl files)
    ↓
utils/           Validation, exceptions, response helpers
```

| Layer | Responsibility |
|-------|----------------|
| `app.py` | App factory, CORS, blueprint registration, process entry |
| `config.py` | Paths, host/port, model filenames |
| `routes/` | `/api/health`, `/api/predict` |
| `services/model_loader.py` | Load `preprocessor.pkl` and `final_model.pkl` once |
| `services/prediction_service.py` | Transform → predict → risk band → recommendation text |
| `models/prediction_schema.py` | Feature names, types, allowed categoricals |
| `utils/validators.py` | Reject incomplete/invalid payloads before inference |

Root `models/` holds **serialized sklearn objects**. Backend `models/` holds **Python schemas**. Do not mix them.

## Machine learning pipeline (`ml/`)

Scripts are ordered; notebooks in `notebooks/` are the exploratory twin, not the source of production artifacts.

| Stage | Folder | Output |
|-------|--------|--------|
| EDA | `ml/eda/` | Figures + notes under `reports/` |
| Preprocessing | `ml/preprocessing/` | Clean tables in `dataset/processed/` |
| Feature engineering | `ml/feature_engineering/` | Feature matrix + transformer steps |
| Training | `ml/training/` | Candidate models |
| Evaluation | `ml/evaluation/` | Metrics tables / plots in `reports/` |
| Tuning | `ml/tuning/` | Best hyperparameters |
| Saving | `ml/saving/` | `models/preprocessor.pkl`, `models/final_model.pkl` |

**Invariant:** whatever `preprocessor.pkl` does at train time must be the only transform used in `prediction_service.py`.

## Dataset, reports, presentation, docs

| Path | Role |
|------|------|
| `dataset/raw/` | Immutable source CSV |
| `dataset/processed/` | Train/test splits and encoded tables |
| `reports/` | Metrics, plots, written analysis |
| `presentation/` | Slide deck and demo script |
| `docs/` | Architecture and contracts (this folder) |

## Technology mapping

- **React + Vite:** SPA, fast HMR during UI work.
- **Tailwind CSS:** Utility styling for a responsive dashboard.
- **React Router:** Home / Predict / Results.
- **Axios:** Single HTTP client with base URL from env.
- **Flask + CORS:** REST for the SPA origin.
- **Pandas / NumPy:** Tabular prep.
- **scikit-learn:** Pipelines, LR, trees, forests, GridSearchCV, metrics.
- **XGBoost:** Gradient-boosted trees.
- **Joblib:** Persist preprocessor and final estimator.

## Design principles applied

1. **Single responsibility** — one concern per module (route vs predict vs validate vs train).
2. **Separation of train and serve** — no training inside request handlers.
3. **Explicit contracts** — feature list and JSON shapes documented in `api-specification.md`.
4. **Reproducible artifacts** — named pickle files; pipeline script order in `ml/run_pipeline.py`.
5. **UI isolation** — presentation components do not know sklearn class names.
