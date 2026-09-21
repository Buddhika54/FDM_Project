# File responsibilities

Map of intended files. Stubs exist; fill in logic during implementation (see `development-roadmap.md`).

---

## Root

| File | Responsibility |
|------|----------------|
| `README.md` | Purpose, install, run frontend/backend, training workflow |
| `requirements.txt` | Combined Python deps for API + ML |
| `.gitignore` | Ignore venv, node_modules, env, large data/pkl |

---

## `docs/`

| File | Responsibility |
|------|----------------|
| `architecture.md` | Layers, dependency rules, tech mapping |
| `file-responsibilities.md` | This catalog |
| `development-roadmap.md` | Phased implementation plan |
| `api-specification.md` | REST contract for `/predict` and `/health` |
| `data-dictionary.md` | Feature definitions and encodings |

---

## `frontend/`

| File | Responsibility |
|------|----------------|
| `package.json` | npm scripts and UI dependencies |
| `vite.config.js` | Vite + React plugin |
| `tailwind.config.js` | Tailwind content paths and theme tokens |
| `postcss.config.js` | PostCSS plugins (Tailwind, Autoprefixer) |
| `index.html` | SPA HTML shell |
| `.env.example` | `VITE_API_BASE_URL` |
| `public/` | Static files copied as-is |

### `frontend/src`

| File | Responsibility |
|------|----------------|
| `main.jsx` | React mount point |
| `App.jsx` | Router and layout wrapper |
| `index.css` | Tailwind directives and global tokens |
| `constants/featureFields.js` | Form field metadata matching the ML schema |

### Pages

| File | Responsibility |
|------|----------------|
| `pages/HomePage.jsx` | Project overview, problem, stack, CTA to predict |
| `pages/PredictionPage.jsx` | Hosts the full feature form |
| `pages/ResultsPage.jsx` | Prediction, risk, probability, recommendation |
| `pages/NotFoundPage.jsx` | Unknown routes |

### Components

| File | Responsibility |
|------|----------------|
| `components/layout/Navbar.jsx` | Global navigation |
| `components/layout/Footer.jsx` | Footer |
| `components/layout/Layout.jsx` | Shell: navbar + outlet + footer |
| `components/forms/PredictionForm.jsx` | All model feature inputs |
| `components/forms/FormField.jsx` | Reusable labeled input/select |
| `components/results/PredictionCard.jsx` | Predicted class display |
| `components/results/RiskBadge.jsx` | Low / Medium / High risk chip |
| `components/results/RecommendationPanel.jsx` | Business recommendation copy |
| `components/ui/Button.jsx` | Shared button |
| `components/ui/Card.jsx` | Shared card container |
| `components/ui/Loader.jsx` | Loading state |

### Services and hooks

| File | Responsibility |
|------|----------------|
| `services/apiClient.js` | Axios instance (base URL, JSON headers) |
| `services/predictionService.js` | `predictReturnRisk(payload)` |
| `hooks/usePrediction.js` | Call API, loading/error, navigate to results |
| `hooks/usePredictionForm.js` | Form state for all features |

### Assets

| File | Responsibility |
|------|----------------|
| `assets/README.md` | Where to put logos, diagrams, screenshots |
| `assets/.gitkeep` | Keep empty folder in git |

---

## `backend/`

| File | Responsibility |
|------|----------------|
| `app.py` | Flask factory, CORS, blueprints, `python backend/app.py` |
| `config.py` | PORT, MODEL_DIR, pickle filenames |
| `.env.example` | Backend env template |
| `requirements.txt` | Flask-side Python packages |
| `wsgi.py` | Optional WSGI entry (`create_app`) |

### Routes

| File | Responsibility |
|------|----------------|
| `routes/__init__.py` | Package marker |
| `routes/health_routes.py` | `GET /api/health` |
| `routes/predict_routes.py` | `POST /api/predict` |

### Services

| File | Responsibility |
|------|----------------|
| `services/__init__.py` | Package marker |
| `services/model_loader.py` | Load and cache preprocessor + model |
| `services/prediction_service.py` | Inference + risk + recommendation |

### Models (schemas, not pickles)

| File | Responsibility |
|------|----------------|
| `models/__init__.py` | Package marker |
| `models/prediction_schema.py` | Required fields, dtypes, category enums |

### Utils

| File | Responsibility |
|------|----------------|
| `utils/__init__.py` | Package marker |
| `utils/validators.py` | Input validation; raise domain errors |
| `utils/exceptions.py` | `ValidationError`, `ModelNotLoadedError` |
| `utils/response_builder.py` | Consistent JSON envelopes |

---

## `ml/`

| File | Responsibility |
|------|----------------|
| `README.md` | How to run the training pipeline |
| `requirements.txt` | sklearn, pandas, xgboost, matplotlib, jupyter |
| `config.py` | Paths to dataset, models, reports, random seed |
| `run_pipeline.py` | Orchestrate stages in order (after modules exist) |

### EDA

| File | Responsibility |
|------|----------------|
| `eda/exploratory_analysis.py` | Univariate/bivariate EDA, class balance, missingness |

### Preprocessing

| File | Responsibility |
|------|----------------|
| `preprocessing/data_cleaner.py` | Nulls, outliers, type coercion |
| `preprocessing/preprocessor_pipeline.py` | sklearn `ColumnTransformer` / `Pipeline` |

### Feature engineering

| File | Responsibility |
|------|----------------|
| `feature_engineering/feature_builder.py` | Derived columns shared by train and (if needed) docs |

### Training

| File | Responsibility |
|------|----------------|
| `training/train_logistic_regression.py` | Fit Logistic Regression |
| `training/train_decision_tree.py` | Fit Decision Tree |
| `training/train_random_forest.py` | Fit Random Forest |
| `training/train_xgboost.py` | Fit XGBoost |
| `training/train_all_models.py` | Train all four on the same split |

### Evaluation

| File | Responsibility |
|------|----------------|
| `evaluation/metrics.py` | Metric helpers |
| `evaluation/evaluate_models.py` | Compare models; write `reports/` |

### Tuning

| File | Responsibility |
|------|----------------|
| `tuning/grid_search.py` | `GridSearchCV` param grids and search |

### Saving

| File | Responsibility |
|------|----------------|
| `saving/save_artifacts.py` | joblib dump of preprocessor + final model |

---

## `dataset/`

| File | Responsibility |
|------|----------------|
| `README.md` | How to obtain/place the CSV; train/test convention |
| `raw/` | Original dataset (not overwritten) |
| `processed/` | Cleaned splits and intermediate tables |

---

## `notebooks/`

| File | Responsibility |
|------|----------------|
| `README.md` | Notebook vs `ml/` script usage |
| `01_eda.ipynb` | Exploratory analysis |
| `02_preprocessing.ipynb` | Cleaning and encoding experiments |
| `03_feature_engineering.ipynb` | Feature ideas |
| `04_model_training.ipynb` | Fit candidate models |
| `05_model_evaluation.ipynb` | Metrics and plots |
| `06_hyperparameter_tuning.ipynb` | Grid search experiments |

---

## `models/` (artifacts)

| File | Responsibility |
|------|----------------|
| `README.md` | Contract: `preprocessor.pkl`, `final_model.pkl` |
| `.gitkeep` | Preserve directory |

---

## `reports/`

| File | Responsibility |
|------|----------------|
| `README.md` | What to store (metrics, figures, write-up) |
| `templates/model_comparison.md` | Table template for four algorithms |
| `templates/final_report_outline.md` | Mini-project report outline |

---

## `presentation/`

| File | Responsibility |
|------|----------------|
| `README.md` | Slide outline and demo checklist |
