# E-Commerce Product Return Risk Prediction System

A university Data Mining mini project that predicts whether an online order is likely to be returned **before fulfillment**. The system helps e-commerce teams reduce return-related costs and make better operational decisions (fulfillment priority, packaging, customer communication, and coupon policy).

| Item | Detail |
|------|--------|
| Domain | E-commerce operations / Data Mining |
| Task | Binary classification (`returned`: 0 = Not Returned, 1 = Returned) |
| Frontend | React (Vite), Tailwind CSS, React Router, Axios |
| Backend | Flask, Flask-CORS |
| ML | Logistic Regression, Decision Tree, Random Forest, XGBoost + GridSearchCV |
| Artifacts | `models/final_model.pkl`, `models/preprocessor.pkl` |

This repository is a **professional architecture scaffold**. Folders, file names, contracts, and a development roadmap are in place. Application and training logic are intentionally left as stubs for students to implement.

---

## Project purpose

Returned orders create reverse-logistics cost, inventory friction, and lost margin. This system estimates **return probability** from order, customer, product, and session features so operations can:

- Flag high-risk orders before they ship
- Apply targeted interventions (confirmation, packaging, delivery option)
- Compare model families and justify a production choice with evaluation metrics

**Target variable:** `returned`  
- `0` — Not Returned  
- `1` — Returned  

**Input features:** `customer_age`, `product_price`, `discount_percent`, `product_rating`, `past_purchase_count`, `past_return_rate`, `delivery_delay_days`, `session_length_minutes`, `num_product_views`, `device_type`, `product_category`, `shipping_method`, `payment_method`, `used_coupon`  
**Not features:** `order_id` (drop), `returned` (target)

---

## Repository layout

```
project-root/
├── frontend/          React (Vite) dashboard
├── backend/           Flask REST API
├── ml/                Training pipeline scripts (EDA → save)
├── dataset/           Raw and processed data
├── notebooks/         Exploratory Jupyter notebooks
├── models/            Trained model + preprocessor artifacts
├── reports/           Evaluation write-ups and figures
├── presentation/      Slides / demo materials
├── docs/              Architecture, API, roadmap, file map
├── requirements.txt   Python dependencies (API + ML)
└── README.md
```

Clean-architecture notes: [docs/architecture.md](docs/architecture.md)  
File-by-file responsibilities: [docs/file-responsibilities.md](docs/file-responsibilities.md)  
Implementation roadmap: [docs/development-roadmap.md](docs/development-roadmap.md)  
API contract: [docs/api-specification.md](docs/api-specification.md)

---

## Prerequisites

- Python 3.10+ recommended
- Node.js 18+ and npm
- Git (optional)

---

## Installation

### 1. Clone / open the project

```bash
cd FDM_Project
```

### 2. Python environment (backend + ML)

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Backend-only install:

```bash
pip install -r backend/requirements.txt
```

ML-pipeline-only install:

```bash
pip install -r ml/requirements.txt
```

### 3. Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Environment files

Copy examples and adjust if ports differ:

- `backend/.env.example` → `backend/.env`
- `frontend/.env.example` → `frontend/.env`

---

## Running the backend

From the **project root** (so `models/` paths resolve as configured):

```bash
# with venv active
python backend/app.py
```

Default: `http://127.0.0.1:5000`

Health check (once implemented): `GET /api/health`  
Prediction: `POST /api/predict`

The API is expected to:

1. Validate JSON input against the feature contract
2. Load `models/preprocessor.pkl` and `models/final_model.pkl`
3. Transform features and predict class + probability
4. Return JSON: prediction, risk level, probability, recommendation

See [docs/api-specification.md](docs/api-specification.md).

---

## Running the frontend

```bash
cd frontend
npm run dev
```

Default Vite URL: `http://127.0.0.1:5173`

Planned routes:

| Path | Page | Responsibility |
|------|------|----------------|
| `/` | Home | Project overview and business context |
| `/predict` | Prediction | Form for all model features |
| `/results` | Results | Prediction, risk, probability, recommendation |
| `*` | Not Found | Unknown routes |

Build for production:

```bash
cd frontend
npm run build
npm run preview
```

---

## Model training workflow

Training lives in `ml/` (reproducible scripts) and is mirrored by notebooks in `notebooks/` for exploration.

**Recommended order:**

1. Place the dataset in `dataset/raw/` (CSV). Document columns in `dataset/README.md`.
2. **EDA** — `ml/eda/exploratory_analysis.py` and `notebooks/01_eda.ipynb`  
   Distributions, missing values, class balance, categorical vs numeric split.
3. **Preprocessing** — `ml/preprocessing/`  
   Cleaning, encoding, scaling; fit a scikit-learn `ColumnTransformer` / `Pipeline`.
4. **Feature engineering** — `ml/feature_engineering/feature_builder.py`  
   Derived features (if any) applied consistently at train and inference time.
5. **Train baselines** — `ml/training/`  
   Logistic Regression, Decision Tree, Random Forest, XGBoost on the same split.
6. **Evaluate** — `ml/evaluation/`  
   Accuracy, precision, recall, F1, ROC-AUC, confusion matrix; write `reports/`.
7. **Tune** — `ml/tuning/grid_search.py`  
   `GridSearchCV` on the selected model(s); persist best params.
8. **Save artifacts** — `ml/saving/save_artifacts.py`  
   Write `models/preprocessor.pkl` and `models/final_model.pkl` (joblib).
9. **Serve** — Flask loads those two files only; it must not retrain on request.

Orchestrator stub: `ml/run_pipeline.py` (wire the steps after each module is implemented).

Do not commit large `.pkl` files unless the course requires it; keep `models/README.md` as the artifact contract.

---

## Frontend / backend contract (summary)

```
Browser  →  Axios service layer  →  POST /api/predict
                                      ↓
                                 validators
                                      ↓
                          preprocessor.transform
                                      ↓
                              model.predict / predict_proba
                                      ↓
                                 JSON response
```

---

## Development status

Scaffold only: folder structure, named modules, configuration, and documentation. Implement modules following [docs/development-roadmap.md](docs/development-roadmap.md).

---

## Academic notes

Use this layout for the mini-project report and demo: business problem → EDA → models → evaluation → deployed prediction UI. Keep preprocessing identical between training (`ml/`) and inference (`backend/services/`).
