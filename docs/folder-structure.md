# Full folder structure

```
FDM_Project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── README.md
│   ├── architecture.md
│   ├── file-responsibilities.md
│   ├── development-roadmap.md
│   ├── api-specification.md
│   ├── data-dictionary.md
│   └── folder-structure.md
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── index.html
│   ├── .env.example
│   ├── public/
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css
│       ├── constants/
│       │   └── featureFields.js
│       ├── pages/
│       │   ├── HomePage.jsx
│       │   ├── PredictionPage.jsx
│       │   ├── ResultsPage.jsx
│       │   └── NotFoundPage.jsx
│       ├── components/
│       │   ├── layout/
│       │   │   ├── Layout.jsx
│       │   │   ├── Navbar.jsx
│       │   │   └── Footer.jsx
│       │   ├── forms/
│       │   │   ├── PredictionForm.jsx
│       │   │   └── FormField.jsx
│       │   ├── results/
│       │   │   ├── PredictionCard.jsx
│       │   │   ├── RiskBadge.jsx
│       │   │   └── RecommendationPanel.jsx
│       │   └── ui/
│       │       ├── Button.jsx
│       │       ├── Card.jsx
│       │       └── Loader.jsx
│       ├── services/
│       │   ├── apiClient.js
│       │   └── predictionService.js
│       ├── hooks/
│       │   ├── usePrediction.js
│       │   └── usePredictionForm.js
│       └── assets/
│
├── backend/
│   ├── app.py
│   ├── wsgi.py
│   ├── config.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── routes/
│   │   ├── health_routes.py
│   │   └── predict_routes.py
│   ├── services/
│   │   ├── model_loader.py
│   │   └── prediction_service.py
│   ├── models/
│   │   └── prediction_schema.py
│   └── utils/
│       ├── validators.py
│       ├── exceptions.py
│       └── response_builder.py
│
├── ml/
│   ├── README.md
│   ├── requirements.txt
│   ├── config.py
│   ├── run_pipeline.py
│   ├── eda/
│   │   └── exploratory_analysis.py
│   ├── preprocessing/
│   │   ├── data_cleaner.py
│   │   └── preprocessor_pipeline.py
│   ├── feature_engineering/
│   │   └── feature_builder.py
│   ├── training/
│   │   ├── train_logistic_regression.py
│   │   ├── train_decision_tree.py
│   │   ├── train_random_forest.py
│   │   ├── train_xgboost.py
│   │   └── train_all_models.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate_models.py
│   ├── tuning/
│   │   └── grid_search.py
│   └── saving/
│       └── save_artifacts.py
│
├── dataset/
│   ├── README.md
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── README.md
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│
├── models/
│   └── README.md                 # preprocessor.pkl, final_model.pkl (after training)
│
├── reports/
│   ├── README.md
│   ├── figures/
│   └── templates/
│       ├── model_comparison.md
│       └── final_report_outline.md
│
└── presentation/
    └── README.md
```

`ml/` holds training scripts (required by the ML workflow). It complements `notebooks/` and is not a replacement for `frontend/` or `backend/`.
