# Machine learning pipeline

Production-oriented **scripts** for the return-risk classifier. Use `notebooks/` for exploration; persist artifacts only from this package (or an equivalent notebook that writes the same paths).

## Stages

| Order | Package | Script | Role |
|------:|---------|--------|------|
| 1 | `eda/` | `exploratory_analysis.py` | Profiles, plots, class balance |
| 2 | `preprocessing/` | `data_cleaner.py`, `preprocessor_pipeline.py` | Clean + sklearn pipeline |
| 3 | `feature_engineering/` | `feature_builder.py` | Derived features |
| 4 | `training/` | `train_*.py`, `train_all_models.py` | LR, DT, RF, XGBoost |
| 5 | `evaluation/` | `metrics.py`, `evaluate_models.py` | Compare models |
| 6 | `tuning/` | `grid_search.py` | GridSearchCV |
| 7 | `saving/` | `save_artifacts.py` | `preprocessor.pkl`, `final_model.pkl` |

Orchestrator: `run_pipeline.py` (wire after each stage exists). Paths and seed: `config.py`.

## Run (after implementation)

From project root, with venv active:

```bash
pip install -r ml/requirements.txt
python -m ml.run_pipeline
```

Or run a single stage as a module once `__main__` blocks are added.

## Contract with the API

The Flask service must load:

- `models/preprocessor.pkl`
- `models/final_model.pkl`

Column names and encodings must match `docs/data-dictionary.md`.
