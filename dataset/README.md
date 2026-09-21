# Dataset

## Folders

| Path | Rule |
|------|------|
| `raw/` | Original CSV only. Do not overwrite after cleaning. |
| `processed/` | Train/test splits and intermediate tables from `ml/preprocessing`. |

## Raw file

`dataset/raw/train.csv`

- 200,000 rows, 16 columns, no missing values
- Target `returned` is balanced (~47.5% / 52.5%)

Columns:

```
customer_age, product_price, discount_percent, product_rating,
past_purchase_count, past_return_rate, delivery_delay_days,
session_length_minutes, num_product_views, device_type,
product_category, shipping_method, payment_method, used_coupon,
returned, order_id
```

- Predictors (14): everything except `returned` and `order_id`
- Target: `returned`
- Drop before modeling: `order_id`

Definitions and cleaning *decisions*: `docs/data-dictionary.md`.  
Do not apply cleaning in this folder (Phase 0 is contract only).

## Processed outputs (Phase 2)

Produced by `python -m ml.preprocess` (see `ml/preprocess.py`):

| File | Contents |
|------|----------|
| `train.csv` / `test.csv` | Cleaned features + `returned`; `order_id` dropped; categoricals still labels |
| `X_train.npy` / `X_test.npy` | Scaled numerics + one-hot categoricals + `used_coupon` |
| `y_train.npy` / `y_test.npy` | Target arrays |
| `feature_names.json` | Column names after `ColumnTransformer` |
| `preprocessing_log.json` | Clip counts, train p99 price cap, shapes |

Fitted transformer: `models/preprocessor.pkl` (clean + scale + one-hot). Raw `train.csv` is never overwritten.

Keep `random_state` from `ml/config.py` (42, 80/20, stratified on `returned`).
