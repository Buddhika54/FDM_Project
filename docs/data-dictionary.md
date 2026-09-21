# Data dictionary

Canonical feature contract for training, the Flask API, and the React form.

**Source:** `dataset/raw/train.csv`  
**Rows / columns:** 200,000 × 16  
**Missing values:** none  
**Target balance:** `returned = 0` ≈ 52.5% (105,081), `returned = 1` ≈ 47.5% (94,919) — **no class-weighting required**

Cleaning rules below are **decisions only**. Apply them in the preprocessing phase; do not mutate `dataset/raw/`.

Observed min/max are from the raw file (before any fix).

---

## Column catalog

| Column | Type | Valid range | Cleaning decision | Rationale |
|--------|------|-------------|-------------------|-----------|
| `customer_age` | int | 10–87 (years) | **Keep** | All values are non-negative integers in a plausible shopper-age band; no invalid sign or type issues. |
| `product_price` | float | ≥ 0 | **Fix** — `abs()` or clip to 0 (27,285 negatives) | Price cannot be negative; those rows are data errors, not a real “refund” or “credit” signal. |
| `discount_percent` | float | 0–100 | **Fix** — `abs()` or clip to 0 (1,686 negatives) | A percent off cannot be negative; the column is a magnitude (observed max ≈ 78.8), not a signed residual. |
| `product_rating` | float | ≈ 0.45–5.45 (observed) | **Keep** | No invalid-sign problem was reported; slight overshoot of a 1–5 scale is left as-is until a later rating policy is chosen. |
| `past_purchase_count` | int | ≥ 1 (observed 1–26) | **Keep** | Counts are already non-negative integers with a sensible history range. |
| `past_return_rate` | float | 0–1 | **Fix** — `abs()` or clip to 0 (2,220 negatives) | A rate is a proportion; negatives are impossible and would distort “how often this customer returned before.” Observed max ≈ 0.87, so the upper bound is already valid. |
| `delivery_delay_days` | float | signed (observed ≈ −9.05 to 7.96) | **Keep as signal** — do **not** abs/clip | Negative delay means the order arrived **early** (ahead of the promised date). Early vs late delivery is a real pre-fulfillment / logistics signal for return risk; folding it to absolute days would erase that direction. See note below. |
| `session_length_minutes` | float | ≥ 0 | **Fix** — `abs()` or clip to 0 (1,487 negatives) | Session duration cannot be negative; sign errors are noise, not “short sessions.” |
| `num_product_views` | int | ≥ 0 | **Fix** — `abs()` or clip to 0 (21,443 negatives) | View counts cannot be negative; treat as magnitude after correcting the sign. |
| `device_type` | categorical (str) | `mobile`, `desktop`, `tablet` | **Keep** | Closed set; encode later (fit on train only). |
| `product_category` | categorical (str) | `toys`, `beauty`, `electronics`, `home`, `clothing`, `sports` | **Keep** | Closed set; encode later (fit on train only). |
| `shipping_method` | categorical (str) | `standard`, `express`, `same_day` | **Keep** | Closed set; encode later (fit on train only). |
| `payment_method` | categorical (str) | `debit_card`, `credit_card`, `apple_pay`, `paypal` | **Keep** | Closed set; encode later (fit on train only). |
| `used_coupon` | binary int | `0`, `1` | **Keep** | Already encoded; coupon use is known at order time. |
| `returned` | binary int | `0`, `1` | **Target only** — never send on `/api/predict` | `0` = Not Returned, `1` = Returned. Supervised label; not a predictor. |
| `order_id` | int | identifier (0–249,999 observed) | **Drop before modeling** | Unique/row id only; no generalization value and would leak identity if used as a feature. |

---

## Why `delivery_delay_days` negatives are kept

`delivery_delay_days` is a **signed residual**: promised arrival minus actual (or equivalent “days late”).

- **Positive** → late (customer waited longer than expected).
- **Zero** → on time.
- **Negative** → **early** (arrived before the promised window).

Unlike `product_price` or `session_length_minutes`, the minus sign is **not a data-entry glitch**. About 39% of rows are negative (78,564 / 200,000), which is consistent with a large share of early deliveries, not with rare corruption.

Taking `abs()` would map “2 days early” and “2 days late” to the same number and hide a business-relevant pattern (early delivery can change satisfaction and return behaviour). **Keep the signed value** as a numeric feature; scale it later without forcing non-negativity.

---

## Predictors (API + form)

These 14 fields are the only inputs to the model and to `POST /api/predict`. Exact names:

`customer_age`, `product_price`, `discount_percent`, `product_rating`, `past_purchase_count`, `past_return_rate`, `delivery_delay_days`, `session_length_minutes`, `num_product_views`, `device_type`, `product_category`, `shipping_method`, `payment_method`, `used_coupon`

**Excluded from the prediction schema**

| Column | Why |
|--------|-----|
| `order_id` | Identifier, not a feature |
| `returned` | Label |

---

## Category value inventory

Exact strings (match `frontend/src/constants/featureFields.js` and `backend/models/prediction_schema.py`):

| Column | Allowed values |
|--------|----------------|
| `device_type` | `mobile`, `desktop`, `tablet` |
| `product_category` | `toys`, `beauty`, `electronics`, `home`, `clothing`, `sports` |
| `shipping_method` | `standard`, `express`, `same_day` |
| `payment_method` | `debit_card`, `credit_card`, `apple_pay`, `paypal` |

---

## Leakage checklist

- Drop `order_id` before fit/transform.
- Never use `returned` as an input feature.
- Do not add post-return fields (return reason, refund amount, reverse-logistics flags) if they appear later.
- `past_return_rate` and `past_purchase_count` are valid **history** features if they exclude the current order.
- `delivery_delay_days` is kept as a signed operational signal; confirm in modeling notes whether the production system knows this value **before fulfillment** (promised vs typical delay vs realized delay). If only realized delay exists after delivery, document that limitation in the report.
