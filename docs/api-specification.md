# API specification

Base URL (local): `http://127.0.0.1:5000`  
Frontend should set `VITE_API_BASE_URL` to that origin (no trailing slash).

All application routes are prefixed with `/api`.

---

## `GET /api/health`

Liveness and artifact status.

**Response `200`**

```json
{
  "status": "ok",
  "service": "return-risk-prediction",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

If pickles are missing, still return `200` with `model_loaded: false` **or** `503` — pick one and document it in `health_routes.py` when implementing.

---

## `POST /api/predict`

Predict return risk for a single order **before fulfillment**.

**Headers:** `Content-Type: application/json`

### Request body

Keys must match training feature names (snake_case).

```json
{
  "customer_age": 32,
  "product_price": 149.99,
  "discount_percent": 15.0,
  "product_rating": 4.2,
  "past_purchase_count": 8,
  "past_return_rate": 0.12,
  "delivery_delay_days": -1.5,
  "session_length_minutes": 12.5,
  "num_product_views": 7,
  "device_type": "mobile",
  "product_category": "electronics",
  "shipping_method": "express",
  "payment_method": "credit_card",
  "used_coupon": 1
}
```

Keys match `docs/data-dictionary.md` predictors only (`order_id` and `returned` are not accepted).

| Field | Type | Notes |
|-------|------|--------|
| `customer_age` | integer | Years |
| `product_price` | number | ≥ 0 after cleaning |
| `discount_percent` | number | 0–100 (percent, not a 0–1 rate) |
| `product_rating` | number | Observed ~0.45–5.45 |
| `past_purchase_count` | integer | Prior order count |
| `past_return_rate` | number | 0–1 |
| `delivery_delay_days` | number | **Signed**: negative = arrived early; do not abs() |
| `session_length_minutes` | number | ≥ 0 after cleaning |
| `num_product_views` | integer | ≥ 0 after cleaning |
| `device_type` | string | `mobile` \| `desktop` \| `tablet` |
| `product_category` | string | `toys` \| `beauty` \| `electronics` \| `home` \| `clothing` \| `sports` |
| `shipping_method` | string | `standard` \| `express` \| `same_day` |
| `payment_method` | string | `debit_card` \| `credit_card` \| `apple_pay` \| `paypal` |
| `used_coupon` | integer | `0` / `1` |

Exact category labels: `docs/data-dictionary.md`.

### Success `200`

```json
{
  "prediction": 1,
  "prediction_label": "Returned",
  "risk_level": "High",
  "probability": 0.82,
  "probability_percent": 82.0,
  "recommendation": "Hold fulfillment pending customer confirmation; avoid extra promotional shipping."
}
```

| Field | Meaning |
|-------|---------|
| `prediction` | `0` or `1` |
| `prediction_label` | `"Not Returned"` or `"Returned"` |
| `risk_level` | `"Low"` \| `"Medium"` \| `"High"` (thresholds in `prediction_service`) |
| `probability` | P(returned = 1), 0–1 |
| `probability_percent` | 0–100 for the UI |
| `recommendation` | Operations-facing sentence |

### Error `400` — validation

```json
{
  "error": "validation_error",
  "message": "Invalid or missing fields.",
  "details": {
    "product_price": "must be a non-negative number",
    "device_type": "field is required"
  }
}
```

### Error `503` — artifacts missing

```json
{
  "error": "model_unavailable",
  "message": "Model or preprocessor is not loaded. Train and save artifacts to models/."
}
```

### Error `500`

```json
{
  "error": "internal_error",
  "message": "Prediction failed."
}
```

---

## Risk bands (proposed — confirm in implementation)

| Probability P(return) | `risk_level` |
|----------------------|--------------|
| `< 0.35` | Low |
| `0.35` – `< 0.65` | Medium |
| `≥ 0.65` | High |

Tune after you see probability calibration.

---

## CORS

Allow the Vite origin (`http://127.0.0.1:5173`). Methods: `GET`, `POST`, `OPTIONS`.
