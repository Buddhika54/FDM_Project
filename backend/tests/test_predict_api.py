"""
Manual API checks with Flask's test client (no live server required).

From the project root:
    python backend/tests/test_predict_api.py

Equivalent curl (server running at 127.0.0.1:5000):

    curl http://127.0.0.1:5000/api/health

    curl -s -X POST http://127.0.0.1:5000/api/predict ^
      -H "Content-Type: application/json" ^
      -d "{\"customer_age\":32,\"product_price\":149.99,\"discount_percent\":15,\"product_rating\":4.2,\"past_purchase_count\":8,\"past_return_rate\":0.12,\"delivery_delay_days\":-1.5,\"session_length_minutes\":12.5,\"num_product_views\":7,\"device_type\":\"mobile\",\"product_category\":\"electronics\",\"shipping_method\":\"express\",\"payment_method\":\"credit_card\",\"used_coupon\":1}"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

from models.prediction_schema import REQUIRED_FEATURES  # noqa: E402
from app import create_app  # noqa: E402


def _valid_row_from_csv() -> dict:
    df = pd.read_csv(ROOT / "dataset" / "raw" / "train.csv")
    ok = (
        (df["product_price"] >= 0)
        & (df["discount_percent"].between(0, 100))
        & (df["product_rating"].between(1, 5))
        & (df["past_return_rate"].between(0, 1))
        & (df["session_length_minutes"] >= 0)
        & (df["num_product_views"] >= 0)
        & (df["customer_age"].between(1, 120))
    )
    row = df.loc[ok, REQUIRED_FEATURES].iloc[0]
    payload = {}
    for col in REQUIRED_FEATURES:
        value = row[col]
        if col in {
            "customer_age",
            "past_purchase_count",
            "num_product_views",
            "used_coupon",
        }:
            payload[col] = int(value)
        elif col in {
            "device_type",
            "product_category",
            "shipping_method",
            "payment_method",
        }:
            payload[col] = str(value)
        else:
            payload[col] = float(value)
    return payload


def main() -> int:
    app = create_app()
    client = app.test_client()
    failed = 0

    health = client.get("/api/health")
    print("GET /api/health", health.status_code, health.get_json())
    if health.status_code != 200 or not health.get_json().get("model_loaded"):
        print("FAIL health")
        failed += 1

    valid = _valid_row_from_csv()
    happy = client.post("/api/predict", json=valid)
    body = happy.get_json()
    print("POST /api/predict valid", happy.status_code)
    print(json.dumps(body, indent=2))
    if happy.status_code != 200:
        print("FAIL happy path")
        failed += 1
    else:
        for key in ("return_risk_probability", "flagged", "model", "threshold"):
            if key not in body:
                print("FAIL missing", key)
                failed += 1
        if body.get("threshold") == 0.5:
            print("FAIL threshold should not be hardcoded 0.5")
            failed += 1
        expected_flag = body["return_risk_probability"] >= body["threshold"]
        if body["flagged"] != expected_flag:
            print("FAIL flagged vs threshold")
            failed += 1

    bad_category = dict(valid)
    bad_category["product_category"] = "food"
    cat_resp = client.post("/api/predict", json=bad_category)
    cat_body = cat_resp.get_json()
    print("POST /api/predict bad category", cat_resp.status_code, cat_body)
    if cat_resp.status_code != 400 or "product_category" not in (cat_body.get("details") or {}):
        print("FAIL bad category")
        failed += 1

    bad_rating = dict(valid)
    bad_rating["product_rating"] = 9.0
    rating_resp = client.post("/api/predict", json=bad_rating)
    rating_body = rating_resp.get_json()
    print("POST /api/predict out-of-range rating", rating_resp.status_code, rating_body)
    if rating_resp.status_code != 400 or "product_rating" not in (rating_body.get("details") or {}):
        print("FAIL bad rating")
        failed += 1

    leaked = dict(valid)
    leaked["discount_amount"] = 10.0
    leak_resp = client.post("/api/predict", json=leaked)
    leak_body = leak_resp.get_json()
    print("POST /api/predict engineered field", leak_resp.status_code, leak_body)
    if leak_resp.status_code != 400 or "discount_amount" not in (leak_body.get("details") or {}):
        print("FAIL engineered field must be rejected")
        failed += 1

    if failed:
        print(f"{failed} check(s) failed")
        return 1
    print("All API checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
