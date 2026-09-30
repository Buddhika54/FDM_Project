"""
Input validation for POST /api/predict.

Rejects invalid live traffic. Does not clip or coerce — Phase 2 already
encoded clip/winsorize inside preprocessor.pkl for *valid* rows.
"""

from __future__ import annotations

import math

from models.prediction_schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_VALUES,
    ENGINEERED_FEATURES,
    REQUIRED_FEATURES,
)
from utils.exceptions import ValidationError

# Live-traffic bounds matching Phase 2 clip limits (reject, do not clip).
RANGES = {
    "customer_age": (1, 120),
    "product_price": (0, None),
    "discount_percent": (0, 100),
    "product_rating": (1, 5),
    "past_purchase_count": (0, None),
    "past_return_rate": (0, 1),
    "session_length_minutes": (0, None),
    "num_product_views": (0, None),
}

INTEGER_FIELDS = {
    "customer_age",
    "past_purchase_count",
    "num_product_views",
    "used_coupon",
}

FORBIDDEN_FIELDS = {"order_id", "returned"}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_int(field: str, value) -> int:
    if isinstance(value, bool) or not _is_number(value):
        raise ValidationError(
            "Invalid or missing fields.",
            {field: "must be an integer"},
        )
    if isinstance(value, float) and not value.is_integer():
        raise ValidationError(
            "Invalid or missing fields.",
            {field: "must be an integer"},
        )
    return int(value)


def _as_float(field: str, value) -> float:
    if isinstance(value, bool) or not _is_number(value):
        raise ValidationError(
            "Invalid or missing fields.",
            {field: "must be a number"},
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(
            "Invalid or missing fields.",
            {field: "must be a finite number"},
        )
    return number


def validate_prediction_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValidationError("JSON object required.", {})

    details = {}
    for forbidden in FORBIDDEN_FIELDS:
        if forbidden in payload:
            details[forbidden] = "not an allowed predictor (identifier or target)"

    extra = set(payload.keys()) - set(REQUIRED_FEATURES) - FORBIDDEN_FIELDS
    for name in sorted(extra):
        if name in ENGINEERED_FEATURES:
            details[name] = "generated internally from raw fields; do not send"
        else:
            details[name] = "unknown field"

    for field in REQUIRED_FEATURES:
        if field not in payload:
            details[field] = "field is required"
        elif payload[field] is None:
            details[field] = "field is required"

    if details:
        raise ValidationError("Invalid or missing fields.", details)

    cleaned = {}
    for field in REQUIRED_FEATURES:
        value = payload[field]
        if field in CATEGORICAL_FEATURES:
            if not isinstance(value, str):
                details[field] = "must be a string"
                continue
            allowed = CATEGORICAL_VALUES[field]
            if value not in allowed:
                details[field] = f"must be one of: {', '.join(allowed)}"
                continue
            cleaned[field] = value
            continue

        if field in BINARY_FEATURES:
            number = _safe_int(field, value, details)
            if number is None:
                continue
            if number not in (0, 1):
                details[field] = "must be 0 or 1"
                continue
            cleaned[field] = number
            continue

        if field in INTEGER_FIELDS:
            number = _safe_int(field, value, details)
            if number is None:
                continue
        else:
            number = _safe_float(field, value, details)
            if number is None:
                continue

        low, high = RANGES.get(field, (None, None))
        if low is not None and number < low:
            details[field] = f"must be >= {low}"
            continue
        if high is not None and number > high:
            details[field] = f"must be <= {high}"
            continue
        cleaned[field] = number

    # delivery_delay_days: signed, any finite number
    if "delivery_delay_days" in cleaned:
        pass

    if details:
        raise ValidationError("Invalid or missing fields.", details)
    return cleaned


def _safe_int(field, value, details):
    try:
        return _as_int(field, value)
    except ValidationError as err:
        details.update(err.details)
        return None


def _safe_float(field, value, details):
    try:
        return _as_float(field, value)
    except ValidationError as err:
        details.update(err.details)
        return None
