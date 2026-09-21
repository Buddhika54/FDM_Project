"""
Prediction feature schema.

Canonical names match dataset/raw/train.csv and docs/data-dictionary.md.
Keep in sync with frontend/src/constants/featureFields.js.

Excluded from this contract:
- order_id — identifier only; drop before modeling
- returned — target label; never an API input
"""

TARGET_COLUMN = "returned"
IDENTIFIER_COLUMNS = ["order_id"]

REQUIRED_FEATURES = [
    "customer_age",
    "product_price",
    "discount_percent",
    "product_rating",
    "past_purchase_count",
    "past_return_rate",
    "delivery_delay_days",
    "session_length_minutes",
    "num_product_views",
    "device_type",
    "product_category",
    "shipping_method",
    "payment_method",
    "used_coupon",
]

NUMERIC_FEATURES = [
    "customer_age",
    "product_price",
    "discount_percent",
    "product_rating",
    "past_purchase_count",
    "past_return_rate",
    "delivery_delay_days",
    "session_length_minutes",
    "num_product_views",
]

CATEGORICAL_FEATURES = [
    "device_type",
    "product_category",
    "shipping_method",
    "payment_method",
]

BINARY_FEATURES = [
    "used_coupon",
]

CATEGORICAL_VALUES = {
    "device_type": ["mobile", "desktop", "tablet"],
    "product_category": [
        "toys",
        "beauty",
        "electronics",
        "home",
        "clothing",
        "sports",
    ],
    "shipping_method": ["standard", "express", "same_day"],
    "payment_method": ["debit_card", "credit_card", "apple_pay", "paypal"],
}

FEATURE_COLUMN_ORDER = list(REQUIRED_FEATURES)
