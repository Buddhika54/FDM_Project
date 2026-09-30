"""
Feature Engineering Module for E-Commerce Product Return Prediction.

Provides the FeatureBuilder class compatible with scikit-learn pipelines.
Applies business-driven feature transformations without data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ENGINEERED_NUMERIC_FEATURES = [
    "discount_amount",
    "customer_risk_score",
    "engagement_score",
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "price_category",
    "coupon_category_interaction",
    "shipping_payment_interaction",
]

PRICE_CATEGORY_LABELS = ["Low", "Medium", "High"]

PRODUCT_CATEGORIES = [
    "toys",
    "beauty",
    "electronics",
    "home",
    "clothing",
    "sports",
]

SHIPPING_METHODS = ["standard", "express", "same_day"]
PAYMENT_METHODS = ["debit_card", "credit_card", "apple_pay", "paypal"]

COUPON_CATEGORY_CATEGORIES = [
    f"{prefix}_{cat}"
    for prefix in ["coupon", "nocoupon"]
    for cat in PRODUCT_CATEGORIES
]

SHIPPING_PAYMENT_CATEGORIES = [
    f"{ship}_{pay}"
    for ship in SHIPPING_METHODS
    for pay in PAYMENT_METHODS
]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible Feature Engineering transformer.

    Engineers 6 business-driven features:
    1. discount_amount = product_price * (discount_percent / 100)
    2. customer_risk_score = past_purchase_count * past_return_rate
    3. engagement_score = session_length_minutes * num_product_views
    4. price_category: Low / Medium / High based on training set quantiles
    5. coupon_category_interaction: used_coupon + product_category
    6. shipping_payment_interaction: shipping_method + payment_method

    All quantiles are fitted on training data to strictly prevent data leakage.
    """

    def __init__(self, price_quantiles: tuple[float, float] = (1 / 3, 2 / 3)):
        self.price_quantiles = price_quantiles

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> FeatureBuilder:
        frame = self._ensure_dataframe(X)

        # Store price quantile cutoffs learned exclusively from training set
        q_low, q_high = self.price_quantiles
        self.price_q33_ = float(frame["product_price"].quantile(q_low))
        self.price_q66_ = float(frame["product_price"].quantile(q_high))
        if self.price_q66_ <= self.price_q33_:
            self.price_q66_ = self.price_q33_ + 1e-9

        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns.tolist(), dtype=object)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if not hasattr(self, "price_q33_"):
            raise RuntimeError("FeatureBuilder is not fitted. Call fit() before transform().")

        out = self._ensure_dataframe(X).copy()

        # 1. discount_amount
        out["discount_amount"] = out["product_price"] * (out["discount_percent"] / 100.0)

        # 2. customer_risk_score
        out["customer_risk_score"] = out["past_purchase_count"] * out["past_return_rate"]

        # 3. engagement_score
        out["engagement_score"] = out["session_length_minutes"] * out["num_product_views"]

        # 4. price_category
        bins = [-np.inf, self.price_q33_, self.price_q66_, np.inf]
        out["price_category"] = pd.cut(
            out["product_price"],
            bins=bins,
            labels=PRICE_CATEGORY_LABELS,
            include_lowest=True,
        ).astype(str)

        # 5. coupon_category_interaction
        coupon_prefix = np.where(out["used_coupon"].astype(int) == 1, "coupon", "nocoupon")
        out["coupon_category_interaction"] = (
            coupon_prefix + "_" + out["product_category"].astype(str)
        )

        # 6. shipping_payment_interaction
        out["shipping_payment_interaction"] = (
            out["shipping_method"].astype(str) + "_" + out["payment_method"].astype(str)
        )

        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is not None:
            base_names = list(input_features)
        elif hasattr(self, "feature_names_in_"):
            base_names = list(self.feature_names_in_)
        else:
            base_names = []

        all_names = base_names + ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
        return np.asarray(all_names, dtype=object)

    @staticmethod
    def _ensure_dataframe(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        raise TypeError("FeatureBuilder expects input to be a pandas DataFrame.")


# Set module string so joblib pickles this as ml.feature_engineering.feature_builder.FeatureBuilder
FeatureBuilder.__module__ = "ml.feature_engineering.feature_builder"


def add_engineered_features(df: pd.DataFrame, builder: FeatureBuilder | None = None) -> pd.DataFrame:
    """Convenience helper function for manual feature engineering."""
    if builder is None:
        builder = FeatureBuilder()
        return builder.fit_transform(df)
    return builder.transform(df)
