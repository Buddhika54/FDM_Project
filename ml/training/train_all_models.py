"""Train all four families via ml.train (no production pickle selection)."""

from ml.train import (
    train_decision_tree,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)


def train_all_models(X_train, y_train):
    return {
        "logistic_regression": train_logistic_regression(X_train, y_train),
        "decision_tree": train_decision_tree(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train),
        "xgboost": train_xgboost(X_train, y_train),
    }
