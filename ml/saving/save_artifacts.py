"""Persist preprocessor and final estimator for the API."""

import joblib


def save_preprocessor(preprocessor, path):
    joblib.dump(preprocessor, path)


def save_model(model, path):
    joblib.dump(model, path)


def save_artifacts(preprocessor, model, preprocessor_path, model_path):
    save_preprocessor(preprocessor, preprocessor_path)
    save_model(model, model_path)
