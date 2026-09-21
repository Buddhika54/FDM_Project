"""Delegates to ml.preprocess (Phase 2)."""

from ml.preprocess import apply_deterministic_clips


def clean_dataset(df):
    return apply_deterministic_clips(df)
