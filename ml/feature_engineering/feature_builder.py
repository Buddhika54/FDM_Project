"""
Feature engineering.

Responsibilities:
- Add derived columns (e.g. price×discount) only if they help and are leakage-safe
- Same function must be documented so the API can expect raw features only
  (preferred: bake engineering into the sklearn Pipeline)

If engineering stays outside the pickle, backend must duplicate it — avoid that.
"""


def add_engineered_features(df):
    raise NotImplementedError("Phase 2: optional feature builder; prefer Pipeline steps.")
