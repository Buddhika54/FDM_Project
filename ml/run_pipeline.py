"""
Pipeline orchestrator.

Responsibilities:
- Run stages in order: EDA → clean → features → preprocess fit
  → train all → evaluate → tune → save artifacts
- Keep side effects (files) under dataset/processed, reports/, models/

Do not import Flask. Do not start the API.
"""


def main():
    # TODO Phase 6: call each stage's public function with ml.config paths
    raise NotImplementedError(
        "Wire ml.eda, preprocessing, feature_engineering, training, "
        "evaluation, tuning, and saving after those modules are implemented."
    )


if __name__ == "__main__":
    main()
