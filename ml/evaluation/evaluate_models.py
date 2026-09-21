"""Delegates to ml.evaluate (Phase 5)."""

from ml.evaluate import run_evaluation


def evaluate_models(models=None, X_test=None, y_test=None, reports_dir=None):
    """Ignore unused args; Phase 5 loads pickles and the Phase 2 test set itself."""
    return run_evaluation()
