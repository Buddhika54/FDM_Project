"""Domain errors mapped to HTTP in routes (400 / 503)."""


class ValidationError(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


class ModelNotLoadedError(Exception):
    """Raised when preprocessor.pkl or final_model.pkl cannot be used."""
