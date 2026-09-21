"""Normalised provider errors. `code` is what lands in model_calls.error_code.

Messages must never carry model output, prompts or credentials.
"""


class ConfigurationError(Exception):
    """A required setting (API key, ...) is missing. Fails closed instead of guessing."""


class PromptError(Exception):
    """Unknown/invalid prompt version or role."""


class ModelError(Exception):
    code = "model_error"

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.code)


class ModelTimeout(ModelError):
    code = "timeout"


class ModelRateLimited(ModelError):
    code = "rate_limited"

    def __init__(self, message: str = "", *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ModelUnavailable(ModelError):
    """Provider unreachable or failing. `retryable` is False for errors a retry cannot fix
    (401/403/404, i.e. configuration), so they never burn the turn's call budget."""

    code = "unavailable"

    def __init__(self, message: str = "", *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class StructuredOutputError(ModelError):
    """Provider output could not be parsed into the requested Pydantic model. Never usable."""

    code = "structured_output_invalid"

    def __init__(self, message: str = "", *, fields: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.fields = fields  # locations only, never model output (repair note uses these)


class EmbeddingDimensionError(ModelError):
    code = "embedding_dimension"


class EmbeddingInvalidError(ModelError):
    code = "embedding_invalid"
