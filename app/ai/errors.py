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
    code = "unavailable"


class StructuredOutputError(ModelError):
    """Provider output could not be parsed into the requested Pydantic model. Never usable."""

    code = "structured_output_invalid"


class EmbeddingDimensionError(ModelError):
    code = "embedding_dimension"


class EmbeddingInvalidError(ModelError):
    code = "embedding_invalid"
