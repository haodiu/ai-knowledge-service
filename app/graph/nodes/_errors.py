from app.ai.errors import ModelError, ModelRateLimited, StructuredOutputError
from app.graph.state import TurnError


def to_turn_error(exc: ModelError) -> TurnError:
    """Invalid output is a block (never used); everything else is a temporary unavailability."""
    if isinstance(exc, StructuredOutputError):
        return TurnError(kind="blocked", code=exc.code)
    retry_after = exc.retry_after_seconds if isinstance(exc, ModelRateLimited) else None
    return TurnError(kind="unavailable", code=exc.code, retry_after_seconds=retry_after)
