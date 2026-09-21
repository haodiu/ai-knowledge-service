"""ChatModelClient: the generative-model seam (Plan §11.3). Separate from EmbeddingClient."""
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ValidationError

from app.ai.errors import StructuredOutputError
from app.ai.types import ModelMessage, ModelResponse, Purpose


class ChatModelClient(Protocol):
    provider: str
    model_name: str

    async def complete_structured[T: BaseModel](
        self,
        *,
        messages: Sequence[ModelMessage],
        response_model: type[T],
        purpose: Purpose,
        timeout_seconds: float,
    ) -> ModelResponse[T]:
        """One structured call. Returns a validated `response_model` or raises a ModelError.

        No retries and no repair here: a retry is a *generative call* that must be counted against
        MAX_GENERATIVE_LLM_CALLS by the caller (Week 4 graph), so it cannot hide in an adapter.
        """
        ...


def parse_structured[T: BaseModel](raw: str, response_model: type[T]) -> T:
    """The single place provider text becomes a trusted-shape object. Shared by real and fake.

    The error deliberately names only field locations: the raw output may echo untrusted
    evidence text and must not flow into logs or `model_calls`.
    """
    try:
        return response_model.model_validate_json(raw)
    except ValidationError as exc:
        where = sorted({".".join(str(p) for p in err["loc"]) or "<root>" for err in exc.errors()})
        raise StructuredOutputError(
            f"{response_model.__name__} rejected model output ({exc.error_count()} error(s) "
            f"at: {', '.join(where)})",
            fields=tuple(where),
        ) from None
