"""Scriptable fake ChatModelClient — able to misbehave on purpose.

Scripted items go through the SAME `parse_structured` as the real adapter, so a test that scripts
bad JSON / a wrong schema / an unknown tool / an invented citation exercises the real rejection
path. An exception item is raised as-is (timeouts, 429s). Running past the script is an
AssertionError, which is how tests prove "the grader was never called".
"""
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.ai.chat.base import parse_structured
from app.ai.schemas import AnswerDraft, CitationRef, EvidenceGrade, QueryPlan
from app.ai.types import ModelMessage, ModelResponse, Purpose, Usage

Scripted = BaseModel | dict[str, Any] | str | BaseException
Responder = Callable[[Sequence[ModelMessage], type[BaseModel], Purpose], Scripted]


@dataclass(frozen=True)
class FakeCall:
    purpose: Purpose
    messages: list[ModelMessage]
    response_model: type[BaseModel]
    timeout_seconds: float


class FakeChatModelClient:
    def __init__(
        self,
        script: Iterable[Scripted] = (),
        *,
        responder: Responder | None = None,
        provider: str = "fake",
        model_name: str = "fake-chat",
    ) -> None:
        self._script = list(script)
        self._responder = responder
        self.provider = provider
        self.model_name = model_name
        self.calls: list[FakeCall] = []

    async def complete_structured[T: BaseModel](
        self,
        *,
        messages: Sequence[ModelMessage],
        response_model: type[T],
        purpose: Purpose,
        timeout_seconds: float,
    ) -> ModelResponse[T]:
        self.calls.append(FakeCall(purpose, list(messages), response_model, timeout_seconds))
        item = self._next(messages, response_model, purpose)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            raw = item
        elif isinstance(item, dict):
            raw = json.dumps(item)
        else:
            raw = item.model_dump_json()
        parsed = parse_structured(raw, response_model)
        usage = Usage(
            input_tokens=max(1, sum(len(m.content) for m in messages) // 4),
            output_tokens=max(1, len(raw) // 4),
        )
        return ModelResponse(parsed, self.provider, self.model_name, usage, latency_ms=0)

    def _next(
        self, messages: Sequence[ModelMessage], response_model: type[BaseModel], purpose: Purpose
    ) -> Scripted:
        if self._script:
            return self._script.pop(0)
        if self._responder is not None:
            return self._responder(messages, response_model, purpose)
        raise AssertionError("fake chat client called more times than scripted")


_QUESTION = re.compile(r"Question:\n(.*?)(?:\n\n|\Z)", re.S)
_SOURCE = re.compile(
    r"\[source nonce=\w+ document_version_id=([0-9a-f-]{36}) chunk_id=([0-9a-f-]{36})"
)


def demo_responder(
    messages: Sequence[ModelMessage], response_model: type[BaseModel], purpose: Purpose
) -> Scripted:
    """Deterministic stand-in for the CLI's `--fake` mode: plausible, always schema-valid.

    Not a model of any behaviour — it only lets the plan -> retrieve -> grade -> generate ->
    validate flow run end to end without a provider.
    """
    user = messages[-1].content
    if response_model is QueryPlan:
        m = _QUESTION.search(user)
        question = (m.group(1).strip() if m else "") or "question"
        return QueryPlan(intent="policy", retrieval_query=question[:500], needs_retrieval=True)
    sources = _SOURCE.findall(user)
    if response_model is EvidenceGrade:
        return EvidenceGrade(
            sufficient=bool(sources), confidence=0.9 if sources else 0.1, reason="demo fake grader"
        )
    if response_model is AnswerDraft:
        if not sources:
            return AnswerDraft(status="insufficient_evidence")
        version_id, chunk_id = sources[0]
        return AnswerDraft(
            status="answered",
            answer="Demo answer from the fake model: see the cited source.",
            citations=[CitationRef(document_version_id=version_id, chunk_id=chunk_id)],
        )
    raise AssertionError(f"demo_responder has no answer for {response_model.__name__}")
