"""Conversation API (Plan §6, §11.8). JWT-authenticated, rate-limited chat turns over SSE.

Invariant #8 holds structurally here, not by convention: `run_turn()` only returns after the
graph has finished AND the turn/citations are persisted (app/graph/runner.py), so nothing this
module streams before that point can be an unvalidated draft -- there is no draft in scope until
`run_turn()` returns. Only phase-status events go out before then.
"""
import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncEngine

from app.ai.prompts.loader import Prompts
from app.ai.registry import ModelRegistry
from app.api.dependencies import (
    enforce_rate_limit,
    get_auth,
    get_engine,
    get_models,
    get_prompts,
    get_settings_dep,
    get_tool_client,
)
from app.auth.policies import AuthorizationContext
from app.db import repositories
from app.graph.limits import MAX_QUESTION_CHARS
from app.graph.result import TurnResult, TurnStatus
from app.graph.runner import run_turn
from app.graph.runtime import Phase
from app.settings import Settings
from app.tools.subscription import SubscriptionToolClient

_LOG = logging.getLogger(__name__)

router = APIRouter(tags=["conversations"])


class CreateConversationResponse(BaseModel):
    id: uuid.UUID


class TurnRequest(BaseModel):
    # A generous outer bound: the graph's own MAX_QUESTION_CHARS is the real semantic limit
    # (blocked/invalid_question); this just stops an absurd payload from reaching it at all.
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class SourceResponse(BaseModel):
    source_id: uuid.UUID
    document_title: str
    version_no: int
    text_snapshot: str


@router.post(
    "/v1/conversations",
    response_model=CreateConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    auth: Annotated[AuthorizationContext, Depends(get_auth)],
    engine: Annotated[AsyncEngine, Depends(get_engine)],
) -> CreateConversationResponse:
    conversation_id = await repositories.create_conversation(engine, auth.user_id)
    return CreateConversationResponse(id=conversation_id)


def _sse(event: str, data: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


# Every TurnStatus maps to exactly one terminal SSE event name (Plan §11.8's `event: answer` is
# the "answered" case; the others are named for what they are, not lumped under one generic name).
_RESULT_EVENT: dict[TurnStatus, str] = {
    "answered": "answer",
    "clarification": "clarification",
    "insufficient_evidence": "insufficient_evidence",
    "blocked": "blocked",
    "temporarily_unavailable": "unavailable",
}


def _result_payload(result: TurnResult) -> dict[str, object]:
    if result.status == "answered":
        return {
            "answer": result.answer,
            "citations": [
                {
                    "source_id": str(s.source_id),
                    "document_title": s.document_title,
                    "version_no": s.version_no,
                }
                for s in result.source_snapshots
            ],
        }
    if result.status == "clarification":
        return {"clarification_question": result.clarification_question}
    payload: dict[str, object] = {"detail": result.detail}
    if result.retry_after_seconds is not None:
        payload["retry_after_seconds"] = result.retry_after_seconds
    return payload


async def _stream_turn(
    *,
    engine: AsyncEngine,
    conversation_id: uuid.UUID,
    turn_id: uuid.UUID,
    question: str,
    auth: AuthorizationContext,
    models: ModelRegistry,
    prompts: Prompts,
    settings: Settings,
    tool_client: SubscriptionToolClient | None,
) -> AsyncIterator[bytes]:
    queue: asyncio.Queue[Phase | None] = asyncio.Queue()

    async def on_phase(phase: Phase) -> None:
        await queue.put(phase)

    async def run() -> TurnResult:
        try:
            return await run_turn(
                engine=engine, conversation_id=conversation_id, turn_id=turn_id,
                question=question, auth=auth, models=models, prompts=prompts,
                embedding_model=settings.embedding_model,
                chat_timeout_seconds=settings.chat_timeout_seconds,
                tool_client=tool_client,
                tool_timeout_seconds=settings.subscription_tool_timeout_seconds,
                on_phase=on_phase,
            )
        finally:
            await queue.put(None)  # sentinel: no more phase events, whatever the outcome

    # A detached task, not tied to this generator's lifecycle: if the client disconnects and this
    # generator is closed early, run_turn() keeps running and still persists the turn (Plan §8's
    # graph timeout is the only thing bounding it, same as if nobody were streaming at all).
    task = asyncio.create_task(run())
    while True:
        item = await queue.get()
        if item is None:
            break
        yield _sse("status", {"phase": item})
    try:
        result = await task
    except Exception:
        _LOG.exception("turn %s: streaming failed unexpectedly", turn_id)
        yield _sse("unavailable", {"detail": "internal_error"})
        yield _sse("done", {})
        return
    yield _sse(_RESULT_EVENT[result.status], _result_payload(result))
    yield _sse("done", {})


@router.post("/v1/conversations/{conversation_id}/turns")
async def post_turn(
    conversation_id: uuid.UUID,
    body: TurnRequest,
    auth: Annotated[AuthorizationContext, Depends(get_auth)],
    _rate_limit: Annotated[None, Depends(enforce_rate_limit)],
    engine: Annotated[AsyncEngine, Depends(get_engine)],
    models: Annotated[ModelRegistry, Depends(get_models)],
    prompts: Annotated[Prompts, Depends(get_prompts)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    tool_client: Annotated[SubscriptionToolClient | None, Depends(get_tool_client)],
) -> StreamingResponse:
    owner = await repositories.conversation_owner(engine, conversation_id)
    if owner is None or owner != auth.user_id:
        # Same 404 either way (invariant #4): a 403 would confirm the conversation exists.
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="conversation not found")
    turn_id = await repositories.add_turn(engine, conversation_id, body.question)
    return StreamingResponse(
        _stream_turn(
            engine=engine, conversation_id=conversation_id, turn_id=turn_id,
            question=body.question, auth=auth, models=models, prompts=prompts, settings=settings,
            tool_client=tool_client,
        ),
        media_type="text/event-stream",
        # A citation's source_id (in the "answer" event) is only openable via GET
        # /v1/turns/{turn_id}/sources/{source_id} -- the client needs turn_id to call it, and
        # nothing else in the stream carries it (a gap in Plan §6's SSE example, found testing
        # this live end to end).
        headers={"X-Turn-Id": str(turn_id)},
    )


@router.get("/v1/turns/{turn_id}/sources/{source_id}", response_model=SourceResponse)
async def get_source(
    turn_id: uuid.UUID,
    source_id: uuid.UUID,
    auth: Annotated[AuthorizationContext, Depends(get_auth)],
    engine: Annotated[AsyncEngine, Depends(get_engine)],
) -> SourceResponse:
    snapshot = await repositories.get_turn_source(
        engine, turn_id, source_id, user_id=auth.user_id
    )
    if snapshot is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="source not found")
    return SourceResponse(
        source_id=snapshot.source_id, document_title=snapshot.document_title,
        version_no=snapshot.version_no, text_snapshot=snapshot.text_snapshot,
    )
