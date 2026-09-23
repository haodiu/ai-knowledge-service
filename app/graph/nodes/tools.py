"""call_tool (deterministic): execute the planner's proposed tool call, if any, exactly once
(Plan §8.2, §10; CLAUDE.md invariant #11 "business tools are read-only... fixed allowlist").

Nothing here trusts the proposal blindly: the tool name is re-checked against the allowlist
(defence in depth -- `ToolRequest.name: Literal["get_subscription"]` already makes an unknown name
unreachable via Pydantic), and the identity used to authorize the call comes only from `ctx.auth`,
never from the LLM's proposal (invariant #1). A 404 (not found OR not authorized -- the host
collapses both) contributes zero evidence and is deliberately NOT reported as an error: it falls
through to the ordinary `no_evidence` path, the same branch an empty retrieval takes, so the two
cases are indistinguishable end to end (invariant #4). See `app/tools/errors.py::ToolNotFound` for
the longer version of this note.
"""
import time
from typing import Any, Literal

from app.ai.errors import ModelError
from app.ai.schemas import TOOL_ALLOWLIST
from app.graph.nodes._errors import to_turn_error
from app.graph.nodes.evidence import build_evidence
from app.graph.runtime import GraphRuntimeContext
from app.graph.state import RAGState, TurnError
from app.tools.errors import ToolAmbiguous, ToolError, ToolNotFound, ToolRateLimited
from app.tools.recorder import IdentifierKind, ToolCallRecord
from app.tools.subscription import subscription_to_evidence

_AMBIGUOUS_CLARIFICATION = (
    "There is more than one subscription matching that identifier. Could you share the exact "
    "subscription ID or a more specific customer identifier?"
)


async def tool_node(state: RAGState, ctx: GraphRuntimeContext) -> dict[str, Any]:
    plan = state.get("plan")
    assert plan is not None and plan.tool_request is not None  # the router only sends us here then
    request = plan.tool_request
    assert request.name in TOOL_ALLOWLIST  # defence in depth; Pydantic already enforces this

    await ctx.emit("calling_tool")

    if ctx.tool_client is None:
        # No Payment/Subscription host is configured (Plan §18 Tuan 7: none exists yet). This must
        # fail closed for a turn whose plan actually needs the tool -- but only THAT turn: it must
        # never crash a request whose plan never proposes one (see get_tool_client, which returns
        # None instead of raising for exactly this reason). No attempt was made: no budget spent,
        # no tool_calls row, same rule as a budget-exhausted rejection below.
        return {"tool_executed": True,
                "error": TurnError(kind="unavailable", code="tool_unavailable")}

    try:
        ctx.budget.acquire_tool()  # raises BEFORE any request (same rule as generative calls)
    except ModelError as exc:  # BudgetExhausted; no attempt was made, so no tool_calls row either
        return {"tool_executed": True, "error": to_turn_error(exc)}

    args = request.arguments
    identifier_kind: IdentifierKind = (
        "subscription_id" if args.subscription_id is not None else "customer_id"
    )

    async def record(
        status: Literal["ok", "error"], error_code: str | None, started: float
    ) -> None:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        await ctx.tool_recorder.record(
            ToolCallRecord(request.name, identifier_kind, elapsed_ms, status, error_code)
        )

    started = time.perf_counter()
    try:
        snapshot = await ctx.tool_client.get_subscription(
            args, user_id=ctx.auth.user_id,
            timeout_seconds=ctx.budget.call_timeout(ctx.tool_timeout_seconds),
        )
    except ToolNotFound:
        await record("error", ToolNotFound.code, started)
        # invariant #4: no error, no evidence -> falls through to the ordinary no_evidence path
        return {"tool_executed": True}
    except ToolAmbiguous as exc:
        await record("error", exc.code, started)
        return {
            "tool_executed": True,
            "error": TurnError(
                kind="clarification", code=exc.code, clarification_question=_AMBIGUOUS_CLARIFICATION
            ),
        }
    except ToolError as exc:  # ToolTimeout, ToolRateLimited, ToolUnavailable (and any future kind)
        await record("error", exc.code, started)
        retry_after = exc.retry_after_seconds if isinstance(exc, ToolRateLimited) else None
        return {
            "tool_executed": True,
            "error": TurnError(kind="unavailable", code=exc.code, retry_after_seconds=retry_after),
        }

    await record("ok", None, started)
    evidence = subscription_to_evidence(snapshot, request_id=ctx.request_id)
    merged = build_evidence(state.get("evidence", []), [evidence])
    return {"tool_executed": True, "evidence": merged}
