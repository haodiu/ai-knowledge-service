"""Normalised subscription-tool errors (Plan §10). `code` is what lands in tool_calls.error_code.

Subclasses `ModelError`, not `ModelTimeout`/`ModelRateLimited`/`ModelUnavailable`: those three are
reserved for the LLM-provider adapters that `app/ai/roles.py::_call` and
`app/graph/nodes/_errors.py::to_turn_error` do `isinstance` checks against. A `ToolError` must
never be mistaken for one of those by that machinery. Subclassing `ModelError` itself still buys
automatic coverage from `tests/unit/graph/test_failure_matrix.py`'s recursive
`ModelError.__subclasses__()` walk, which asserts every `.code` lands in `KNOWN_DETAILS`.

Messages must never carry the host's raw response body, credentials or the identifier value looked
up (Plan §17 "no raw sensitive payload in observability rows").
"""
from app.ai.errors import ModelError


class ToolError(ModelError):
    code = "tool_error"


class ToolTimeout(ToolError):
    code = "tool_timeout"


class ToolRateLimited(ToolError):
    code = "tool_rate_limited"

    def __init__(self, message: str = "", *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ToolUnavailable(ToolError):
    """Host unreachable, misconfigured, or returned an HTTP status this client cannot parse."""

    code = "tool_unavailable"


class ToolNotFound(ToolError):
    """404: the record does not exist OR the asking user is not authorized for it -- the host
    collapses both into the same status code (invariant #4), and this class mirrors that: nothing
    downstream can tell the two cases apart either.

    Registered here (and added to KNOWN_DETAILS) only so the failure-matrix subclass walk stays
    complete. The graph node that raises it always catches it itself and never lets it become
    `TurnError`/`TurnResult.detail` -- a 404 contributes zero evidence and falls through to the
    ordinary "insufficient_evidence" path, same as retrieval finding nothing. Do not "clean up" the
    apparent dead branch this implies; see `app/graph/nodes/tools.py`.
    """

    code = "tool_not_found"


class ToolAmbiguous(ToolError):
    """409: the identifier alone does not resolve to one record. Maps to a clarification turn."""

    code = "tool_ambiguous"
