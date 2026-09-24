"""Correlation-id middleware (Plan §17, Week 8).

Tags every HTTP request/response with a `request_id`, purely for log correlation -- reads
`X-Request-Id` if the host already sends one, else generates one. Deliberately NOT the same thing
as `GraphRuntimeContext.request_id` (= `str(turn_id)`, set in `app.graph.runner.run_turn`): that
one is load-bearing (`app.tools.subscription` derives deterministic UUIDs from it) and only exists
once a turn begins, so this middleware must never rename or feed into it -- see
`app.logging_setup`'s module docstring for the same note from the other side.
"""
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.logging_setup import bind_request_id, reset_request_id


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        token = bind_request_id(request_id)
        try:
            response = await call_next(request)
        finally:
            reset_request_id(token)
        response.headers["X-Request-Id"] = request_id
        return response
