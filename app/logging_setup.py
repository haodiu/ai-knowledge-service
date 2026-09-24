"""Structured (JSON) logging config (Plan §17). Stdlib only -- no structlog, no new dependency
(CLAUDE.md: no new infra without an observed need).

One JSON object per line: `timestamp`, `level`, `logger`, `message`, plus whatever fields a call
site attaches via `extra={...}` -- so `request_id`/`turn_id`/`job_id`/`status`/`latency_ms`/etc.
become real, queryable fields instead of substrings of a `%s`-formatted message. Existing call
sites (`app/graph/runner.py`, `app/rate_limit/redis.py`, ...) are converted to `extra={...}`
incrementally as later Week 8 items touch them (Plan §17's own rule carries over unchanged: never
let a converted call site start including a raw prompt, JWT, or full subscription payload as a
field).

Wired once per process: `app.main.create_app()` for the API, `app.worker.celery_app` (via
`after_setup_logger`/`after_setup_task_logger`, not import time, so it does not fight Celery's own
logging bootstrap) for the worker.
"""
import json
import logging
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

from app.settings import Settings

# HTTP-level correlation id (Plan §17), set by app.api.middleware.RequestIdMiddleware for the
# lifetime of one request. Deliberately separate from `GraphRuntimeContext.request_id` (=
# str(turn_id)): that one is load-bearing -- app.tools.subscription derives deterministic UUIDs
# from it -- and only exists once a turn begins, whereas this exists for every HTTP request
# (including ones that 401/429 before a turn is ever minted) and is for log correlation only.
_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)

# Every attribute a stock LogRecord already carries -- anything else on a record came from a call
# site's `extra={...}` and is surfaced below as its own JSON field.
_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                payload[key] = value
        return json.dumps(payload, default=str)  # default=str: never let a stray object 500 logging


class RequestIdLogFilter(logging.Filter):
    """Stamps the current HTTP request's correlation id (if any) onto every log record emitted
    while handling it, so it comes out as a real field via JsonLogFormatter without hand-threading
    it through every function call in between."""

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = _REQUEST_ID.get()
        if request_id is not None:
            record.request_id = request_id
        return True


def bind_request_id(request_id: str) -> Token[str | None]:
    """Set the current request's correlation id; returns a token for `reset_request_id`. Used by
    `app.api.middleware.RequestIdMiddleware` -- kept here (not in the middleware module) so the
    ContextVar has exactly one owner and `app.logging_setup` does not depend on `app.api`."""
    return _REQUEST_ID.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    _REQUEST_ID.reset(token)


def current_request_id() -> str | None:
    return _REQUEST_ID.get()


_configured = False


def configure_logging(settings: Settings) -> None:
    """Idempotent -- `create_app()` is built many times per test session (and, in production, once
    per process), so a second call must not duplicate handlers or fight a level another call set.
    """
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    handler.addFilter(RequestIdLogFilter())
    root.handlers = [handler]
    _configured = True
