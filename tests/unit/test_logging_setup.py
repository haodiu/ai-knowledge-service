"""Structured JSON logging (Plan §17, Week 8). Stdlib only -- see app/logging_setup.py."""
import json
import logging

import app.logging_setup as logging_setup
from app.logging_setup import JsonLogFormatter, configure_logging


def test_json_formatter_surfaces_extra_fields_as_json() -> None:
    record = logging.LogRecord(
        name="app.graph.runner", level=logging.INFO, pathname=__file__, lineno=1,
        msg="turn summary", args=(), exc_info=None,
    )
    record.turn_id = "11111111-1111-1111-1111-111111111111"
    record.latency_ms = 42

    line = JsonLogFormatter().format(record)
    payload = json.loads(line)

    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.graph.runner"
    assert payload["message"] == "turn summary"
    assert payload["turn_id"] == "11111111-1111-1111-1111-111111111111"
    assert payload["latency_ms"] == 42
    assert "timestamp" in payload


def test_json_formatter_never_raises_on_an_unserialisable_extra_field() -> None:
    record = logging.LogRecord(
        name="x", level=logging.WARNING, pathname=__file__, lineno=1,
        msg="oops", args=(), exc_info=None,
    )
    record.weird = object()  # not JSON-serialisable by default

    line = JsonLogFormatter().format(record)  # must not raise

    assert json.loads(line)["weird"]  # stringified via default=str, not dropped


def test_configure_logging_is_idempotent(settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(logging_setup, "_configured", False)
    configure_logging(settings)
    handlers_after_first = list(logging.getLogger().handlers)

    configure_logging(settings)  # must not add a second handler

    assert logging.getLogger().handlers == handlers_after_first
