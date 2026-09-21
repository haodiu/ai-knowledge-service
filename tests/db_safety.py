"""Fail-closed guard for destructive database operations in tests.

`alembic downgrade base` drops every table. A test running it must never be able to reach a real
database by accident: it once wiped the dev database because DATABASE_URL was exported. Like an
authorization check this defaults to REFUSING — only database names that clearly follow the
throwaway convention (`chatbot_test_<hex>` / `chatbot_scratch_<id>`) pass; anything else,
including anything unparsable, raises. Do not loosen the pattern to make a test pass.
"""
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.engine import make_url

from app import settings as app_settings
from app.settings import get_settings

DISPOSABLE_DB_NAME = re.compile(r"chatbot_(?:test|scratch)_[0-9a-z]+")
_ENV_KEYS = ("DATABASE_URL", "REDIS_URL", "RABBITMQ_URL")


class UnsafeDatabaseError(RuntimeError):
    """A destructive operation was requested on a database that is not clearly disposable."""


def require_disposable_database(url: str) -> None:
    if not isinstance(url, str) or not url.strip():
        raise UnsafeDatabaseError("refusing destructive operation: no database URL given")
    try:
        name = make_url(url).database
    except Exception:
        raise UnsafeDatabaseError(
            "refusing destructive operation: the database URL cannot be parsed") from None
    if not name or not DISPOSABLE_DB_NAME.fullmatch(name):
        raise UnsafeDatabaseError(  # the name is safe to show; the URL (password) is not shown
            f"refusing destructive operation on database {name!r}: only names matching "
            f"{DISPOSABLE_DB_NAME.pattern!r} are disposable"
        )


@contextmanager
def alembic_env(url: str, *, destructive: bool = False) -> Iterator[None]:
    """Point migrations/env.py (which reads get_settings()) at `url`, then restore the environment.

    With `destructive=True` (any downgrade) the database must be disposable — checked on the
    argument AND on the URL alembic will actually resolve, before the body runs.
    """
    if destructive:
        require_disposable_database(url)
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    os.environ["DATABASE_URL"] = url
    os.environ.setdefault("REDIS_URL", "redis://localhost:1/0")
    os.environ.setdefault("RABBITMQ_URL", "amqp://u:p@localhost:1//")
    app_settings.get_settings.cache_clear()
    try:
        if destructive:
            require_disposable_database(get_settings().database_url.get_secret_value())
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        app_settings.get_settings.cache_clear()
