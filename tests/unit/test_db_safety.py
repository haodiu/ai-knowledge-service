"""Fail-closed guard for destructive database operations in tests.

`alembic downgrade base` drops every table. A test that runs it must NEVER be able to point at a
real database by accident (it once wiped the dev database when DATABASE_URL was exported). Like an
authorization check, the guard defaults to refusing: only names that clearly follow the throwaway
convention pass, everything else — including anything unparsable — is rejected.
"""
import os

import pytest
from pydantic import SecretStr

from tests import db_safety
from tests.db_safety import UnsafeDatabaseError, alembic_env, require_disposable_database

BASE = "postgresql+psycopg://user:s3cret@127.0.0.1:5434/{db}"


@pytest.mark.parametrize(
    "db",
    ["chatbot", "postgres", "template1", "chatbot_prod", "chatbot_production", "chatbot_dev",
     "chatbot_test", "chatbot_test_", "chatbot_scratch", "chatbot_scratch_", "xchatbot_test_ab12",
     "chatbot_test_AB12", "chatbot_test_ab-12", "chatbot_test_ab12; DROP DATABASE chatbot",
     "my_chatbot_test_ab12", "chatbot_testing_ab12"],
)
def test_real_or_ambiguous_database_names_are_refused(db: str) -> None:
    with pytest.raises(UnsafeDatabaseError):
        require_disposable_database(BASE.format(db=db))


@pytest.mark.parametrize("db", ["chatbot_test_1a2b3c4d5e6f", "chatbot_scratch_12345",
                                "chatbot_test_a", "chatbot_scratch_9f3"])
def test_the_throwaway_naming_convention_is_accepted(db: str) -> None:
    require_disposable_database(BASE.format(db=db))


@pytest.mark.parametrize("url", ["", "not a url", "postgresql+psycopg://user:pw@host", None,
                                 "postgresql+psycopg://user:pw@host/"])
def test_missing_or_unparsable_urls_are_refused(url: str | None) -> None:
    with pytest.raises(UnsafeDatabaseError):
        require_disposable_database(url)  # type: ignore[arg-type]


def test_the_refusal_message_names_the_database_but_never_the_password() -> None:
    with pytest.raises(UnsafeDatabaseError) as exc:
        require_disposable_database(BASE.format(db="chatbot"))
    assert "chatbot" in str(exc.value) and "s3cret" not in str(exc.value)


def test_a_destructive_run_on_an_unsafe_url_is_refused_before_anything_is_touched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused UP FRONT: neither the environment nor the settings cache may be touched at all.
    (Checking only that the environment ends up unchanged is not enough: a later guard plus the
    `finally` restore would also produce that.)"""
    from app import settings as app_settings

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://a:b@h/keep_me")
    touched: list[str] = []
    monkeypatch.setattr(app_settings.get_settings, "cache_clear", lambda: touched.append("clear"))
    env_writes: list[str] = []
    real_setitem = os.environ.__class__.__setitem__

    def spy(self, key, value):  # type: ignore[no-untyped-def]
        env_writes.append(key)
        real_setitem(self, key, value)

    monkeypatch.setattr(os.environ.__class__, "__setitem__", spy)
    with pytest.raises(UnsafeDatabaseError):
        with alembic_env(BASE.format(db="chatbot"), destructive=True):
            pytest.fail("the body of a refused destructive run must never execute")
    monkeypatch.undo()
    assert touched == [] and env_writes == []


def test_the_effective_settings_url_is_checked_not_just_the_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """migrations/env.py reads get_settings(). If that resolves to a different database than the
    one we validated, we must refuse: the guard protects what alembic will actually touch."""

    class _Settings:
        database_url = SecretStr(BASE.format(db="chatbot"))  # the REAL database

    monkeypatch.setattr(db_safety, "get_settings", lambda: _Settings())
    with pytest.raises(UnsafeDatabaseError):
        with alembic_env(BASE.format(db="chatbot_test_abc123"), destructive=True):
            pytest.fail("must not run")


def test_environment_is_restored_after_use_and_after_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://a:b@h/original")
    monkeypatch.delenv("REDIS_URL", raising=False)
    safe = BASE.format(db="chatbot_test_abc123")

    with alembic_env(safe, destructive=True):
        assert os.environ["DATABASE_URL"] == safe
    assert os.environ["DATABASE_URL"] == "postgresql+psycopg://a:b@h/original"
    assert "REDIS_URL" not in os.environ

    with pytest.raises(RuntimeError):
        with alembic_env(safe, destructive=True):
            raise RuntimeError("boom")
    assert os.environ["DATABASE_URL"] == "postgresql+psycopg://a:b@h/original"


def test_a_non_destructive_run_may_target_any_database_name() -> None:
    """Only downgrades are guarded; migrating a fresh throwaway DB or a dev DB up is not."""
    with alembic_env(BASE.format(db="chatbot"), destructive=False):
        assert os.environ["DATABASE_URL"].endswith("/chatbot")
