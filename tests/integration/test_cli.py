"""The Week 2 CLI: synchronous ingest through the same service the Week 5 task will call."""
from pathlib import Path

import pytest
from sqlalchemy import Engine, text

from app.ingestion.cli import main
from app.settings import get_settings

pytestmark = pytest.mark.integration

GENERAL = "---\ntitle: Refund policy\ntier: general\n---\nRefunds within 14 days.\n"
INTERNAL = "---\ntitle: Fraud rules\ntier: internal\n---\nBlock cards after 3 chargebacks.\n"
UNLABELLED = "---\ntitle: No tier here\n---\nSomething.\n"


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, test_db_url: str, db_engine: Engine) -> Engine:
    monkeypatch.setenv("DATABASE_URL", test_db_url)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:1/0")
    monkeypatch.setenv("RABBITMQ_URL", "amqp://u:p@localhost:1//")
    get_settings.cache_clear()
    yield db_engine  # type: ignore[misc]
    get_settings.cache_clear()


def _docs(tmp_path: Path) -> Path:
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "refund.md").write_text(GENERAL, encoding="utf-8")
    (tmp_path / "fraud.txt").write_text(INTERNAL, encoding="utf-8")
    (tmp_path / "unlabelled.md").write_text(UNLABELLED, encoding="utf-8")
    return tmp_path


def test_ingest_directory_rejects_unlabelled_files_but_ingests_the_rest(
    cli_env: Engine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ingest", str(_docs(tmp_path))]) == 1  # non-zero: one file was rejected

    out = capsys.readouterr()
    assert "unlabelled.md" in out.err and "REJECTED" in out.err
    with cli_env.connect() as conn:
        rows = dict(conn.execute(text("SELECT external_id, tier FROM documents")).all())
    assert rows == {"policies/refund.md": "general", "fraud.txt": "internal"}  # tier from file


def test_running_the_cli_twice_is_idempotent(
    cli_env: Engine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = _docs(tmp_path)
    (docs / "unlabelled.md").unlink()

    assert main(["ingest", str(docs)]) == 0
    first = capsys.readouterr().out
    assert first.count("activated") == 2

    assert main(["ingest", str(docs)]) == 0
    assert capsys.readouterr().out.count("unchanged") == 2
    with cli_env.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM document_versions")).scalar_one() == 2
        kv = conn.execute(text("SELECT knowledge_version FROM knowledge_base_state")).scalar_one()
        assert kv == 2


def test_missing_path_is_a_usage_error(cli_env: Engine, tmp_path: Path) -> None:
    assert main(["ingest", str(tmp_path / "nope")]) == 2
