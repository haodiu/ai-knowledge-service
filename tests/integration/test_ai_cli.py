"""Week 3 demo: `ask` = plan -> retrieval -> grade -> generate -> validated answer, on fakes.

Data is seeded with raw SQL (like the retrieval tests) with the Week 2 fake embedder, so the
fake query embedding hits the same vector space. Superseded and internal chunks are *better*
textual matches, so a broken version/tier filter would show up in the printed sources.
"""
import pytest
from sqlalchemy import Engine, text

from app.ai.cli import main
from app.ingestion.fake_embedder import fake_embed
from app.settings import get_settings
from tests.integration.conftest import make_chunk, make_document, make_version, point_active

pytestmark = pytest.mark.integration


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, test_db_url: str, db_engine: Engine) -> Engine:
    monkeypatch.setenv("DATABASE_URL", test_db_url)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:1/0")
    monkeypatch.setenv("RABBITMQ_URL", "amqp://u:p@localhost:1//")
    get_settings.cache_clear()
    yield db_engine  # type: ignore[misc]
    get_settings.cache_clear()


def _chunk(conn, version, idx, body):  # type: ignore[no-untyped-def]
    return str(make_chunk(conn, version, idx, body, fake_embed([body])[0]))


@pytest.fixture
def seeded(cli_env: Engine) -> dict[str, str]:
    with cli_env.begin() as conn:
        pol = make_document(conn, "policy", tier="general")
        old = make_version(conn, pol, 1, "superseded")
        new = make_version(conn, pol, 2, "active")
        point_active(conn, pol, new)
        ids = {
            "active": _chunk(conn, new, 0, "refund policy allows refunds within 14 days"),
            "superseded": _chunk(conn, old, 0, "refund policy refund refund allows 30 days"),
        }
        internal = make_document(conn, "fraud", tier="internal")
        iv = make_version(conn, internal, 1, "active")
        point_active(conn, internal, iv)
        ids["internal"] = _chunk(conn, iv, 0, "refund policy refund fraud thresholds internal")
    return ids


def test_general_tier_gets_a_validated_answer_from_the_active_version_only(
    seeded: dict[str, str], cli_env: Engine, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ask", "refund policy", "--tier", "general", "--fake"]) == 0
    out = capsys.readouterr().out

    assert "status: answered" in out
    assert seeded["active"] in out
    assert seeded["superseded"] not in out and seeded["internal"] not in out

    with cli_env.connect() as conn:
        calls = conn.execute(
            text("SELECT purpose, provider, prompt_version, status, latency_ms "
                 "FROM model_calls ORDER BY created_at, id")
        ).all()
        turn = conn.execute(
            text("SELECT graph_status, answer, retrieval_attempts FROM turns")
        ).one()
        user = conn.execute(text("SELECT user_id FROM conversations")).scalar_one()
    assert [c.purpose for c in calls] == ["plan", "grade", "answer"]
    assert all(
        c.prompt_version == "v2" and c.status == "ok" and c.provider == "fake" for c in calls
    )
    assert turn.graph_status == "answered" and turn.answer and turn.retrieval_attempts == 1
    assert user == "cli"


def test_internal_tier_can_see_internal_but_still_not_superseded(
    seeded: dict[str, str], cli_env: Engine, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ask", "refund policy", "--tier", "internal", "--fake"]) == 0
    out = capsys.readouterr().out
    assert seeded["internal"] in out and seeded["superseded"] not in out


def test_tier_defaults_to_general_fail_closed(
    seeded: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ask", "refund policy", "--fake"]) == 0
    assert seeded["internal"] not in capsys.readouterr().out


def test_unknown_tier_is_rejected_by_the_parser(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ask", "q", "--tier", "admin", "--fake"])
    assert exc.value.code == 2


def test_no_active_documents_means_insufficient_evidence_and_only_the_planner_ran(
    cli_env: Engine, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ask", "refund policy", "--fake"]) == 1
    assert "status: insufficient_evidence" in capsys.readouterr().out
    with cli_env.connect() as conn:
        purposes = [r[0] for r in conn.execute(text("SELECT purpose FROM model_calls"))]
        status = conn.execute(text("SELECT graph_status FROM turns")).scalar_one()
    assert purposes == ["plan"]  # empty evidence -> grader and generator never called
    assert status == "insufficient_evidence"


def test_no_raw_prompt_or_answer_text_is_written_to_model_calls(
    seeded: dict[str, str], cli_env: Engine
) -> None:
    main(["ask", "refund policy", "--fake"])
    with cli_env.connect() as conn:
        dump = conn.execute(text("SELECT model_calls::text FROM model_calls")).scalars().all()
    assert dump and all("refund policy" not in row for row in dump)
