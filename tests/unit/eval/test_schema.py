import json
from pathlib import Path

import pytest

from app.eval.schema import GoldenQuestionError, load_golden_questions
from app.retrieval.schemas import Tier

VALID = [
    {
        "id": "gq-1",
        "question": "What is the refund window?",
        "tier": "general",
        "expected_type": "answered",
        "expected_chunks": [{"external_id": "refund-policy.md", "chunk_index": 0}],
        "expected_citations": [{"external_id": "refund-policy.md", "chunk_index": 0}],
        "notes": "docs-only",
    },
    {
        "id": "gq-2",
        "question": "What is my current plan?",
        "tier": "general",
        "expected_type": "answered",
        "expected_tool": "get_subscription",
    },
]


def _write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_round_trips_a_valid_file(tmp_path: Path) -> None:
    questions = load_golden_questions(_write(tmp_path, VALID))

    assert len(questions) == 2
    first = questions[0]
    assert first.id == "gq-1"
    assert first.tier == Tier.GENERAL
    assert first.expected_type == "answered"
    assert first.expected_chunks[0].external_id == "refund-policy.md"
    assert first.expected_chunks[0].chunk_index == 0
    second = questions[1]
    assert second.expected_tool == "get_subscription"
    assert second.expected_chunks == ()  # defaults to empty, not required


def test_rejects_an_out_of_range_expected_type(tmp_path: Path) -> None:
    bad = [{**VALID[0], "expected_type": "maybe"}]
    with pytest.raises(GoldenQuestionError, match="expected_type"):
        load_golden_questions(_write(tmp_path, bad))


def test_rejects_an_unknown_tier(tmp_path: Path) -> None:
    bad = [{**VALID[0], "tier": "vip"}]
    with pytest.raises(GoldenQuestionError):
        load_golden_questions(_write(tmp_path, bad))


def test_rejects_a_missing_required_field(tmp_path: Path) -> None:
    bad = [{k: v for k, v in VALID[0].items() if k != "question"}]
    with pytest.raises(GoldenQuestionError, match="question"):
        load_golden_questions(_write(tmp_path, bad))


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    dup = [VALID[0], {**VALID[0]}]
    with pytest.raises(GoldenQuestionError, match="duplicate"):
        load_golden_questions(_write(tmp_path, dup))


def test_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(GoldenQuestionError):
        load_golden_questions(path)


def test_rejects_a_non_array_top_level(tmp_path: Path) -> None:
    with pytest.raises(GoldenQuestionError, match="array"):
        load_golden_questions(_write(tmp_path, {"not": "an array"}))
