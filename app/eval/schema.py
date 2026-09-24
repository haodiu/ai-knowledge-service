"""Golden-question schema (Plan §16.4).

JSON, not YAML: PyYAML is only a transitive dependency today (not declared in pyproject.toml), and
stdlib `json` avoids adding one for a feature that does not need it.

A golden question references chunks as `(external_id, chunk_index)`, never a raw chunk UUID:
`chunks.id`/`document_versions.id` are assigned at ingestion time and are not knowable when a
human authors a question by hand. `chunk_index` is deterministic given the chunker and the
document content (app/ingestion/chunking.py), and `external_id` is the same stable identifier
app/ingestion/parsing.py already derives from a corpus file's path relative to the ingest root. The
runner resolves each reference to a live `(document_version_id, chunk_id)` pair once, at load
time, against the active version -- see `resolve_expected_chunks` in app/eval/runner.py.
"""
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.retrieval.schemas import Tier

ExpectedType = Literal["answered", "insufficient_evidence", "clarification", "blocked"]
_EXPECTED_TYPES: frozenset[str] = frozenset(
    ("answered", "insufficient_evidence", "clarification", "blocked")
)


@dataclass(frozen=True)
class ExpectedChunk:
    external_id: str
    chunk_index: int


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    tier: Tier
    expected_type: ExpectedType
    expected_chunks: tuple[ExpectedChunk, ...] = ()
    expected_citations: tuple[ExpectedChunk, ...] = ()
    expected_tool: str | None = None
    notes: str = ""


class GoldenQuestionError(ValueError):
    """The questions file is malformed. Fail loud, at load time, never silently skip a question."""


def _parse_chunks(raw: Sequence[dict[str, object]], *, where: str) -> tuple[ExpectedChunk, ...]:
    out = []
    for i, item in enumerate(raw):
        try:
            out.append(
                ExpectedChunk(
                    external_id=str(item["external_id"]),
                    chunk_index=int(item["chunk_index"]),  # type: ignore[call-overload]
                )
            )
        except KeyError as exc:
            raise GoldenQuestionError(f"{where}[{i}]: missing field {exc}") from exc
    return tuple(out)


def _parse_one(raw: dict[str, object], *, index: int) -> GoldenQuestion:
    where = f"question[{index}]"
    try:
        qid = str(raw["id"])
        question = str(raw["question"])
        tier = Tier(str(raw["tier"]))
        expected_type = str(raw["expected_type"])
    except KeyError as exc:
        raise GoldenQuestionError(f"{where}: missing field {exc}") from exc
    except ValueError as exc:
        raise GoldenQuestionError(f"{where}: {exc}") from exc
    if expected_type not in _EXPECTED_TYPES:
        raise GoldenQuestionError(
            f"{where}: expected_type must be one of {sorted(_EXPECTED_TYPES)}, "
            f"got {expected_type!r}"
        )
    expected_tool = raw.get("expected_tool")
    if expected_tool is not None and not isinstance(expected_tool, str):
        raise GoldenQuestionError(f"{where}: expected_tool must be a string or null")
    return GoldenQuestion(
        id=qid,
        question=question,
        tier=tier,
        expected_type=expected_type,  # type: ignore[arg-type]  # validated above
        expected_chunks=_parse_chunks(
            raw.get("expected_chunks", []),  # type: ignore[arg-type]
            where=f"{where}.expected_chunks",
        ),
        expected_citations=_parse_chunks(
            raw.get("expected_citations", []),  # type: ignore[arg-type]
            where=f"{where}.expected_citations",
        ),
        expected_tool=expected_tool,
        notes=str(raw.get("notes", "")),
    )


def load_golden_questions(path: Path) -> list[GoldenQuestion]:
    """Load and validate every question in `path` (a JSON array). Duplicate ids are rejected --
    the eval report keys rows by id, and a silent collision would hide one question's result."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GoldenQuestionError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(raw, list):
        raise GoldenQuestionError(f"{path}: top level must be a JSON array of questions")

    questions = [_parse_one(item, index=i) for i, item in enumerate(raw)]
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            raise GoldenQuestionError(f"duplicate question id {q.id!r}")
        seen.add(q.id)
    return questions
