"""Invariant #3 (release-blocking): citations are checked by plain set membership, no LLM."""
import uuid

from app.ai.citations import validate_citations
from app.ai.schemas import AnswerDraft, CitationRef
from tests.unit.ai.helpers import make_evidence


def _draft(*refs: CitationRef, status: str = "answered") -> AnswerDraft:
    return AnswerDraft(
        status=status,  # type: ignore[arg-type]
        answer="text" if status == "answered" else None,
        citations=list(refs),
    )


def _ref(e) -> CitationRef:  # type: ignore[no-untyped-def]
    return CitationRef(document_version_id=e.document_version_id, chunk_id=e.chunk_id)


def test_citations_inside_the_evidence_set_are_valid() -> None:
    ev = make_evidence(3)
    assert validate_citations(_draft(_ref(ev[0]), _ref(ev[2])), ev).valid


def test_citation_to_a_chunk_outside_the_evidence_is_blocked() -> None:
    ev = make_evidence(2)
    invented = CitationRef(document_version_id=ev[0].document_version_id, chunk_id=uuid.uuid4())
    check = validate_citations(_draft(_ref(ev[0]), invented), ev)  # one good, one invented
    assert not check.valid
    assert check.unknown == frozenset({(invented.document_version_id, invented.chunk_id)})


def test_right_chunk_under_the_wrong_version_is_blocked() -> None:
    """The key is the PAIR: a real chunk id cited against another version must not pass."""
    ev = make_evidence(2)
    mixed = CitationRef(document_version_id=ev[1].document_version_id, chunk_id=ev[0].chunk_id)
    assert not validate_citations(_draft(mixed), ev).valid


def test_answered_draft_without_citations_is_blocked_by_the_validator_too() -> None:
    """The schema forbids it, but the validator must not rely on that (defence in depth)."""
    ev = make_evidence(1)
    draft = AnswerDraft.model_construct(status="answered", answer="x", citations=[])
    assert not validate_citations(draft, ev).valid


def test_any_citation_against_empty_evidence_is_blocked() -> None:
    ev = make_evidence(1)
    assert not validate_citations(_draft(_ref(ev[0])), []).valid


def test_non_answer_statuses_pass_with_no_citations() -> None:
    ev = make_evidence(1)
    assert validate_citations(_draft(status="insufficient_evidence"), ev).valid
