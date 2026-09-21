"""Deterministic citation validation (invariant #3, Plan §9.3). No model is consulted, ever.

The key is the PAIR (document_version_id, chunk_id): a real chunk id cited under another version
does not pass. Do not "simplify" this to chunk ids only.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from app.ai.schemas import AnswerDraft
from app.retrieval.schemas import Evidence

SourceKey = tuple[UUID, UUID]


@dataclass(frozen=True)
class CitationCheck:
    valid: bool
    reason: str
    unknown: frozenset[SourceKey] = frozenset()


def validate_citations(draft: AnswerDraft, evidence: Sequence[Evidence]) -> CitationCheck:
    allowed: set[SourceKey] = {(e.document_version_id, e.chunk_id) for e in evidence}
    cited: set[SourceKey] = {(c.document_version_id, c.chunk_id) for c in draft.citations}

    if draft.status != "answered":
        # clarification / insufficient_evidence carry no claims, so they must carry no citations
        if cited:
            return CitationCheck(False, "non_answer_with_citations", frozenset(cited))
        return CitationCheck(True, "no_citations_expected")

    if not cited:  # the schema forbids this; the validator must not depend on the schema
        return CitationCheck(False, "answer_without_citations")
    unknown = cited - allowed
    if unknown:
        return CitationCheck(False, "citation_outside_evidence", frozenset(unknown))
    return CitationCheck(True, "ok")
