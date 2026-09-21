"""validate_citations (deterministic) and safe_fallback.

`validate_node` is the ONLY place `answer` is written, and only after set-membership validation
(invariants #3, #8). Snapshots are built here from the evidence; the runner persists them together
with the turn, atomically.
"""
import logging
import uuid
from typing import Any

from app.ai.citations import validate_citations
from app.ai.schemas import CitationRef
from app.graph.nodes.evidence import evidence_key
from app.graph.result import BUG_DETAILS
from app.graph.routing import fallback_outcome
from app.graph.runtime import GraphRuntimeContext
from app.graph.state import RAGState
from app.retrieval.schemas import SourceSnapshot

log = logging.getLogger(__name__)


async def validate_node(state: RAGState, ctx: GraphRuntimeContext) -> dict[str, Any]:
    draft = state.get("draft")
    evidence = state.get("evidence", [])
    if draft is None:  # the router never sends us here without a draft; fail closed regardless
        return {"citations_valid": False}
    if not validate_citations(draft, evidence).valid:
        return {"citations_valid": False}
    if draft.status != "answered":
        return {"citations_valid": True}

    by_key = {evidence_key(e): e for e in evidence}
    citations: list[CitationRef] = []
    snapshots: list[SourceSnapshot] = []
    seen: set[tuple[object, object]] = set()
    for ref in draft.citations:
        key = (ref.document_version_id, ref.chunk_id)
        if key in seen:
            continue  # a chunk cited twice is one source
        seen.add(key)
        e = by_key[key]
        citations.append(ref)
        snapshots.append(
            SourceSnapshot(
                source_id=uuid.uuid4(), document_id=e.document_id,
                document_version_id=e.document_version_id, chunk_id=e.chunk_id,
                document_title=e.title, version_no=e.version_no, text_snapshot=e.text,
                metadata_snapshot={"chunk_index": e.chunk_index, "score": e.score},
            )
        )
    return {"citations_valid": True, "status": "answered", "detail": "ok",
            "answer": draft.answer, "citations": citations, "source_snapshots": snapshots}


async def fallback_node(state: RAGState, ctx: GraphRuntimeContext) -> dict[str, Any]:
    outcome = fallback_outcome(state)
    if outcome.detail in BUG_DETAILS:
        log.error("turn %s reached an impossible state: %s", ctx.request_id, outcome.detail)
    return {
        "status": outcome.status,
        "detail": outcome.detail,
        "clarification_question": outcome.clarification_question,
        "retry_after_seconds": outcome.retry_after_seconds,
        "answer": None,  # a fallback never carries an answer, whatever the draft said
        "citations": [],
        "source_snapshots": [],
    }
