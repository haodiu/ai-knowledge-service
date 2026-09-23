"""Regression: tool evidence must survive a SECOND retrieve_node call (the bounded-rewrite loop),
not just the first build_evidence merge right after the tool ran.

Without pinning, `build_evidence(existing, new)` ranks ALL of `new` ahead of ALL of `existing`
regardless of score (app/graph/nodes/evidence.py's own docstring: "new... placed ahead of
existing... the cap trims the older evidence"). Once tool evidence has been folded into
`state["evidence"]` and a rewrite triggers a second retrieval, the tool evidence becomes
`existing` on that second `retrieve_node` call -- a full page of freshly retrieved chunks could
fill MAX_RETRIEVED_CHUNKS/MAX_EVIDENCE_CHARS before ever reaching it, silently dropping the one
fact a "subscription"/"hybrid" question actually needed. `is_tool_evidence` +
`app/graph/nodes/retrieval.py`'s pinning is what prevents that.
"""
from app.graph.limits import MAX_EVIDENCE_CHARS, MAX_RETRIEVED_CHUNKS
from app.graph.nodes.retrieval import retrieve_node
from app.retrieval.schemas import Evidence
from app.tools.subscription import SubscriptionSnapshot, subscription_to_evidence
from tests.unit.ai.helpers import make_evidence
from tests.unit.graph.harness import Harness


def _big_evidence(n: int, chars_each: int) -> list[Evidence]:
    big = make_evidence(n, text_prefix="x" * chars_each)
    assert sum(len(e.text) for e in big) > MAX_EVIDENCE_CHARS  # actually stresses the cap
    return big


async def test_tool_evidence_survives_a_second_retrieve_that_alone_fills_the_cap() -> None:
    snapshot = SubscriptionSnapshot(
        subscription_id="sub_1", customer_id="cus_1", status="active",
        observed_at="2026-09-23T00:00:00Z",
    )
    tool_evidence = subscription_to_evidence(snapshot, request_id="test")
    big_page = _big_evidence(MAX_RETRIEVED_CHUNKS, chars_each=MAX_EVIDENCE_CHARS // 4)

    h = Harness(retrievals=[big_page])
    state = {
        "retrieval_query": "cancel policy", "retrieval_attempts": 1,  # simulates the rewrite's pass
        "evidence": [tool_evidence],
    }
    out = await retrieve_node(state, h.ctx())  # type: ignore[arg-type]

    assert tool_evidence in out["evidence"]
    assert any(e.chunk_id == tool_evidence.chunk_id for e in out["evidence"])
    # the cap is still respected -- pinning does not turn the cap off, just protects the pin
    assert sum(len(e.text) for e in out["evidence"]) <= \
        MAX_EVIDENCE_CHARS + len(tool_evidence.text)


async def test_no_pinned_evidence_is_a_no_op_pass_through() -> None:
    """When there is no tool evidence to pin, behaviour is byte-identical to before this fix."""
    h = Harness(evidence=make_evidence(1))
    state = {"retrieval_query": "q", "retrieval_attempts": 0, "evidence": []}
    out = await retrieve_node(state, h.ctx())  # type: ignore[arg-type]
    assert len(out["evidence"]) == 1
