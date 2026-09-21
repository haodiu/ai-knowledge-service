"""Invariant #7: the hard limits are module constants with these exact values (Plan §8.4)."""
from app.graph import limits


def test_hard_limits_have_the_plan_values() -> None:
    assert limits.MAX_RETRIEVAL_ATTEMPTS == 2
    assert limits.MAX_TOOL_CALLS == 1
    assert limits.MAX_GENERATIVE_LLM_CALLS == 4
    assert limits.MAX_RETRIEVED_CHUNKS == 8
    assert limits.GRAPH_TIMEOUT_SECONDS == 30
