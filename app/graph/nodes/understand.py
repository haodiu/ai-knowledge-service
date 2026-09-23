"""understand_query (deterministic): normalise the question and bound its size."""
import re
from typing import Any

from app.graph.limits import MAX_QUESTION_CHARS
from app.graph.runtime import GraphRuntimeContext
from app.graph.state import RAGState, TurnError


async def understand_node(state: RAGState, ctx: GraphRuntimeContext) -> dict[str, Any]:
    question = re.sub(r"\s+", " ", state.get("question", "")).strip()
    if not question or len(question) > MAX_QUESTION_CHARS:
        return {"error": TurnError(kind="blocked", code="invalid_question")}
    # recent_turns is seeded by run_graph() before this node runs (Week 6, Plan §11.5); omitting
    # the key here leaves that seeded value untouched rather than overwriting it with [].
    return {"normalized_question": question}
