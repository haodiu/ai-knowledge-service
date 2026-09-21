"""The compiled LangGraph (Plan §8.3). The ONLY module that imports langgraph.

Nodes are plain `async (state, ctx)` functions and routers are pure; this file only wires them.
The graph itself holds no per-request data: the runtime context arrives via `ainvoke(context=)`.
"""
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from app.graph.nodes.evidence import grade_node, rewrite_node
from app.graph.nodes.generation import generate_node
from app.graph.nodes.planning import plan_node
from app.graph.nodes.retrieval import retrieve_node
from app.graph.nodes.understand import understand_node
from app.graph.nodes.validation import fallback_node, validate_node
from app.graph.routing import (
    route_after_generate,
    route_after_grade,
    route_after_plan,
    route_after_retrieve,
    route_after_understand,
    route_after_validate,
)
from app.graph.runtime import GraphRuntimeContext
from app.graph.state import RAGState

Node = Callable[[RAGState, GraphRuntimeContext], Awaitable[dict[str, Any]]]


def _bind(node: Node) -> Any:
    """Adapt `(state, ctx)` to LangGraph's `(state, runtime)`. Typed Any: langgraph's overloads
    key on the parameter *name* `runtime`, which a Callable type cannot express."""

    async def run(state: RAGState, runtime: Runtime[GraphRuntimeContext]) -> dict[str, Any]:
        return await node(state, runtime.context)

    run.__name__ = node.__name__
    return run


@lru_cache
def build_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    g: StateGraph[Any, Any, Any, Any] = StateGraph(RAGState, context_schema=GraphRuntimeContext)
    g.add_node("understand", _bind(understand_node))
    g.add_node("plan", _bind(plan_node))
    g.add_node("retrieve", _bind(retrieve_node))
    g.add_node("grade", _bind(grade_node))
    g.add_node("rewrite", _bind(rewrite_node))
    g.add_node("generate", _bind(generate_node))
    g.add_node("validate", _bind(validate_node))
    g.add_node("fallback", _bind(fallback_node))

    g.add_edge(START, "understand")
    g.add_conditional_edges(
        "understand", route_after_understand, {"plan": "plan", "fallback": "fallback"})
    g.add_conditional_edges(
        "plan", route_after_plan, {"retrieve": "retrieve", "fallback": "fallback"})
    g.add_conditional_edges(
        "retrieve", route_after_retrieve, {"grade": "grade", "fallback": "fallback"})
    g.add_conditional_edges(
        "grade", route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"})
    g.add_edge("rewrite", "retrieve")  # the one bounded loop: bounded by route_after_grade
    g.add_conditional_edges(
        "generate", route_after_generate, {"validate": "validate", "fallback": "fallback"})
    g.add_conditional_edges("validate", route_after_validate, {"end": END, "fallback": "fallback"})
    g.add_edge("fallback", END)
    return g.compile()


class GraphLoopError(Exception):
    """LangGraph's recursion backstop fired. The routing bounds should make this unreachable."""


async def run_graph(question: str, ctx: GraphRuntimeContext, *, recursion_limit: int) -> RAGState:
    try:
        state: RAGState = await build_graph().ainvoke(  # type: ignore[assignment]
            {"question": question}, context=ctx, config={"recursion_limit": recursion_limit}
        )
    except GraphRecursionError:
        raise GraphLoopError from None
    return state
