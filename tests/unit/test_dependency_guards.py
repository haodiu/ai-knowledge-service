"""LangChain rule (CLAUDE.md): langgraph is required and drags langchain-core in transitively,
but our own code never imports LangChain/LangSmith; and only graph/workflow.py knows langgraph."""
import ast
import os
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_app_never_imports_langchain_or_langsmith() -> None:
    offenders = {
        str(p.relative_to(APP)): m
        for p in APP.rglob("*.py")
        for m in _imports(p)
        if m.split(".")[0] in {"langchain", "langchain_core", "langsmith", "langchain_protocol"}
    }
    assert not offenders, offenders


def test_only_the_workflow_module_imports_langgraph() -> None:
    offenders = {
        str(p.relative_to(APP))
        for p in APP.rglob("*.py")
        if any(m.split(".")[0] == "langgraph" for m in _imports(p))
    }
    assert offenders == {"graph/workflow.py"}


def test_importing_the_graph_package_switches_tracing_off_by_default() -> None:
    import app.graph  # noqa: F401

    assert os.environ.get("LANGSMITH_TRACING") == "false"
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"
