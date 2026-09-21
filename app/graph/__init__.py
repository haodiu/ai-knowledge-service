"""Online path: the LangGraph workflow (Plan §8).

LangGraph pulls `langchain-core` and `langsmith` in transitively. Our code never imports them
(CLAUDE.md), and LangSmith tracing — which would ship prompts and evidence off-box — is switched
OFF by default here, before anything from langgraph is imported. Opting in is a deliberate act.
"""
import os

os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
