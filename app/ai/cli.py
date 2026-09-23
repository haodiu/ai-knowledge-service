"""Demo CLI: one question through the online LangGraph workflow (Plan §8), citation-validated.

    python -m app.ai.cli ask "What is the refund window?" --tier general --fake
    python -m app.ai.cli ask "What is the refund window?" --tier general      # needs GEMINI_API_KEY

`--tier` is an operator flag standing in for the verified JWT claim of Week 6 (default: general,
i.e. fail closed). It never comes from the model. Exit codes: 0 answered/clarification,
1 insufficient_evidence/blocked, 2 usage/config, 3 temporarily_unavailable, 4 unexpected error.
"""
import argparse
import asyncio
import sys
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncEngine

from app.ai.errors import ConfigurationError, PromptError
from app.ai.prompts.loader import load_prompts
from app.ai.registry import build_fake_registry, build_registry
from app.auth.policies import AuthorizationContext
from app.db import repositories
from app.db.session import create_engine
from app.graph.result import TurnResult
from app.graph.runner import run_turn
from app.graph.runtime import Phase
from app.ingestion.fake_embedder import FAKE_EMBEDDING_MODEL
from app.retrieval.schemas import Tier
from app.settings import get_settings
from app.tools.fake import FakeSubscriptionToolClient
from app.tools.subscription import SubscriptionToolClient, build_tool_client

_EXIT = {
    "answered": 0, "clarification": 0, "insufficient_evidence": 1, "blocked": 1,
    "temporarily_unavailable": 3,
}


def _print(result: TurnResult) -> None:
    print(f"status: {result.status} ({result.detail})")
    print(f"retrieval attempts: {result.retrieval_attempts}")
    if result.plan:
        print(f"plan: intent={result.plan.intent} query={result.plan.retrieval_query!r}")
    if result.proposed_tool:
        state = "executed" if result.tool_executed else "NOT executed"
        print(f"tool proposed ({state}): {result.proposed_tool.model_dump_json()}")
    if result.evidence:
        print("evidence:")
        for e in result.evidence:
            print(f"- document_version_id={e.document_version_id} chunk_id={e.chunk_id} "
                  f"title={e.title!r} v{e.version_no}")
    if result.grade:
        print(f"grade: sufficient={result.grade.sufficient} confidence={result.grade.confidence}"
              f" reason={result.grade.reason!r}")
    if result.answer:
        print(f"answer: {result.answer}")
        print("sources:")
        for c in result.citations:
            print(f"- document_version_id={c.document_version_id} chunk_id={c.chunk_id}")
    if result.clarification_question:
        print(f"clarification: {result.clarification_question}")
    if result.retry_after_seconds is not None:
        print(f"retry after: {result.retry_after_seconds:g}s")


async def _ask(question: str, *, tier: Tier, fake: bool) -> int:
    settings = get_settings()
    try:
        models = build_fake_registry() if fake else build_registry(settings)
        prompts = load_prompts(settings.prompt_version)
    except (ConfigurationError, PromptError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    tool_client: SubscriptionToolClient | None
    if fake:
        tool_client = FakeSubscriptionToolClient()
    else:
        try:
            tool_client = build_tool_client(settings)
        except ConfigurationError:
            # Unlike the chat models above, not every question needs the tool (no
            # Payment/Subscription host exists yet) -- only a question whose plan actually
            # proposes one pays for that, as `temporarily_unavailable`/`tool_unavailable`.
            tool_client = None

    async def show_phase(phase: Phase) -> None:  # phase events only; never any answer text
        print(f"[{phase}]", file=sys.stderr)

    engine: AsyncEngine = create_engine(settings)
    turn_id = None
    try:
        created = await repositories.create_turn(engine, user_id="cli", question=question)
        turn_id = created.turn_id
        result = await run_turn(
            engine=engine, conversation_id=created.conversation_id, turn_id=turn_id,
            question=question,
            auth=AuthorizationContext(user_id="cli", tier=tier),
            models=models, prompts=prompts,
            embedding_model=FAKE_EMBEDDING_MODEL if fake else settings.embedding_model,
            chat_timeout_seconds=settings.chat_timeout_seconds,
            tool_client=tool_client,
            tool_timeout_seconds=settings.subscription_tool_timeout_seconds,
            on_phase=show_phase,
        )
        _print(result)
        return _EXIT[result.status]
    except Exception as exc:
        if turn_id is not None:
            await repositories.finish_turn(
                engine, turn_id, graph_status="error", answer=None, sources=[],
                retrieval_attempts=0, latency_ms=0,
            )
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ai.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="run one question through the sequential Week 3 flow")
    ask.add_argument("question")
    ask.add_argument("--tier", choices=[t.value for t in Tier], default=Tier.GENERAL.value)
    ask.add_argument("--fake", action="store_true", help="use fake models (no API key needed)")
    args = parser.parse_args(argv)
    return asyncio.run(_ask(args.question, tier=Tier(args.tier), fake=args.fake))


if __name__ == "__main__":
    raise SystemExit(main())
