"""Week 3 demo CLI: plan -> retrieval -> grade -> generate -> citation-validated answer.

    python -m app.ai.cli ask "What is the refund window?" --tier general --fake
    python -m app.ai.cli ask "What is the refund window?" --tier general      # needs GEMINI_API_KEY

`--tier` is an operator flag standing in for the verified JWT claim of Week 6 (default: general,
i.e. fail closed). It never comes from the model. Exit codes: 0 answered/clarification,
1 insufficient_evidence/blocked, 2 usage/config, 3 temporarily_unavailable, 4 unexpected error.
"""
import argparse
import asyncio
import sys
import time
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncEngine

from app.ai.errors import ConfigurationError, PromptError
from app.ai.pipeline import TurnResult, run_turn
from app.ai.prompts.loader import load_prompts
from app.ai.recorder import SqlModelCallRecorder
from app.ai.registry import build_fake_registry, build_registry
from app.auth.policies import allowed_tiers
from app.db import repositories
from app.db.session import create_engine
from app.ingestion.fake_embedder import FAKE_EMBEDDING_MODEL
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.schemas import Evidence, Tier
from app.settings import get_settings

MAX_RETRIEVED_CHUNKS = 8  # Plan §8.4
_EXIT = {
    "answered": 0, "clarification": 0, "insufficient_evidence": 1, "blocked": 1,
    "temporarily_unavailable": 3,
}


def _print(result: TurnResult) -> None:
    print(f"status: {result.status} ({result.detail})")
    if result.plan:
        print(f"plan: intent={result.plan.intent} query={result.plan.retrieval_query!r}")
    if result.proposed_tool:
        print(f"tool proposed (NOT executed): {result.proposed_tool.model_dump_json()}")
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
    embedding_model = FAKE_EMBEDDING_MODEL if fake else settings.embedding_model
    tiers = allowed_tiers(tier)

    engine: AsyncEngine = create_engine(settings)
    started = time.perf_counter()
    turn_id = None
    try:
        turn_id = await repositories.create_turn(engine, user_id="cli", question=question)

        async def retrieve(query: str, embedding: Sequence[float]) -> Sequence[Evidence]:
            return await hybrid_search(
                engine, query_text=query, query_embedding=embedding,
                allowed_tiers=tiers, limit=MAX_RETRIEVED_CHUNKS,
            )

        result = await run_turn(
            question=question, models=models, prompts=prompts,
            recorder=SqlModelCallRecorder(engine, turn_id), retrieve=retrieve,
            embedding_model=embedding_model, timeout_seconds=settings.chat_timeout_seconds,
        )
        await repositories.finish_turn(
            engine, turn_id, graph_status=result.status, answer=result.answer,
            sources=[
                {"document_version_id": str(c.document_version_id), "chunk_id": str(c.chunk_id)}
                for c in result.citations
            ],
            retrieval_attempts=result.retrieval_attempts,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        _print(result)
        return _EXIT[result.status]
    except Exception as exc:
        if turn_id is not None:
            await repositories.finish_turn(
                engine, turn_id, graph_status="error", answer=None, sources=[],
                retrieval_attempts=0, latency_ms=int((time.perf_counter() - started) * 1000),
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
