"""Eval runner CLI (Plan §16.4).

    python -m app.eval.cli run --questions app/eval/golden/questions.json \
        --out ./eval_reports/2026-09-23 --fake
    python -m app.eval.cli run --questions app/eval/golden/questions.json \
        --out ./eval_reports/2026-09-23      # needs GEMINI_API_KEY

`--fake` is plumbing-only: `build_fake_registry()`'s deterministic responder produces meaningless
correctness/retrieval numbers, same caveat as `app.ai.cli ask --fake`. Point `DATABASE_URL` at a
DB with the golden corpus already ingested (see README) before running for real -- this CLI never
ingests anything itself.

Exit codes: 0 on a clean run (regardless of how questions scored -- this is a report, not a pass/
fail gate), 2 on a usage/config error, 4 on an unexpected error.
"""
import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from app.ai.errors import ConfigurationError, PromptError
from app.ai.prompts.loader import load_prompts
from app.ai.registry import build_fake_registry, build_registry
from app.db.session import create_engine
from app.eval.report import write_report
from app.eval.runner import run_eval
from app.eval.schema import GoldenQuestionError, load_golden_questions
from app.ingestion.fake_embedder import FAKE_EMBEDDING_MODEL
from app.settings import get_settings
from app.tools.fake import FakeSubscriptionToolClient
from app.tools.subscription import SubscriptionSnapshot, SubscriptionToolClient, build_tool_client

# Canned, documented snapshots for --fake mode's tool-required golden questions (no real
# Payment/Subscription host exists to eval against -- an unscripted FakeSubscriptionToolClient
# raises on every call). One more entry than the corpus has tool-required questions, so a stray
# extra call surfaces as a graceful "insufficient_evidence"/error, not a silent script exhaustion.
_FAKE_TOOL_SCRIPT: tuple[SubscriptionSnapshot, ...] = tuple(
    SubscriptionSnapshot(
        subscription_id="sub_eval_demo", customer_id="eval", status="active",
        plan_name="Pro", current_period_end="2026-12-31T00:00:00Z",
        observed_at="2026-09-23T00:00:00Z",
    )
    for _ in range(5)
)


async def _run(questions_path: Path, out_dir: Path, *, fake: bool) -> int:
    settings = get_settings()
    try:
        questions = load_golden_questions(questions_path)
    except GoldenQuestionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not questions:
        print(f"error: {questions_path} has no questions", file=sys.stderr)
        return 2

    try:
        models = build_fake_registry() if fake else build_registry(settings)
        prompts = load_prompts(settings.prompt_version)
    except (ConfigurationError, PromptError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    tool_client: SubscriptionToolClient | None
    if fake:
        tool_client = FakeSubscriptionToolClient(_FAKE_TOOL_SCRIPT)
    else:
        try:
            tool_client = build_tool_client(settings)
        except ConfigurationError:
            tool_client = None

    engine = create_engine(settings)
    try:
        print(f"running {len(questions)} questions...", file=sys.stderr)
        results = await run_eval(
            engine, models, prompts, tool_client, questions,
            embedding_model=FAKE_EMBEDDING_MODEL if fake else settings.embedding_model,
            chat_timeout_seconds=settings.chat_timeout_seconds,
            tool_timeout_seconds=settings.subscription_tool_timeout_seconds,
        )
        write_report(results, out_dir)
        print(f"wrote {out_dir / 'results.json'} and {out_dir / 'summary.md'}")
        matched = sum(1 for r in results if r.status_matches_expected)
        print(f"status matched expected for {matched}/{len(results)} questions", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.eval.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the golden question set through the real graph")
    run.add_argument("--questions", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--fake", action="store_true", help="use fake models (no API key needed)")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args.questions, args.out, fake=args.fake))


if __name__ == "__main__":
    raise SystemExit(main())
