"""Ingestion CLI (Plan §12.8). Every subcommand calls into `app.ingestion.service`/`app.ingestion
.tasks` directly -- there is no second pipeline here (Plan §12.8: "CLI không được có pipeline
riêng khác Celery task").

    python -m app.ingestion.cli ingest ./documents
    python -m app.ingestion.cli ingest ./documents --queue
    python -m app.ingestion.cli retry-failed
    python -m app.ingestion.cli republish-stale
    python -m app.ingestion.cli cleanup-versions --keep-last 2 --older-than-days 30

By default `ingest` embeds with the deterministic fake embedder and processes synchronously
in-process, so vector ranking over the data is meaningless and only full-text search is
meaningful; no RabbitMQ/worker is needed. `--embeddings gemini` uses the real EmbeddingClient
(needs GEMINI_API_KEY) still in-process. `--queue` instead creates the job and hands it to a
Celery worker over RabbitMQ (needs GEMINI_API_KEY -- the worker always embeds for real) -- the
only way, until the HTTP ingestion endpoint exists, to demo redelivery/worker-restart resilience
end to end.
"""
import argparse
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import Engine, text

from app.ai.embeddings.gemini import GeminiEmbeddingClient
from app.ai.embeddings.sync import SyncEmbedder
from app.db.session import create_sync_engine
from app.ingestion.fake_embedder import FAKE_EMBEDDING_MODEL, fake_embed
from app.ingestion.parsing import ParsingError, discover_documents, parse_document
from app.ingestion.service import Embedder, IngestionError, create_ingestion_job, ingest_document
from app.settings import get_settings


def _ingest(path: Path, embeddings: str, *, queue: bool) -> int:
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2
    files = discover_documents(path)
    if not files:
        print(f"error: no .md/.txt files under {path}", file=sys.stderr)
        return 2

    settings = get_settings()
    if queue:
        # The worker (app/ingestion/tasks.py) always embeds for real; --embeddings is ignored in
        # --queue mode (there is nothing to embed in this process) but the API key check still
        # applies, so a missing key is caught at submission time rather than wedging every job.
        if settings.gemini_api_key is None:
            print("error: GEMINI_API_KEY is required for --queue", file=sys.stderr)
            return 2
        embedding_model = settings.embedding_model
        real: SyncEmbedder | None = None
        embed: Embedder | None = None
    elif embeddings == "gemini":
        if settings.gemini_api_key is None:
            print("error: GEMINI_API_KEY is required for --embeddings gemini", file=sys.stderr)
            return 2
        real = SyncEmbedder(
            GeminiEmbeddingClient(
                api_key=settings.gemini_api_key.get_secret_value(),
                timeout_seconds=settings.embedding_timeout_seconds,
            ),
            model_version=settings.embedding_model,
        )
        embed, embedding_model = real, settings.embedding_model
    else:
        print(
            "warning: using the fake deterministic embedder; vector search is NOT semantic",
            file=sys.stderr,
        )
        real = None
        embed, embedding_model = fake_embed, FAKE_EMBEDDING_MODEL

    engine = create_sync_engine(settings)
    failures = 0
    try:
        for file in files:
            try:
                doc = parse_document(file, path)
                if queue:
                    job = create_ingestion_job(
                        engine,
                        external_id=doc.external_id,
                        title=doc.title,
                        tier=doc.tier,
                        content=doc.content,
                        embedding_model=embedding_model,
                    )
                    if job.status == "completed":
                        print(f"{doc.external_id}: unchanged (v{job.version_no})")
                    else:
                        assert job.job_id is not None
                        _dispatch(engine, job.job_id)
                        print(f"{doc.external_id}: queued (job {job.job_id}, v{job.version_no})")
                else:
                    assert embed is not None
                    result = ingest_document(
                        engine,
                        external_id=doc.external_id,
                        title=doc.title,
                        tier=doc.tier,
                        content=doc.content,
                        embed=embed,
                        embedding_model=embedding_model,
                    )
                    print(f"{doc.external_id}: {result.outcome} (v{result.version_no})")
            except (ParsingError, IngestionError) as exc:
                failures += 1
                print(f"{file}: REJECTED - {exc}", file=sys.stderr)
            except Exception as exc:  # one bad file must not stop the batch
                failures += 1
                print(f"{file}: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        engine.dispose()
        if real is not None:
            real.close()
    return 1 if failures else 0


def _dispatch(engine: Engine, job_id: uuid.UUID) -> None:
    """Publish `job_id` to the worker and record the task id + a re-publish cooldown on the job
    row, so `republish-stale` does not immediately re-select something just dispatched."""
    from app.ingestion.tasks import ingest_document_task  # local: needs RABBITMQ_URL resolvable

    result = ingest_document_task.delay(str(job_id))
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE ingestion_jobs SET celery_task_id = :t, "
                "available_at = now() + interval '5 minutes' WHERE id = :j"
            ),
            {"t": result.id, "j": job_id},
        )


def _retry_failed(engine: Engine) -> int:
    """Re-queue jobs that failed for a retryable reason (Plan §12.8). `superseded_content` is
    excluded on purpose: it means the submitted content itself is wrong (identical to a version
    that was already superseded), not a transient failure -- retrying it would just fail again."""
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "UPDATE ingestion_jobs SET status = 'queued', error_code = NULL, "
                "error_message = NULL, completed_at = NULL, retry_count = retry_count + 1 "
                "WHERE status = 'failed' "
                "AND (error_code IS NULL OR error_code <> 'superseded_content') "
                "RETURNING id, document_version_id"
            )
        ).all()
        # The job row alone isn't enough: run_ingestion_job() re-activates its linked version, and
        # activate_version() only accepts a `building` one -- a permanent failure left it `failed`
        # (_mark_version_failed), so a retry must reset that too, in the same transaction.
        for row in rows:
            if row.document_version_id is not None:
                conn.execute(
                    text(
                        "UPDATE document_versions SET status = 'building' "
                        "WHERE id = :v AND status = 'failed'"
                    ),
                    {"v": row.document_version_id},
                )
    for row in rows:
        _dispatch(engine, row.id)
    print(f"retried {len(rows)} job(s)")
    return 0


def _republish_stale(engine: Engine) -> int:
    """Publisher recovery (Plan §12.3): a job can be `queued` in PostgreSQL with no Celery message
    ever having reached RabbitMQ (the process died between the DB commit and the publish), or with
    a message that was lost. Idempotent: `run_ingestion_job()` no-ops a job that is no longer
    `queued`/`retrying` by the time a worker picks it up, so publishing twice is harmless."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id FROM ingestion_jobs WHERE status IN ('queued', 'retrying') "
                "AND (celery_task_id IS NULL OR available_at < now())"
            )
        ).all()
    for row in rows:
        _dispatch(engine, row.id)
    print(f"republished {len(rows)} job(s)")
    return 0


def _cleanup_versions(engine: Engine, keep_last: int, older_than_days: int) -> int:
    """Retention (Plan §12.7): delete `superseded`/`failed` versions older than
    `older_than_days`, keeping at least `keep_last` most-recent versions (by version_no) per
    document regardless of age, and never touching `active`/`building` versions (they are never
    candidates -- excluded by the status filter below, not by the rank).

    A version's status is monotonic once `superseded`/`failed` -- nothing reactivates it -- so the
    plain `DELETE ... WHERE status IN (...)` is race-safe on its own without a separate
    SELECT-FOR-UPDATE step. `chunks` cascade-deletes with it; `ingestion_jobs.document_version_id`
    is nulled (ON DELETE SET NULL), not cascaded; `turn_sources` has no FK at all (invariant #6) --
    citation history keeps working after this runs.
    """
    with engine.begin() as conn:
        ranked = conn.execute(
            text(
                "SELECT id, status, "
                "ROW_NUMBER() OVER (PARTITION BY document_id ORDER BY version_no DESC) AS rank "
                "FROM document_versions"
            )
        ).all()
        deleted = 0
        for row in ranked:
            if row.status not in ("superseded", "failed") or row.rank <= keep_last:
                continue
            result = conn.execute(
                text(
                    "DELETE FROM document_versions WHERE id = :v "
                    "AND status IN ('superseded', 'failed') "
                    "AND created_at < now() - make_interval(days => :days)"
                ),
                {"v": row.id, "days": older_than_days},
            )
            deleted += result.rowcount
    print(f"deleted {deleted} version(s)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ingestion.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingest a directory (or one file) of .md/.txt documents")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--embeddings", choices=["fake", "gemini"], default="fake")
    ingest.add_argument(
        "--queue",
        action="store_true",
        help="hand off to a Celery worker over RabbitMQ instead of processing in-process",
    )

    sub.add_parser("retry-failed", help="re-queue jobs that failed for a retryable reason")
    sub.add_parser("republish-stale", help="re-publish queued jobs Celery may never have received")

    cleanup = sub.add_parser("cleanup-versions", help="delete old superseded/failed versions")
    cleanup.add_argument("--keep-last", type=int, default=2)
    cleanup.add_argument("--older-than-days", type=int, default=30)

    args = parser.parse_args(argv)

    if args.command == "ingest":
        return _ingest(args.path, args.embeddings, queue=args.queue)

    engine = create_sync_engine(get_settings())
    try:
        if args.command == "retry-failed":
            return _retry_failed(engine)
        if args.command == "republish-stale":
            return _republish_stale(engine)
        if args.command == "cleanup-versions":
            return _cleanup_versions(engine, args.keep_last, args.older_than_days)
        raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
