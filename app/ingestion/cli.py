"""Ingestion CLI (Plan §12.8). Synchronous and direct: it calls the same `ingest_document()` the
Week 5 Celery task will call — there is no second pipeline here.

    python -m app.ingestion.cli ingest ./documents

By default this embeds with the deterministic fake embedder, so vector ranking over the data is
meaningless and only full-text search is meaningful. `--embeddings gemini` uses the real
EmbeddingClient (needs GEMINI_API_KEY); it backs off, boundedly, on rate limits.
"""
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from app.ai.embeddings.gemini import GeminiEmbeddingClient
from app.ai.embeddings.sync import SyncEmbedder
from app.db.session import create_sync_engine
from app.ingestion.fake_embedder import FAKE_EMBEDDING_MODEL, fake_embed
from app.ingestion.parsing import ParsingError, discover_documents, parse_document
from app.ingestion.service import Embedder, IngestionError, ingest_document
from app.settings import get_settings


def _ingest(path: Path, embeddings: str) -> int:
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2
    files = discover_documents(path)
    if not files:
        print(f"error: no .md/.txt files under {path}", file=sys.stderr)
        return 2

    settings = get_settings()
    real: SyncEmbedder | None = None
    embed: Embedder
    if embeddings == "gemini":
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
        embed, embedding_model = fake_embed, FAKE_EMBEDDING_MODEL
    engine = create_sync_engine(settings)
    failures = 0
    try:
        for file in files:
            try:
                doc = parse_document(file, path)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ingestion.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="ingest a directory (or one file) of .md/.txt documents")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--embeddings", choices=["fake", "gemini"], default="fake")
    args = parser.parse_args(argv)
    return _ingest(args.path, args.embeddings)


if __name__ == "__main__":
    raise SystemExit(main())
