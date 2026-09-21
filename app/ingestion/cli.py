"""Ingestion CLI (Plan §12.8). Synchronous and direct: it calls the same `ingest_document()` the
Week 5 Celery task will call — there is no second pipeline here.

    python -m app.ingestion.cli ingest ./documents

Week 2 embeds with the deterministic fake embedder (no provider exists until Week 3), so vector
ranking over data ingested here is meaningless; only full-text search is meaningful.
"""
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from app.db.session import create_sync_engine
from app.ingestion.fake_embedder import FAKE_EMBEDDING_MODEL, fake_embed
from app.ingestion.parsing import ParsingError, discover_documents, parse_document
from app.ingestion.service import IngestionError, ingest_document
from app.settings import get_settings


def _ingest(path: Path) -> int:
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2
    files = discover_documents(path)
    if not files:
        print(f"error: no .md/.txt files under {path}", file=sys.stderr)
        return 2

    print(
        "warning: using the fake deterministic embedder; vector search is NOT semantic",
        file=sys.stderr,
    )
    engine = create_sync_engine(get_settings())
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
                    embed=fake_embed,
                    embedding_model=FAKE_EMBEDDING_MODEL,
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
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ingestion.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="ingest a directory (or one file) of .md/.txt documents")
    ingest.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    return _ingest(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
