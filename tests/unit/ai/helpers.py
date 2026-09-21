"""Shared builders for the Week 3 unit tests."""
import uuid

from app.retrieval.schemas import Evidence


def make_evidence(n: int = 2, *, text_prefix: str = "chunk") -> list[Evidence]:
    doc = uuid.uuid4()
    return [
        Evidence(
            chunk_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            document_id=doc,
            title=f"Doc {i}",
            version_no=1,
            chunk_index=i,
            text=f"{text_prefix} {i}",
            score=1.0 / (i + 1),
        )
        for i in range(n)
    ]
