from pathlib import Path

import pytest

from app.ingestion.parsing import ParsingError, discover_documents, parse_document
from app.retrieval.schemas import Tier


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_parses_front_matter_and_relative_posix_external_id(tmp_path: Path) -> None:
    f = _write(
        tmp_path / "policies" / "refund.md", "---\ntitle: Refunds\ntier: internal\n---\nBody.\n"
    )
    doc = parse_document(f, tmp_path)
    assert (doc.external_id, doc.title, doc.tier, doc.content) == (
        "policies/refund.md",
        "Refunds",
        Tier.INTERNAL,
        "Body.",
    )


@pytest.mark.parametrize(
    "text",
    [
        "no front matter\n",
        "---\ntitle: T\n---\nBody\n",  # tier missing -> must NOT default to general
        "---\ntitle: T\ntier:\n---\nBody\n",
        "---\ntitle: T\ntier: public\n---\nBody\n",
        "---\ntier: general\n---\nBody\n",
        "---\ntitle: T\ntier: general\n",  # unclosed
        "---\ntitle: T\ntier: general\n---\n   \n",  # empty body
        "---\ntitle T\ntier: general\n---\nBody\n",  # malformed line
    ],
)
def test_rejects_files_that_are_not_fully_labelled(tmp_path: Path, text: str) -> None:
    with pytest.raises(ParsingError):
        parse_document(_write(tmp_path / "bad.md", text), tmp_path)


def test_rejects_non_utf8(tmp_path: Path) -> None:
    f = tmp_path / "bin.md"
    f.write_bytes(b"---\ntitle: T\ntier: general\n---\n\xff\xfe")
    with pytest.raises(ParsingError):
        parse_document(f, tmp_path)


def test_discovers_supported_files_sorted(tmp_path: Path) -> None:
    _write(tmp_path / "b.md", "x")
    _write(tmp_path / "a" / "c.txt", "x")
    _write(tmp_path / "ignore.pdf", "x")
    assert [p.relative_to(tmp_path).as_posix() for p in discover_documents(tmp_path)] == [
        "a/c.txt",
        "b.md",
    ]


def test_symlink_escaping_the_root_is_skipped(tmp_path: Path) -> None:
    outside = _write(tmp_path / "outside" / "secret.md", "x")
    root = tmp_path / "docs"
    _write(root / "ok.md", "x")
    (root / "link.md").symlink_to(outside)
    assert [p.name for p in discover_documents(root)] == ["ok.md"]
