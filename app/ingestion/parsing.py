"""Read documents from disk (Plan §12.8 CLI). Metadata comes from a small front-matter block:

    ---
    title: Refund policy
    tier: general
    ---
    body...

`tier` is required and has no default: a missing/unknown tier rejects the file. Defaulting to
`general` would be fail-open (an unlabeled internal document would become world-readable).
"""
from dataclasses import dataclass
from pathlib import Path

from app.retrieval.schemas import Tier

SUPPORTED_SUFFIXES = (".md", ".txt")


class ParsingError(ValueError):
    """The file cannot be ingested as written; nothing was touched."""


@dataclass(frozen=True)
class ParsedDocument:
    external_id: str  # path relative to the ingest root, POSIX separators
    title: str
    tier: Tier
    content: str


def discover_documents(root: Path) -> list[Path]:
    """Supported files under `root` (or `root` itself if it is a file), sorted, no escapes."""
    if root.is_file():
        return [root]
    resolved_root = root.resolve()
    found = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and p.resolve().is_relative_to(resolved_root)  # skip symlinks pointing outside the tree
    ]
    return sorted(found)


def _split_front_matter(raw: str, path: Path) -> tuple[dict[str, str], str]:
    lines = raw.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        raise ParsingError(
            f"{path}: missing front matter (need a leading '---' block with title/tier)"
        )
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise ParsingError(f"{path}: front matter is not closed with '---'") from None
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ParsingError(
                f"{path}: malformed front-matter line {line!r} (expected 'key: value')"
            )
        meta[key.strip().lower()] = value.strip()
    return meta, "\n".join(lines[end + 1 :]).strip()


def parse_document(path: Path, root: Path) -> ParsedDocument:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ParsingError(f"{path}: not valid UTF-8") from exc

    meta, body = _split_front_matter(raw, path)
    title = meta.get("title", "")
    if not title:
        raise ParsingError(f"{path}: front matter has no 'title'")
    tier_raw = meta.get("tier", "")
    try:
        tier = Tier(tier_raw)
    except ValueError:
        allowed = ", ".join(t.value for t in Tier)
        raise ParsingError(f"{path}: 'tier' must be one of [{allowed}], got {tier_raw!r}") from None
    if not body:
        raise ParsingError(f"{path}: document body is empty")

    base = root if root.is_dir() else root.parent
    return ParsedDocument(
        external_id=path.relative_to(base).as_posix(), title=title, tier=tier, content=body
    )
