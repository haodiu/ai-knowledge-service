"""Deterministic paragraph chunker (Plan §16.1: "Chunking deterministic").

Same input -> same chunks, always: no randomness, no dependence on locale or time. Bump
CHUNKING_VERSION (or change MAX_CHUNK_CHARS) and index_config_hash changes with it, which forces
a re-index of unchanged content (Plan §12.6 rule 2).
"""
import re

CHUNKING_VERSION = "paragraph-v1"
MAX_CHUNK_CHARS = 1200

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Split one over-long paragraph on whitespace; hard-slice any single word that is too long."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for word in paragraph.split():
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def chunk_text(content: str, *, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Greedily pack whole paragraphs into chunks of at most `max_chars` characters."""
    normalised = content.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(normalised) if p.strip()]

    chunks: list[str] = []
    current = ""
    for piece in (part for p in paragraphs for part in _split_long(p, max_chars)):
        if not current:
            current = piece
        elif len(current) + 2 + len(piece) <= max_chars:
            current += "\n\n" + piece
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks
