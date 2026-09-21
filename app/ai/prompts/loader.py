"""Versioned prompts: app/ai/prompts/<version>/<role>.md, one directory per released version.

A released version is never edited in place; a wording change is a new `vN/` directory. The active
version comes from PROMPT_VERSION and is what `model_calls.prompt_version` records.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.ai.errors import PromptError

_ROOT = Path(__file__).parent
_VERSION = re.compile(r"v\d+")


@dataclass(frozen=True)
class Prompts:
    version: str
    planner: str
    grader: str
    answer: str


@lru_cache
def load_prompts(version: str) -> Prompts:
    if not _VERSION.fullmatch(version):  # also blocks path traversal
        raise PromptError(f"invalid prompt version {version!r} (expected e.g. 'v1')")
    directory = _ROOT / version
    if not directory.is_dir():
        raise PromptError(f"unknown prompt version {version!r}")
    try:
        texts = {
            role: (directory / f"{role}.md").read_text(encoding="utf-8").strip()
            for role in ("planner", "grader", "answer")
        }
    except FileNotFoundError as exc:
        raise PromptError(f"prompt version {version!r} is incomplete: {exc.filename}") from None
    return Prompts(version=version, **texts)
