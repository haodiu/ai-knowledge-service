"""Pydantic JSON Schema -> the subset OpenAI-compatible providers accept for `json_schema`.

Inlines `$ref`, turns `anyOf [X, null]` into `X + nullable`, and drops keywords providers tend to
reject (`title`, `default`, `additionalProperties`, `format`). This schema is only a *hint* that
steers the model; `parse_structured` (Pydantic) remains the gate, and it is the only place that can
enforce cross-field rules anyway.
"""
from typing import Any

from pydantic import BaseModel

_DROPPED = {"title", "default", "additionalProperties", "format", "$defs"}


def to_provider_schema(model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema()
    return _convert(raw, raw.get("$defs", {}))  # type: ignore[no-any-return]


def _convert(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_convert(item, defs) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        target = _convert(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
        extra = {k: v for k, v in node.items() if k != "$ref" and k not in _DROPPED}
        return {**target, **_convert(extra, defs)}

    if "anyOf" in node:
        variants = node["anyOf"]
        concrete = [v for v in variants if v.get("type") != "null"]
        if len(concrete) != 1:
            raise ValueError("only `T | None` unions can be expressed for providers")
        inner = _convert(concrete[0], defs)
        if len(concrete) != len(variants):
            inner["nullable"] = True
        rest = {k: v for k, v in node.items() if k != "anyOf" and k not in _DROPPED}
        return {**inner, **_convert(rest, defs)}

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROPPED:
            continue
        if key == "const":
            out["enum"] = [value]
        elif key == "properties":  # keys are field names, not keywords: never drop by name
            out[key] = {name: _convert(sub, defs) for name, sub in value.items()}
        else:
            out[key] = _convert(value, defs)
    return out
