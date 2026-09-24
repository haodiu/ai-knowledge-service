"""Estimated $/token pricing (Plan §16.4's "estimated cost/turn" metric).

Plain module constants, edited by hand whenever list pricing changes -- not a runtime `Settings`
field (pricing is not a per-deployment knob, it's a small table reviewed via diff, the same
treatment app/graph/limits.py gives its hard limits) and not fetched from any provider API (no new
dependency, no network call in the eval runner's hot path).

This is the ONLY place in the codebase that computes a dollar figure: `model_calls`/
`embedding_calls` deliberately store tokens/batch size only, never cost (see
app/db/models.py::EmbeddingCall's docstring) -- so pricing has exactly one home instead of two
drifting apart.

Prices below are placeholders as of this file's writing; update them from the provider's current
published pricing before trusting a report's cost column.
"""
from dataclasses import dataclass

ESTIMATED = True  # marker used by app/eval/report.py to label the cost column


@dataclass(frozen=True)
class ModelPricing:
    input_per_1k_tokens_usd: float
    output_per_1k_tokens_usd: float


# ESTIMATED, USD per 1,000 tokens. Keyed by the exact pinned model name (Settings rejects
# "-latest" aliases for the same reproducibility reason, see app/settings.py::_pinned_model_name).
CHAT_PRICING: dict[str, ModelPricing] = {
    "gemini-3.1-flash-lite": ModelPricing(
        input_per_1k_tokens_usd=0.00010, output_per_1k_tokens_usd=0.00040
    ),
    "gemini-3.6-flash": ModelPricing(
        input_per_1k_tokens_usd=0.00030, output_per_1k_tokens_usd=0.00250
    ),
}

EMBEDDING_PRICING_PER_1K_TOKENS_USD: dict[str, float] = {
    "gemini-embedding-001": 0.00013,
}


def estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float | None:
    """None for an unpinned/unknown model name -- fail loud (a blank report cell) rather than
    silently guess a wrong number."""
    pricing = CHAT_PRICING.get(model_name)
    if pricing is None:
        return None
    return (
        input_tokens / 1000 * pricing.input_per_1k_tokens_usd
        + output_tokens / 1000 * pricing.output_per_1k_tokens_usd
    )


def estimate_embedding_cost_usd(model_version: str, token_estimate: int) -> float | None:
    """Embedding calls don't record tokens (only batch_size -- see EmbeddingCall's docstring), so
    this takes a caller-supplied token estimate rather than reading one from the DB."""
    price = EMBEDDING_PRICING_PER_1K_TOKENS_USD.get(model_version)
    if price is None:
        return None
    return token_estimate / 1000 * price
