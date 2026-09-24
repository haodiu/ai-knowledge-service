from app.eval.pricing import CHAT_PRICING, estimate_cost_usd


def test_returns_none_for_an_unknown_model() -> None:
    assert estimate_cost_usd("some-model-nobody-pinned", 1000, 1000) is None


def test_computes_cost_for_a_known_model() -> None:
    model = next(iter(CHAT_PRICING))
    pricing = CHAT_PRICING[model]

    cost = estimate_cost_usd(model, 1000, 1000)

    assert cost == pricing.input_per_1k_tokens_usd + pricing.output_per_1k_tokens_usd


def test_zero_tokens_costs_zero_for_a_known_model() -> None:
    model = next(iter(CHAT_PRICING))
    assert estimate_cost_usd(model, 0, 0) == 0.0
