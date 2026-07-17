from sage.generation.cost import estimate_cost_usd


def test_estimate_cost_usd_uses_configured_rates():
    cost = estimate_cost_usd("gemini-2.5-flash", prompt_tokens=1000, completion_tokens=1000)
    assert cost == 0.0003 + 0.0025


def test_estimate_cost_usd_falls_back_to_zero_for_unknown_model():
    assert (
        estimate_cost_usd("some-unlisted-model", prompt_tokens=1000, completion_tokens=1000) == 0.0
    )


def test_estimate_cost_usd_scales_with_token_count():
    cost_500 = estimate_cost_usd("gemini-2.5-flash", prompt_tokens=500, completion_tokens=0)
    cost_1000 = estimate_cost_usd("gemini-2.5-flash", prompt_tokens=1000, completion_tokens=0)
    assert cost_1000 == cost_500 * 2
