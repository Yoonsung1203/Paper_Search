from __future__ import annotations

import pytest

from paper_search.llm.client import PRICING, LlmUsage, estimate_cost


def test_estimate_cost_opus5() -> None:
    # 1M 입력 + 1M 출력 = $5 + $25
    assert estimate_cost("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)


def test_cache_read_is_cheaper_than_fresh_input() -> None:
    fresh = estimate_cost("claude-opus-5", 1_000_000, 0)
    cached = estimate_cost("claude-opus-5", 0, 0, cache_read_tokens=1_000_000)
    assert cached < fresh
    assert cached == pytest.approx(fresh * 0.1)


def test_unknown_model_falls_back_to_default_pricing() -> None:
    assert estimate_cost("mystery-model", 1_000_000, 0) == pytest.approx(
        estimate_cost("claude-opus-5", 1_000_000, 0)
    )


def test_pricing_table_has_the_configured_default() -> None:
    assert "claude-opus-5" in PRICING


def test_usage_tracks_per_stage_and_cache_ratio() -> None:
    usage = LlmUsage()
    usage.add("score", 200, 50, 800, 0.01)
    usage.add("summarize", 1000, 300, 0, 0.02)

    assert usage.calls == 2
    assert usage.cost_usd == pytest.approx(0.03)
    assert usage.per_stage == {"score": pytest.approx(0.01), "summarize": pytest.approx(0.02)}
    assert usage.cache_hit_ratio == pytest.approx(800 / 2000)


def test_cache_ratio_is_zero_when_nothing_recorded() -> None:
    assert LlmUsage().cache_hit_ratio == 0.0
