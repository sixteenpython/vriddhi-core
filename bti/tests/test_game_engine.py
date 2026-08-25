from __future__ import annotations

import json

import pytest

from bti.game_engine import BTIGame, GameRuleError


def _fund_one_month(game: BTIGame) -> dict:
    state = game.public_state()
    buying_power = state["cash_paise"] + state["monthly_amount_rupees"] * 100
    stocks = game.market_view()["stocks"]
    affordable = [stock for stock in stocks if stock["close_paise"] <= buying_power]
    stock = min(affordable, key=lambda item: item["close_paise"])
    shares = buying_power // stock["close_paise"]
    return game.submit_move(
        [{"side": "BUY", "ticker": stock["ticker"], "shares": shares}],
        expected_month=state["moves_completed"],
    )


def test_same_seed_is_exactly_reproducible() -> None:
    left = BTIGame.create(25_000, 24, "determinism")
    right = BTIGame.create(25_000, 24, "determinism")
    for _ in range(4):
        assert _fund_one_month(left) == _fund_one_month(right)
    assert left.to_json() == right.to_json()
    restored = BTIGame.from_json(left.to_json())
    assert restored.public_state() == left.public_state()


def test_execution_accounting_and_whole_share_rules() -> None:
    game = BTIGame.create(50_000, 36, "accounting")
    market = {s["ticker"]: s for s in game.market_view()["stocks"]}
    ticker = min(market, key=lambda name: market[name]["close_paise"])
    price = market[ticker]["close_paise"]
    shares = 5_000_000 // price
    result = game.submit_move([{"side": "BUY", "ticker": ticker, "shares": shares}])
    assert result["execution"][0]["cash_movement_paise"] == -(shares * price)
    assert game.public_state()["cash_paise"] == 5_000_000 - shares * price
    with pytest.raises(GameRuleError, match="whole numbers"):
        BTIGame.create(50_000, 24, "fraction").submit_move(
            [{"side": "BUY", "ticker": ticker, "shares": 1.5}]
        )


def test_validation_immutability_and_resignation() -> None:
    game = BTIGame.create(10_000, 24, "rules")
    with pytest.raises(GameRuleError, match="Cannot sell"):
        game.submit_move([{"side": "SELL", "ticker": "BPCL", "shares": 1}])
    with pytest.raises(GameRuleError, match="90%"):
        game.submit_move([{"side": "BUY", "ticker": "BPCL", "shares": 1}])
    _fund_one_month(game)
    frozen = json.dumps(game.public_state()["last_result"], sort_keys=True)
    with pytest.raises(GameRuleError, match="stale"):
        game.submit_move([{"side": "BUY", "ticker": "BPCL", "shares": 1}], expected_month=0)
    assert json.dumps(game.public_state()["last_result"], sort_keys=True) == frozen
    outcome = game.resign()
    assert outcome["recorded_as"] == "LOSS"
    with pytest.raises(GameRuleError, match="active"):
        _fund_one_month(game)


def test_public_payload_never_exposes_private_oracle() -> None:
    game = BTIGame.create(25_000, 48, "privacy")
    _fund_one_month(game)
    payload = json.dumps({"state": game.public_state(), "market": game.market_view()}).lower()
    forbidden = (
        "reference_weights",
        "reference_holdings",
        "optimal_view",
        "efficient_frontier",
        "target_weight",
        "regret",
        "utility",
    )
    assert not any(term in payload for term in forbidden)
    assert game.market_view()["label"] == "SIMULATED MARKET"


def test_public_stock_research_projection_is_rich_and_chartable() -> None:
    game = BTIGame.create(25_000, 36, "research-projection")
    stock = game.market_view()["stocks"][0]
    assert stock["overall_rank"] > 0
    assert isinstance(stock["historical_cagr_pct"], float)
    assert len(stock["ohlc_history"]) == 252
    assert stock["ohlc_history"][-1]["close_paise"] == stock["close_paise"]
    assert [point["months"] for point in stock["forecast_curve"]] == [12, 24, 36, 48, 60]
    assert 0 <= stock["rsi_14"] <= 100
    assert 0 <= stock["sentiment_score"] <= 100
    assert stock["volume_index"] > 0
    assert stock["roe_pct"] == pytest.approx(stock["pb"] / stock["pe"] * 100, rel=1e-2)


@pytest.mark.parametrize("horizon", [24, 36, 48, 60])
def test_complete_campaign_smoke(horizon: int) -> None:
    game = BTIGame.create(10_000, horizon, f"smoke-{horizon}")
    for _ in range(horizon):
        result = _fund_one_month(game)
        assert result["classification"] in {
            "BRILLIANT",
            "EXCELLENT",
            "GOOD",
            "INACCURACY",
            "MISTAKE",
            "BLUNDER",
        }
        assert 0 <= result["score"] <= 100
    assert game.status == "COMPLETED"
    final = game.final_result()
    assert final["months_completed"] == horizon
    assert final["total_invested_paise"] == horizon * 1_000_000
    assert final["portfolio_value_paise"] > 0
    assert final["benchmark_value_paise"] > 0


def test_rating_is_capital_neutral_for_equivalent_decisions() -> None:
    small = BTIGame.create(10_000, 24, "capital-neutral")
    large = BTIGame.create(100_000, 24, "capital-neutral")
    # Both choose the same one-stock process; ratings depend on quality, never rupee profit.
    small_result = _fund_one_month(small)
    large_result = _fund_one_month(large)
    assert small_result["score"] == pytest.approx(large_result["score"], abs=8)
    assert abs(small.public_state()["rating"] - large.public_state()["rating"]) <= 32


def test_rated_history_is_immutable_reconstructable_and_chase_ready() -> None:
    game = BTIGame.create(25_000, 36, "rated-history")
    first = _fund_one_month(game)
    second = _fund_one_month(game)
    state_before_review = game.to_json()
    review = game.review_move(1)

    assert game.public_state()["gameplay_mode"] == "RATED"
    assert game.public_state()["can_repeat_last_move"] is True
    assert game.public_state()["last_move_instructions"]
    assert first["notation"].startswith("M01 ·")
    assert -3 <= first["position_evaluation"]["value"] <= 3
    assert len(game.performance_series()) == 2
    assert game.performance_series()[-1]["move"] == 2
    assert review["review_mode"] is True
    assert review["selected_move"] == 1
    assert len(review["performance_series"]) == 1
    assert review["result"] == first
    assert review["market"]["month"] == 1
    assert game.to_json() == state_before_review
    assert second["regime"]["difficulty"] >= first["regime"]["difficulty"]


def test_market_regime_path_is_precomputed_and_portfolio_independent() -> None:
    left = BTIGame.create(25_000, 24, "fair-market")
    right = BTIGame.create(25_000, 24, "fair-market")
    left.state["holdings"] = {"BPCL": 10}
    right.state["holdings"] = {"TCS": 10}
    assert left.state["regime_schedule"] == right.state["regime_schedule"]
    assert left._market_at_month(6) == right._market_at_month(6)
