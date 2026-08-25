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
    assert len(stock["ohlc_history"]) >= 12
    assert stock["ohlc_history"][-1]["close_paise"] == stock["close_paise"]
    assert [point["months"] for point in stock["forecast_curve"]] == [12, 24, 36, 48, 60]


@pytest.mark.parametrize("horizon", [24, 36, 48, 60])
def test_complete_campaign_smoke(horizon: int) -> None:
    game = BTIGame.create(10_000, horizon, f"smoke-{horizon}")
    for _ in range(horizon):
        result = _fund_one_month(game)
        assert result["classification"] in {
            "BEST MOVE",
            "EXCELLENT",
            "GOOD",
            "INACCURACY",
            "MISS",
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
