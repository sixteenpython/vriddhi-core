"""Capital-neutral decision-quality evaluation, separate from market outcomes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

SCORING_VERSION = "bti-score-v2"


def weights(holdings: dict[str, int], market: dict[str, Any]) -> dict[str, float]:
    values = {
        ticker: shares * market[ticker]["price_paise"]
        for ticker, shares in holdings.items()
        if shares and ticker in market
    }
    total = sum(values.values())
    return {ticker: value / total for ticker, value in values.items()} if total else {}


def portfolio_quality(holdings: dict[str, int], market: dict[str, Any]) -> dict[str, float]:
    allocation = weights(holdings, market)
    if not allocation:
        return {
            "utility": -50.0,
            "forecast": 0.0,
            "risk": 100.0,
            "concentration": 1.0,
            "sector_concentration": 1.0,
            "valuation": 0.0,
        }
    forecast = sum(w * market[t]["forecast_pct"] for t, w in allocation.items())
    risk = sum(w * market[t]["volatility_pct"] for t, w in allocation.items())
    concentration = sum(w * w for w in allocation.values())
    sectors: dict[str, float] = defaultdict(float)
    for ticker, weight in allocation.items():
        sectors[market[ticker]["sector"]] += weight
    sector_concentration = sum(w * w for w in sectors.values())
    valuation = sum(
        w
        * (
            min(3.0, 1 / max(0.15, market[t]["peg"]))
            if market[t].get("asset_class", "EQUITY") == "EQUITY"
            else 1.0
        )
        for t, w in allocation.items()
    )
    asset_classes = {
        market[ticker].get("asset_class", "EQUITY") for ticker in allocation
    }
    cross_asset_bonus = min(5.0, max(0, len(asset_classes) - 1) * 1.75)
    # Utility recognises many economically equivalent portfolios; it does not score weight imitation.
    utility = (
        forecast * 2.25
        + valuation * 8.0
        - risk * 0.38
        - concentration * 30.0
        - sector_concentration * 12.0
        + cross_asset_bonus
    )
    return {
        "utility": utility,
        "forecast": forecast,
        "risk": risk,
        "concentration": concentration,
        "sector_concentration": sector_concentration,
        "valuation": valuation,
    }


def classify(score: float) -> str:
    if score >= 98:
        return "BRILLIANT"
    if score >= 90:
        return "EXCELLENT"
    if score >= 80:
        return "GOOD"
    if score >= 65:
        return "INACCURACY"
    if score >= 45:
        return "MISTAKE"
    return "BLUNDER"


def position_evaluation(score: float) -> dict[str, Any]:
    """Translate deterministic portfolio quality into a chess-like evaluation."""
    value = round(max(-3.0, min(3.0, (score - 70.0) / 15.0)), 2)
    absolute = abs(value)
    if absolute <= 0.2:
        label = "LEVEL"
    elif absolute < 1.0:
        label = "SLIGHT ADVANTAGE" if value > 0 else "SLIGHT DISADVANTAGE"
    elif absolute < 2.0:
        label = "STRONG ADVANTAGE" if value > 0 else "SERIOUS DISADVANTAGE"
    else:
        label = "DOMINANT" if value > 0 else "CRITICAL"
    return {
        "value": value,
        "display": f"{value:+.2f}",
        "label": label,
        "player_advantage": value >= 0,
    }


def portfolio_health(holdings: dict[str, int], market: dict[str, Any]) -> dict[str, Any]:
    """Public portfolio X-Ray derived only from information visible to the player."""
    allocation = weights(holdings, market)
    quality = portfolio_quality(holdings, market)
    public_quality = {key: value for key, value in quality.items() if key != "utility"}
    if not allocation:
        return {
            **public_quality,
            "sharpe": 0.0,
            "drawdown_pct": 0.0,
            "var_95_pct": 0.0,
            "positions": 0,
            "sectors": 0,
            "top_weight_pct": 0.0,
            "health_score": 0.0,
            "health_label": "UNINVESTED",
        }
    sharpe = sum(weight * market[ticker]["sharpe"] for ticker, weight in allocation.items())
    drawdown = sum(
        weight * market[ticker]["drawdown_pct"] for ticker, weight in allocation.items()
    )
    value_at_risk = sum(
        weight * market[ticker]["var_95_pct"] for ticker, weight in allocation.items()
    )
    sectors = {market[ticker]["sector"] for ticker in allocation}
    top_weight = max(allocation.values()) * 100
    health_score = max(
        0.0,
        min(
            100.0,
            50
            + quality["forecast"] * 1.2
            + sharpe * 8
            + quality["valuation"] * 4
            - quality["risk"] * 0.35
            - top_weight * 0.35
            - max(0, 4 - len(sectors)) * 4,
        ),
    )
    label = (
        "ROBUST"
        if health_score >= 80
        else "HEALTHY"
        if health_score >= 65
        else "FRAGILE"
        if health_score >= 45
        else "CRITICAL"
    )
    return {
        **{key: round(value, 4) for key, value in public_quality.items()},
        "sharpe": round(sharpe, 4),
        "drawdown_pct": round(drawdown, 4),
        "var_95_pct": round(value_at_risk, 4),
        "positions": len(allocation),
        "sectors": len(sectors),
        "top_weight_pct": round(top_weight, 2),
        "health_score": round(health_score, 1),
        "health_label": label,
    }


def evaluate(
    player: dict[str, int], reference: dict[str, int], market: dict[str, Any]
) -> dict[str, Any]:
    actual = portfolio_quality(player, market)
    optimum = portfolio_quality(reference, market)
    regret = max(0.0, optimum["utility"] - actual["utility"])
    score = round(max(0.0, 100.0 - regret * 3.0), 1)
    label = classify(score)
    positives = []
    improvements = []
    if actual["concentration"] <= 0.18:
        positives.append(
            "You spread risk across the portfolio instead of depending on one company."
        )
    else:
        improvements.append("Your portfolio depends too heavily on a small number of holdings.")
    if actual["valuation"] >= 1.0:
        positives.append("You balanced the price paid with the growth available.")
    else:
        improvements.append("Look harder for growth that is not already fully priced in.")
    if actual["risk"] <= optimum["risk"] * 1.15:
        positives.append("You kept the expected risk close to the strongest feasible alternatives.")
    else:
        improvements.append(
            "A less volatile mix offered similar growth with a smoother expected ride."
        )
    asset_classes = {
        market[ticker].get("asset_class", "EQUITY")
        for ticker in player
        if player[ticker] and ticker in market
    }
    if len(asset_classes) > 1:
        positives.append("You used more than one economic return driver instead of relying only on equities.")
    headline = {
        "BRILLIANT": "You found a portfolio within the strongest feasible decision band.",
        "EXCELLENT": "You captured nearly all of the available risk-adjusted opportunity.",
        "GOOD": "This is a sound portfolio, with one meaningful improvement still available.",
        "INACCURACY": "The portfolio is workable, but gives away avoidable risk-adjusted return.",
        "MISTAKE": "A materially stronger portfolio was available from the same information.",
        "BLUNDER": "This move takes risk without enough diversified growth to justify it.",
    }[label]
    return {
        "score": score,
        "classification": label,
        "scoring_version": SCORING_VERSION,
        "position_evaluation": position_evaluation(score),
        "portfolio_health": portfolio_health(player, market),
        "decision_quality": {
            "headline": headline,
            "did_well": positives[:2] or ["You completed a valid, fully funded monthly decision."],
            "improve": improvements[:2]
            or ["Keep applying the same valuation, growth and diversification discipline."],
        },
    }
