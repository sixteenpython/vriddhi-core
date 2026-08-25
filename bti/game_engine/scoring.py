"""Capital-neutral decision-quality evaluation, separate from market outcomes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

SCORING_VERSION = "bti-score-v1"


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
    valuation = sum(w * min(3.0, 1 / max(0.15, market[t]["peg"])) for t, w in allocation.items())
    # Utility recognises many economically equivalent portfolios; it does not score weight imitation.
    utility = (
        forecast * 2.25
        + valuation * 8.0
        - risk * 0.38
        - concentration * 30.0
        - sector_concentration * 12.0
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
        return "BEST MOVE"
    if score >= 90:
        return "EXCELLENT"
    if score >= 80:
        return "GOOD"
    if score >= 65:
        return "INACCURACY"
    if score >= 45:
        return "MISS"
    return "BLUNDER"


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
    headline = {
        "BEST MOVE": "You found a portfolio within the strongest feasible decision band.",
        "EXCELLENT": "You captured nearly all of the available risk-adjusted opportunity.",
        "GOOD": "This is a sound portfolio, with one meaningful improvement still available.",
        "INACCURACY": "The portfolio is workable, but gives away avoidable risk-adjusted return.",
        "MISS": "A materially stronger portfolio was available from the same information.",
        "BLUNDER": "This move takes risk without enough diversified growth to justify it.",
    }[label]
    return {
        "score": score,
        "classification": label,
        "scoring_version": SCORING_VERSION,
        "decision_quality": {
            "headline": headline,
            "did_well": positives[:2] or ["You completed a valid, fully funded monthly decision."],
            "improve": improvements[:2]
            or ["Keep applying the same valuation, growth and diversification discipline."],
        },
    }
