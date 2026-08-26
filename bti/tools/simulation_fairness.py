"""Deterministic BTI strategy tournament used as a release fairness gate.

This is not a return forecast. It checks whether one simplistic public-signal
rule dominates every simulated path and whether the market rewards several
coherent investment disciplines without leaking the hidden reference.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from statistics import mean

from bti.game_engine import BTIGame


def _rank(stocks: list[dict], strategy: str, seed: str, month: int) -> list[str]:
    if strategy == "low_peg":
        ordered = sorted(stocks, key=lambda s: (s["peg"], -s["forecast_pct"]))
    elif strategy == "high_forecast":
        ordered = sorted(stocks, key=lambda s: (-s["forecast_pct"], s["volatility_pct"]))
    elif strategy == "high_sharpe":
        ordered = sorted(stocks, key=lambda s: (-s["sharpe"], s["drawdown_pct"]))
    elif strategy == "low_risk":
        ordered = sorted(stocks, key=lambda s: (s["volatility_pct"], s["drawdown_pct"]))
    elif strategy == "momentum":
        ordered = sorted(stocks, key=lambda s: (-s["momentum_90d_pct"], s["peg"]))
    elif strategy == "value":
        ordered = sorted(stocks, key=lambda s: (s["pe"] + s["pb"] * 2, -s["roe_pct"]))
    elif strategy == "composite":
        ordered = sorted(
            stocks,
            key=lambda s: -(
                s["forecast_pct"] / max(s["peg"], 0.2)
                + s["sharpe"] * 5
                - s["volatility_pct"] / 4
                + s["sentiment_score"] / 20
            ),
        )
    elif strategy == "random_diversified":
        ordered = list(stocks)
        random.Random(f"{seed}:{month}").shuffle(ordered)
    else:
        ordered = sorted(stocks, key=lambda s: s["close_paise"])

    if strategy in {"low_peg", "composite", "random_diversified"}:
        selected, sectors = [], set()
        for stock in ordered:
            if stock["sector"] not in sectors:
                selected.append(stock["ticker"])
                sectors.add(stock["sector"])
            if len(selected) == 8:
                break
        if len(selected) < 8:
            selected.extend(s["ticker"] for s in ordered if s["ticker"] not in selected)
        return selected[:8]
    return [stock["ticker"] for stock in ordered[:8]]


def _target_move(game: BTIGame, tickers: list[str]) -> list[dict]:
    state = game.public_state()
    prices = {s["ticker"]: s["close_paise"] for s in game.market_view()["stocks"]}
    holdings = state["holdings"]
    capital = state["cash_paise"] + state["monthly_amount_rupees"] * 100 + sum(
        shares * prices[ticker] for ticker, shares in holdings.items()
    )
    target_each = capital / len(tickers)
    desired = {ticker: int(target_each // prices[ticker]) for ticker in tickers}
    sells: dict[str, int] = {}
    buys: dict[str, int] = {}
    for ticker, held in holdings.items():
        delta = desired.get(ticker, 0) - held
        if delta < 0:
            sells[ticker] = -delta
    for ticker, target in desired.items():
        delta = target - holdings.get(ticker, 0)
        if delta > 0:
            buys[ticker] = delta

    buying_power = state["cash_paise"] + state["monthly_amount_rupees"] * 100 + sum(
        shares * prices[ticker] for ticker, shares in sells.items()
    )
    buy_total = sum(shares * prices[ticker] for ticker, shares in buys.items())
    remaining = buying_power - buy_total
    cheapest = min(tickers, key=prices.get)
    extra = max(0, remaining // prices[cheapest])
    if extra:
        buys[cheapest] = buys.get(cheapest, 0) + extra
    return [
        *({"side": "SELL", "ticker": ticker, "shares": shares} for ticker, shares in sells.items()),
        *({"side": "BUY", "ticker": ticker, "shares": shares} for ticker, shares in buys.items()),
    ]


def run_tournament(horizon: int, seeds: int) -> dict:
    strategies = (
        "low_peg",
        "high_forecast",
        "high_sharpe",
        "low_risk",
        "momentum",
        "value",
        "composite",
        "random_diversified",
        "cheapest_concentrated",
    )
    outcomes: dict[str, list[dict]] = defaultdict(list)
    for index in range(seeds):
        seed = f"fairness-{horizon}-{index}"
        for strategy in strategies:
            game = BTIGame.create(25_000, horizon, seed)
            for month in range(1, horizon + 1):
                selected = _rank(game.market_view()["stocks"], strategy, seed, month)
                game.submit_move(_target_move(game, selected))
            final = game.final_result()
            outcomes[strategy].append(
                {
                    "wealth_alpha_pct": final["wealth_alpha_pct"],
                    "average_move_score": final["average_move_score"],
                    "won": final["verdict"] == "BEAT_INDEX",
                }
            )
    summary = {
        strategy: {
            "mean_wealth_alpha_pct": round(mean(x["wealth_alpha_pct"] for x in rows), 2),
            "mean_move_score": round(mean(x["average_move_score"] for x in rows), 1),
            "win_rate_pct": round(mean(x["won"] for x in rows) * 100, 1),
        }
        for strategy, rows in outcomes.items()
    }
    leaders = sorted(summary, key=lambda name: summary[name]["mean_wealth_alpha_pct"], reverse=True)
    return {
        "horizon": horizon,
        "scenario_seeds": seeds,
        "strategies": summary,
        "highest_mean_alpha": leaders[0],
        "lowest_peg_is_universal_exploit": summary["low_peg"]["win_rate_pct"] == 100.0
        and leaders[0] == "low_peg",
        "purpose": "Simulation-game fairness diagnostic; not investment research or a forecast.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=(24, 36, 48, 60), default=24)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run_tournament(args.horizon, args.seeds), indent=2))
