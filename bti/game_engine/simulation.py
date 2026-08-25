"""Independent deterministic forward simulation for the BTI showcase."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

SIMULATION_VERSION = "bti-forward-2026-08-v1"


def _rng(seed: str, *parts: object) -> random.Random:
    digest = hashlib.sha256("|".join([seed, *map(str, parts)]).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def initial_market(stocks: dict[str, dict[str, Any]], horizon: int) -> dict[str, Any]:
    market = {}
    for ticker, stock in stocks.items():
        price = max(100, round(float(stock["Current_Price"]) * 100))
        forecast = float(stock[f"Forecast_{horizon}M"])
        vol = min(
            55.0, max(12.0, 13.0 + abs(float(stock["Risk_Adjusted_Return"]) - forecast) * 0.65)
        )
        market[ticker] = {
            "ticker": ticker,
            "sector": stock["sector"],
            "price_paise": price,
            "open_paise": price,
            "high_paise": price,
            "low_paise": price,
            "close_paise": price,
            "pe": float(stock["PE_Ratio"]),
            "pb": float(stock["PB_Ratio"]),
            "peg": float(stock["PEG_Ratio"]),
            "forecast_pct": forecast,
            "volatility_pct": vol,
            "sharpe": float(stock["Risk_Adjusted_Return"]) / max(vol, 1.0),
            "drawdown_pct": 0.0,
            "var_95_pct": 1.645 * vol / math.sqrt(12),
            "expected_shortfall_95_pct": 2.063 * vol / math.sqrt(12),
            "peak_paise": price,
            "returns": [],
            "history_paise": [price],
        }
    return market


def advance_market(market: dict[str, Any], seed: str, month: int) -> tuple[dict[str, Any], float]:
    """Advance independently of player and reference holdings."""
    common = _rng(seed, "market", month).gauss(0, 1)
    sectors = {
        sector: _rng(seed, "sector", sector, month).gauss(0, 1)
        for sector in {v["sector"] for v in market.values()}
    }
    updated = {}
    for ticker, previous in market.items():
        item = dict(previous)
        drift = max(-0.08, min(0.28, previous["forecast_pct"] / 100))
        annual_vol = previous["volatility_pct"] / 100
        shock = (
            0.58 * common
            + 0.25 * sectors[previous["sector"]]
            + 0.57 * _rng(seed, "stock", ticker, month).gauss(0, 1)
        )
        ret = max(-0.24, min(0.24, drift / 12 + annual_vol / math.sqrt(12) * shock))
        opening = previous["close_paise"]
        close = max(100, round(opening * (1 + ret)))
        intraday = abs(_rng(seed, "ohlc", ticker, month).gauss(0, annual_vol / math.sqrt(252)))
        high = max(opening, close, round(max(opening, close) * (1 + intraday)))
        low = max(100, min(opening, close, round(min(opening, close) * (1 - intraday))))
        returns = [*previous["returns"], ret][-24:]
        avg = sum(returns) / len(returns)
        realised_vol = (
            (sum((r - avg) ** 2 for r in returns) / max(1, len(returns) - 1)) ** 0.5 * math.sqrt(12)
            if len(returns) > 1
            else annual_vol
        )
        peak = max(previous["peak_paise"], close)
        drawdown = max(previous["drawdown_pct"], (peak - close) / peak * 100)
        item.update(
            {
                "price_paise": close,
                "open_paise": opening,
                "high_paise": high,
                "low_paise": low,
                "close_paise": close,
                "returns": returns,
                "history_paise": [*previous["history_paise"], close][-61:],
                "peak_paise": peak,
                "pe": max(1.0, previous["pe"] * (1 + ret)),
                "pb": max(0.1, previous["pb"] * (1 + ret)),
                "peg": max(0.05, previous["peg"] * (1 + ret * 0.35)),
                "forecast_pct": max(-12.0, min(35.0, previous["forecast_pct"] * 0.985 + ret * 90)),
                "volatility_pct": realised_vol * 100,
                "sharpe": (avg * 12 - 0.06) / max(realised_vol, 0.01),
                "drawdown_pct": drawdown,
                "var_95_pct": 1.645 * realised_vol / math.sqrt(12) * 100,
                "expected_shortfall_95_pct": 2.063 * realised_vol / math.sqrt(12) * 100,
            }
        )
        updated[ticker] = item
    return updated, max(-0.18, min(0.18, 0.08 / 12 + 0.135 / math.sqrt(12) * common))


def public_market(market: dict[str, Any], month: int, data_through: str) -> dict[str, Any]:
    fields = {
        "ticker",
        "sector",
        "open_paise",
        "high_paise",
        "low_paise",
        "close_paise",
        "pe",
        "pb",
        "peg",
        "forecast_pct",
        "volatility_pct",
        "sharpe",
        "drawdown_pct",
        "var_95_pct",
        "expected_shortfall_95_pct",
        "history_paise",
    }
    visible = [
        {k: round(v, 4) if isinstance(v, float) else v for k, v in item.items() if k in fields}
        for item in market.values()
    ]
    advancing = sum(item["close_paise"] >= item["open_paise"] for item in visible)
    value_watch = min(visible, key=lambda item: item["peg"])
    risk_watch = max(visible, key=lambda item: item["expected_shortfall_95_pct"])
    return {
        "label": "SIMULATED MARKET",
        "basis": f"Governed Vriddhi information available through {data_through}",
        "month": month + 1,
        "simulation_version": SIMULATION_VERSION,
        "stocks": visible,
        "events": [
            {
                "kind": "MARKET BREADTH",
                "headline": f"{advancing} of {len(visible)} stocks are above their monthly open",
                "detail": "Breadth shows whether strength is widespread or concentrated.",
            },
            {
                "kind": "VALUATION WATCH",
                "headline": f"{value_watch['ticker']} has the market's lowest simulated PEG",
                "detail": "Low PEG is a clue, not an answer; test growth durability and portfolio fit.",
            },
            {
                "kind": "RISK WATCH",
                "headline": f"{risk_watch['ticker']} carries the highest simulated tail-loss estimate",
                "detail": "Expected shortfall describes severe downside in this scenario, not a forecast guarantee.",
            },
        ],
    }
