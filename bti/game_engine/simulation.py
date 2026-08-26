"""Independent deterministic forward simulation for the BTI showcase."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

SIMULATION_VERSION = "bti-calibrated-synthetic-2026-08-v4"

REGIMES = (
    ("SELECTIVE GROWTH", "Growth remains available, but valuation discipline separates leaders.", 0.10, 0.95),
    ("SECTOR ROTATION", "Leadership is rotating; yesterday's strongest sector is no longer enough.", 0.00, 1.08),
    ("EARNINGS DISPERSION", "Company delivery matters more than the broad index direction.", 0.02, 1.15),
    ("RISK OFF", "Liquidity tightens and concentrated portfolios face a harder downside test.", -0.22, 1.25),
    ("VALUATION RESET", "Forecasts remain positive while expensive expectations compress.", -0.12, 1.18),
    ("RECOVERY", "Risk appetite improves, but weak balance sheets still lag the rebound.", 0.18, 1.05),
)


def _rng(seed: str, *parts: object) -> random.Random:
    digest = hashlib.sha256("|".join([seed, *map(str, parts)]).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def build_regime_schedule(seed: str, horizon: int) -> list[dict[str, Any]]:
    """Pre-generate a fair campaign path that cannot inspect player holdings."""
    schedule = []
    for month in range(1, horizon + 1):
        rng = _rng(seed, "regime", month)
        index = (rng.randrange(len(REGIMES)) + (month - 1) // 6) % len(REGIMES)
        label, narrative, bias, volatility = REGIMES[index]
        difficulty = round(1.0 + (month / horizon) * 0.35, 3)
        schedule.append(
            {
                "month": month,
                "label": label,
                "narrative": narrative,
                "difficulty": difficulty,
                "market_bias": bias,
                "volatility_multiplier": round(volatility * difficulty, 3),
            }
        )
    return schedule


def _simulated_lookback(
    ticker: str, price: int, annual_vol: float, horizon: int
) -> list[dict[str, int]]:
    """Create 252 deterministic daily candles ending at the opening quote."""
    rng = _rng("lookback", ticker, horizon)
    closes = [price]
    for _ in range(251):
        reverse_return = max(-0.06, min(0.06, rng.gauss(0.00035, annual_vol / math.sqrt(252))))
        closes.insert(0, max(100, round(closes[0] / (1 + reverse_return))))
    candles: list[dict[str, int]] = []
    for index, close in enumerate(closes):
        opening = closes[index - 1] if index else close
        spread = abs(rng.gauss(0, annual_vol / math.sqrt(252))) * 0.65
        high = max(opening, close, round(max(opening, close) * (1 + spread)))
        low = max(100, min(opening, close, round(min(opening, close) * (1 - spread))))
        candles.append(
            {
                "month": index - len(closes) + 1,
                "open_paise": opening,
                "high_paise": high,
                "low_paise": low,
                "close_paise": close,
            }
        )
    return candles


def _simulated_month_candles(
    ticker: str,
    opening: int,
    close: int,
    annual_vol: float,
    seed: str,
    month: int,
) -> list[dict[str, int]]:
    """Bridge an investment month with 21 reproducible daily OHLC candles."""
    rng = _rng(seed, "daily", month, ticker)
    raw_logs = [rng.gauss(0, annual_vol / math.sqrt(252)) for _ in range(21)]
    target_log = math.log(max(close, 100) / max(opening, 100))
    adjustment = (target_log - sum(raw_logs)) / len(raw_logs)
    candles: list[dict[str, int]] = []
    prior = opening
    for index, raw_log in enumerate(raw_logs, start=1):
        day_close = (
            close if index == 21 else max(100, round(prior * math.exp(raw_log + adjustment)))
        )
        spread = abs(rng.gauss(0, annual_vol / math.sqrt(252))) * 0.55
        candles.append(
            {
                "month": month * 100 + index,
                "open_paise": prior,
                "high_paise": max(prior, day_close, round(max(prior, day_close) * (1 + spread))),
                "low_paise": max(
                    100,
                    min(prior, day_close, round(min(prior, day_close) * (1 - spread))),
                ),
                "close_paise": day_close,
            }
        )
        prior = day_close
    return candles


def _rsi(closes: list[int], window: int = 14) -> float:
    changes = [
        current - prior for prior, current in zip(closes, closes[1:], strict=False)
    ][-window:]
    if not changes:
        return 50.0
    gains = sum(max(change, 0) for change in changes) / len(changes)
    losses = sum(max(-change, 0) for change in changes) / len(changes)
    if losses == 0:
        return 100.0
    relative_strength = gains / losses
    return 100 - 100 / (1 + relative_strength)


def initial_market(stocks: dict[str, dict[str, Any]], horizon: int) -> dict[str, Any]:
    market = {}
    for ticker, stock in stocks.items():
        price = max(100, round(float(stock["Current_Price"]) * 100))
        forecast = float(stock[f"Forecast_{horizon}M"])
        vol = min(
            55.0, max(12.0, 13.0 + abs(float(stock["Risk_Adjusted_Return"]) - forecast) * 0.65)
        )
        lookback = _simulated_lookback(ticker, price, vol / 100, horizon)
        closes = [item["close_paise"] for item in lookback]
        peg = float(stock["PEG_Ratio"])
        pe = float(stock["PE_Ratio"])
        pb = float(stock["PB_Ratio"])
        fundamentals_rng = _rng("fundamentals", ticker, horizon)
        market[ticker] = {
            "ticker": ticker,
            "sector": stock["sector"],
            "overall_rank": int(stock["Overall_Rank"]),
            "historical_cagr_pct": float(stock["Avg_Historical_CAGR"]),
            "price_paise": price,
            "open_paise": price,
            "high_paise": price,
            "low_paise": price,
            "close_paise": price,
            "pe": pe,
            "pb": pb,
            "peg": peg,
            "forecast_pct": forecast,
            "volatility_pct": vol,
            "sharpe": float(stock["Risk_Adjusted_Return"]) / max(vol, 1.0),
            "drawdown_pct": 0.0,
            "var_95_pct": 1.645 * vol / math.sqrt(12),
            "expected_shortfall_95_pct": 2.063 * vol / math.sqrt(12),
            "peak_paise": price,
            "returns": [],
            "history_paise": closes,
            "ohlc_history": lookback,
            "volume_index": round(fundamentals_rng.uniform(82, 118), 2),
            "sentiment_score": round(max(5, min(95, 50 + forecast * 1.15 - vol * 0.25)), 2),
            "momentum_90d_pct": round((closes[-1] / closes[-63] - 1) * 100, 2),
            "rsi_14": round(_rsi(closes), 2),
            "beta": round(max(0.3, min(2.2, vol / 18 + fundamentals_rng.uniform(-0.2, 0.2))), 2),
            "roe_pct": round(max(0, min(45, pb / max(pe, 1) * 100)), 2),
            "earnings_growth_pct": round(
                max(-15, min(55, pe / peg if peg > 0.05 else forecast)), 2
            ),
            "profit_margin_pct": round(fundamentals_rng.uniform(8, 31), 2),
            "debt_to_equity": round(fundamentals_rng.uniform(0.05, 1.65), 2),
            "dividend_yield_pct": round(fundamentals_rng.uniform(0.2, 5.8), 2),
            "forecast_curve": [
                {
                    "months": months,
                    "annualized_pct": float(stock[f"Forecast_{months}M"]),
                    "cumulative_pct": float(stock[f"Expected_Returns_{months}M"]),
                }
                for months in (12, 24, 36, 48, 60)
            ],
        }
    return market


def advance_market(
    market: dict[str, Any],
    seed: str,
    month: int,
    regime: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    """Advance independently of player and reference holdings."""
    regime = regime or {"market_bias": 0.0, "volatility_multiplier": 1.0}
    common = (
        _rng(seed, "market", month).gauss(0, 1)
        * float(regime.get("volatility_multiplier", 1.0))
        + float(regime.get("market_bias", 0.0))
    )
    sectors = {
        sector: _rng(seed, "sector", sector, month).gauss(0, 1)
        for sector in {v["sector"] for v in market.values()}
    }
    updated = {}
    stock_returns: list[float] = []
    for ticker, previous in market.items():
        item = dict(previous)
        drift = max(-0.08, min(0.28, previous["forecast_pct"] / 100))
        annual_vol = previous["volatility_pct"] / 100
        shock = (
            0.58 * common
            + 0.25 * sectors[previous["sector"]]
            + 0.57 * _rng(seed, "stock", ticker, month).gauss(0, 1)
        )
        recent_return = previous.get("returns", [0.0])[-1] if previous.get("returns") else 0.0
        # Crowded monthly winners and losers are deliberately allowed to mean-revert.
        # This keeps the visible momentum/news boards educationally tempting without
        # turning yesterday's headline into a deterministic shortcut to tomorrow's alpha.
        reversal_strength = 0.28 if abs(recent_return) >= 0.04 else 0.08
        attention_reversal = -reversal_strength * recent_return
        ret = max(
            -0.24,
            min(
                0.24,
                drift / 12
                + annual_vol / math.sqrt(12) * shock
                + attention_reversal,
            ),
        )
        stock_returns.append(ret)
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
        daily_candles = _simulated_month_candles(ticker, opening, close, annual_vol, seed, month)
        combined_candles = [
            *previous.get("ohlc_history", []),
            *daily_candles,
        ][-1260:]
        combined_closes = [candle["close_paise"] for candle in combined_candles]
        item.update(
            {
                "price_paise": close,
                "open_paise": opening,
                "high_paise": high,
                "low_paise": low,
                "close_paise": close,
                "returns": returns,
                "history_paise": combined_closes,
                "ohlc_history": combined_candles,
                "volume_index": max(
                    35.0,
                    min(320.0, previous.get("volume_index", 100) * 0.72 + 28 + abs(ret) * 520),
                ),
                "sentiment_score": max(
                    2.0,
                    min(
                        98.0,
                        previous.get("sentiment_score", 50) * 0.68
                        + 16
                        + ret * 145
                        + previous["forecast_pct"] * 0.42,
                    ),
                ),
                "momentum_90d_pct": (combined_closes[-1] / combined_closes[-63] - 1) * 100,
                "rsi_14": _rsi(combined_closes),
                "earnings_growth_pct": max(
                    -20.0,
                    min(60.0, previous.get("earnings_growth_pct", 10) * 0.96 + ret * 22),
                ),
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
                "forecast_curve": [
                    {
                        "months": point["months"],
                        "annualized_pct": max(
                            -12.0,
                            min(35.0, point["annualized_pct"] * 0.99 + ret * 18),
                        ),
                        "cumulative_pct": (
                            (1 + max(-0.95, point["annualized_pct"] * 0.99 + ret * 18) / 100)
                            ** (point["months"] / 12)
                            - 1
                        )
                        * 100,
                    }
                    for point in previous.get("forecast_curve", [])
                ],
            }
        )
        updated[ticker] = item
    # The simulated Nifty must participate in the same market as its constituents. A
    # mostly cross-sectional return proxy prevents the old fixed 8% drift from becoming
    # an artificially easy target when the playable universe has materially stronger
    # growth. Trimming extremes approximates a broad large-cap index without disclosing
    # or inventing real-world index weights.
    ordered_returns = sorted(stock_returns)
    broad_market_return = sum(ordered_returns[5:-5]) / max(1, len(ordered_returns[5:-5]))
    macro_index_return = 0.08 / 12 + 0.135 / math.sqrt(12) * common
    benchmark_return = 0.85 * broad_market_return + 0.15 * macro_index_return
    return updated, max(-0.18, min(0.18, benchmark_return))


def public_market(market: dict[str, Any], month: int, data_through: str) -> dict[str, Any]:
    fields = {
        "ticker",
        "sector",
        "overall_rank",
        "historical_cagr_pct",
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
        "ohlc_history",
        "forecast_curve",
        "volume_index",
        "sentiment_score",
        "momentum_90d_pct",
        "rsi_14",
        "beta",
        "roe_pct",
        "earnings_growth_pct",
        "profit_margin_pct",
        "debt_to_equity",
        "dividend_yield_pct",
    }
    visible = [
        {k: round(v, 4) if isinstance(v, float) else v for k, v in item.items() if k in fields}
        for item in market.values()
    ]
    for item in visible:
        history = item.get("history_paise", [item["close_paise"]])
        item.setdefault("volume_index", 100.0)
        item.setdefault("sentiment_score", 50.0)
        item.setdefault(
            "momentum_90d_pct",
            (history[-1] / history[max(0, len(history) - 63)] - 1) * 100,
        )
        item.setdefault("rsi_14", _rsi(history))
        item.setdefault("beta", max(0.3, min(2.2, item["volatility_pct"] / 18)))
        item.setdefault("roe_pct", max(0, min(45, item["pb"] / max(item["pe"], 1) * 100)))
        item.setdefault(
            "earnings_growth_pct",
            max(-15, min(55, item["pe"] / item["peg"] if item["peg"] > 0.05 else 0)),
        )
        item.setdefault("profit_margin_pct", 15.0)
        item.setdefault("debt_to_equity", 0.65)
        item.setdefault("dividend_yield_pct", 1.8)
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
