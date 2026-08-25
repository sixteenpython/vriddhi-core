"""Atomic campaign lifecycle for the deterministic BTI web game."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .artifacts import VriddhiArtifacts
from .scoring import evaluate
from .simulation import advance_market, initial_market, public_market


class GameRuleError(ValueError):
    """A proposed action violates a published campaign rule."""


def _value(holdings: dict[str, int], market: dict[str, Any]) -> int:
    return sum(shares * market[ticker]["price_paise"] for ticker, shares in holdings.items())


def _annualised_sip_return(contribution: int, months: int, final_value: int) -> float:
    if not months or not contribution or final_value <= 0:
        return 0.0
    low, high = -0.95, 3.0
    for _ in range(80):
        annual = (low + high) / 2
        monthly = (1 + annual) ** (1 / 12) - 1
        projected = sum(contribution * (1 + monthly) ** (months - i) for i in range(months))
        if projected < final_value:
            low = annual
        else:
            high = annual
    return round((low + high) * 50, 2)


class BTIGame:
    ENGINE_VERSION = "bti-game-v1"

    @classmethod
    def create(
        cls,
        monthly_amount_rupees: int,
        horizon_months: int,
        seed: str,
        repository_root: str | Path | None = None,
        campaign_id: str | None = None,
    ) -> "BTIGame":
        if horizon_months not in {24, 36, 48, 60}:
            raise GameRuleError("Choose a 24, 36, 48 or 60 month campaign")
        if (
            not isinstance(monthly_amount_rupees, int)
            or not 10_000 <= monthly_amount_rupees <= 100_000
        ):
            raise GameRuleError(
                "Monthly investment must be a whole-rupee amount from ₹10,000 to ₹1,00,000"
            )
        if not seed:
            raise GameRuleError("A scenario seed is required")
        self = cls.__new__(cls)
        self.artifacts = VriddhiArtifacts(repository_root)
        identity = f"{self.artifacts.release_id}|{horizon_months}|{monthly_amount_rupees}|{seed}"
        self.state = {
            "campaign_id": campaign_id
            or hashlib.sha256(identity.encode()).hexdigest()[:12].upper(),
            "status": "ACTIVE",
            "engine_version": cls.ENGINE_VERSION,
            "release_id": self.artifacts.release_id,
            "data_through": self.artifacts.data_through,
            "seed": seed,
            "horizon_months": horizon_months,
            "monthly_amount_paise": monthly_amount_rupees * 100,
            "current_month": 0,
            "cash_paise": 0,
            "holdings": {},
            "market": initial_market(self.artifacts.stocks, horizon_months),
            "benchmark_value_paise": 0,
            "total_invested_paise": 0,
            "moves": [],
            "rating": 1200,
        }
        return self

    @property
    def status(self) -> str:
        return self.state["status"]

    def market_view(self) -> dict[str, Any]:
        return public_market(
            self.state["market"], self.state["current_month"], self.state["data_through"]
        )

    def _reference_holdings(self, total_capital: int) -> dict[str, int]:
        """Solve a whole-share reference feasible for this campaign's actual capital."""
        base_weights = self.artifacts._reference_weights(self.state["horizon_months"])
        market = self.state["market"]
        # Vriddhi supplies the governed horizon portfolio. The private game adapter then lets
        # its weights respond to the information available in the current simulated month.
        # This creates an evolving reference trajectory without consulting the next outcome.
        signals: dict[str, float] = {}
        for ticker in base_weights:
            stock = market[ticker]
            growth = max(0.1, (stock["forecast_pct"] + 12.0) / 30.0)
            valuation = min(3.0, 1.0 / max(0.2, stock["peg"]))
            resilience = max(0.15, stock["sharpe"] + 1.25) / max(
                0.5, stock["volatility_pct"] / 20.0
            )
            signals[ticker] = growth * valuation * resilience
        signal_total = sum(signals.values()) or 1.0
        weights = {
            ticker: 0.72 * base_weight + 0.28 * signals[ticker] / signal_total
            for ticker, base_weight in base_weights.items()
        }
        weight_total = sum(weights.values()) or 1.0
        weights = {ticker: weight / weight_total for ticker, weight in weights.items()}
        holdings = {
            ticker: int(total_capital * weight // market[ticker]["price_paise"])
            for ticker, weight in weights.items()
        }
        spent = _value(holdings, market)
        # Spend residual where it most reduces target underweight; expensive shares remain genuinely infeasible.
        while True:
            candidates = []
            for ticker, target in weights.items():
                price = market[ticker]["price_paise"]
                if price <= total_capital - spent:
                    actual = holdings.get(ticker, 0) * price / total_capital
                    candidates.append((target - actual, -price, ticker))
            if not candidates:
                break
            ticker = max(candidates)[2]
            holdings[ticker] = holdings.get(ticker, 0) + 1
            spent += market[ticker]["price_paise"]
        return {ticker: shares for ticker, shares in holdings.items() if shares}

    def submit_move(
        self, instructions: list[dict[str, Any]], expected_month: int | None = None
    ) -> dict[str, Any]:
        if self.status != "ACTIVE":
            raise GameRuleError("Only an active campaign accepts moves")
        month = self.state["current_month"]
        if expected_month is not None and expected_month != month:
            raise GameRuleError("This move is stale; reload the current campaign month")
        if not instructions:
            raise GameRuleError("A move needs at least one BUY or SELL instruction")
        market = self.state["market"]
        holdings = dict(self.state["holdings"])
        seen: set[tuple[str, str]] = set()
        sells: list[tuple[str, int]] = []
        buys: list[tuple[str, int]] = []
        for raw in instructions:
            side = str(raw.get("side", "")).upper()
            ticker = str(raw.get("ticker", "")).upper()
            shares = raw.get("shares")
            if side not in {"BUY", "SELL"} or ticker not in market:
                raise GameRuleError(
                    "Every instruction needs a valid BUY/SELL side and market ticker"
                )
            if isinstance(shares, bool) or not isinstance(shares, int) or shares <= 0:
                raise GameRuleError("Share quantities must be positive whole numbers")
            if (side, ticker) in seen:
                raise GameRuleError("Combine duplicate instructions for the same stock and side")
            seen.add((side, ticker))
            (sells if side == "SELL" else buys).append((ticker, shares))
        cash = self.state["cash_paise"] + self.state["monthly_amount_paise"]
        opening_value = self.state["cash_paise"] + _value(holdings, market)
        execution = []
        for ticker, shares in sells:
            if shares > holdings.get(ticker, 0):
                raise GameRuleError(
                    f"Cannot sell {shares} {ticker}; only {holdings.get(ticker, 0)} held"
                )
            amount = shares * market[ticker]["price_paise"]
            holdings[ticker] -= shares
            if not holdings[ticker]:
                del holdings[ticker]
            cash += amount
            execution.append(
                {
                    "side": "SELL",
                    "ticker": ticker,
                    "shares": shares,
                    "price_paise": market[ticker]["price_paise"],
                    "cash_movement_paise": amount,
                }
            )
        buying_power = cash
        buy_total = 0
        for ticker, shares in buys:
            amount = shares * market[ticker]["price_paise"]
            if amount > cash:
                raise GameRuleError(f"Not enough cash to buy {shares} {ticker}")
            holdings[ticker] = holdings.get(ticker, 0) + shares
            cash -= amount
            buy_total += amount
            execution.append(
                {
                    "side": "BUY",
                    "ticker": ticker,
                    "shares": shares,
                    "price_paise": market[ticker]["price_paise"],
                    "cash_movement_paise": -amount,
                }
            )
        if buy_total * 10 < buying_power * 9:
            raise GameRuleError("Deploy at least 90% of available buying power this month")
        if cash + _value(holdings, market) != opening_value + self.state["monthly_amount_paise"]:
            raise RuntimeError("Portfolio accounting invariant failed")
        reference = self._reference_holdings(opening_value + self.state["monthly_amount_paise"])
        decision = evaluate(holdings, reference, market)
        next_market, benchmark_return = advance_market(market, self.state["seed"], month + 1)
        before_outcome = cash + _value(holdings, market)
        after_outcome = cash + _value(holdings, next_market)
        portfolio_return = after_outcome / before_outcome - 1
        benchmark_before = self.state["benchmark_value_paise"] + self.state["monthly_amount_paise"]
        benchmark_after = round(benchmark_before * (1 + benchmark_return))
        result = {
            **decision,
            "move": month + 1,
            "execution": execution,
            "market_outcome": {
                "headline": "What happened next is an outcome, not a retroactive judgment of your decision.",
                "portfolio_return_pct": round(portfolio_return * 100, 2),
                "benchmark_return_pct": round(benchmark_return * 100, 2),
                "alpha_pct": round((portfolio_return - benchmark_return) * 100, 2),
            },
            "progress": {
                "move": month + 1,
                "total": self.state["horizon_months"],
                "portfolio_value_paise": after_outcome,
                "benchmark_value_paise": benchmark_after,
            },
        }
        self.state.update(
            {
                "cash_paise": cash,
                "holdings": holdings,
                "market": next_market,
                "benchmark_value_paise": benchmark_after,
                "total_invested_paise": self.state["total_invested_paise"]
                + self.state["monthly_amount_paise"],
                "current_month": month + 1,
            }
        )
        self.state["moves"].append(deepcopy(result))
        average = sum(item["score"] for item in self.state["moves"]) / len(self.state["moves"])
        self.state["rating"] = max(600, round(1200 + (average - 70) * 4))
        if self.state["current_month"] == self.state["horizon_months"]:
            self.state["status"] = "COMPLETED"
            result["final_result"] = self.final_result()
            self.state["moves"][-1] = deepcopy(result)
        return deepcopy(result)

    def public_state(self) -> dict[str, Any]:
        market = self.state["market"]
        portfolio = self.state["cash_paise"] + _value(self.state["holdings"], market)
        benchmark = self.state["benchmark_value_paise"]
        return {
            "campaign_id": self.state["campaign_id"],
            "status": self.status,
            "horizon_months": self.state["horizon_months"],
            "monthly_amount_rupees": self.state["monthly_amount_paise"] // 100,
            "current_move": min(self.state["current_month"] + 1, self.state["horizon_months"]),
            "moves_completed": self.state["current_month"],
            "holdings": dict(self.state["holdings"]),
            "cash_paise": self.state["cash_paise"],
            "portfolio_value_paise": portfolio,
            "benchmark_value_paise": benchmark,
            "alpha_pct": round((portfolio / benchmark - 1) * 100, 2) if benchmark else 0.0,
            "total_invested_paise": self.state["total_invested_paise"],
            "rating": self.state["rating"],
            "last_result": deepcopy(self.state["moves"][-1]) if self.state["moves"] else None,
            "release_id": self.state["release_id"],
            "market_label": "SIMULATED MARKET",
        }

    def resign(self) -> dict[str, Any]:
        if self.status != "ACTIVE":
            raise GameRuleError("Only an active campaign can be resigned")
        self.state["status"] = "RESIGNED"
        self.state["rating"] = max(600, self.state["rating"] - 35)
        return {"status": "RESIGNED", "recorded_as": "LOSS", "rating": self.state["rating"]}

    def final_result(self) -> dict[str, Any]:
        portfolio = self.state["cash_paise"] + _value(self.state["holdings"], self.state["market"])
        months = self.state["current_month"]
        contribution = self.state["monthly_amount_paise"]
        benchmark = self.state["benchmark_value_paise"]
        return {
            "status": self.status,
            "months_completed": months,
            "total_invested_paise": self.state["total_invested_paise"],
            "portfolio_value_paise": portfolio,
            "benchmark_value_paise": benchmark,
            "wealth_alpha_paise": portfolio - benchmark,
            "portfolio_money_weighted_annual_return_pct": _annualised_sip_return(
                contribution, months, portfolio
            ),
            "benchmark_money_weighted_annual_return_pct": _annualised_sip_return(
                contribution, months, benchmark
            ),
            "average_move_score": round(sum(m["score"] for m in self.state["moves"]) / months, 1)
            if months
            else 0.0,
            "rating": self.state["rating"],
            "decision_support_only": True,
        }

    def to_json(self) -> str:
        return json.dumps(self.state, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str, repository_root: str | Path | None = None) -> "BTIGame":
        self = cls.__new__(cls)
        self.artifacts = VriddhiArtifacts(repository_root)
        self.state = json.loads(payload)
        if self.state.get("engine_version") != cls.ENGINE_VERSION:
            raise GameRuleError("Unsupported saved campaign engine version")
        if self.state.get("release_id") != self.artifacts.release_id:
            raise GameRuleError("Saved campaign requires a different governed Vriddhi release")
        return self
