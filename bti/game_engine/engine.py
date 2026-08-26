"""Atomic campaign lifecycle for the deterministic BTI web game."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .artifacts import VriddhiArtifacts
from .scoring import evaluate, portfolio_health
from .simulation import advance_market, build_regime_schedule, initial_market, public_market


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
    ENGINE_VERSION = "bti-game-v3"

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
            "gameplay_mode": "RATED",
            "regime_schedule": build_regime_schedule(seed, horizon_months),
        }
        return self

    @property
    def status(self) -> str:
        return self.state["status"]

    def market_view(self) -> dict[str, Any]:
        view = public_market(
            self.state["market"], self.state["current_month"], self.state["data_through"]
        )
        view["regime"] = deepcopy(self._regime_for_move(self.state["current_month"] + 1))
        return view

    def _regime_for_move(self, move: int) -> dict[str, Any]:
        schedule = self.state.setdefault(
            "regime_schedule",
            build_regime_schedule(self.state["seed"], self.state["horizon_months"]),
        )
        index = max(0, min(len(schedule) - 1, move - 1))
        public = deepcopy(schedule[index])
        public.pop("market_bias", None)
        public.pop("volatility_multiplier", None)
        return public

    def _regime_internal(self, move: int) -> dict[str, Any]:
        schedule = self.state.setdefault(
            "regime_schedule",
            build_regime_schedule(self.state["seed"], self.state["horizon_months"]),
        )
        return schedule[max(0, min(len(schedule) - 1, move - 1))]

    def _market_at_month(self, completed_months: int) -> dict[str, Any]:
        """Reconstruct a historical public information set without storing giant snapshots."""
        market = initial_market(self.artifacts.stocks, self.state["horizon_months"])
        for month in range(1, completed_months + 1):
            market, _ = advance_market(
                market, self.state["seed"], month, self._regime_internal(month)
            )
        return market

    @staticmethod
    def _notation(move: int, execution: list[dict[str, Any]]) -> str:
        orders = " · ".join(
            f"{'+' if item['side'] == 'BUY' else '−'}{item['ticker']} {item['shares']}"
            for item in execution
        )
        return f"M{move:02d} · {orders}"

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
        holdings_before = dict(holdings)
        cash_before = self.state["cash_paise"]
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
        regime = self._regime_internal(month + 1)
        next_market, benchmark_return = advance_market(
            market, self.state["seed"], month + 1, regime
        )
        before_outcome = cash + _value(holdings, market)
        after_outcome = cash + _value(holdings, next_market)
        portfolio_return = after_outcome / before_outcome - 1
        benchmark_before = self.state["benchmark_value_paise"] + self.state["monthly_amount_paise"]
        benchmark_after = round(benchmark_before * (1 + benchmark_return))
        invested_after = self.state["total_invested_paise"] + self.state["monthly_amount_paise"]
        move_number = month + 1
        rating_average = (
            sum(item["score"] for item in self.state["moves"]) + decision["score"]
        ) / (len(self.state["moves"]) + 1)
        rating_after = max(600, round(1200 + (rating_average - 70) * 4))
        result = {
            **decision,
            "move": move_number,
            "execution": execution,
            "notation": self._notation(move_number, execution),
            "rating_after": rating_after,
            "regime": self._regime_for_move(move_number),
            "portfolio_before": {
                "holdings": holdings_before,
                "cash_paise": cash_before,
                "value_paise": cash_before + _value(holdings_before, market),
            },
            "portfolio_after_execution": {
                "holdings": dict(holdings),
                "cash_paise": cash,
                "value_paise": before_outcome,
                "health": portfolio_health(holdings, market),
            },
            "market_outcome": {
                "headline": "What happened next is an outcome, not a retroactive judgment of your decision.",
                "portfolio_return_pct": round(portfolio_return * 100, 2),
                "benchmark_return_pct": round(benchmark_return * 100, 2),
                "alpha_pct": round((portfolio_return - benchmark_return) * 100, 2),
            },
            "progress": {
                "move": move_number,
                "total": self.state["horizon_months"],
                "total_invested_paise": invested_after,
                "portfolio_value_paise": after_outcome,
                "benchmark_value_paise": benchmark_after,
                "portfolio_xirr_pct": _annualised_sip_return(
                    self.state["monthly_amount_paise"], move_number, after_outcome
                ),
                "benchmark_xirr_pct": _annualised_sip_return(
                    self.state["monthly_amount_paise"], move_number, benchmark_after
                ),
                "projected_annual_return_pct": round(
                    decision["portfolio_health"]["forecast"], 2
                ),
                "benchmark_projected_annual_return_pct": 8.0,
                "position_evaluation": deepcopy(decision["position_evaluation"]),
            },
        }
        self.state.update(
            {
                "cash_paise": cash,
                "holdings": holdings,
                "market": next_market,
                "benchmark_value_paise": benchmark_after,
                "total_invested_paise": invested_after,
                "current_month": month + 1,
            }
        )
        self.state["moves"].append(deepcopy(result))
        self.state["rating"] = rating_after
        result["match_summary"] = self.match_summary()
        if self.state["current_month"] == self.state["horizon_months"]:
            self.state["status"] = "COMPLETED"
            result["final_result"] = self.final_result()
        self.state["moves"][-1] = deepcopy(result)
        return deepcopy(result)

    def performance_series(self) -> list[dict[str, Any]]:
        series = []
        contribution = self.state["monthly_amount_paise"]
        for index, move in enumerate(self.state["moves"], start=1):
            progress = move.get("progress", {})
            portfolio = int(progress.get("portfolio_value_paise", 0))
            benchmark = int(progress.get("benchmark_value_paise", 0))
            invested = int(progress.get("total_invested_paise", contribution * index))
            series.append(
                {
                    "move": index,
                    "total_invested_paise": invested,
                    "portfolio_value_paise": portfolio,
                    "benchmark_value_paise": benchmark,
                    "wealth_gap_paise": portfolio - benchmark,
                    "alpha_pct": round((portfolio / benchmark - 1) * 100, 2)
                    if benchmark
                    else 0.0,
                    "portfolio_xirr_pct": progress.get(
                        "portfolio_xirr_pct",
                        _annualised_sip_return(contribution, index, portfolio),
                    ),
                    "benchmark_xirr_pct": progress.get(
                        "benchmark_xirr_pct",
                        _annualised_sip_return(contribution, index, benchmark),
                    ),
                    "projected_annual_return_pct": progress.get(
                        "projected_annual_return_pct", 0.0
                    ),
                    "benchmark_projected_annual_return_pct": progress.get(
                        "benchmark_projected_annual_return_pct", 8.0
                    ),
                    "position_evaluation": deepcopy(
                        move.get(
                            "position_evaluation",
                            {"value": 0.0, "display": "+0.00", "label": "LEVEL"},
                        )
                    ),
                }
            )
        return series

    @staticmethod
    def _max_drawdown_pct(series: list[dict[str, Any]]) -> float:
        """Measure drawdown on the contribution-adjusted wealth multiple."""
        peak = 0.0
        worst = 0.0
        for point in series:
            invested = point["total_invested_paise"]
            multiple = point["portfolio_value_paise"] / invested if invested else 1.0
            peak = max(peak, multiple)
            if peak:
                worst = min(worst, multiple / peak - 1)
        return round(abs(worst) * 100, 2)

    def match_summary(self, move_number: int | None = None) -> dict[str, Any]:
        completed = len(self.state["moves"])
        move = completed if move_number is None else min(move_number, completed)
        series = self.performance_series()[:move]
        latest = series[-1] if series else {
            "total_invested_paise": 0,
            "portfolio_value_paise": 0,
            "benchmark_value_paise": 0,
            "wealth_gap_paise": 0,
            "portfolio_xirr_pct": 0.0,
            "benchmark_xirr_pct": 0.0,
        }
        moves = self.state["moves"][:move]
        distribution: dict[str, int] = {}
        for item in moves:
            label = item["classification"]
            distribution[label] = distribution.get(label, 0) + 1
        gap = latest["wealth_gap_paise"]
        benchmark = latest["benchmark_value_paise"]
        rating = moves[-1].get("rating_after", 1200) if moves else 1200
        return {
            "move": move,
            "total_moves": self.state["horizon_months"],
            "overs_remaining": self.state["horizon_months"] - move,
            "total_invested_paise": latest["total_invested_paise"],
            "portfolio_value_paise": latest["portfolio_value_paise"],
            "benchmark_value_paise": benchmark,
            "wealth_gap_paise": gap,
            "wealth_alpha_pct": round((latest["portfolio_value_paise"] / benchmark - 1) * 100, 2)
            if benchmark
            else 0.0,
            "portfolio_xirr_pct": latest["portfolio_xirr_pct"],
            "benchmark_xirr_pct": latest["benchmark_xirr_pct"],
            "xirr_advantage_pct": round(
                latest["portfolio_xirr_pct"] - latest["benchmark_xirr_pct"], 2
            ),
            "max_drawdown_pct": self._max_drawdown_pct(series),
            "average_move_score": round(sum(item["score"] for item in moves) / move, 1)
            if move
            else 0.0,
            "rating": rating,
            "classification_distribution": distribution,
            "position": "LEADING" if gap > 0 else "TRAILING" if gap < 0 else "LEVEL",
        }

    def move_history(self) -> list[dict[str, Any]]:
        return [
            {
                "move": index,
                "notation": move.get("notation", self._notation(index, move["execution"])),
                "score": move["score"],
                "classification": move["classification"],
                "position_evaluation": deepcopy(
                    move.get(
                        "position_evaluation",
                        {"value": 0.0, "display": "+0.00", "label": "LEVEL"},
                    )
                ),
                "alpha_pct": move.get("market_outcome", {}).get("alpha_pct", 0.0),
                "rating_after": move.get("rating_after", self.state["rating"]),
                "regime": deepcopy(move.get("regime", self._regime_for_move(index))),
            }
            for index, move in enumerate(self.state["moves"], start=1)
        ]

    def review_move(self, move_number: int) -> dict[str, Any]:
        if move_number < 1 or move_number > len(self.state["moves"]):
            raise GameRuleError("That completed move is not available for review")
        market = self._market_at_month(move_number - 1)
        market_view = public_market(market, move_number - 1, self.state["data_through"])
        market_view["regime"] = deepcopy(self._regime_for_move(move_number))
        return {
            "review_mode": True,
            "selected_move": move_number,
            "live_move": min(
                self.state["current_month"] + 1, self.state["horizon_months"]
            ),
            "result": deepcopy(self.state["moves"][move_number - 1]),
            "market": market_view,
            "performance_series": self.performance_series()[:move_number],
            "move_history": self.move_history(),
            "match_summary": self.match_summary(move_number),
        }

    def public_state(self) -> dict[str, Any]:
        market = self.state["market"]
        portfolio = self.state["cash_paise"] + _value(self.state["holdings"], market)
        benchmark = self.state["benchmark_value_paise"]
        result = {
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
            "gameplay_mode": "RATED",
            "current_regime": deepcopy(
                self._regime_for_move(
                    min(self.state["current_month"] + 1, self.state["horizon_months"])
                )
            ),
            "move_history": self.move_history(),
            "performance_series": self.performance_series(),
            "can_repeat_last_move": bool(self.state["moves"] and self.status == "ACTIVE"),
            "last_move_instructions": [
                {
                    "side": item["side"],
                    "ticker": item["ticker"],
                    "shares": item["shares"],
                }
                for item in self.state["moves"][-1]["execution"]
            ]
            if self.state["moves"]
            else [],
            "release_id": self.state["release_id"],
            "market_label": "SIMULATED MARKET",
        }
        result["match_summary"] = self.match_summary()
        result["final_result"] = self.final_result() if self.status == "COMPLETED" else None
        return result

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
        summary = self.match_summary()
        gap = portfolio - benchmark
        gap_pct = abs(gap / benchmark * 100) if benchmark else 0.0
        if gap_pct <= 0.5:
            verdict = "PHOTO_FINISH"
            headline = "PHOTO FINISH"
        elif gap > 0:
            verdict = "BEAT_INDEX"
            headline = "YOU BEAT THE INDEX"
        else:
            verdict = "INDEX_WON"
            headline = "NIFTY WON THIS CAMPAIGN"
        ranked = sorted(
            (
                {
                    "move": index,
                    "score": item["score"],
                    "classification": item["classification"],
                    "notation": item.get("notation", ""),
                }
                for index, item in enumerate(self.state["moves"], start=1)
            ),
            key=lambda item: item["score"],
        )
        average = summary["average_move_score"]
        process = (
            "ELITE DECISION DISCIPLINE"
            if average >= 90
            else "STRONG DECISION DISCIPLINE"
            if average >= 80
            else "SOUND, IMPROVABLE PROCESS"
            if average >= 70
            else "INCONSISTENT DECISION PROCESS"
            if average >= 60
            else "FRAGILE DECISION PROCESS"
        )
        if gap > 0 and average < 70:
            lesson = "You won the outcome, but the move record shows avoidable decision risk. Do not confuse a winning simulation with a repeatable process."
        elif gap <= 0 and average >= 80:
            lesson = "Your process was strong even though this market path won. Preserve the discipline; one campaign outcome is not the same as decision quality."
        elif gap > 0:
            lesson = "You converted disciplined portfolio decisions into a lead over the simulated index. Study the strongest moves and make that process repeatable."
        else:
            lesson = "The index finished ahead. Use the weakest moves to identify where valuation, risk or diversification discipline broke down."
        return {
            "status": self.status,
            "verdict": verdict,
            "headline": headline,
            "months_completed": months,
            "total_invested_paise": self.state["total_invested_paise"],
            "portfolio_value_paise": portfolio,
            "benchmark_value_paise": benchmark,
            "wealth_alpha_paise": portfolio - benchmark,
            "portfolio_gain_paise": portfolio - self.state["total_invested_paise"],
            "benchmark_gain_paise": benchmark - self.state["total_invested_paise"],
            "wealth_alpha_pct": summary["wealth_alpha_pct"],
            "portfolio_money_weighted_annual_return_pct": _annualised_sip_return(
                contribution, months, portfolio
            ),
            "benchmark_money_weighted_annual_return_pct": _annualised_sip_return(
                contribution, months, benchmark
            ),
            "xirr_advantage_pct": summary["xirr_advantage_pct"],
            "max_drawdown_pct": summary["max_drawdown_pct"],
            "average_move_score": average,
            "rating": self.state["rating"],
            "process_verdict": process,
            "strategic_lesson": lesson,
            "best_move": ranked[-1] if ranked else None,
            "weakest_move": ranked[0] if ranked else None,
            "classification_distribution": summary["classification_distribution"],
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
        self.state.setdefault("gameplay_mode", "RATED")
        self.state.setdefault(
            "regime_schedule",
            build_regime_schedule(self.state["seed"], self.state["horizon_months"]),
        )
        return self
