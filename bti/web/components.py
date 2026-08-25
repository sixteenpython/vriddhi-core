"""Small presentation helpers for the BTI Streamlit showcase."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def money(paise: int | float) -> str:
    rupees = float(paise) / 100
    if abs(rupees) >= 10_000_000:
        return f"₹{rupees / 10_000_000:.2f}Cr"
    if abs(rupees) >= 100_000:
        return f"₹{rupees / 100_000:.2f}L"
    return f"₹{rupees:,.0f}"


def rupees(value: int | float) -> str:
    return money(float(value) * 100)


def campaign_header(state: dict[str, Any]) -> None:
    move = min(state["current_move"], state["horizon_months"])
    st.markdown(
        f"<div class='bti-kicker'>BTI CAMPAIGN · {state['campaign_id']}</div>"
        f"<h1>Move {move} <span style='color:#859188'>/ {state['horizon_months']}</span></h1>",
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    cols[0].metric("Portfolio", money(state["portfolio_value_paise"]))
    cols[1].metric("Nifty benchmark", money(state["benchmark_value_paise"]))
    cols[2].metric("Wealth alpha", f"{state['alpha_pct']:+.1f}%")
    cols[3].metric("BTI performance", state["rating"])
    cols[4].metric("Monthly capital", rupees(state["monthly_amount_rupees"]))


def market_frame(stocks: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for stock in stocks:
        close = stock["close_paise"] / 100
        opening = stock["open_paise"] / 100
        rows.append(
            {
                "Stock": stock["ticker"],
                "Sector": stock["sector"],
                "Price ₹": close,
                "Move %": (close / opening - 1) * 100 if opening else 0,
                "PE": stock["pe"],
                "PB": stock["pb"],
                "PEG": stock["peg"],
                "Forecast %": stock["forecast_pct"],
                "Sharpe": stock["sharpe"],
                "Volatility %": stock["volatility_pct"],
                "Drawdown %": stock["drawdown_pct"],
                "VaR 95%": stock["var_95_pct"],
                "Expected shortfall %": stock["expected_shortfall_95_pct"],
            }
        )
    return pd.DataFrame(rows).sort_values(["Forecast %", "Sharpe"], ascending=False)


def disclosure(market: dict[str, Any]) -> None:
    st.warning(
        f"**{market['label']} · Month {market['month']}** — {market['basis']}. "
        "Prices and events are deterministic game scenarios, not live quotes or investment advice."
    )


def decision_bullets(title: str, items: list[str], positive: bool) -> None:
    icon = "✓" if positive else "→"
    st.markdown(f"#### {title}")
    for item in items:
        st.markdown(f"{icon} {item}")


def score_colour(classification: str) -> str:
    if classification in {"BEST MOVE", "EXCELLENT"}:
        return "#c9f277"
    if classification == "GOOD":
        return "#85d5ae"
    if classification == "INACCURACY":
        return "#f0c66a"
    return "#f28b82"
