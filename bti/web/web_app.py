"""Desktop-first playable Beat the Index showcase."""

from __future__ import annotations

import json
import re
import secrets
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from bti.game_engine import BTIGame, GameRuleError
from bti.web.components import (
    campaign_header,
    decision_bullets,
    disclosure,
    market_frame,
    money,
    rupees,
    score_colour,
)
from bti.web.styles import CSS

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTENT = REPO_ROOT / "bti" / "content"
SAVE_DIR = REPO_ROOT / ".bti_saves"
AMOUNTS = [10_000, 25_000, 50_000, 75_000, 100_000]
HORIZONS = [24, 36, 48, 60]


def _defaults() -> None:
    defaults = {
        "screen": "landing",
        "player": "",
        "game_json": None,
        "draft": [],
        "last_result": None,
        "move_history": [],
        "market_history": [],
        "show_resign": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.game_json:
        token = str(st.query_params.get("campaign", "")).upper()
        if re.fullmatch(r"[A-F0-9]{16}", token):
            path = SAVE_DIR / f"{token}.json"
            if path.exists():
                try:
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    game_payload = json.dumps(saved["game"], separators=(",", ":"), sort_keys=True)
                    game = BTIGame.from_json(game_payload, repository_root=REPO_ROOT)
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    st.query_params.pop("campaign", None)
                else:
                    st.session_state.game_json = game_payload
                    st.session_state.player = str(saved.get("player", "Returning player"))[:40]
                    st.session_state.move_history = [
                        {
                            "move": item["move"],
                            "score": item["score"],
                            "classification": item["classification"],
                        }
                        for item in game.state["moves"]
                    ]
                    st.session_state.screen = "game" if game.status == "ACTIVE" else "final"


def _load_game() -> BTIGame | None:
    payload = st.session_state.game_json
    return BTIGame.from_json(payload, repository_root=REPO_ROOT) if payload else None


def _resume_campaign(token: str) -> bool:
    token = token.strip().upper()
    if not re.fullmatch(r"[A-F0-9]{16}", token):
        return False
    path = SAVE_DIR / f"{token}.json"
    if not path.exists():
        return False
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        game_payload = json.dumps(saved["game"], separators=(",", ":"), sort_keys=True)
        game = BTIGame.from_json(game_payload, repository_root=REPO_ROOT)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    st.session_state.game_json = game_payload
    st.session_state.player = str(saved.get("player", "Returning player"))[:40]
    st.session_state.move_history = [
        {"move": item["move"], "score": item["score"], "classification": item["classification"]}
        for item in game.state["moves"]
    ]
    st.session_state.screen = "game" if game.status == "ACTIVE" else "final"
    st.query_params["campaign"] = token
    return True


def _save_game(game: BTIGame) -> None:
    st.session_state.game_json = game.to_json()
    token = game.public_state()["campaign_id"]
    if re.fullmatch(r"[A-F0-9]{16}", token):
        SAVE_DIR.mkdir(exist_ok=True)
        envelope = {"player": st.session_state.player, "game": json.loads(game.to_json())}
        (SAVE_DIR / f"{token}.json").write_text(
            json.dumps(envelope, separators=(",", ":"), sort_keys=True), encoding="utf-8"
        )
        st.query_params["campaign"] = token


def _snapshot_history(game: BTIGame) -> None:
    market = game.market_view()
    st.session_state.market_history.append(
        {stock["ticker"]: stock["close_paise"] / 100 for stock in market["stocks"]}
    )


def _reset_campaign() -> None:
    for key in ("game_json", "draft", "last_result", "move_history", "market_history"):
        st.session_state[key] = [] if key in {"draft", "move_history", "market_history"} else None
    st.session_state.screen = "setup"
    st.query_params.pop("campaign", None)


def _landing() -> None:
    st.markdown(
        "<div class='bti-kicker'>VRIDDHI INTELLIGENCE · BTI GAME</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='bti-hero'>Can you <span>beat the index?</span></div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='bti-lede'>A skill-based investment strategy game. Investigate a simulated "
        "Nifty 50 market, execute one complete portfolio move every month and discover whether "
        "your process can outperform the benchmark.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='bti-strip'><span class='bti-pill'>No real money</span>"
        "<span class='bti-pill'>No stock tips</span><span class='bti-pill'>Deterministic scenarios</span>"
        "<span class='bti-pill'>Powered by hidden Vriddhi intelligence</span></div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown("### One month. One complete move.")
        for index, (title, body) in enumerate(
            [
                (
                    "Investigate",
                    "Use valuation, growth, risk and market evidence as your information weapons.",
                ),
                (
                    "Decide",
                    "Construct an executable whole-share portfolio. The hidden engine never gives you the answer.",
                ),
                (
                    "Learn",
                    "See decision quality separately from market luck, improve your rating and play the next move.",
                ),
            ],
            start=1,
        ):
            st.markdown(
                f"<div class='bti-panel'><span class='bti-step'>{index}</span><b>{title}</b>"
                f"<div class='bti-note' style='margin:.55rem 0 0 2.35rem'>{body}</div></div>",
                unsafe_allow_html=True,
            )
    with right:
        st.markdown(
            "<div class='bti-dark'><div class='bti-kicker' style='color:#c9f277'>ENTER THE BOARD</div><h2>Your first campaign starts here.</h2><p>There is no clock. Think carefully. Once submitted, every move is permanent.</p></div>",
            unsafe_allow_html=True,
        )
        with st.form("showcase_identity"):
            player = st.text_input(
                "Player name",
                value=st.session_state.player,
                placeholder="How should BTI address you?",
            )
            accepted = st.checkbox(
                "I understand this is a simulated educational game, not investment advice."
            )
            start = st.form_submit_button("Enter BTI", type="primary", width="stretch")
        if start:
            if not player.strip():
                st.error("Enter a player name to begin.")
            elif not accepted:
                st.error("Please confirm the simulation boundary.")
            else:
                st.session_state.player = player.strip()[:40]
                st.session_state.screen = "setup"
                st.rerun()
        with st.expander("Resume a saved campaign"):
            resume_code = st.text_input("Campaign code", placeholder="16-character campaign code")
            if st.button("Resume campaign", width="stretch"):
                if _resume_campaign(resume_code):
                    st.rerun()
                else:
                    st.error("That campaign code is not available on this deployment.")


def _setup() -> None:
    st.markdown("<div class='bti-kicker'>CREATE A CAMPAIGN</div>", unsafe_allow_html=True)
    st.title(f"Choose your long game, {st.session_state.player}.")
    st.caption(
        "Amount and horizon freeze when the campaign begins. There is no clock; completing the campaign is part of the challenge."
    )
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        amount = st.select_slider(
            "Monthly investment", options=AMOUNTS, value=25_000, format_func=lambda x: rupees(x)
        )
        horizon = st.radio(
            "Campaign horizon",
            HORIZONS,
            horizontal=True,
            format_func=lambda x: f"{x // 12} years · {x} moves",
        )
        st.selectbox("Market scenario", ["Balanced India · August 2026"], disabled=True)
        if st.button("Start campaign", type="primary", width="stretch"):
            seed = f"bti-balanced-aug-2026|{horizon}"
            campaign_id = secrets.token_hex(8).upper()
            game = BTIGame.create(
                amount, horizon, seed, repository_root=REPO_ROOT, campaign_id=campaign_id
            )
            _save_game(game)
            st.session_state.draft = []
            st.session_state.last_result = None
            st.session_state.move_history = []
            st.session_state.market_history = []
            _snapshot_history(game)
            st.session_state.screen = "game"
            st.rerun()
    with right:
        level = {24: "Foundation", 36: "Club", 48: "Advanced", 60: "Grand campaign"}[horizon]
        st.markdown(
            f"<div class='bti-dark'><div class='bti-kicker' style='color:#c9f277'>{level}</div>"
            f"<div class='bti-score' style='font-size:3.6rem'>{horizon}</div><h3>monthly decisions</h3>"
            f"<p>{rupees(amount)} arrives before every move. Total committed capital: "
            f"<b>{rupees(amount * horizon)}</b>.</p><p>Your reference opponent is horizon-specific and constrained by the same capital and whole-share rules.</p></div>",
            unsafe_allow_html=True,
        )
    if st.button("← Back"):
        st.session_state.screen = "landing"
        st.rerun()


def _market_tab(game: BTIGame) -> None:
    market = game.market_view()
    disclosure(market)
    event_cols = st.columns(3)
    for col, event in zip(event_cols, market.get("events", []), strict=False):
        col.markdown(
            f"<div class='bti-panel'><div class='bti-kicker'>{event['kind']}</div>"
            f"<b>{event['headline']}</b><div class='bti-note' style='margin-top:.45rem'>{event['detail']}</div></div>",
            unsafe_allow_html=True,
        )
    stocks = market["stocks"]
    frame = market_frame(stocks)
    sectors = ["All sectors", *sorted(frame["Sector"].unique())]
    query_col, sector_col = st.columns([2, 1])
    query = query_col.text_input(
        "Find a stock", placeholder="Ticker…", label_visibility="collapsed"
    )
    sector = sector_col.selectbox("Sector", sectors, label_visibility="collapsed")
    shown = frame
    if query:
        shown = shown[shown["Stock"].str.contains(query.upper(), regex=False)]
    if sector != "All sectors":
        shown = shown[shown["Sector"] == sector]
    st.dataframe(
        shown,
        width="stretch",
        hide_index=True,
        height=455,
        column_config={
            "Price ₹": st.column_config.NumberColumn(format="₹%.2f"),
            "Move %": st.column_config.NumberColumn(format="%+.2f%%"),
            "PE": st.column_config.NumberColumn(format="%.1f"),
            "PB": st.column_config.NumberColumn(format="%.1f"),
            "PEG": st.column_config.NumberColumn(format="%.2f"),
            "Forecast %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Volatility %": st.column_config.NumberColumn(format="%.1f%%"),
            "Drawdown %": st.column_config.NumberColumn(format="%.1f%%"),
            "VaR 95%": st.column_config.NumberColumn(format="%.1f%%"),
            "Expected shortfall %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.markdown("### Inspect a stock")
    selected = st.selectbox("Stock", frame["Stock"].tolist(), label_visibility="collapsed")
    stock = next(item for item in stocks if item["ticker"] == selected)
    price = stock["close_paise"] / 100
    movement = (stock["close_paise"] / stock["open_paise"] - 1) * 100 if stock["open_paise"] else 0
    st.markdown(
        f"<div class='bti-panel'><div class='bti-kicker'>{stock['sector']}</div><h2>{selected} · {rupees(price)}</h2><div class='{'bti-up' if movement >= 0 else 'bti-down'}'>{movement:+.2f}% this month</div></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(6)
    for col, label, value in zip(
        cols,
        ["Open", "High", "Low", "PE", "PEG", "Forecast"],
        [
            rupees(stock["open_paise"] / 100),
            rupees(stock["high_paise"] / 100),
            rupees(stock["low_paise"] / 100),
            f"{stock['pe']:.1f}×",
            f"{stock['peg']:.2f}",
            f"{stock['forecast_pct']:+.1f}%",
        ],
        strict=True,
    ):
        col.metric(label, value)
    risk_cols = st.columns(5)
    risk_cols[0].metric("Sharpe", f"{stock['sharpe']:.2f}")
    risk_cols[1].metric("Volatility", f"{stock['volatility_pct']:.1f}%")
    risk_cols[2].metric("Max drawdown", f"{stock['drawdown_pct']:.1f}%")
    risk_cols[3].metric("Monthly VaR 95%", f"{stock['var_95_pct']:.1f}%")
    risk_cols[4].metric("Expected shortfall", f"{stock['expected_shortfall_95_pct']:.1f}%")
    history = [value / 100 for value in stock.get("history_paise", [])]
    if len(history) > 1:
        st.line_chart(
            pd.DataFrame({selected: history}, index=range(1, len(history) + 1)), height=230
        )
    else:
        st.caption("The price history will build as this campaign advances.")
    with st.expander("How to read these numbers"):
        st.markdown(
            "**PEG** asks whether the price is reasonable for the growth. **Sharpe** asks how much return "
            "the risk has historically bought. **Drawdown** is the deepest fall from a previous high. "
            "**VaR** and **expected shortfall** describe ordinary and more severe downside—not guarantees."
        )


def _draft_cash(
    game: BTIGame, state: dict[str, Any], market_by_ticker: dict[str, dict[str, Any]]
) -> tuple[int, int, int]:
    buying_power = state["cash_paise"] + state["monthly_amount_rupees"] * 100
    buys = 0
    for item in st.session_state.draft:
        amount = item["shares"] * market_by_ticker[item["ticker"]]["close_paise"]
        if item["side"] == "SELL":
            buying_power += amount
        else:
            buys += amount
    return buying_power, buys, buying_power - buys


def _add_instruction(side: str, ticker: str, shares: int) -> None:
    draft = list(st.session_state.draft)
    for item in draft:
        if item["side"] == side and item["ticker"] == ticker:
            item["shares"] += shares
            st.session_state.draft = draft
            return
    draft.append({"side": side, "ticker": ticker, "shares": shares})
    st.session_state.draft = draft


def _portfolio_tab(game: BTIGame) -> None:
    state = game.public_state()
    stocks = game.market_view()["stocks"]
    by_ticker = {stock["ticker"]: stock for stock in stocks}
    st.markdown("### Construct this month’s complete move")
    st.caption(
        "SELL instructions release cash first. BUY instructions then deploy it. HOLD requires no instruction."
    )
    if state["holdings"]:
        rows = []
        for ticker, shares in sorted(state["holdings"].items()):
            value = shares * by_ticker[ticker]["close_paise"]
            rows.append(
                {
                    "Stock": ticker,
                    "Shares held": shares,
                    "Price ₹": by_ticker[ticker]["close_paise"] / 100,
                    "Value ₹": value / 100,
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Price ₹": st.column_config.NumberColumn(format="₹%.2f"),
                "Value ₹": st.column_config.NumberColumn(format="₹%.0f"),
            },
        )
    else:
        st.info("This is your opening move. You do not hold any shares yet.")
    side_col, stock_col, qty_col, add_col = st.columns(
        [0.7, 1.35, 0.8, 0.7], vertical_alignment="bottom"
    )
    staged_sells = {
        item["ticker"]: item["shares"] for item in st.session_state.draft if item["side"] == "SELL"
    }
    sellable = [
        ticker
        for ticker, held in sorted(state["holdings"].items())
        if held > staged_sells.get(ticker, 0)
    ]
    side = side_col.selectbox("Action", ["BUY", *(["SELL"] if sellable else [])])
    ticker_options = sellable if side == "SELL" else sorted(by_ticker)
    ticker = stock_col.selectbox("Stock", ticker_options)
    price = by_ticker[ticker]["close_paise"]
    total_buying_power, _, available_after_draft = _draft_cash(game, state, by_ticker)
    maximum = (
        max(1, available_after_draft // price)
        if side == "BUY"
        else state["holdings"][ticker] - staged_sells.get(ticker, 0)
    )
    # Start near a 20% slice so the interface invites portfolio construction instead of
    # accidentally steering a novice into an all-in single-stock move.
    suggested = (
        max(1, min(maximum, round(total_buying_power * 0.2 / price)))
        if side == "BUY"
        else max(1, maximum)
    )
    shares = qty_col.number_input(
        "Whole shares", min_value=1, max_value=int(maximum), value=int(suggested), step=1
    )
    if add_col.button("Add step", width="stretch"):
        _add_instruction(side, ticker, int(shares))
        st.rerun()
    buying_power, buys, remaining = _draft_cash(game, state, by_ticker)
    utilization = buys / buying_power * 100 if buying_power else 0
    st.markdown("### Monthly execution sheet")
    if st.session_state.draft:
        execution_rows = []
        running = state["cash_paise"] + state["monthly_amount_rupees"] * 100
        ordered = sorted(st.session_state.draft, key=lambda item: item["side"] != "SELL")
        for step, item in enumerate(ordered, start=1):
            amount = item["shares"] * by_ticker[item["ticker"]]["close_paise"]
            movement = amount if item["side"] == "SELL" else -amount
            running += movement
            execution_rows.append(
                {
                    "Step": step,
                    "Instruction": f"{item['side']} {item['shares']} shares of {item['ticker']}",
                    "Estimated price ₹": by_ticker[item["ticker"]]["close_paise"] / 100,
                    "Cash movement ₹": movement / 100,
                    "Cash after step ₹": running / 100,
                }
            )
        st.dataframe(
            pd.DataFrame(execution_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Estimated price ₹": st.column_config.NumberColumn(format="₹%.2f"),
                "Cash movement ₹": st.column_config.NumberColumn(format="%+.0f"),
                "Cash after step ₹": st.column_config.NumberColumn(format="₹%.0f"),
            },
        )
    else:
        st.markdown(
            "<div class='bti-panel bti-note'>Your execution sheet is empty. Add BUY or SELL instructions above.</div>",
            unsafe_allow_html=True,
        )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Opening buying power", money(buying_power))
    metric_cols[1].metric("BUY value", money(buys))
    metric_cols[2].metric("Expected cash left", money(remaining))
    metric_cols[3].metric("Utilisation", f"{utilization:.1f}%", "minimum 90%")
    if remaining < 0:
        st.error("This draft spends more cash than is available.")
    elif utilization < 90:
        st.warning(
            f"Deploy another {money(max(0, round(buying_power * 0.9 - buys)))} to make this move valid."
        )
    clear_col, submit_col = st.columns([1, 2])
    if clear_col.button("Clear sheet", width="stretch", disabled=not st.session_state.draft):
        st.session_state.draft = []
        st.rerun()
    confirmed = st.checkbox(
        "I have reviewed this complete move. I understand it cannot be changed after submission."
    )
    if submit_col.button(
        "MAKE YOUR MOVE",
        type="primary",
        width="stretch",
        disabled=not st.session_state.draft or not confirmed or remaining < 0 or utilization < 90,
    ):
        try:
            result = game.submit_move(
                list(st.session_state.draft), expected_month=state["moves_completed"]
            )
        except GameRuleError as exc:
            st.error(str(exc))
        else:
            st.session_state.last_result = result
            st.session_state.move_history.append(
                {
                    "move": result["move"],
                    "score": result["score"],
                    "classification": result["classification"],
                }
            )
            st.session_state.draft = []
            _snapshot_history(game)
            _save_game(game)
            st.session_state.screen = "result"
            st.rerun()


def _analysis_tab() -> None:
    st.markdown("### Your decision record")
    st.caption(
        "BTI evaluates the process using the information available at the time. The market outcome is reported separately."
    )
    if not st.session_state.move_history:
        st.info("Complete your first move to open the Analysis Board.")
        return
    history = pd.DataFrame(st.session_state.move_history).set_index("move")
    st.line_chart(history[["score"]], height=260)
    for result in reversed(st.session_state.move_history[-6:]):
        with st.expander(
            f"Move {result['move']} · {result['classification']} · {result['score']:.0f}"
        ):
            st.write(
                "This compact record preserves the decision classification. Full evidence is shown immediately after each move."
            )


def _skill_tab(game: BTIGame) -> None:
    state = game.public_state()
    history = st.session_state.move_history
    average = sum(item["score"] for item in history) / len(history) if history else 0
    strong = sum(item["classification"] in {"BEST MOVE", "EXCELLENT", "GOOD"} for item in history)
    cols = st.columns(4)
    cols[0].metric("BTI performance", state["rating"])
    cols[1].metric("Moves completed", state["moves_completed"])
    cols[2].metric("Average quality", f"{average:.1f}" if history else "—")
    cols[3].metric("Strong moves", f"{strong}/{len(history)}" if history else "—")
    st.markdown("### Skill map")
    st.info(
        "A reliable personal skill map needs repeated decisions. BTI will infer valuation, growth, downside, diversification and rebalancing patterns as this campaign develops—it will not invent strengths from insufficient evidence."
    )
    dimensions = pd.DataFrame(
        {
            "Dimension": [
                "Valuation discipline",
                "Growth assessment",
                "Downside awareness",
                "Diversification",
                "Rebalancing",
            ],
            "Evidence collected": [min(len(history) / 8, 1)] * 5,
        }
    ).set_index("Dimension")
    st.bar_chart(dimensions, horizontal=True, height=260)


def _game() -> None:
    game = _load_game()
    if game is None:
        st.session_state.screen = "setup"
        st.rerun()
    state = game.public_state()
    if state["status"] == "COMPLETED":
        st.session_state.screen = "final"
        st.rerun()
    campaign_header(state)
    tab_market, tab_portfolio, tab_analysis, tab_skill = st.tabs(
        ["MARKET BOARD", "BUILD YOUR MOVE", "ANALYSIS BOARD", "SKILL PROFILE"]
    )
    with tab_market:
        _market_tab(game)
    with tab_portfolio:
        _portfolio_tab(game)
    with tab_analysis:
        _analysis_tab()
    with tab_skill:
        _skill_tab(game)
    st.divider()
    with st.expander("Campaign controls"):
        st.caption(
            "There is no clock and inactivity has no penalty. Resignation is explicit and records this campaign as a loss."
        )
        if st.button("Resign campaign", type="secondary"):
            st.session_state.show_resign = True
        if st.session_state.show_resign:
            st.warning(
                "This permanently ends the campaign and records a loss. It cannot be undone."
            )
            left, right = st.columns(2)
            if left.button("Confirm resignation", type="primary"):
                game.resign()
                _save_game(game)
                st.session_state.screen = "final"
                st.session_state.show_resign = False
                st.rerun()
            if right.button("Keep playing"):
                st.session_state.show_resign = False
                st.rerun()


def _result() -> None:
    game = _load_game()
    result = st.session_state.last_result
    if game is None or result is None:
        st.session_state.screen = "game"
        st.rerun()
    colour = score_colour(result["classification"])
    st.markdown(
        f"<div class='bti-dark' style='text-align:center;padding:2.5rem'>"
        f"<div class='bti-class' style='color:{colour}'>{result['classification']}</div>"
        f"<div class='bti-score'>{result['score']:.0f}</div>"
        f"<p style='font-size:1.15rem;max-width:720px;margin:1rem auto'>{result['decision_quality']['headline']}</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("## Your move, analysed")
    left, right = st.columns(2, gap="large")
    with left:
        decision_bullets("What you did well", result["decision_quality"]["did_well"], True)
    with right:
        decision_bullets("Your next improvement", result["decision_quality"]["improve"], False)
    st.divider()
    st.markdown("### Decision quality is not the same as market luck")
    st.caption(result["market_outcome"]["headline"])
    outcome = result["market_outcome"]
    progress = result["progress"]
    cols = st.columns(5)
    cols[0].metric("Portfolio this month", f"{outcome['portfolio_return_pct']:+.2f}%")
    cols[1].metric("Nifty this month", f"{outcome['benchmark_return_pct']:+.2f}%")
    cols[2].metric("Monthly alpha", f"{outcome['alpha_pct']:+.2f}%")
    cols[3].metric("Portfolio", money(progress["portfolio_value_paise"]))
    cols[4].metric("Benchmark", money(progress["benchmark_value_paise"]))
    with st.expander("Review the committed execution sheet"):
        rows = []
        for step, item in enumerate(result["execution"], start=1):
            rows.append(
                {
                    "Step": step,
                    "Instruction": f"{item['side']} {item['shares']} shares of {item['ticker']}",
                    "Execution price ₹": item["price_paise"] / 100,
                    "Cash movement ₹": item["cash_movement_paise"] / 100,
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Execution price ₹": st.column_config.NumberColumn(format="₹%.2f"),
                "Cash movement ₹": st.column_config.NumberColumn(format="%+.0f"),
            },
        )
    if game.status == "COMPLETED":
        label = "See final campaign result"
        target = "final"
    else:
        label = f"Continue to move {progress['move'] + 1}"
        target = "game"
    if st.button(label, type="primary", width="stretch"):
        st.session_state.screen = target
        st.rerun()


def _final() -> None:
    game = _load_game()
    if game is None:
        _reset_campaign()
        st.rerun()
    final = game.final_result()
    completed = game.status == "COMPLETED"
    verdict = "CAMPAIGN COMPLETE" if completed else "CAMPAIGN RESIGNED"
    won = completed and final["wealth_alpha_paise"] > 0
    st.markdown(f"<div class='bti-kicker'>{verdict}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='bti-hero'>{'You beat the index.' if won else ('The index held the advantage.' if completed else 'This campaign records a loss.')}</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    cols[0].metric("Total invested", money(final["total_invested_paise"]))
    cols[1].metric("Final portfolio", money(final["portfolio_value_paise"]))
    cols[2].metric("Nifty benchmark", money(final["benchmark_value_paise"]))
    cols[3].metric("Wealth alpha", money(final["wealth_alpha_paise"]))
    cols[4].metric("Final BTI rating", final["rating"])
    st.markdown("### What the return labels mean")
    ret_cols = st.columns(2)
    ret_cols[0].metric(
        "Portfolio annualised money-weighted return",
        f"{final['portfolio_money_weighted_annual_return_pct']:.2f}%",
    )
    ret_cols[1].metric(
        "Benchmark annualised money-weighted return",
        f"{final['benchmark_money_weighted_annual_return_pct']:.2f}%",
    )
    st.caption(
        "Money-weighted return reflects the monthly cash-flow journey. These are simulated campaign outcomes, not CAGR promises or real-world forecasts."
    )
    st.markdown("### Campaign analysis")
    st.metric("Average decision quality", f"{final['average_move_score']:.1f} / 100")
    if st.session_state.move_history:
        st.line_chart(
            pd.DataFrame(st.session_state.move_history).set_index("move")[["score"]], height=260
        )
    download = game.to_json().encode()
    action_cols = st.columns(2)
    action_cols[0].download_button(
        "Download campaign record",
        download,
        file_name=f"BTI-{game.public_state()['campaign_id']}.json",
        mime="application/json",
        width="stretch",
    )
    if action_cols[1].button("Play another campaign", type="primary", width="stretch"):
        _reset_campaign()
        st.rerun()


def _learning_drawer() -> None:
    with st.sidebar:
        st.markdown("## BTI Learning Room")
        st.caption(
            "Short lessons and puzzles support the game. They never provide the hidden portfolio answer."
        )
        with (CONTENT / "lessons.json").open(encoding="utf-8") as handle:
            lessons = json.load(handle)["lessons"]
        lesson = st.selectbox("Starter lesson", lessons, format_func=lambda item: item["title"])
        st.markdown(lesson["plain_english"])
        st.info(lesson["decision_rule"])
        with st.expander("Check your understanding"):
            st.write(lesson["check_question"])
            st.caption(lesson["check_answer"])
        st.divider()
        st.caption("Educational simulation · no real-money execution · no return promises")


def main() -> None:
    st.set_page_config(
        page_title="Beat the Index · Vriddhi",
        page_icon="♟",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    _defaults()
    _learning_drawer()
    screen = st.session_state.screen
    if screen == "landing":
        _landing()
    elif screen == "setup":
        _setup()
    elif screen == "game":
        _game()
    elif screen == "result":
        _result()
    elif screen == "final":
        _final()
    else:
        st.session_state.screen = "landing"
        st.rerun()
    st.divider()
    st.caption(
        "BTI is an educational strategy game powered by governed Vriddhi research. Simulated outcomes are not investment advice, performance promises or live market quotations."
    )


if __name__ == "__main__":
    main()
