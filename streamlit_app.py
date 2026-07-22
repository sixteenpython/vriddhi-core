import importlib
import os
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

import vriddhi_core

# Streamlit can rerun this entry point in a long-lived worker after deploying
# updated source. Reload only when that worker still holds the pre-ledger core
# module, so the new exports below are resolved from the current checkout.
if not all(
    hasattr(vriddhi_core, name)
    for name in ("build_recommendation_ledger_replay", "load_recommendation_ledger")
):
    vriddhi_core = importlib.reload(vriddhi_core)

from vriddhi_core import (
    build_oos_sip_replay,
    build_recommendation_ledger_replay,
    load_benchmark_series,
    load_portfolio_bundle,
    load_previous_bundle,
    load_recommendation_ledger,
    load_release_manifest,
    scale_allocations,
)

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Vriddhi.png")

st.set_page_config(
    page_title="Vriddhi | Nifty 50 Portfolio Decision Aid",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else None,
    layout="wide",
)

# Brand logo (top-left of the app + sidebar). st.logo is available on newer
# Streamlit; fall back to a sidebar image on older runtimes.
if os.path.exists(LOGO_PATH):
    try:
        st.logo(LOGO_PATH, size="large")
    except Exception:
        st.sidebar.image(LOGO_PATH, width="stretch")

# ---- Optional simple password gate (set APP_PASSWORD in Streamlit secrets) ----
try:
    required_pw = st.secrets.get("APP_PASSWORD", None)
except Exception:
    required_pw = None  # No secrets.toml configured -> gate disabled
if required_pw:
    pw = st.sidebar.text_input("App password", type="password")
    if pw != required_pw:
        st.warning("Enter the app password to continue.")
        st.stop()


# ===============================
# HELPERS
# ===============================
def pct(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}%"


def binding_walk_forward(bundle):
    """Return the walk-forward record matching the bundle's horizon."""
    wf = bundle.get("walk_forward", {})
    return wf.get(str(bundle["horizon_years"])) or {}


def show_disclaimer():
    st.caption(
        "Educational decision aid in beta. Backtests and walk-forward use "
        "corporate-action-adjusted historical prices; past performance does not "
        "guarantee future results. Not investment advice."
    )


def render_landing_vision():
    """Explain Vriddhi's purpose before asking the investor for inputs."""
    st.markdown("## Invest with evidence, not emotion.")
    st.markdown(
        "Vriddhi is an **AI-assisted portfolio decision engine**. It helps you build "
        "and maintain a long-term investment portfolio using quantitative analysis "
        "instead of guesswork."
    )
    st.markdown(
        "Every month, it analyses market data, evaluates the health of companies, "
        "and recommends how your portfolio should evolve. It does not try to predict "
        "tomorrow's winning stock."
    )
    st.success(
        "**The question Vriddhi asks**\n\n"
        "Given everything we know today, which portfolio maximizes the probability "
        "of superior long-term, risk-adjusted returns?"
    )
    st.success(
        "**What Vriddhi does**\n\n"
        "✔ Builds diversified portfolios  \n"
        "✔ Reviews your portfolio every month  \n"
        "✔ Recommends what to buy, reduce, hold or exit  \n"
        "✔ Balances return potential with investment risk  \n"
        "✔ Helps remove emotion from investing"
    )


def render_rebalance_explainer():
    """Give a first-time investor the vocabulary and purpose of rebalancing."""
    st.markdown("### What should I do this month?")
    st.markdown(
        "Each month, Vriddhi reviews your portfolio using the latest market information. "
        "Some companies become more attractive; others become less attractive."
    )
    st.markdown(
        "You do not rebuild the portfolio from scratch. Vriddhi recommends only the "
        "changes needed to keep it aligned with the latest analysis - a regular "
        "**health check-up for your investments**, not a complete makeover."
    )

    st.markdown("#### Your action guide")
    st.markdown(
        """
        <style>
        .vriddhi-action-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
            margin: 0.5rem 0 1rem;
        }
        .vriddhi-action-card {
            background: #ecf8f1;
            border: 1px solid #b7dfc8;
            border-left: 0.35rem solid #2e8b57;
            border-radius: 0.5rem;
            padding: 0.8rem 0.9rem;
            min-height: 8rem;
        }
        .vriddhi-action-card strong { color: #17643b; }
        .vriddhi-action-card p { margin: 0.35rem 0 0; line-height: 1.4; }
        </style>
        <div class="vriddhi-action-grid">
          <div class="vriddhi-action-card"><strong>🌱 PICK · Start investing</strong><p>A new company has entered the portfolio. Begin investing this month's allocation.</p></div>
          <div class="vriddhi-action-card"><strong>🚪 DROP · Exit</strong><p>The company is no longer recommended. Stop future investments and sell your existing holding.</p></div>
          <div class="vriddhi-action-card"><strong>⬆️ TOP-UP · Increase</strong><p>The company deserves a larger allocation. Invest slightly more than before.</p></div>
          <div class="vriddhi-action-card"><strong>⬇️ TRIM · Reduce</strong><p>The company remains, but at a smaller allocation. Continue investing at the lower amount.</p></div>
          <div class="vriddhi-action-card"><strong>✅ HOLD · No change</strong><p>No action is required. Continue investing exactly as you did last month.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("See a ₹50,000/month example", expanded=False):
        st.markdown(
            "Suppose you invest **₹50,000 every month**. Last month, Company A received "
            "**₹5,000**; this month Vriddhi recommends **₹7,000**. That is a **TOP-UP**."
        )
        st.markdown(
            "If another company falls from **₹5,000 to ₹2,500**, that is a **TRIM**. "
            "A completely new company is a **PICK**; one that disappears is a **DROP**. "
            "Everything else remains **HOLD**."
        )
        st.caption(
            "The objective is to make only the necessary changes each month while "
            "keeping your long-term investment strategy on track."
        )

    st.success(
        "**Why rebalance every month?**\n\n"
        "Markets, companies and economic conditions change. Vriddhi periodically "
        "reassesses every company using the latest available information and adjusts "
        "the portfolio only where necessary - without chasing short-term price moves.\n\n"
        "Small, disciplined improvements made consistently over time can have a "
        "significant impact on long-term investment outcomes."
    )
    st.success(
        "**The most important question is not:** Which stock will go up tomorrow?\n\n"
        "**It is:** Given everything we know today, is this still the best portfolio "
        "for the next several years?\n\n"
        "That is the question Vriddhi answers every month."
    )


def verdict_banner(bundle):
    v = bundle["verdict"]
    h = bundle["horizon_years"]
    base = bundle["scenarios"]["base"]
    beat = bundle["benchmark"].get("beat_after_costs")
    if v["recommended"]:
        st.success(
            f"### RECOMMENDED for a {h}-year horizon\n"
            f"This {bundle['num_stocks']}-stock portfolio cleared every credibility gate: "
            f"~{pct(base)} validated out-of-sample CAGR, beats Nifty 50 by {pct(beat)} "
            f"after costs, "
            f"with risk inside our limits. Would you trust it with new monthly money? "
            f"The evidence says yes."
        )
    elif v.get("insufficient_history"):
        st.warning(
            f"### NOT VALIDATED for a {h}-year horizon\n"
            f"A {h}-year window is too short for a credible out-of-sample (walk-forward) test, "
            f"so we will not pretend to certify it. Validated portfolios begin at a 3-year "
            f"horizon - please lengthen the horizon."
        )
    else:
        reasons = ", ".join(v["failing_gates"]) if v["failing_gates"] else "one or more gates"
        st.error(
            f"### NOT RECOMMENDED for a {h}-year horizon\n"
            f"This basket failed: **{reasons}**. We only recommend portfolios that pass "
            f"every gate, so we are not putting our name on this one for {h} years. "
            f"The full analysis is shown below for transparency."
        )


def finance_doctor_note(bundle, monthly_investment):
    """Plain-English, balanced 'finance doctor' narrative for a common investor:
    what to do, what to realistically expect, and the honest risks - tuned to
    the portfolio's actual health (verdict + metrics)."""
    h = bundle["horizon_years"]
    n = bundle["num_stocks"]
    v = bundle["verdict"]
    s = bundle["scenarios"]
    pm = bundle["portfolio_metrics"]
    bench = bundle["benchmark"]
    wf = binding_walk_forward(bundle)
    amt = f"\u20b9{monthly_investment:,}"

    base = s.get("base")
    beat = bench.get("beat_after_costs")
    nifty = bench["metrics"].get(f"cagr_{h}y")
    sharpe = wf.get("oos_sharpe")
    dd = pm.get("max_drawdown")
    stocks = sorted(bundle["stocks"], key=lambda x: x["weight"], reverse=True)
    top3 = sum(x["weight"] for x in stocks[:3]) * 100 if stocks else 0
    sectors = bundle.get("sector_allocation", {})
    n_sectors = len({k for k, val in sectors.items() if val > 0.0001})

    st.markdown("#### Your plan, in plain English")

    # --- Case 1: too short to validate honestly ---
    if v.get("insufficient_history"):
        st.warning(
            f"**Let me be straight with you.** You want to invest **{amt}/month** for "
            f"just **{h} year{'s' if h > 1 else ''}**. That is too short a window for me to "
            f"*honestly* validate a stock portfolio - one bad quarter can wreck a one- or "
            f"two-year plan, and I refuse to dress up a guess as evidence. "
            f"Equities reward **patience**: my validated, trustworthy portfolios start at a "
            f"**3-year** horizon. Please lengthen the horizon in the sidebar and I'll show you "
            f"a plan I can actually stand behind."
        )
        return

    # --- Case 2: validated but failed one or more gates ---
    if not v["recommended"]:
        reasons = "; ".join(v.get("failing_gates", [])) or "one or more credibility checks"
        st.warning(
            f"**My honest advice: don't put new money into a {h}-year version of this today.** "
            f"I ran the full battery of tests and it stumbled on **{reasons}**. "
            f"A portfolio that can't clear these bars is one I won't ask you to trust with your "
            f"hard-earned **{amt}/month**. The good news: a longer horizon often clears them - "
            f"try switching the horizon in the sidebar. (Full numbers are below for the curious.)"
        )
        return

    # --- Case 3: recommended ---
    st.success(
        f"**Here's the deal, in everyday terms.**\n\n"
        f"You're planning to invest **{amt} every month for {h} years**. "
        f"Your job is simple: **build the {n}-stock basket shown below and top it up in the "
        f"same proportions every month**, with a quick monthly rebalance. No market timing, "
        f"no daily tinkering - just steady, disciplined buying.\n\n"
        f"**What you can realistically expect:** in out-of-sample testing (on periods the model "
        f"had *not* seen), this basket compounded at about **{pct(base)} a year** - versus only "
        f"**{pct(nifty)}** for the Nifty 50 over the same stretch. It's not a fluke either: it "
        f"cleared every credibility check - it **beat the index by ~{pct(beat)}** after costs, "
        f"scored a healthy risk-adjusted (Sharpe) **{sharpe:.2f}**, and kept its worst "
        f"peak-to-trough fall to about **{pct(dd)}**, spread across **{n_sectors} sectors**. "
        f"So yes - the historical evidence supports a **validated out-of-sample CAGR of "
        f"roughly {pct(base)}** *if* you stay the course.\n\n"
        f"**Now the honest risks - because a good doctor doesn't only give good news:** this is "
        f"100% equity, so it *will* have ugly years; expect a 20%+ drop at some point and don't "
        f"panic-sell when it happens. It leans on its biggest 3 names (~{top3:.0f}% of the money). "
        f"And **{pct(base)} is a backtested figure, not a promise** - the future can be worse. "
        f"I deliberately ignore the model's rosier forecast and anchor to what actually held up.\n\n"
        f"**Bottom line:** if you can stay invested through the rough patches for the full {h} "
        f"years, this is a portfolio I'd be comfortable putting new monthly money into."
    )


def panel_summary(bundle, monthly_investment):
    finance_doctor_note(bundle, monthly_investment)
    st.markdown("#### A. Portfolio Summary")
    s = bundle["scenarios"]
    pm = bundle["portfolio_metrics"]
    wf = binding_walk_forward(bundle)
    bench = bundle["benchmark"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validated OOS CAGR", pct(s["base"]),
              help="Annualized compound growth observed in the binding walk-forward "
                   "out-of-sample test. This is strategy evidence, not your personal XIRR.")
    c2.metric("Illustrative annual range", f"{pct(s['best'])} / {pct(s['worst'])}",
              help="A volatility-based sensitivity band around the validated OOS CAGR; "
                   "it is not a promised best or worst outcome.")
    c3.metric("Max Drawdown", pct(pm.get("max_drawdown")),
              help="Largest peak-to-trough loss over the backtest.")
    c4.metric("Sharpe (walk-forward)", f"{wf.get('oos_sharpe', 0):.2f}" if wf.get("oos_sharpe") is not None else "n/a")

    c5, c6, c7 = st.columns(3)
    c5.metric(f"Historical portfolio CAGR ({bundle['horizon_years']}yr)",
              pct(pm.get(f"cagr_{bundle['horizon_years']}y")),
              help="Lump-sum historical annual growth of the basket (one-time investment "
                   "held for the period) - distinct from your monthly-contribution projection.")
    c6.metric("Historical Nifty 50 CAGR",
              pct(bench["metrics"].get(f"cagr_{bundle['horizon_years']}y")),
              help=f"Nifty 50 compound annual growth over the same {bundle['horizon_years']}-year window.")
    c7.metric("Beat after costs", pct(bench.get("beat_after_costs")),
              delta="vs Nifty 50")

    st.caption(
        f"A forward model signals a higher annualized return (~{pct(s.get('forecast_signal'))}), "
        f"but we treat that as optimistic and **do not rely on it** - the headline rate "
        f"above is deliberately anchored to validated out-of-sample walk-forward returns."
    )
    st.caption(
        "**Why CAGR, not XIRR?** CAGR describes the strategy's annualized compound growth "
        "without investor cash-flow timing. Vriddhi will label a figure XIRR only when it is "
        "calculated from the actual dated contributions and ending value in the prospective "
        "recommendation ledger."
    )

    # ---- TL;DR: the whole story in one headline + two pictures ----
    st.markdown("---")
    n = bundle.get("num_stocks", "a handful of")
    beat = bench.get("beat_after_costs")
    ov = bundle.get("optimal_view") or {}
    opt_n = (ov.get("optimum") or {}).get("n_stocks")

    st.markdown(
        "### \U0001F31F The 30-second story: your money, aimed straight at the *North Star*"
    )

    narrative = (
        f"Two pictures below tell you almost everything.\n\n"
        f"**1. Where your money actually goes.** Your **\u20b9{monthly_investment:,}/month** is "
        f"never dumped into one hot tip. It's split across **{n} hand-picked Nifty 50 companies**, "
        f"with a hard **15% cap** on any single name, so no one stock can sink your plan. The first "
        f"chart is the exact rupee-by-rupee split - small, deliberate, diversified bets.\n\n"
        f"**2. Why *this* exact mix.** The second chart is a map of *every* way you could invest. "
        f"Out of thousands of possible combinations (the coloured cloud), Vriddhi's blend lands "
        f"right on the **efficient frontier** - the *North Star* line of **maximum reward for the "
        f"least risk** - and sits **well above and to the left of the Nifty 50** (more return, "
        f"similar-or-less risk). Healthier portfolios sit up-and-to-the-left, and that's exactly "
        f"where ours lands."
    )
    if opt_n:
        narrative += (
            f" The pure-maths 'optimum' would gamble everything on just **~{opt_n} names**; we run "
            f"the *same return engine* but spread it across **{n}** so the ride is survivable."
        )
    if beat is not None:
        narrative += (
            f"\n\nThe payoff: a portfolio that has **beaten the index by ~{pct(beat)} after costs** "
            f"and earns a **validated out-of-sample CAGR of ~{pct(s['base'])}** on money the "
            f"model had **never seen** "
            f"- not a forecast, a track record."
        )
    st.success(narrative)

    alloc = scale_allocations(bundle, monthly_investment)
    st.markdown("**1. Where every \u20b9 of your SIP goes each month**")
    st.pyplot(build_monthly_allocation_figure(alloc))

    st.markdown("**2. Why this exact mix - the North Star view**")
    fig_ov = build_optimal_view_figure(bundle)
    if fig_ov is not None:
        st.pyplot(fig_ov)
    else:
        st.caption("Optimization chart unavailable in this bundle.")


def _render_actual_track_record(bundle, monthly_investment):
    st.markdown("### Actual Vriddhi Track Record")
    ledger = load_recommendation_ledger()
    if not ledger or not ledger.get("snapshots"):
        st.warning(
            "Prospective recommendation tracking has not started. Walk-forward evidence remains "
            "available below, but no live monthly release ledger can be claimed yet."
        )
        return

    snapshots = ledger["snapshots"]
    policy = ledger.get("execution_policy", {})
    recorded = len(snapshots)
    required = int(policy.get("minimum_releases_for_performance", 12))
    remaining = max(required - recorded, 0)
    tracking_started = pd.Timestamp(ledger["tracking_started"]).strftime("%d %b %Y")

    c1, c2, c3 = st.columns(3)
    c1.metric("Recorded releases", f"{recorded} of {required}")
    c2.metric("Tracking since", tracking_started)
    c3.metric("Evidence status", "Unlocked" if recorded >= required else "Collecting")
    st.progress(min(recorded / required, 1.0))

    release_rows = []
    horizon_key = str(bundle["horizon_years"])
    for snapshot in reversed(snapshots[-12:]):
        horizon = snapshot.get("horizons", {}).get(horizon_key, {})
        release_rows.append({
            "Recorded release": pd.Timestamp(snapshot["data_through"]).strftime("%d %b %Y"),
            "Decision": "Recommended" if horizon.get("recommended") else "Not recommended",
            "Holdings": horizon.get("num_stocks", 0),
            "Snapshot source": snapshot.get("provenance", "monthly_refresh").replace("_", " "),
        })
    st.dataframe(pd.DataFrame(release_rows), width="stretch", hide_index=True)

    if recorded < required:
        st.info(
            f"**Prospective evidence collection is active: {recorded} of {required} validated "
            f"monthly releases are recorded ({remaining} remaining).**\n\n"
            "Each future transactional refresh appends its published holdings, weights, prices "
            "and PICK / DROP / TOP-UP / TRIM / HOLD actions. Older entries are not overwritten.\n\n"
            "When the evidence gate is reached, this section will automatically unlock the "
            "actual recorded-recommendation SIP replay. Walk-forward evidence will remain below "
            "as a separate test of the methodology."
        )
        return

    replay = build_recommendation_ledger_replay(
        ledger, bundle["horizon_years"], monthly_investment
    )
    if not replay.get("available"):
        st.warning(
            "The release-count gate has been reached, but an actual replay cannot yet be shown: "
            f"{replay.get('reason', 'recorded evidence is incomplete')}."
        )
        return

    st.success(
        "**The prospective evidence gate is complete.** The figures below use only immutable "
        "recommendation snapshots that Vriddhi actually published month by month."
    )
    r1, r2, r3 = st.columns(3)
    r1.metric("Total invested", f"\u20b9{replay['total_invested']:,.0f}")
    r2.metric("Recorded-replay value", f"\u20b9{replay['ending_value']:,.0f}")
    r3.metric("Historical gain", f"\u20b9{replay['gain']:,.0f}")
    st.caption(
        f"Recorded releases from {replay['start_date']} through {replay['end_date']}. "
        "Assumptions: one contribution before each release rebalance, fractional shares, full "
        "investment and no transaction-cost deduction. This remains historical evidence, not "
        "a forecast."
    )


def panel_backtest(bundle, benchmark_df, monthly_investment):
    st.markdown("#### B. Backtest Evidence")
    st.info(
        "**In plain English: did this actually work in the past - or does it just look good on paper?**\n\n"
        "Anyone can build a portfolio that looks brilliant *after* the fact. The honest test is harder: "
        "we hand the model **only old data**, let it pick a portfolio, then fast-forward and watch how it "
        "did on dates it had **never seen** - exactly like real life, where you invest today and the future "
        "is unknown. That 'never seen' result is the **Out-of-sample** column below, and it's the only one "
        "we trust. (The **In-sample** column is the model graded on its own homework - always flattering, so "
        "we don't lean on it.) A big gap where in-sample looks amazing but out-of-sample is weak is a red "
        "flag; we want the out-of-sample number to stand on its own."
    )
    pm = bundle["portfolio_metrics"]
    bench = bundle["benchmark"]["metrics"]
    wf_all = bundle.get("walk_forward", {})

    rows = []
    for yr in (1, 2, 3, 4, 5):
        wf = wf_all.get(str(yr), {})
        rows.append({
            "Window": f"{yr}-year",
            "Historical portfolio CAGR": pct(pm.get(f"cagr_{yr}y")),
            "Historical Nifty 50 CAGR": pct(bench.get(f"cagr_{yr}y")),
            "In-sample CAGR": pct(wf.get("in_sample_cagr")),
            "Validated OOS CAGR": pct(wf.get("oos_cagr")),
            "OOS Sharpe": f"{wf.get('oos_sharpe'):.2f}" if wf.get("oos_sharpe") is not None else "n/a",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "In-sample = model fit on data it has seen. Out-of-sample = performance on "
        "unseen periods (walk-forward). We judge credibility on the out-of-sample column."
    )

    # Walk-forward chart: out-of-sample portfolio equity vs Nifty 50
    wf = binding_walk_forward(bundle)
    eq = wf.get("equity_curve")
    if eq and eq.get("dates"):
        dates = pd.to_datetime(eq["dates"])
        port = np.array(eq["portfolio"], dtype=float)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates, port, color="#A23B72", linewidth=2,
                label="Portfolio (out-of-sample)")
        if benchmark_df is not None:
            bsub = benchmark_df[benchmark_df["Date"].isin(dates)].copy()
            if len(bsub) > 1:
                bvals = bsub["Nifty50_Normalized"].values
                bvals = bvals / bvals[0]
                ax.plot(bsub["Date"], bvals, color="#2E86AB", linewidth=2,
                        linestyle="--", label="Nifty 50")
        ax.axhline(1.0, color="gray", alpha=0.4, linewidth=1)
        ax.set_title(f"Walk-Forward: Out-of-Sample Growth of \u20b91 ({bundle['horizon_years']}-yr lookback)",
                     fontweight="bold")
        ax.set_ylabel("Growth multiple")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.success(
            "**How to read this chart (the simple version):** imagine you put **\u20b91** into the portfolio "
            "at the start of the test, on data the model had never seen. The **solid line is your money**; the "
            "**dashed line is the Nifty 50** (the overall market). The flat grey line at **1.0 is your starting "
            "point - break-even**. Two things tell you it worked: (1) the solid line ends **well above 1.0** "
            "(your \u20b91 grew into more), and (2) it stays **above the dashed line** (you beat simply buying "
            "the market). Dips along the way are normal - what matters is the destination and that it kept "
            "ahead of the market on data it couldn't have memorised."
        )

        _render_actual_track_record(bundle, monthly_investment)

        st.markdown("### Illustrative out-of-sample SIP replay")
        st.markdown("**What if you had started earlier?**")
        st.caption(
            "See how a monthly contribution would have grown across the available "
            "out-of-sample return stream."
        )
        replay = build_oos_sip_replay(eq, monthly_investment)
        replay_end = dates.max().strftime("%d %b %Y")
        replay_rows = []
        for row in replay.to_dict("records"):
            available = row["Replay value"] is not None and not pd.isna(row["Replay value"])
            replay_rows.append({
                "Started": f"{row['Months']} months ago",
                "Historical start": (
                    pd.Timestamp(row["Start date"]).strftime("%d %b %Y")
                    if available else "Not enough OOS history"
                ),
                "Total invested": (
                    f"\u20b9{row['Total invested']:,.0f}" if available else "\u2014"
                ),
                f"Value at replay end ({replay_end})": (
                    f"\u20b9{row['Replay value']:,.0f}" if available else "\u2014"
                ),
            })
        st.dataframe(pd.DataFrame(replay_rows), width="stretch", hide_index=True)
        st.caption(
            f"Illustration uses your selected \u20b9{monthly_investment:,} monthly contribution. "
            "A contribution is added at the first available OOS observation each month. "
            "Longer rows remain unavailable when the stored unseen-data period is shorter."
        )
        st.warning(
            "**Important transparency note:** this table is a SIP cash-flow view of the stored "
            "walk-forward return stream. It is **not** the actual recommendation ledger shown "
            "above. The ledger now preserves published monthly PICK, DROP, TOP-UP, TRIM and "
            "HOLD evidence prospectively and will unlock its own replay after 12 validated "
            "releases."
        )

        st.markdown("### How is this calculated?")
        st.success(
            "Imagine you had started investing on one of the available dates above.\n\n"
            "At the beginning of each month, your chosen contribution is added to Vriddhi's "
            "stored walk-forward return stream. Each contribution then experiences only the "
            "out-of-sample market returns that follow it.\n\n"
            f"The value shown is what those contributions would have been worth at the replay's "
            f"final historical date, **{replay_end}**."
        )
        st.info(
            "**This is not a prediction of future returns.** It is an illustration based on "
            "historical walk-forward results, where each test period used only the price "
            "history available before that unseen period began."
        )

        st.markdown("### Why should I trust these results?")
        trust_cards = [
            (
                "This is NOT a forecast",
                "The figures are not predictions. They replay the stored out-of-sample return "
                "stream using historical market prices as they occurred.",
            ),
            (
                "No future prices were used",
                "Each walk-forward portfolio was fitted using earlier prices, then measured on "
                "the next unseen period. Future test-period prices were not available when its "
                "weights were chosen.",
            ),
            (
                "Disciplined re-optimization",
                "The walk-forward test rebuilds portfolio weights at its train/test boundaries. "
                "It does not simply apply one fixed set of weights across the entire chart.",
            ),
            (
                "Tested on unseen data",
                "The displayed stream comes from Walk-Forward Out-of-Sample Validation, where "
                "performance is measured after fitting rather than on the data used to fit.",
            ),
            (
                "Built for probability, not certainty",
                "Markets are uncertain and future returns will differ. Vriddhi aims to improve "
                "the probability of superior long-term risk-adjusted returns through disciplined "
                "portfolio construction, not to guarantee profits.",
            ),
        ]
        for title, body in trust_cards:
            st.success(f"**{title}**\n\n{body}")

        st.markdown("### How Vriddhi thinks")
        st.info(
            "Vriddhi does not try to predict tomorrow's best stock.\n\n"
            "Instead, it evaluates available market information and refreshes your portfolio "
            "through a disciplined monthly process.\n\n"
            "**The objective is simple:** given everything we know today, which portfolio "
            "maximizes the probability of superior long-term, risk-adjusted returns?"
        )
    else:
        _render_actual_track_record(bundle, monthly_investment)
        st.info(
            "### What if you had started earlier?\n\n"
            "This horizon does not contain enough out-of-sample history for a credible wealth "
            "replay. Vriddhi leaves the illustration unavailable rather than filling it with "
            "in-sample or forecast values."
        )


def _inr(x):
    """Compact INR formatter for axes (L = lakh, Cr = crore)."""
    if x >= 1e7:
        return f"\u20b9{x/1e7:.1f}Cr"
    if x >= 1e5:
        return f"\u20b9{x/1e5:.1f}L"
    if x >= 1e3:
        return f"\u20b9{x/1e3:.0f}K"
    return f"\u20b9{x:.0f}"


def _draw_monthly_allocation(ax, alloc):
    """Horizontal bar of each holding's monthly rupee allocation. Shared by the
    Final Portfolio projection figure and the standalone Summary re-display."""
    a = alloc.sort_values("Monthly Allocation (INR)", ascending=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(a)))
    ax.barh(a["Ticker"], a["Monthly Allocation (INR)"], color=colors)
    for i, v in enumerate(a["Monthly Allocation (INR)"]):
        ax.text(v, i, f" {_inr(v)}", va="center", fontsize=8)
    ax.set_xlabel("Monthly \u20b9")
    ax.set_title("Monthly Stock Allocation", fontweight="bold")
    ax.margins(x=0.18)


def build_monthly_allocation_figure(alloc):
    """Standalone version of the monthly-allocation bar chart for re-display."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_monthly_allocation(ax, alloc)
    fig.tight_layout()
    return fig


def portfolio_table_view(alloc):
    """Format a scaled allocation for consistent display across portfolio views."""
    display = alloc.copy()
    display["Weight"] = display["Weight"].apply(lambda value: f"{value*100:.1f}%")
    display["Monthly Allocation (INR)"] = display["Monthly Allocation (INR)"].apply(
        lambda value: f"₹{value:,.0f}"
    )
    display["Current_Price"] = display["Current_Price"].apply(
        lambda value: f"₹{value:,.0f}"
    )
    display = display[
        [
            "Ticker",
            "Sector",
            "Weight",
            "Monthly Allocation (INR)",
            "Current_Price",
            "Whole_Shares",
        ]
    ]
    display.columns = ["Stock", "Sector", "Weight", "Monthly ₹", "Price", "Whole Shares"]
    return display


def display_release_date(value):
    """Format a bundle date as the compact release label used in the UI."""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return str(value or "date unavailable")
    return f"{timestamp.day}-{timestamp.strftime('%b-%Y')}"


def build_projection_figure(bundle, monthly_investment, alloc):
    """projection.png-style visual: SIP growth journey + year-wise breakdown +
    monthly stock allocation. Uses the validated walk-forward OOS CAGR as an
    illustrative annual-rate input (not the optimistic forecast)."""
    horizon_years = bundle["horizon_years"]
    base_cagr = bundle["scenarios"]["base"] or 0.0
    horizon_months = horizon_years * 12
    monthly_rate = (base_cagr / 100.0) / 12.0

    months = np.arange(1, horizon_months + 1)
    invested = monthly_investment * months
    if monthly_rate > 0:
        projected = monthly_investment * (((1 + monthly_rate) ** months - 1) / monthly_rate)
    else:
        projected = invested.astype(float)
    gains = projected - invested

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.35, wspace=0.25)

    # ---- Investment journey (top, full width) ----
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(months, invested, projected, alpha=0.30, color="#5BBA6F",
                     label="Potential gains")
    ax1.plot(months, invested, "--", lw=2.5, color="#2E86AB", label="Total invested")
    ax1.plot(months, projected, "-", lw=2.5, color="#A23B72",
             label=f"Illustrative value ({base_cagr:.1f}% annual-rate input)")
    for m in [y for y in (12, 24, 36, 48, 60) if y <= horizon_months]:
        ax1.annotate(_inr(projected[m - 1]), xy=(m, projected[m - 1]),
                     xytext=(m, projected[m - 1] + max(projected) * 0.07),
                     ha="center", fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#A23B72", lw=1.2),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3B0", alpha=0.8))
    ax1.set_title(f"Your Investment Journey: \u20b9{monthly_investment:,}/month for "
                  f"{horizon_months} months", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Months"); ax1.set_ylabel("Amount")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _inr(v)))
    ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.3)

    # ---- Year-wise breakdown (bottom-left) ----
    ax2 = fig.add_subplot(gs[1, 0])
    yr_months = [y for y in (12, 24, 36, 48, 60) if y <= horizon_months]
    yrs = list(range(1, len(yr_months) + 1))
    yr_inv = [invested[m - 1] for m in yr_months]
    yr_gain = [gains[m - 1] for m in yr_months]
    x = np.arange(len(yrs)); w = 0.38
    ax2.bar(x - w / 2, [v / 1e5 for v in yr_inv], w, label="Invested", color="#2E86AB", alpha=0.85)
    ax2.bar(x + w / 2, [v / 1e5 for v in yr_gain], w, label="Gains", color="#F18F01", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels([f"Yr {y}" for y in yrs])
    ax2.set_ylabel("\u20b9 Lakhs"); ax2.set_title("Year-wise Breakdown", fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")

    # ---- Monthly stock allocation (bottom-right) ----
    ax3 = fig.add_subplot(gs[1, 1])
    _draw_monthly_allocation(ax3, alloc)
    return fig, invested[-1], projected[-1], gains[-1]


def _stock_peg(stock):
    """Read the retained PEG value from a stock's generated explanation."""
    fundamental = stock.get("explanation", {}).get("fundamental", "")
    if not fundamental.startswith("PEG "):
        return None
    token = fundamental.removeprefix("PEG ").split(" ", 1)[0]
    try:
        return float(token)
    except ValueError:
        return None


def _render_stock_thesis_card(stock, bundle, monthly_investment):
    """Render one holding as an evidence-led, plain-English decision card."""
    horizon = int(bundle["horizon_years"])
    weight = float(stock.get("weight", 0) or 0)
    sector = stock.get("sector", "Unknown")
    backtest = stock.get("backtest", {})
    cagr = backtest.get(f"cagr_{horizon}y")
    volatility = backtest.get("volatility")
    drawdown = backtest.get("max_drawdown")
    sharpe = backtest.get("sharpe")
    portfolio_volatility = bundle.get("portfolio_metrics", {}).get("volatility")
    portfolio_drawdown = bundle.get("portfolio_metrics", {}).get("max_drawdown")
    peg = _stock_peg(stock)
    pe = float(stock.get("pe_ratio", 0) or 0)
    pb = float(stock.get("pb_ratio", 0) or 0)
    contribution_points = None if cagr is None else float(cagr) * weight

    if weight >= 0.145:
        role = "a core holding at the 15% safety cap"
    elif weight >= 0.10:
        role = "a meaningful core holding"
    else:
        role = "a supporting holding sized to limit single-stock risk"

    if peg is None:
        peg_reading = "PEG is unavailable, so no price-versus-growth claim is made."
    elif peg < 1:
        peg_reading = (
            f"PEG is **{peg:.2f}**. Below 1.0 usually means the price looks inexpensive "
            "relative to the company's historical growth."
        )
    elif peg <= 1.3:
        peg_reading = (
            f"PEG is **{peg:.2f}**—roughly fair relative to the company's historical growth."
        )
    else:
        peg_reading = (
            f"PEG is **{peg:.2f}**, which is expensive relative to historical growth. "
            "It must therefore earn its place through quality or portfolio balance—not cheapness."
        )

    if pe <= 0:
        pe_reading = "PE is unavailable."
    elif pe >= 45:
        pe_reading = (
            f"PE is **{pe:.1f}**: the price is roughly equivalent to {pe:.1f} years of "
            "today's annual profit. That is expensive, so very high expectations are already "
            "built into the price."
        )
    elif pe >= 25:
        pe_reading = (
            f"PE is **{pe:.1f}**: the price is roughly equivalent to {pe:.1f} years of "
            "today's annual profit, which implies above-average expectations."
        )
    else:
        pe_reading = (
            f"PE is **{pe:.1f}**: the price is roughly equivalent to {pe:.1f} years of "
            "today's annual profit—a less demanding earnings valuation."
        )

    if pb <= 0:
        pb_reading = "PB is unavailable."
    elif pb >= 8:
        pb_reading = (
            f"PB is **{pb:.1f}**: the market pays about ₹{pb:.1f} for each ₹1 of accounting "
            "net worth, which is a rich premium."
        )
    elif pb >= 4:
        pb_reading = (
            f"PB is **{pb:.1f}**: the market pays about ₹{pb:.1f} for each ₹1 of accounting "
            "net worth, so a meaningful quality premium is present."
        )
    else:
        pb_reading = (
            f"PB is **{pb:.1f}**: the market pays about ₹{pb:.1f} for each ₹1 of accounting "
            "net worth—a comparatively moderate premium."
        )

    cagr_text = "n/a" if cagr is None else f"{float(cagr):.1f}%"
    sharpe_text = "n/a" if sharpe is None else f"{float(sharpe):.2f}"
    volatility_text = "n/a" if volatility is None else f"{float(volatility):.1f}%"
    drawdown_text = "n/a" if drawdown is None else f"{float(drawdown):.1f}%"
    portfolio_volatility_text = (
        "n/a" if portfolio_volatility is None else f"{float(portfolio_volatility):.1f}%"
    )
    portfolio_drawdown_text = (
        "n/a" if portfolio_drawdown is None else f"{float(portfolio_drawdown):.1f}%"
    )
    contribution_text = (
        "n/a" if contribution_points is None else f"{contribution_points:.1f} pts"
    )
    if cagr is not None and float(cagr) > -100:
        growth_multiple = (1 + float(cagr) / 100) ** horizon
        growth_story = (
            f"If ₹1 had followed that exact historical path, it would have become about "
            f"**₹{growth_multiple:.2f} after {horizon} years**."
        )
    else:
        growth_story = "There is not enough retained history to translate growth into a ₹1 example."

    with st.container(border=True):
        st.markdown(f"#### 🌿 {stock['ticker']} — {sector}")
        weight_col, money_col, growth_col, sharpe_col = st.columns(4)
        weight_col.metric("Portfolio weight", f"{weight*100:.1f}%")
        money_col.metric("Monthly allocation", f"₹{weight*monthly_investment:,.0f}")
        growth_col.metric(f"Historical {horizon}Y growth", cagr_text)
        sharpe_col.metric("Growth contribution", contribution_text)

        st.markdown("**Why this stock earned its place**")
        st.markdown(
            f"**{stock['ticker']} is {role}.** It passed the value/growth screen and then "
            "earned a place when the optimizer considered how all shortlisted stocks work "
            "together. No single number below is the reason on its own."
        )

        st.markdown("**Valuation in plain English**")
        st.info(f"{peg_reading}\n\n{pe_reading}\n\n{pb_reading}")

        st.markdown("**Growth and contribution in plain English**")
        st.success(
            f"The stock's retained {horizon}-year history grew at **{cagr_text} a year**. "
            f"{growth_story} At its **{weight*100:.1f}% portfolio weight**, its standalone "
            f"growth accounts for roughly **{contribution_text}** of the portfolio's weighted "
            "historical growth calculation. That is attribution—not cash credited to your "
            "account and not a future promise."
        )

        st.markdown("**Risk in plain English**")
        st.warning(
            f"**Volatility {volatility_text}** describes how bumpy the stock's usual journey "
            f"was; it is not a loss forecast. **Max drawdown {drawdown_text}** was the worst "
            f"drop from a previous high that an investor had to sit through. Its **Sharpe "
            f"score was {sharpe_text}**—a summary of how much historical return it earned for "
            "the bumpiness taken, where higher is better. The combined portfolio was steadier "
            f"at **{portfolio_volatility_text} volatility** with a **{portfolio_drawdown_text} "
            "worst fall**, showing why position sizing and diversification matter."
        )

        st.markdown("**Why it fits the whole portfolio**")
        st.info(
            f"**How this supports Vriddhi's philosophy:** at a **{weight*100:.1f}% weight**, "
            f"{stock['ticker']} supplies **{sector} exposure** and about **{contribution_text}** "
            "of weighted historical growth. Its return pattern and risk were considered "
            "alongside every other holding. An attractive stock in isolation is not enough; "
            "it remains only because the evidence says it improves today's overall balance "
            "of long-term return and risk."
        )
        st.caption(
            f"Forward model signal (context only, not relied upon): ~"
            f"{float(stock.get('forecast', 0) or 0):.0f}% annualized. The recommendation "
            "is anchored to historical and walk-forward evidence, not this forecast."
        )


def panel_portfolio(bundle, monthly_investment):
    st.markdown("#### C. Final Portfolio")
    alloc = scale_allocations(bundle, monthly_investment)

    screened = bundle.get("screened_count")
    if screened:
        st.caption(
            f"Funnel: screened **{screened}** Nifty 50 candidates -> optimized down to "
            f"**{bundle['num_stocks']}** holdings (Markowitz max-Sharpe, "
            f"15% single-name cap, 5% floor)."
        )
    st.info(
        "**In plain English: how were these stocks chosen - and is it just guesswork?**\n\n"
        "No hunches, no tips, no news headlines. It's a two-step sieve. **Step 1 - value for money:** out of "
        "the Nifty 50 we keep only companies that are *reasonably priced for how fast they've actually grown* "
        f"(the PEG idea, explained below) - that left **{screened or 'a shortlist of'} candidates**. "
        "**Step 2 - the smartest mix:** a maths engine (Markowitz optimisation) then blends them into the "
        f"combination that historically gave the **best return for the least bumpiness**, landing on these "
        f"**{bundle['num_stocks']}**. We deliberately **cap any single stock at 15%** so no one company can "
        "sink you, and spread the money across several sectors. Think of it as a recipe, not a gamble: "
        "the same inputs always produce the same portfolio."
    )

    # ---- projection.png-style growth visual (validated OOS rate as input) ----
    fig, total_invested, final_value, total_gains = build_projection_figure(
        bundle, monthly_investment, alloc)
    m1, m2, m3 = st.columns(3)
    m1.metric("Total invested", _inr(total_invested),
              help=f"\u20b9{monthly_investment:,}/month for {bundle['horizon_years']} years.")
    m2.metric("Illustrative SIP value", _inr(final_value),
              help="A hypothetical monthly-contribution outcome using the validated "
                   "walk-forward OOS CAGR as the annual-rate input. This is not an XIRR "
                   "or a forecast of your actual result.")
    m3.metric("Illustrative gains", _inr(total_gains))
    st.pyplot(fig)
    st.caption(
        "Illustrative only. This applies the portfolio's validated out-of-sample CAGR "
        f"({pct(bundle['scenarios']['base'])}) as the annual-rate input for a hypothetical "
        f"{bundle['horizon_years']}-year monthly SIP. It is neither your personal XIRR nor "
        "a guaranteed future outcome."
    )
    st.markdown("---")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"**{bundle['num_stocks']} stocks | \u20b9{monthly_investment:,}/month**")
        st.dataframe(portfolio_table_view(alloc), width="stretch", hide_index=True)

    with col2:
        sectors = bundle.get("sector_allocation", {})
        sectors = {k: v for k, v in sectors.items() if v > 0.0001}
        if sectors:
            fig, ax = plt.subplots(figsize=(5, 5))
            colors = plt.cm.Set3(np.linspace(0, 1, len(sectors)))
            ax.pie(list(sectors.values()), labels=list(sectors.keys()),
                   autopct="%1.0f%%", colors=colors, startangle=90)
            ax.set_title("Allocation by Sector", fontweight="bold")
            st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Why these stocks?")
    st.caption("Per-stock rationale + how to read the numbers")
    st.markdown("#### How to read each stock decision")
    st.info(
        "**Quick guide to the jargon, in everyday terms:**\n"
        "- **PEG** - *are you paying a fair price for the growth?* Below **1.0** = a bargain "
        "(you're paying less than \u20b91 for each \u20b91 of growth); around **1** = fair; "
        "well above **1** = expensive, so we only keep it for quality or balance, not because it's cheap.\n"
        "- **PE (price-to-earnings)** - roughly *how many years of the company's profit you're paying for "
        "one share*. Lower is cheaper; a very high PE means big expectations are already baked in.\n"
        "- **PB (price-to-book)** - *price versus the company's accounting net worth*. A high PB means the "
        "market is paying a big premium over what's on the books.\n"
        "- **Volatility** - *how bumpy the ride is* month to month. **Max drawdown** - *the worst drop from "
        "a previous high* you'd have had to sit through.\n"
        "- **Contribution** - how much this one holding actually added to the portfolio's past growth.\n\n"
        "_Below: the plain-language reason each stock earned its place._"
    )
    st.markdown("### Stock-by-stock decision cards")
    st.caption(
        "Each card separates the investment case, the risk you must accept, and the "
        "stock's role in Vriddhi's overall long-term return-versus-risk philosophy."
    )
    for s in bundle["stocks"]:
        _render_stock_thesis_card(s, bundle, monthly_investment)


def panel_risk(bundle):
    st.markdown("#### D. Risk View")
    pm = bundle["portfolio_metrics"]
    bench = bundle["benchmark"]["metrics"]
    stocks = sorted(bundle["stocks"], key=lambda x: x["weight"], reverse=True)
    sectors = {
        name: float(weight)
        for name, weight in bundle.get("sector_allocation", {}).items()
        if float(weight) > 0.0001
    }

    top1 = stocks[0]["weight"] if stocks else 0
    top3 = sum(s["weight"] for s in stocks[:3])
    max_sector = max(sectors.items(), key=lambda kv: kv[1]) if sectors else ("n/a", 0)
    portfolio_volatility = float(pm.get("volatility") or 0)
    benchmark_volatility = float(bench.get("volatility") or 0)
    portfolio_drawdown = float(pm.get("max_drawdown") or 0)
    benchmark_drawdown = float(bench.get("max_drawdown") or 0)
    portfolio_sharpe = float(pm.get("sharpe") or 0)
    benchmark_sharpe = float(bench.get("sharpe") or 0)
    volatility_gap = portfolio_volatility - benchmark_volatility
    drawdown_gap = portfolio_drawdown - benchmark_drawdown
    rupee_after_drawdown = 100_000 * (1 - portfolio_drawdown / 100)
    weighted_stock_volatility = sum(
        float(stock.get("weight", 0) or 0)
        * float(stock.get("backtest", {}).get("volatility") or 0)
        for stock in stocks
    )

    st.markdown("### Your risk, in one sentence")
    summary_method = st.success if portfolio_drawdown < 25 and portfolio_sharpe > 1 else st.warning
    summary_method(
        f"The portfolio stayed inside Vriddhi's **25% historical drawdown guardrail** "
        f"at **{portfolio_drawdown:.1f}%**, while producing a historical risk-adjusted "
        f"score of **{portfolio_sharpe:.2f}**. It was "
        f"**{abs(volatility_gap):.1f} percentage points "
        f"{'bumpier' if volatility_gap >= 0 else 'steadier'} than Nifty 50**, so this is "
        "a disciplined equity portfolio—not a low-risk product."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Largest holding", f"{top1*100:.1f}%", help="Hard cap: 15% per stock.")
    c2.metric("Top-3 concentration", f"{top3*100:.1f}%")
    c3.metric("Largest sector", f"{max_sector[0]} {max_sector[1]*100:.0f}%")
    c4.metric("Portfolio volatility", f"{portfolio_volatility:.1f}%")

    st.markdown("### Risk scorecard")
    drawdown_reading = (
        f"Passed the 25% gate; {abs(drawdown_gap):.1f} points "
        f"{'worse' if drawdown_gap > 0 else 'better'} than Nifty in this history."
    )
    volatility_reading = (
        f"{abs(volatility_gap):.1f} points "
        f"{'bumpier' if volatility_gap > 0 else 'steadier'} than Nifty."
    )
    scorecard = pd.DataFrame([
        {
            "Risk lens": "Single-company exposure",
            "Vriddhi portfolio": f"{top1*100:.1f}% largest holding",
            "Reference": "15% hard cap",
            "Plain-English reading": "At the limit, but no company exceeds it.",
        },
        {
            "Risk lens": "Top-three exposure",
            "Vriddhi portfolio": f"{top3*100:.1f}%",
            "Reference": ", ".join(stock["ticker"] for stock in stocks[:3]),
            "Plain-English reading": f"₹{top3*100_000:,.0f} of each ₹1 lakh sits in the top three.",
        },
        {
            "Risk lens": "Sector exposure",
            "Vriddhi portfolio": f"{max_sector[0]} {max_sector[1]*100:.1f}%",
            "Reference": f"{len(sectors)} sectors represented",
            "Plain-English reading": "No single industry owns the entire outcome.",
        },
        {
            "Risk lens": "Usual market swings",
            "Vriddhi portfolio": f"{portfolio_volatility:.1f}% volatility",
            "Reference": f"Nifty 50 {benchmark_volatility:.1f}%",
            "Plain-English reading": volatility_reading,
        },
        {
            "Risk lens": "Worst historical fall",
            "Vriddhi portfolio": f"{portfolio_drawdown:.1f}% drawdown",
            "Reference": f"25% gate; Nifty 50 {benchmark_drawdown:.1f}%",
            "Plain-English reading": drawdown_reading,
        },
        {
            "Risk lens": "Return earned per unit of risk",
            "Vriddhi portfolio": f"Sharpe {portfolio_sharpe:.2f}",
            "Reference": f"Nifty 50 {benchmark_sharpe:.2f}",
            "Plain-English reading": "Higher was better historically; this is not a future promise.",
        },
    ])
    st.dataframe(scorecard, width="stretch", hide_index=True)

    st.markdown("### What could hurt this portfolio?")
    with st.container(border=True):
        st.markdown("#### 🧺 Concentration risk — several eggs still share a basket")
        st.markdown(
            f"The top three names—**{', '.join(stock['ticker'] for stock in stocks[:3])}**—"
            f"hold **{top3*100:.1f}%** of the money. The largest sector is "
            f"**{max_sector[0]} at {max_sector[1]*100:.1f}%**. These are meaningful "
            "exposures, even though the 15% stock cap and sector spread prevent a single "
            "company or industry from controlling everything."
        )

    with st.container(border=True):
        st.markdown("#### 📉 A bad market can still produce an uncomfortable loss")
        st.warning(
            f"The worst historical portfolio fall was **{portfolio_drawdown:.1f}%**. In "
            f"plain money terms, **₹1,00,000 could temporarily have shown about "
            f"₹{rupee_after_drawdown:,.0f}**. Nifty 50's historical fall in the same data "
            f"was **{benchmark_drawdown:.1f}%**—"
            f"{abs(drawdown_gap):.1f} percentage points "
            f"{'less' if drawdown_gap > 0 else 'more'} than Vriddhi. Passing the 25% gate "
            "does not mean the journey will feel comfortable."
        )

    if stocks:
        most_volatile = max(
            stocks,
            key=lambda stock: float(stock.get("backtest", {}).get("volatility") or 0),
        )
        deepest_fall = max(
            stocks,
            key=lambda stock: float(stock.get("backtest", {}).get("max_drawdown") or 0),
        )
        premium_stocks = sorted(
            (stock for stock in stocks if (_stock_peg(stock) or 0) > 1.3),
            key=lambda stock: _stock_peg(stock) or 0,
            reverse=True,
        )
        with st.container(border=True):
            st.markdown("#### 🔍 Single-stock hotspots — where to keep your eyes open")
            st.markdown(
                f"**{most_volatile['ticker']}** had the largest usual swings at "
                f"**{float(most_volatile['backtest']['volatility']):.1f}% volatility**. "
                f"**{deepest_fall['ticker']}** had the deepest historical fall at "
                f"**{float(deepest_fall['backtest']['max_drawdown']):.1f}%**."
            )
            if premium_stocks:
                st.markdown(
                    "The richest growth valuations are "
                    + ", ".join(
                        f"**{stock['ticker']} (PEG {_stock_peg(stock):.2f})**"
                        for stock in premium_stocks[:3]
                    )
                    + ". These are not automatically bad holdings, but expectations are "
                    "higher and disappointment can hurt more. Their position limits matter."
                )

    with st.container(border=True):
        st.markdown("#### 🛡️ What diversification is doing for you")
        st.success(
            f"The weighted average volatility of the individual holdings was about "
            f"**{weighted_stock_volatility:.1f}%**, while the combined portfolio measured "
            f"**{portfolio_volatility:.1f}%** across **{len(stocks)} stocks and "
            f"{len(sectors)} sectors**. Historically, the holdings did not all move in the "
            "same direction at the same time—that is the risk-reduction job of diversification."
        )

    st.markdown("### If markets turn ugly: your investor playbook")
    st.info(
        f"1. **Expect a fall before it happens.** A temporary decline around the historical "
        f"**{portfolio_drawdown:.1f}%** level should not surprise you—and the future could be worse.\n\n"
        "2. **Do not redesign the portfolio during panic.** Use the Monthly Rebalance tab; "
        "do not react to headlines or one bad week.\n\n"
        "3. **Keep the time horizon honest.** Only invest money you can leave untouched for "
        f"the full **{bundle['horizon_years']} years**.\n\n"
        "4. **Review the evidence, not your emotions.** A change is justified when the monthly "
        "data and portfolio fit change—not simply because prices fell."
    )

    st.caption(
        "Important limitation: historical volatility and drawdown describe what happened in "
        "the retained data. They are not a ceiling on future losses, and taxes, liquidity and "
        "investor behavior can make real outcomes worse."
    )
    st.success(
        "**How this supports Vriddhi's philosophy:** risk is not something Vriddhi pretends "
        "to eliminate. It asks whether today's complete portfolio offers the strongest "
        "evidence-based probability of superior long-term returns **for the risk you must "
        "live through**. The caps, diversification and monthly review exist to ensure that "
        "risk is deliberate rather than accidental."
    )


def _rebalance_rationale(ticker, action, previous_weight, current_weight, bundle, previous):
    """Explain an action from optimizer inputs and retained health metrics."""
    current_stocks = {stock["ticker"]: stock for stock in bundle.get("stocks", [])}
    previous_stocks = {stock["ticker"]: stock for stock in previous.get("stocks", [])}
    current_candidates = {
        stock["ticker"]: stock
        for stock in bundle.get("optimal_view", {}).get("candidates", [])
    }
    previous_candidates = {
        stock["ticker"]: stock
        for stock in previous.get("optimal_view", {}).get("candidates", [])
    }

    current_stock = current_stocks.get(ticker)
    previous_stock = previous_stocks.get(ticker)
    current_candidate = current_candidates.get(ticker)
    previous_candidate = previous_candidates.get(ticker)
    horizon = int(bundle.get("horizon_years", 5))

    if action.startswith("PICK"):
        decision = (
            "This company earned a place in today's final portfolio at "
            f"{current_weight*100:.1f}%. When considered alongside the other holdings, "
            "it improved the overall balance between potential return and risk."
        )
        alignment = (
            "Adding it helps build the mix that today's evidence says has the best "
            "chance of superior long-term returns for the risk taken."
        )
    elif action.startswith("DROP"):
        if current_candidate:
            decision = (
                "The company still passed the first health screen, but it did not make "
                "today's final portfolio once every candidate was compared together, "
                "including how their prices tend to move alongside one another. This is "
                "about its fit in the whole portfolio, not a claim that it is unhealthy."
            )
        else:
            decision = (
                "The latest evidence no longer kept the company in the screened shortlist, "
                "so its portfolio weight moved to zero."
            )
        alignment = (
            "Removing it makes room for the combination that today's evidence says has "
            "the better chance of superior long-term returns for the risk taken."
        )
    elif action == "TOP-UP":
        decision = (
            "The company still belongs in the portfolio, and today's analysis gives it "
            f"a slightly larger role: {previous_weight*100:.1f}% to "
            f"{current_weight*100:.1f}%."
        )
        alignment = (
            "Putting a little more money behind it moves the whole portfolio toward the "
            "best evidence-based balance of long-term return and risk available today."
        )
    elif action == "TRIM":
        decision = (
            "The company still adds value, but today's analysis gives it a smaller role: "
            f"{previous_weight*100:.1f}% to {current_weight*100:.1f}%."
        )
        alignment = (
            "Keeping it with less money preserves its benefit while moving the whole "
            "portfolio toward a better long-term return-versus-risk balance."
        )
    else:
        decision = (
            "The company still fits today's portfolio, and its suggested weight barely "
            f"changed ({previous_weight*100:.1f}% to {current_weight*100:.1f}%). The "
            "movement was less than 1 percentage point, so no action is needed."
        )
        alignment = (
            "Leaving it unchanged avoids needless trading while preserving the mix that "
            "today's evidence supports for superior long-term, risk-adjusted returns."
        )

    evidence = []
    if current_candidate:
        evidence.append(
            "historical annualized return input "
            f"{float(current_candidate.get('ret', 0))*100:.1f}%"
        )
        evidence.append(
            "volatility (the usual ups and downs) "
            f"{float(current_candidate.get('vol', 0))*100:.1f}%"
        )
    if previous_candidate and current_candidate:
        evidence.append(
            "risk-adjusted score (Sharpe; higher is better) moved from "
            f"{float(previous_candidate.get('sharpe', 0)):.2f} to "
            f"{float(current_candidate.get('sharpe', 0)):.2f}"
        )
    elif current_candidate:
        evidence.append(
            "risk-adjusted score (Sharpe; higher is better) "
            f"{float(current_candidate.get('sharpe', 0)):.2f}"
        )

    health_stock = current_stock or previous_stock
    if health_stock:
        backtest = health_stock.get("backtest", {})
        cagr = backtest.get(f"cagr_{horizon}y")
        drawdown = backtest.get("max_drawdown")
        if cagr is not None:
            prefix = "historical" if current_stock else "previously recorded historical"
            evidence.append(
                f"{prefix} {horizon}-year growth (CAGR) {float(cagr):.1f}% a year"
            )
        if drawdown is not None:
            prefix = "worst historical fall" if current_stock else "previously recorded worst fall"
            evidence.append(f"{prefix} {float(drawdown):.1f}%")

        fundamental = health_stock.get("explanation", {}).get("fundamental", "")
        if fundamental.startswith("PEG "):
            peg = fundamental.removeprefix("PEG ").split(" ", 1)[0]
            prefix = "PEG" if current_stock else "previously recorded PEG"
            evidence.append(f"{prefix} (price paid for growth) {peg}")

    evidence_text = "; ".join(evidence)
    evidence_section = (
        f"\n\n**Evidence in plain English:** {evidence_text}." if evidence else ""
    )
    return (
        f"{decision}{evidence_section}\n\n"
        f"**How this supports Vriddhi's philosophy:** {alignment}"
    )


def _rebalance_short_reason(action):
    """One-line scan aid for the compact comparison table."""
    return {
        "PICK (new buy)": "Earned a place in today's strongest overall mix",
        "DROP (exit)": "Makes room for a better overall portfolio fit",
        "TOP-UP": "A larger role improves the return-risk balance",
        "TRIM": "A smaller role improves the return-risk balance",
        "HOLD": "Still fits; no meaningful weight change",
    }[action]


def _build_surplus_purchase_plan(rows, surplus, max_actions=3):
    """Allocate surplus to at most `max_actions` whole-share PICK/TOP-UP buys.

    The target rupee increases determine the desired split. We enumerate the
    feasible stock subsets, then add whole shares to the subset whose final
    spend stays closest to that split while leaving as little cash idle as the
    available share prices allow.
    """
    budget = float(surplus or 0)
    if budget <= 0 or max_actions <= 0:
        return []

    candidates = [
        {
            "ticker": row["Stock"],
            "action": row["Action"],
            "price": float(row.get("_Action price", 0) or 0),
            "target_increase": float(row.get("_Target increase", 0) or 0),
        }
        for row in rows
        if row["Action"] in {"PICK (new buy)", "TOP-UP"}
        and float(row.get("_Action price", 0) or 0) > 0
        and float(row.get("_Target increase", 0) or 0) > 0
    ]
    candidates = [candidate for candidate in candidates if candidate["price"] <= budget]
    if not candidates:
        return []

    candidates.sort(
        key=lambda item: (
            0 if item["action"] == "PICK (new buy)" else 1,
            -item["target_increase"],
            item["ticker"],
        )
    )
    total_target = sum(item["target_increase"] for item in candidates)
    desired = {
        item["ticker"]: budget * item["target_increase"] / total_target
        for item in candidates
    }

    # Use three distinct actions whenever the budget can buy at least one share
    # of a feasible three-name combination; otherwise gracefully fall back.
    feasible_subsets = []
    for count in range(min(max_actions, len(candidates)), 0, -1):
        feasible_subsets = [
            subset
            for subset in combinations(candidates, count)
            if sum(item["price"] for item in subset) <= budget
        ]
        if feasible_subsets:
            break

    best = None
    for subset in feasible_subsets:
        quantities = {item["ticker"]: 1 for item in subset}
        spent = {item["ticker"]: item["price"] for item in subset}
        total_spent = sum(spent.values())

        while True:
            fitting = [item for item in subset if item["price"] <= budget - total_spent + 1e-9]
            if not fitting:
                break

            def incremental_score(item):
                trial = dict(spent)
                trial[item["ticker"]] += item["price"]
                deviation = sum(
                    abs(trial.get(candidate["ticker"], 0) - desired[candidate["ticker"]])
                    for candidate in candidates
                )
                leftover = budget - total_spent - item["price"]
                return deviation + 0.20 * leftover, item["price"], item["ticker"]

            chosen = min(fitting, key=incremental_score)
            quantities[chosen["ticker"]] += 1
            spent[chosen["ticker"]] += chosen["price"]
            total_spent += chosen["price"]

        deviation = sum(
            abs(spent.get(candidate["ticker"], 0) - desired[candidate["ticker"]])
            for candidate in candidates
        )
        leftover = budget - total_spent
        score = deviation + 0.20 * leftover
        candidate_result = (score, leftover, subset, quantities, spent)
        if best is None or candidate_result[:2] < best[:2]:
            best = candidate_result

    if best is None:
        return []

    _, _, subset, quantities, spent = best
    action_priority = {"PICK (new buy)": 0, "TOP-UP": 1}
    plan = [
        {
            "Stock": item["ticker"],
            "Source action": item["action"],
            "Whole shares": quantities[item["ticker"]],
            "Price ₹": item["price"],
            "Estimated cost ₹": spent[item["ticker"]],
            "Target increase ₹": item["target_increase"],
        }
        for item in subset
    ]
    plan.sort(
        key=lambda item: (
            action_priority[item["Source action"]],
            -item["Target increase ₹"],
            -item["Price ₹"],
            item["Stock"],
        )
    )
    return plan


def _render_rebalance_action_card(row):
    """Render one action as a readable, wrapped decision card."""
    action = row["Action"]
    icon, label = {
        "PICK (new buy)": ("🌱", "PICK · Start investing"),
        "DROP (exit)": ("🚪", "DROP · Exit"),
        "TOP-UP": ("⬆️", "TOP-UP · Increase"),
        "TRIM": ("⬇️", "TRIM · Reduce"),
        "HOLD": ("✅", "HOLD · No change"),
    }[action]

    with st.container(border=True):
        st.markdown(f"#### {icon} {row['Stock']} — {label}")
        previous_col, current_col, change_col, shares_col = st.columns(4)
        previous_col.metric("Last month", row["Last month ₹"])
        current_col.metric("This month", row["This month ₹"])
        change_col.metric("Target change", row["Target change ₹"])
        shares_col.metric("Change in shares", row["Change in shares"])
        st.markdown("**Why Vriddhi recommends this action**")
        if action == "DROP (exit)":
            st.error(row["Rationale"])
        elif action == "TRIM":
            st.warning(row["Rationale"])
        elif action == "HOLD":
            st.info(row["Rationale"])
        else:
            st.success(row["Rationale"])


def panel_rebalance(bundle, monthly_investment):
    st.markdown("#### E. Monthly Rebalance")
    render_rebalance_explainer()
    prev = load_previous_bundle(bundle["horizon_years"])

    if prev is None:
        st.success(
            "**This is your very first portfolio**, so there's nothing to rebalance against yet - just buy "
            "the basket in the Final Portfolio tab. From next month, this tab will show your exact "
            "buy / sell / top-up / hold actions versus the previous month."
        )
        return

    cur_w = {s["ticker"]: s["weight"] for s in bundle["stocks"]}
    prev_w = {s["ticker"]: s["weight"] for s in prev["stocks"]}
    cur_price = {
        s["ticker"]: float(s.get("current_price") or 0)
        for s in bundle["stocks"]
    }
    prev_price = {
        s["ticker"]: float(s.get("current_price") or 0)
        for s in prev["stocks"]
    }

    THRESH = 0.01  # ignore weight wiggles under 1%
    PRIORITY = {"PICK (new buy)": 0, "DROP (exit)": 1, "TOP-UP": 2, "TRIM": 3, "HOLD": 4}
    rows, counts = [], {"PICK (new buy)": 0, "DROP (exit)": 0, "TOP-UP": 0, "TRIM": 0, "HOLD": 0}
    turnover = 0.0
    for t in set(cur_w) | set(prev_w):
        pw, cw = prev_w.get(t, 0.0), cur_w.get(t, 0.0)
        turnover += abs(cw - pw)
        prev_amt, cur_amt = pw * monthly_investment, cw * monthly_investment
        d_amt = cur_amt - prev_amt
        action_price = cur_price.get(t, 0) or prev_price.get(t, 0)
        fractional_share_change = d_amt / action_price if action_price > 0 else None
        if pw == 0 and cw > 0:
            action = "PICK (new buy)"
        elif cw == 0 and pw > 0:
            action = "DROP (exit)"
        elif cw - pw > THRESH:
            action = "TOP-UP"
        elif pw - cw > THRESH:
            action = "TRIM"
        else:
            action = "HOLD"
        counts[action] += 1
        rows.append({
            "Stock": t,
            "Action": action,
            "Last month \u20b9": f"\u20b9{prev_amt:,.0f}",
            "This month \u20b9": f"\u20b9{cur_amt:,.0f}",
            "Target change \u20b9": f"{'+' if d_amt >= 0 else '-'}\u20b9{abs(d_amt):,.0f}",
            "Change in shares": (
                f"{fractional_share_change:+.1f}"
                if fractional_share_change is not None
                else "n/a"
            ),
            "_Action price": action_price,
            "_Target increase": max(d_amt, 0),
            "Reason at a glance": _rebalance_short_reason(action),
            "Rationale": _rebalance_rationale(t, action, pw, cw, bundle, prev),
        })
    rows.sort(key=lambda r: (PRIORITY[r["Action"]], r["Stock"]))

    pdate = prev.get("data_through", "last month")
    cdate = bundle.get("data_through", "this month")
    st.caption(f"Comparing **{pdate}** (last month) -> **{cdate}** (this month), "
               f"at \u20b9{monthly_investment:,}/month.")

    summary = (f"**This month's plan:** {counts['PICK (new buy)']} new pick(s), "
               f"{counts['DROP (exit)']} exit(s), {counts['TOP-UP']} top-up(s), "
               f"{counts['TRIM']} trim(s), {counts['HOLD']} hold(s). "
               f"Turnover ~{turnover/2*100:.0f}% of the portfolio.")
    if counts["PICK (new buy)"] or counts["DROP (exit)"]:
        st.warning(summary)
    else:
        st.success(summary + " A quiet month - mostly just keep buying as usual.")

    st.markdown("#### At-a-glance comparison")
    compact_columns = [
        "Stock",
        "Action",
        "Last month ₹",
        "This month ₹",
        "Target change ₹",
        "Change in shares",
        "Reason at a glance",
    ]
    st.dataframe(
        pd.DataFrame(rows)[compact_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "Reason at a glance": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "How to act: **PICK** - start buying it with this month's money. **DROP** - stop buying and sell "
        "what you hold of it. **TOP-UP / TRIM** - buy a bit more / less than before. **HOLD** - no change, "
        "keep buying the same. **Target change ₹** is the change in intended monthly allocation. "
        "**Change in shares** translates that rupee change into a one-decimal share equivalent at "
        "the applicable stock price. Fractional values are planning guidance; actual execution "
        "depends on whether your broker supports fractional shares."
    )

    st.markdown("#### Top 3 recommended actions")
    released_cash = sum(
        max(
            float(prev_w.get(row["Stock"], 0) - cur_w.get(row["Stock"], 0))
            * monthly_investment,
            0,
        )
        for row in rows
        if row["Action"] in {"DROP (exit)", "TRIM"}
    )
    suggested_surplus = int(round(released_cash))
    surplus = st.number_input(
        "Available surplus this month (₹)",
        min_value=0,
        max_value=1_000_000,
        value=suggested_surplus,
        step=500,
        help=(
            "Defaults to the target cash released by this month's DROP and TRIM actions. "
            "Override it if you have additional cash available."
        ),
    )
    st.caption(
        "Vriddhi considers only this month's **PICK** and **TOP-UP** names, uses complete "
        "shares, and selects up to three buys that stay close to the intended rebalance "
        "split while minimizing idle cash."
    )
    surplus_plan = _build_surplus_purchase_plan(rows, surplus)
    if surplus <= 0:
        st.info("Enter a surplus above ₹0 to generate a whole-share purchase plan.")
    elif not surplus_plan:
        st.warning(
            "The available surplus cannot buy one whole share of any current PICK or "
            "TOP-UP candidate. Keep it as cash or combine it with a future contribution."
        )
    else:
        deployed = sum(item["Estimated cost ₹"] for item in surplus_plan)
        remainder = float(surplus) - deployed
        d1, d2, d3 = st.columns(3)
        d1.metric("Surplus available", f"₹{float(surplus):,.0f}")
        d2.metric("Deployed into whole shares", f"₹{deployed:,.0f}")
        d3.metric("Cash remaining", f"₹{remainder:,.0f}")

        for rank, item in enumerate(surplus_plan, start=1):
            shares = int(item["Whole shares"])
            source = "new PICK" if item["Source action"] == "PICK (new buy)" else "TOP-UP"
            st.success(
                f"**#{rank}: Buy {shares} whole share{'s' if shares != 1 else ''} of "
                f"{item['Stock']}** — approximately ₹{item['Estimated cost ₹']:,.0f} "
                f"at ₹{item['Price ₹']:,.2f}/share. This executes part of the current "
                f"**{source}** allocation."
            )
        st.caption(
            "Best-fit execution aid, not a guarantee of exact exhaustion: whole-share prices "
            "can leave a small remainder, and live market prices may differ at execution."
        )
    st.markdown("#### Actions to take now")
    actionable_rows = [row for row in rows if row["Action"] != "HOLD"]
    if actionable_rows:
        st.caption(
            f"Focus on these **{len(actionable_rows)} changes** first. Each card shows "
            "the money movement, the evidence behind it, and how it improves the whole portfolio."
        )
        for row in actionable_rows:
            _render_rebalance_action_card(row)
    else:
        st.success("No changes are needed this month. Keep investing as before.")

    hold_rows = [row for row in rows if row["Action"] == "HOLD"]
    if hold_rows:
        with st.expander(
            f"✅ No change needed — review {len(hold_rows)} HOLD rationale(s)",
            expanded=False,
        ):
            for row in hold_rows:
                _render_rebalance_action_card(row)

    st.caption(
        "How to read the evidence: the cards translate stored health metrics and optimizer "
        "inputs into plain English. Final weights use the full return/covariance matrix, "
        "so no action is attributed to one number in isolation."
    )

    st.markdown("---")
    st.markdown("#### The two portfolios being compared")
    st.caption(
        f"Both snapshots use your selected **₹{monthly_investment:,}/month** contribution. "
        "Weights and prices come from each recorded release, so you can see the complete "
        "before-and-after portfolios behind the action table above."
    )

    st.markdown(f"##### Current portfolio — {display_release_date(cdate)}")
    current_allocation = scale_allocations(bundle, monthly_investment)
    st.dataframe(
        portfolio_table_view(current_allocation),
        width="stretch",
        hide_index=True,
    )

    st.markdown(f"##### Previous portfolio — {display_release_date(pdate)}")
    previous_allocation = scale_allocations(prev, monthly_investment)
    st.dataframe(
        portfolio_table_view(previous_allocation),
        width="stretch",
        hide_index=True,
    )


def _passing_horizons(exclude=None):
    """Return the list of horizons whose bundle currently clears every gate."""
    passing = []
    for h in (3, 4, 5):
        if h == exclude:
            continue
        b = load_portfolio_bundle(h)
        if b and b.get("verdict", {}).get("recommended"):
            passing.append(h)
    return passing


def build_optimal_view_figure(bundle):
    """Efficient-frontier chart (cloud + frontier + candidates + CAL + optimum vs
    recommended vs benchmark). Returns a matplotlib fig, or None if the bundle has
    no optimal_view. Shared by the Optimal View tab and the Summary re-display."""
    ov = bundle.get("optimal_view")
    if not ov:
        return None

    cloud = np.array(ov.get("cloud", []), dtype=float)   # columns: [vol, ret, sharpe]
    frontier = ov.get("frontier", [])
    cands = ov.get("candidates", [])
    opt = ov.get("optimum", {})
    rec = ov.get("recommended", {})
    bench = ov.get("benchmark")
    rf = float(ov.get("risk_free", 0.065))

    fig, ax = plt.subplots(figsize=(10, 6))

    if cloud.size:
        sc = ax.scatter(cloud[:, 0] * 100, cloud[:, 1] * 100, c=cloud[:, 2],
                        cmap="viridis", s=8, alpha=0.45)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Sharpe ratio (higher = better risk-adjusted)")

    if frontier:
        ax.plot([p["vol"] * 100 for p in frontier], [p["ret"] * 100 for p in frontier],
                color="#333333", linewidth=1.6, label="Efficient frontier")

    # Split candidates into held (in the portfolio) vs screened-out, coloured
    # differently so the 18 -> 12 funnel is visible at a glance.
    held_set = {s["ticker"] for s in bundle.get("stocks", [])}
    held_pts = [s for s in cands if s["ticker"] in held_set]
    out_pts = [s for s in cands if s["ticker"] not in held_set]
    if held_pts:
        ax.scatter([s["vol"] * 100 for s in held_pts], [s["ret"] * 100 for s in held_pts],
                   color="orange", edgecolor="black", s=48, zorder=5,
                   label="Stocks Vriddhi holds")
    if out_pts:
        ax.scatter([s["vol"] * 100 for s in out_pts], [s["ret"] * 100 for s in out_pts],
                   color="#9aa0a6", edgecolor="#555555", s=40, zorder=4,
                   label="Screened out (not held)")
    for s in cands:
        is_held = s["ticker"] in held_set
        ax.annotate(s["ticker"], (s["vol"] * 100, s["ret"] * 100),
                    textcoords="offset points", xytext=(3, 3), fontsize=5.5,
                    color="#333333" if is_held else "#9aa0a6")

    # X extent for the Capital Allocation Line.
    xs = list(cloud[:, 0] * 100) if cloud.size else []
    xs += [s["vol"] * 100 for s in cands] + [opt.get("vol", 0) * 100]
    xmax = (max(xs) if xs else 40) * 1.05

    if opt:
        ox, oy = opt["vol"] * 100, opt["ret"] * 100
        if ox > 0:
            slope = (oy - rf * 100) / ox
            ax.plot([0, xmax], [rf * 100, rf * 100 + slope * xmax], "r--",
                    linewidth=1.4, label="Capital Allocation Line")
        ax.scatter([ox], [oy], marker="*", s=340, color="red", edgecolor="black",
                   zorder=6, label=f"Pure-math optimum ({opt.get('n_stocks', '?')} stocks)")

    if rec:
        ax.scatter([rec["vol"] * 100], [rec["ret"] * 100], marker="D", s=130,
                   color="#A23B72", edgecolor="black", zorder=6,
                   label=f"Vriddhi recommended ({rec.get('n_stocks', '?')} stocks)")

    if bench:
        ax.scatter([bench["vol"] * 100], [bench["ret"] * 100], marker="s", s=120,
                   facecolor="none", edgecolor="red", linewidth=2, zorder=6)
        ax.annotate(bench.get("label", "Benchmark"),
                    (bench["vol"] * 100, bench["ret"] * 100),
                    textcoords="offset points", xytext=(6, -11), fontsize=9, color="red")

    ax.set_xlabel("Risk - annualized volatility (%)")
    ax.set_ylabel("Annualized return (%)")
    ax.set_title("Portfolio Optimization (Modern Portfolio Theory)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    return fig


def panel_optimal(bundle):
    """Optimal View: the efficient frontier with the pure-math optimum shown
    next to our deliberately-regularized recommended book - the honest answer to
    'is it always 12 stocks, and is this really optimal?'"""
    ov = bundle.get("optimal_view")
    if not ov:
        st.info(
            "Optimal View is not available in this bundle yet. "
            "Re-run `py build_research_db.py` to generate it."
        )
        return

    st.subheader("Where this portfolio sits on the efficient frontier")

    fig = build_optimal_view_figure(bundle)
    if fig is not None:
        st.pyplot(fig)

    opt = ov.get("optimum", {})
    rec = ov.get("recommended", {})

    st.success(
        "**How to read this chart (the simple version):** think of it as a map of every way you "
        "could invest. **Left-to-right is risk** (how bumpy the ride); **bottom-to-top is return** "
        "(how much you make). Every **small dot is one possible mix** of these stocks, and its "
        "**colour is the Sharpe ratio** - how much return you earn *per unit of risk* (brighter = "
        "better quality). The bigger dots are the individual stocks on their own: **orange = names "
        "Vriddhi holds**, **grey = names it screened out**. \n\n"
        "- The **curved line** is the *efficient frontier* - the best return you can get for each "
        "level of risk. Nothing exists above it; that's the limit of what's possible.\n"
        "- The **red dashed line** is the best risk-vs-reward trade-off line; where it just touches "
        "the frontier is the mathematically 'perfect' mix.\n"
        "- The **red star** is that textbook-perfect portfolio. The **purple diamond** is what "
        "**Vriddhi actually recommends**. They sit almost on top of each other - we give up a "
        "whisker of theoretical return for a lot more safety.\n"
        "- The **red square is the Nifty 50** (the market). Notice both our portfolios sit **well "
        "above and to the left of it** - *more return, similar-or-less risk.* That gap is the edge.\n\n"
        "**The quick intuition:** up-and-to-the-left is *healthier* - more reward for less risk. "
        "Notice the big dots (individual stocks) almost all sit to the **right** of our portfolio "
        "markers: that's because **any single stock on its own is bumpier than a sensible blend**. "
        "**Orange = names Vriddhi holds, grey = names it screened out.** The orange names generally "
        "sit in the healthier upper zone, and where a higher-risk name *is* kept, it earns its place "
        "by *combining* well with the others (low correlation quietly lowers the whole portfolio's "
        "risk). The real magic: by blending the orange names, **Vriddhi's portfolio (the purple "
        "diamond) leaps far up-and-left to sit right beside the mathematical optimum (red star)** - "
        "a healthier spot than *any* individual stock can reach alone. That 'North Star' position is "
        "what you're actually buying - not a bet on one name."
    )
    st.caption(
        "Note: this chart's return axis is the MPT mean-return basis the optimizer works in. The "
        "headline recommendation is still anchored to the validated out-of-sample CAGR shown on the "
        "other tabs."
    )

    # ---- Which stocks? The pure-math 7 vs the Vriddhi 12 (the USP) ----
    st.markdown("#### Which stocks does each one actually hold?")
    opt_w = {k: float(v) for k, v in (opt.get("weights") or {}).items()}
    rec_w = {s["ticker"]: float(s["weight"]) for s in bundle.get("stocks", [])}

    # Union, ordered by the recommended weight (then any optimum-only names).
    ordered = sorted(rec_w, key=lambda t: rec_w[t], reverse=True)
    ordered += [t for t in opt_w if t not in rec_w]

    hold_rows = []
    for t in ordered:
        in_opt, in_rec = t in opt_w, t in rec_w
        if in_opt and in_rec:
            role = "Core idea - kept by both"
        elif in_rec:
            role = "Added by Vriddhi for diversification"
        else:
            role = "Optimum-only (Vriddhi dropped it)"
        hold_rows.append({
            "Stock": t,
            "Pure-math optimum (7)": f"{opt_w[t]*100:.1f}%" if in_opt else "\u2014",
            "Vriddhi recommended (12)": f"{rec_w[t]*100:.1f}%" if in_rec else "\u2014",
            "Role": role,
        })
    st.dataframe(pd.DataFrame(hold_rows), width="stretch", hide_index=True)

    # Dynamic plain-English story of the difference.
    opt_names = sorted(opt_w, key=lambda t: opt_w[t], reverse=True)
    shared = [t for t in opt_names if t in rec_w]
    added = [t for t in ordered if t in rec_w and t not in opt_w]
    capped = [t for t in shared if opt_w[t] > rec_w[t] + 0.005]
    top_nm = opt_names[0] if opt_names else "the top name"
    top_wt = opt_w[top_nm] * 100 if opt_names else 0
    bits = [
        f"**Same winning ideas, spread more safely.** The pure-math optimum crams everything into "
        f"**{len(opt_w)} names** and bets big - **{top_nm} alone at {top_wt:.0f}%**.",
    ]
    if shared:
        bits.append(
            f"Vriddhi keeps **{len(shared)} of those exact same names** "
            f"({', '.join(shared)}) - so we're not throwing away the optimizer's best ideas.")
    if capped:
        bits.append(
            f"But we **trim the over-sized bets** (e.g. {', '.join(capped[:3])}) down to the 15% cap, "
            f"so no single stock can sink the plan.")
    if added:
        bits.append(
            f"Then we **add {len(added)} more solid names** ({', '.join(added)}) that the raw "
            f"optimizer ignored, to spread the risk across more of the market.")
    bits.append(
        "The result: the **same engine of returns, with the single-stock risk defused** - which is "
        "exactly why the 12-stock book holds up better on money the model had never seen.")
    st.info(" ".join(bits))

    # Honest optimum-vs-recommended comparison table.
    oc = ov.get("oos_compare", {})
    reg, unc = oc.get("regularized", {}), oc.get("unconstrained", {})
    lb = oc.get("lookback_years", bundle.get("horizon_years"))
    rec_top2 = sum(sorted((s["weight"] for s in bundle.get("stocks", [])),
                          reverse=True)[:2]) * 100

    def _p(x):
        return "n/a" if x is None else f"{x:.1f}%"

    cmp_rows = [
        {"Metric": "Number of stocks",
         "Pure-math optimum": str(opt.get("n_stocks", "n/a")),
         "Vriddhi recommended": str(rec.get("n_stocks", "n/a"))},
        {"Metric": "Top-2 concentration",
         "Pure-math optimum": f"{opt.get('top2_concentration', 0)*100:.0f}%",
         "Vriddhi recommended": f"{rec_top2:.0f}%"},
        {"Metric": "In-sample Sharpe",
         "Pure-math optimum": f"{opt.get('sharpe', 0):.2f}",
         "Vriddhi recommended": f"{rec.get('sharpe', 0):.2f}"},
        {"Metric": f"Validated OOS CAGR ({lb}-yr walk-forward)",
         "Pure-math optimum": _p(unc.get("oos_cagr")),
         "Vriddhi recommended": _p(reg.get("oos_cagr"))},
        {"Metric": "Out-of-sample max drawdown",
         "Pure-math optimum": _p(unc.get("oos_max_drawdown")),
         "Vriddhi recommended": _p(reg.get("oos_max_drawdown"))},
    ]
    st.markdown("#### Pure-math optimum vs. the Vriddhi book")
    st.dataframe(pd.DataFrame(cmp_rows), width="stretch", hide_index=True)

    st.markdown("#### Is it always 12 stocks? (the honest answer)")
    st.info(ov.get("narrative", ""))


def render_panels(bundle, benchmark_df, monthly_investment):
    tabs = st.tabs(["Summary", "Backtest Evidence", "Final Portfolio",
                    "Optimal View", "Risk", "Monthly Rebalance"])
    with tabs[0]:
        panel_summary(bundle, monthly_investment)
    with tabs[1]:
        panel_backtest(bundle, benchmark_df, monthly_investment)
    with tabs[2]:
        panel_portfolio(bundle, monthly_investment)
    with tabs[3]:
        panel_optimal(bundle)
    with tabs[4]:
        panel_risk(bundle)
    with tabs[5]:
        panel_rebalance(bundle, monthly_investment)


def render_dashboard(horizon_years, monthly_investment):
    bundle = load_portfolio_bundle(horizon_years)
    if bundle is None:
        st.error(
            f"No research bundle found for a {horizon_years}-year horizon. "
            f"Run `py build_research_db.py` to generate the research database."
        )
        return
    benchmark_df = load_benchmark_series()

    verdict_banner(bundle)
    st.caption(
        f"Universe: Nifty 50 | Data through {bundle.get('data_through', 'n/a')} | "
        f"Selection: {bundle.get('selection_method', 'PEG screen + Markowitz')}"
    )

    if bundle["verdict"]["recommended"]:
        # HARD GATE: only a passing portfolio is presented as a recommendation.
        render_panels(bundle, benchmark_df, monthly_investment)
        return

    # Failing horizon: do NOT present it as a recommendation. Point to any
    # horizon that does qualify, and keep the full analysis for transparency only.
    others = _passing_horizons(exclude=horizon_years)
    if others:
        nice = " or ".join(f"{h}-year" for h in others)
        st.info(
            f"A credible portfolio **is** available today at a **{nice}** horizon. "
            f"Switch the horizon in the sidebar to see it."
        )
    else:
        st.info(
            "No horizon clears every gate with this month's data. The honest answer is "
            "to wait - we would rather show nothing than recommend a portfolio we do not "
            "trust with new money."
        )

    with st.expander("See the full analysis for this horizon (for transparency - NOT a recommendation)",
                     expanded=False):
        render_panels(bundle, benchmark_df, monthly_investment)


# ===============================
# PAGE
# ===============================
st.title("Vriddhi - Nifty 50 Portfolio Decision Aid")
render_landing_vision()
show_disclaimer()
st.markdown("---")

# ---- Inputs ----
st.sidebar.header("Your Plan")
horizon_years = st.sidebar.selectbox(
    "Investment horizon (years)", options=[1, 2, 3, 4, 5], index=4,
    help="Matches the 12/24/36/48/60-month forecasts in our database. The "
         "walk-forward validation is matched to your chosen horizon.",
)
monthly_investment = st.sidebar.slider(
    "Monthly contribution (INR)", min_value=50000, max_value=100000,
    step=5000, value=100000,
)
st.sidebar.markdown("---")
generate = st.sidebar.button("Generate Investment Plan", type="primary",
                             width="stretch")
st.sidebar.caption(
    "Gates: validated OOS CAGR >= 18% (1-4yr) / 20% (5yr), max drawdown < 25%, "
    "walk-forward Sharpe > 1.0, and a benchmark beat after costs."
)

if generate:
    st.session_state["plan_generated"] = True

if st.session_state.get("plan_generated"):
    render_dashboard(horizon_years, monthly_investment)
else:
    st.info(
        "Set your **investment horizon** and **monthly contribution** in the sidebar, "
        "then click **Generate Investment Plan** to see whether a credible portfolio "
        "is available - and exactly what to buy and monitor each month."
    )

st.markdown("---")
manifest = load_release_manifest() or {}
release = manifest.get("release_id", "legacy research bundle")
source_commit = (manifest.get("source_commit") or "unknown")[:8]
st.caption(
    f"Release: {release} | Source: {source_commit} | "
    "Methodology: yfinance adjusted prices, damped-trend (Holt) time-series forecasts, "
    "CAGR / drawdown / volatility / Sharpe, walk-forward validation, Markowitz optimization, "
    "pass/fail gates, benchmark comparison, and month-over-month rebalancing. Current limitations: "
    "PE/PB are current rather than point-in-time, taxes/slippage are simplified, and the benchmark "
    "series is undergoing a documented total-return methodology review."
)
