import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

import vriddhi_core
from vriddhi_core import (
    load_portfolio_bundle,
    load_previous_bundle,
    load_benchmark_series,
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
        st.sidebar.image(LOGO_PATH, use_container_width=True)

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


def verdict_banner(bundle):
    v = bundle["verdict"]
    h = bundle["horizon_years"]
    base = bundle["scenarios"]["base"]
    beat = bundle["benchmark"].get("beat_after_costs")
    if v["recommended"]:
        st.success(
            f"### RECOMMENDED for a {h}-year horizon\n"
            f"This {bundle['num_stocks']}-stock portfolio cleared every credibility gate: "
            f"~{pct(base)} walk-forward CAGR, beats Nifty 50 by {pct(beat)} after costs, "
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
        f"So yes - there is **high confidence** in a roughly **{pct(base)} CAGR** *if* you stay the course.\n\n"
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
    c1.metric("Expected CAGR (base)", pct(s["base"]),
              help="Anchored to validated out-of-sample walk-forward returns.")
    c2.metric("Best / Worst case", f"{pct(s['best'])} / {pct(s['worst'])}")
    c3.metric("Max Drawdown", pct(pm.get("max_drawdown")),
              help="Largest peak-to-trough loss over the backtest.")
    c4.metric("Sharpe (walk-forward)", f"{wf.get('oos_sharpe', 0):.2f}" if wf.get("oos_sharpe") is not None else "n/a")

    c5, c6, c7 = st.columns(3)
    c5.metric(f"Portfolio {bundle['horizon_years']}yr CAGR",
              pct(pm.get(f"cagr_{bundle['horizon_years']}y")),
              help="Lump-sum historical annual growth of the basket (one-time investment "
                   "held for the period) - distinct from your monthly-contribution projection.")
    c6.metric("Nifty 50 (same window)",
              pct(bench["metrics"].get(f"cagr_{bundle['horizon_years']}y")))
    c7.metric("Beat after costs", pct(bench.get("beat_after_costs")),
              delta="vs Nifty 50")

    st.caption(
        f"A forward forecast model signals higher upside (~{pct(s.get('forecast_signal'))} "
        f"CAGR), but we treat that as optimistic and **do not rely on it** - the base case "
        f"above is deliberately anchored to validated out-of-sample walk-forward returns."
    )


def panel_backtest(bundle, benchmark_df):
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
            "Portfolio CAGR": pct(pm.get(f"cagr_{yr}y")),
            "Nifty 50 CAGR": pct(bench.get(f"cagr_{yr}y")),
            "In-sample CAGR": pct(wf.get("in_sample_cagr")),
            "Out-of-sample CAGR": pct(wf.get("oos_cagr")),
            "OOS Sharpe": f"{wf.get('oos_sharpe'):.2f}" if wf.get("oos_sharpe") is not None else "n/a",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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


def _inr(x):
    """Compact INR formatter for axes (L = lakh, Cr = crore)."""
    if x >= 1e7:
        return f"\u20b9{x/1e7:.1f}Cr"
    if x >= 1e5:
        return f"\u20b9{x/1e5:.1f}L"
    if x >= 1e3:
        return f"\u20b9{x/1e3:.0f}K"
    return f"\u20b9{x:.0f}"


def build_projection_figure(bundle, monthly_investment, alloc):
    """projection.png-style visual: SIP growth journey + year-wise breakdown +
    monthly stock allocation. Uses the VALIDATED walk-forward base CAGR (not the
    optimistic forecast) so the projection is consistent with the verdict."""
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
             label=f"Portfolio value ({base_cagr:.1f}% validated CAGR)")
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
    a = alloc.sort_values("Monthly Allocation (INR)", ascending=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(a)))
    ax3.barh(a["Ticker"], a["Monthly Allocation (INR)"], color=colors)
    for i, v in enumerate(a["Monthly Allocation (INR)"]):
        ax3.text(v, i, f" {_inr(v)}", va="center", fontsize=8)
    ax3.set_xlabel("Monthly \u20b9"); ax3.set_title("Monthly Stock Allocation", fontweight="bold")
    ax3.margins(x=0.18)
    return fig, invested[-1], projected[-1], gains[-1]


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

    # ---- projection.png-style growth visual (validated CAGR) ----
    fig, total_invested, final_value, total_gains = build_projection_figure(
        bundle, monthly_investment, alloc)
    m1, m2, m3 = st.columns(3)
    m1.metric("Total invested", _inr(total_invested),
              help=f"\u20b9{monthly_investment:,}/month for {bundle['horizon_years']} years.")
    m2.metric("Projected value", _inr(final_value),
              help="Illustrative growth of your monthly contributions at the validated "
                   "walk-forward CAGR (not the optimistic forecast).")
    m3.metric("Projected gains", _inr(total_gains))
    st.pyplot(fig)
    st.caption(
        "Illustrative only. This projects *your monthly contributions* compounding at the "
        f"portfolio's validated {bundle['horizon_years']}-yr walk-forward CAGR "
        f"({pct(bundle['scenarios']['base'])}) - it is not the lump-sum historical return and "
        "not a guarantee."
    )
    st.markdown("---")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"**{bundle['num_stocks']} stocks | \u20b9{monthly_investment:,}/month**")
        disp = alloc.copy()
        disp["Weight"] = disp["Weight"].apply(lambda x: f"{x*100:.1f}%")
        disp["Monthly Allocation (INR)"] = disp["Monthly Allocation (INR)"].apply(lambda x: f"\u20b9{x:,.0f}")
        disp["Current_Price"] = disp["Current_Price"].apply(lambda x: f"\u20b9{x:,.0f}")
        disp = disp[["Ticker", "Sector", "Weight", "Monthly Allocation (INR)", "Current_Price", "Whole_Shares"]]
        disp.columns = ["Stock", "Sector", "Weight", "Monthly \u20b9", "Price", "Whole Shares"]
        st.dataframe(disp, use_container_width=True, hide_index=True)

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

    with st.expander("Why these stocks? (per-stock rationale + how to read the numbers)", expanded=False):
        st.markdown(
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
        st.markdown("---")
        for s in bundle["stocks"]:
            e = s.get("explanation", {})
            st.markdown(
                f"**{s['ticker']}** ({s['sector']}, {s['weight']*100:.1f}%)  \n"
                f"- Fundamental: {e.get('fundamental', '')}  \n"
                f"- Forecast: {e.get('forecast', '')}  \n"
                f"- Risk: {e.get('risk', '')}  \n"
                f"- {e.get('contribution', '')}"
            )


def panel_risk(bundle):
    st.markdown("#### D. Risk View")
    st.info(
        "**In plain English: how badly could this hurt along the way?**\n\n"
        "Returns are only half the story - a good adviser also tells you how rough the ride can get. "
        "Here's how to read the numbers below:\n"
        "- **Largest holding / Top-3 concentration** - *how many eggs are in a few baskets.* The smaller "
        "these are, the less any single company can hurt you (we cap any one stock at 15%).\n"
        "- **Largest sector** - the same idea for industries, so you're not secretly betting on just one theme.\n"
        "- **Annualized volatility** - *how bumpy* the journey is. Higher = bigger swings up **and** down.\n"
        "- **Max drawdown** - the single most important one: *the worst peak-to-trough fall* you'd have had "
        "to live through. If it says 20%, it means at some point \u20b91,00,000 would have shown as ~\u20b980,000 "
        "on your statement before recovering. The honest question is: **could you stay calm and not sell?**"
    )
    pm = bundle["portfolio_metrics"]
    bench = bundle["benchmark"]["metrics"]
    stocks = sorted(bundle["stocks"], key=lambda x: x["weight"], reverse=True)
    sectors = bundle.get("sector_allocation", {})

    top1 = stocks[0]["weight"] if stocks else 0
    top3 = sum(s["weight"] for s in stocks[:3])
    max_sector = max(sectors.items(), key=lambda kv: kv[1]) if sectors else ("n/a", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Largest holding", f"{top1*100:.1f}%")
    c2.metric("Top-3 concentration", f"{top3*100:.1f}%")
    c3.metric("Largest sector", f"{max_sector[0]} {max_sector[1]*100:.0f}%")
    c4.metric("Annualized volatility", pct(pm.get("volatility")))

    st.markdown("**Worst historical period (max drawdown)**")
    d1, d2 = st.columns(2)
    d1.metric("Portfolio max drawdown", pct(pm.get("max_drawdown")))
    d2.metric("Nifty 50 max drawdown", pct(bench.get("max_drawdown")))
    st.caption(
        "Concentration is capped at 15% per stock with a 5% floor, so no single name can "
        "sink the portfolio and every holding is meaningful. Drawdown is held under the 25% gate."
    )
    st.success(
        f"**The takeaway:** notice the portfolio's worst fall ({pct(pm.get('max_drawdown'))}) is compared "
        f"against the Nifty 50's ({pct(bench.get('max_drawdown'))}) - we aim to fall *less* than the market "
        "in bad times while still beating it overall. **Golden rule:** only invest money you won't need for "
        f"the next {bundle['horizon_years']} years. That way a temporary drop is just a number on a screen - "
        "you're never forced to sell at the bottom, and time does the healing."
    )


def panel_rebalance(bundle, monthly_investment):
    st.markdown("#### E. Monthly Rebalance")
    prev = load_previous_bundle(bundle["horizon_years"])

    st.info(
        "**In plain English: what do I actually buy this month?**\n\n"
        "You don't rebuild everything each month - you just nudge. Every month the engine refreshes "
        "and this tab tells you the few changes versus last month: a stock to start buying (**PICK**), one "
        "to stop and sell (**DROP**), a name to buy a little **more** (TOP-UP) or **less** (TRIM), and the "
        "rest you simply **HOLD**. Small changes are good news - low churn means low cost and taxes."
    )

    if prev is None:
        st.success(
            "**This is your very first portfolio**, so there's nothing to rebalance against yet - just buy "
            "the basket in the Final Portfolio tab. From next month, this tab will show your exact "
            "buy / sell / top-up / hold actions versus the previous month."
        )
        return

    cur_w = {s["ticker"]: s["weight"] for s in bundle["stocks"]}
    prev_w = {s["ticker"]: s["weight"] for s in prev["stocks"]}
    price = {s["ticker"]: (s.get("current_price") or 0) for s in bundle["stocks"]}
    for s in prev["stocks"]:
        price.setdefault(s["ticker"], s.get("current_price") or 0)

    THRESH = 0.01  # ignore weight wiggles under 1%
    PRIORITY = {"PICK (new buy)": 0, "DROP (exit)": 1, "TOP-UP": 2, "TRIM": 3, "HOLD": 4}
    rows, counts = [], {"PICK (new buy)": 0, "DROP (exit)": 0, "TOP-UP": 0, "TRIM": 0, "HOLD": 0}
    turnover = 0.0
    for t in set(cur_w) | set(prev_w):
        pw, cw = prev_w.get(t, 0.0), cur_w.get(t, 0.0)
        turnover += abs(cw - pw)
        prev_amt, cur_amt = pw * monthly_investment, cw * monthly_investment
        d_amt = cur_amt - prev_amt
        p = price.get(t, 0) or 0
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
            "Change \u20b9": f"{'+' if d_amt >= 0 else '-'}\u20b9{abs(d_amt):,.0f}",
            "Change (shares)": f"{int(round(d_amt / p)):+d}" if p > 0 else "n/a",
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

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "How to act: **PICK** - start buying it with this month's money. **DROP** - stop buying and sell "
        "what you hold of it. **TOP-UP / TRIM** - buy a bit more / less than before. **HOLD** - no change, "
        "keep buying the same. (Change in shares is per month at your chosen contribution.)"
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

    for s in cands:
        ax.scatter(s["vol"] * 100, s["ret"] * 100, color="orange",
                   edgecolor="black", s=45, zorder=4)
        ax.annotate(s["ticker"], (s["vol"] * 100, s["ret"] * 100),
                    textcoords="offset points", xytext=(4, 3), fontsize=7, color="#333333")

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
    st.pyplot(fig)
    st.caption(
        "Each dot is a portfolio: x = how bumpy the ride (risk), y = annualized return, "
        "colour = Sharpe (risk-adjusted quality). Orange dots are the individual candidate "
        "stocks. This chart uses the MPT mean-return basis the optimizer works in; the "
        "headline recommendation is still anchored to the validated walk-forward CAGR on the "
        "other tabs."
    )

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
        {"Metric": f"Out-of-sample CAGR ({lb}-yr walk-forward)",
         "Pure-math optimum": _p(unc.get("oos_cagr")),
         "Vriddhi recommended": _p(reg.get("oos_cagr"))},
        {"Metric": "Out-of-sample max drawdown",
         "Pure-math optimum": _p(unc.get("oos_max_drawdown")),
         "Vriddhi recommended": _p(reg.get("oos_max_drawdown"))},
    ]
    st.markdown("#### Pure-math optimum vs. the Vriddhi book")
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Is it always 12 stocks? (the honest answer)")
    st.info(ov.get("narrative", ""))


def render_panels(bundle, benchmark_df, monthly_investment):
    tabs = st.tabs(["Summary", "Backtest Evidence", "Final Portfolio",
                    "Optimal View", "Risk", "Monthly Rebalance"])
    with tabs[0]:
        panel_summary(bundle, monthly_investment)
    with tabs[1]:
        panel_backtest(bundle, benchmark_df)
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
st.markdown(
    "**Would you trust this portfolio with new monthly money?** "
    "We screen the Nifty 50, validate with backtest + walk-forward, optimize with "
    "Markowitz, and only recommend portfolios that clear robust return and risk gates."
)
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
                             use_container_width=True)
st.sidebar.caption(
    "Gates: walk-forward CAGR >= 18% (1-4yr) / 20% (5yr), max drawdown < 25%, "
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
st.caption(
    "v1 MVP - genuine: yfinance adjusted prices, damped-trend (Holt) time-series forecasts, "
    "CAGR / drawdown / volatility / Sharpe, walk-forward validation, Markowitz optimization, "
    "pass/fail gates, benchmark beat, and month-over-month rebalancing. Simplified for v1 "
    "(finishing next): point-in-time fundamentals (PE/PB are current, not historical), explicit "
    "transaction-cost & tax modeling, and a richer multi-model forecast ensemble."
)
