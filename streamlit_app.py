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
            f"This 11-12 stock portfolio cleared every credibility gate: "
            f"~{pct(base)} walk-forward CAGR, beats Nifty 50 by {pct(beat)} after costs, "
            f"with risk inside our limits. Would you trust it with new monthly money? "
            f"The evidence says yes."
        )
    else:
        reasons = ", ".join(v["failing_gates"]) if v["failing_gates"] else "one or more gates"
        st.error(
            f"### NOT RECOMMENDED for a {h}-year horizon\n"
            f"This basket failed: **{reasons}**. We only recommend portfolios that pass "
            f"every gate, so we are not putting our name on this one for {h} years. "
            f"The full analysis is shown below for transparency."
        )


def panel_summary(bundle):
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
    pm = bundle["portfolio_metrics"]
    bench = bundle["benchmark"]["metrics"]
    wf_all = bundle.get("walk_forward", {})

    rows = []
    for yr in (3, 4, 5):
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

    with st.expander("Why these stocks? (per-stock rationale)", expanded=False):
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


def render_panels(bundle, benchmark_df, monthly_investment):
    tabs = st.tabs(["Summary", "Backtest Evidence", "Final Portfolio", "Risk"])
    with tabs[0]:
        panel_summary(bundle)
    with tabs[1]:
        panel_backtest(bundle, benchmark_df)
    with tabs[2]:
        panel_portfolio(bundle, monthly_investment)
    with tabs[3]:
        panel_risk(bundle)


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
    "Investment horizon (years)", options=[3, 4, 5], index=2,
    help="The walk-forward gate is matched to your chosen horizon.",
)
monthly_investment = st.sidebar.slider(
    "Monthly contribution (INR)", min_value=50000, max_value=100000,
    step=5000, value=50000,
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Gates: walk-forward CAGR >= 18% (3-4yr) / 20% (5yr), max drawdown < 25%, "
    "walk-forward Sharpe > 1.0, and a benchmark beat after costs."
)

render_dashboard(horizon_years, monthly_investment)

st.markdown("---")
st.caption(
    "v1 MVP - genuine: adjusted price history, CAGR, drawdown, volatility, Sharpe, "
    "walk-forward, Markowitz optimization, pass/fail gates, benchmark beat. Simplified "
    "for v1 (finishing next): ML/time-series forecast ensemble (currently uses precomputed "
    "forecast signals), quarterly fundamentals depth, full turnover/cost modeling."
)
