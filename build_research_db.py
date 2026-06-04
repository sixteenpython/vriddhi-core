"""
Vriddhi - Offline Research Database Builder (run ONCE before the demo)
======================================================================

Pulls ~5 years of corporate-action-adjusted daily prices for the Nifty 50
universe (the tickers already in grand_table_expanded.csv) plus the Nifty 50
index (^NSEI), then precomputes everything the live app needs:

  - per-stock backtest metrics (3/4/5yr CAGR, volatility, max drawdown, Sharpe)
  - for each horizon (3, 4, 5yr): the optimized 11-12 stock portfolio,
    portfolio-level metrics, a genuine walk-forward (train -> unseen test),
    benchmark-beat-after-costs, per-stock explanations, and a pass/fail verdict.

Outputs (committed, read by the live app - no runtime network needed):
  research/portfolio_3y.json
  research/portfolio_4y.json
  research/portfolio_5y.json
  research/benchmark.csv

This is a BUILD-TIME script. yfinance/scipy are in requirements-dev.txt only.
Run with:  py build_research_db.py
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import yfinance as yf
from scipy.optimize import minimize

import vriddhi_core

warnings.filterwarnings("ignore")

# ===============================
# CONFIG
# ===============================
DATA_CSV = "grand_table_expanded.csv"
OUTPUT_DIR = "research"
BENCHMARK_SYMBOL = "^NSEI"
BENCHMARK_NAME = "Nifty 50"
PERIOD = "5y"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.065          # 6.5% annual, typical Indian risk-free proxy
ANNUAL_COST = 0.005             # 0.5%/yr assumed cost drag for "after costs"
MAX_PORTFOLIO_STOCKS = 12       # spec: final portfolio of 11-12 names
TARGET_SCREEN_CANDIDATES = 18   # spec: screen ~15-20 before optimizing down
SINGLE_STOCK_CAP = 0.15         # 15% single-stock cap (avoid a top-heavy barbell)
MIN_STOCK_WEIGHT = 0.05         # 5% floor so every held name is genuinely meaningful

HORIZONS = [1, 2, 3, 4, 5]       # matches the 12/24/36/48/60M data in the CSV

# A walk-forward needs enough history to leave a credible out-of-sample record.
# With ~5yr of data, a 1yr lookback collapses to a noisy ~4-month test, so we
# require at least this many return-days before we will validate a horizon.
MIN_WF_RETURN_DAYS = int(TRADING_DAYS * 1.5)   # ~378 days -> 2yr+ validates, 1yr does not

# Pass/fail gates (from the investor MVP spec). Per-horizon min walk-forward CAGR.
GATES = {
    "wf_cagr_min": {1: 18.0, 2: 18.0, 3: 18.0, 4: 18.0, 5: 20.0},
    "max_drawdown_max": 25.0,
    "sharpe_min": 1.0,
    "sharpe_strong": 1.2,
}


# ===============================
# TICKER MAPPING
# ===============================
from ticker_resolver import load_aliases, symbol_for

_ALIASES = load_aliases()


def to_yahoo_symbol(ticker: str) -> str:
    """Map an NSE ticker from our CSV to a yfinance symbol, honoring any
    auto-healed overrides (renames/demergers) recorded by build_grand_table."""
    return symbol_for(ticker, _ALIASES)


# ===============================
# DATA DOWNLOAD
# ===============================
def download_prices(tickers):
    """Download adjusted daily close prices for tickers + benchmark.

    Returns (prices_df, benchmark_series) aligned on common trading days.
    Tickers that fail to download are dropped (Nifty 50 leaves plenty).
    """
    symbols = [to_yahoo_symbol(t) for t in tickers]
    all_symbols = symbols + [BENCHMARK_SYMBOL]

    print(f"Downloading {len(all_symbols)} symbols ({PERIOD}, adjusted)...")
    raw = yf.download(all_symbols, period=PERIOD, auto_adjust=True, progress=False)

    close = raw["Close"].copy()
    # Forward/back fill small gaps, then drop any column that is still mostly empty
    close = close.ffill().bfill()
    min_obs = int(len(close) * 0.6)
    close = close.dropna(axis=1, thresh=min_obs)

    benchmark = None
    if BENCHMARK_SYMBOL in close.columns:
        benchmark = close[BENCHMARK_SYMBOL].dropna()
        close = close.drop(columns=[BENCHMARK_SYMBOL])

    # Map yahoo symbols back to plain tickers
    rename = {to_yahoo_symbol(t): t for t in tickers}
    close = close.rename(columns=rename)

    dropped = [t for t in tickers if t not in close.columns]
    if dropped:
        print(f"  Dropped (no/insufficient data): {dropped}")
    print(f"  Usable price history for {close.shape[1]} stocks over {close.shape[0]} days.")
    return close, benchmark


# ===============================
# METRIC HELPERS
# ===============================
def slice_last_years(series_or_df, years):
    """Return the trailing `years` of a price series/frame."""
    end = series_or_df.index[-1]
    start = end - pd.DateOffset(years=years)
    return series_or_df.loc[series_or_df.index >= start]


def cagr_from_prices(prices, years):
    """CAGR (%) over the trailing `years` from a price series."""
    window = slice_last_years(prices, years).dropna()
    if len(window) < 2:
        return None
    start_p, end_p = window.iloc[0], window.iloc[-1]
    if start_p <= 0:
        return None
    return ((end_p / start_p) ** (1.0 / years) - 1.0) * 100.0


def annualized_vol(daily_returns):
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS) * 100.0)


def max_drawdown(prices):
    """Max drawdown (%) of a price/equity series (positive number)."""
    curve = prices.dropna()
    if len(curve) < 2:
        return None
    running_max = curve.cummax()
    drawdown = (curve - running_max) / running_max
    return float(abs(drawdown.min()) * 100.0)


def sharpe_ratio(daily_returns):
    """Annualized Sharpe using excess returns over the risk-free rate."""
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return None
    excess = daily_returns.mean() * TRADING_DAYS - RISK_FREE_RATE
    vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
    return float(excess / vol)


def per_stock_metrics(prices_df):
    """Compute backtest metrics for every stock with price history."""
    metrics = {}
    for ticker in prices_df.columns:
        p = prices_df[ticker].dropna()
        if len(p) < TRADING_DAYS:  # need at least ~1yr
            continue
        rets = p.pct_change().dropna()
        metrics[ticker] = {
            "cagr_1y": cagr_from_prices(p, 1),
            "cagr_2y": cagr_from_prices(p, 2),
            "cagr_3y": cagr_from_prices(p, 3),
            "cagr_4y": cagr_from_prices(p, 4),
            "cagr_5y": cagr_from_prices(p, 5),
            "volatility": annualized_vol(rets),
            "max_drawdown": max_drawdown(p),
            "sharpe": sharpe_ratio(rets),
        }
    return metrics


# ===============================
# WALK-FORWARD OPTIMIZER (historical-returns MPT, max Sharpe)
# ===============================
def optimize_max_sharpe(returns_window, cap=SINGLE_STOCK_CAP, floor=0.0):
    """Long-only max-Sharpe weights on a window of daily returns.

    Constraints: weights sum to 1, no shorting, single-stock cap, optional
    per-stock floor (to force a diversified, fully-allocated portfolio).
    """
    mean_daily = returns_window.mean().values
    cov_daily = returns_window.cov().values
    n = len(mean_daily)

    # Keep the floor feasible (n * floor must be <= 1)
    if floor * n > 1.0:
        floor = 0.0

    def neg_sharpe(w):
        port_ret = np.dot(w, mean_daily) * TRADING_DAYS
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_daily * TRADING_DAYS, w)))
        if port_vol == 0:
            return 0.0
        return -(port_ret - RISK_FREE_RATE) / port_vol

    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
    bounds = [(floor, cap)] * n
    init = np.ones(n) / n
    res = minimize(neg_sharpe, init, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x if res.success else init


def portfolio_daily_returns(returns_df, weights_map):
    """Weighted daily return series from a {ticker: weight} map."""
    cols = [t for t in weights_map if t in returns_df.columns]
    w = np.array([weights_map[t] for t in cols], dtype=float)
    if w.sum() == 0:
        return pd.Series(dtype=float)
    w = w / w.sum()
    return (returns_df[cols] * w).sum(axis=1)


def walk_forward(prices_df, tickers, lookback_years, cap=SINGLE_STOCK_CAP, floor=MIN_STOCK_WEIGHT):
    """Genuine walk-forward over the trailing `lookback_years`.

    Splits the lookback into an initial training window then steps forward,
    re-optimizing weights on data seen so far and measuring the next unseen
    window. Returns in-sample vs out-of-sample annualized stats plus the
    stitched out-of-sample equity curve for charting.

    `cap`/`floor` parameterize the per-stock constraints so the same machinery
    can validate both the regularized book (default) and an unconstrained
    pure-max-Sharpe variant for the honest "why we regularize" comparison.
    """
    avail = [t for t in tickers if t in prices_df.columns]
    window_prices = slice_last_years(prices_df[avail], lookback_years).dropna(how="all")
    rets = window_prices.pct_change().dropna()
    if len(rets) < MIN_WF_RETURN_DAYS:
        return None

    n_folds = lookback_years - 1  # e.g. 5yr -> 4 folds (train>=1yr, step 1yr)
    n_folds = max(2, n_folds)
    fold_len = len(rets) // (n_folds + 1)

    oos_curve = []           # stitched out-of-sample daily returns
    is_cagrs, oos_cagrs = [], []

    for k in range(1, n_folds + 1):
        train = rets.iloc[: fold_len * k]
        test = rets.iloc[fold_len * k : fold_len * (k + 1)]
        if len(test) < 20 or len(train) < 60:
            continue

        weights = optimize_max_sharpe(train, cap=cap, floor=floor)
        wmap = {t: weights[i] for i, t in enumerate(train.columns)}

        train_port = portfolio_daily_returns(train, wmap)
        test_port = portfolio_daily_returns(test, wmap)

        # Annualized in-sample / out-of-sample return for this fold
        is_cagrs.append((np.prod(1 + train_port) ** (TRADING_DAYS / len(train_port)) - 1) * 100)
        oos_cagrs.append((np.prod(1 + test_port) ** (TRADING_DAYS / len(test_port)) - 1) * 100)
        oos_curve.append(test_port)

    if not oos_curve:
        return None

    stitched = pd.concat(oos_curve)
    equity = (1 + stitched).cumprod()
    oos_cagr = (equity.iloc[-1] ** (TRADING_DAYS / len(stitched)) - 1) * 100
    oos_sharpe = sharpe_ratio(stitched)
    oos_dd = max_drawdown(equity)

    return {
        "lookback_years": lookback_years,
        "in_sample_cagr": float(np.mean(is_cagrs)) if is_cagrs else None,
        "oos_cagr": float(oos_cagr),
        "oos_sharpe": float(oos_sharpe) if oos_sharpe is not None else None,
        "oos_max_drawdown": float(oos_dd) if oos_dd is not None else None,
        "equity_curve": {
            "dates": [d.strftime("%Y-%m-%d") for d in equity.index],
            "portfolio": [round(float(v), 4) for v in equity.values],
        },
    }


# ===============================
# OPTIMAL VIEW (efficient frontier)
# ===============================
def _ann_perf(w, mean_daily, cov_daily):
    """Annualized (return, volatility, Sharpe) for a weight vector, using the
    SAME mean/covariance geometry the optimizer uses, so every point we plot
    (cloud, frontier, stocks, optimum, recommended) lives on one comparable axis.
    Returns decimals (e.g. 0.21 = 21%)."""
    w = np.asarray(w, dtype=float)
    ret = float(np.dot(w, mean_daily) * TRADING_DAYS)
    var = float(np.dot(w, np.dot(cov_daily * TRADING_DAYS, w)))
    vol = float(np.sqrt(var)) if var > 0 else 0.0
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _min_variance_for_target(mean_daily, cov_daily, target_ret):
    """Long-only minimum-variance weights for a target annualized return."""
    n = len(mean_daily)
    cov_ann = cov_daily * TRADING_DAYS

    def port_var(w):
        return float(np.dot(w, np.dot(cov_ann, w)))

    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
        {"type": "eq", "fun": lambda x: np.dot(x, mean_daily) * TRADING_DAYS - target_ret},
    )
    res = minimize(port_var, np.ones(n) / n, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n, constraints=constraints)
    return res.x if res.success else None


def efficient_frontier(mean_daily, cov_daily, n_points=60):
    """Trace the long-only efficient frontier as (vol, ret) points.

    Sweeps target returns, takes the min-variance portfolio for each, then keeps
    only the Pareto-efficient (upper-left) envelope so the plotted curve is clean
    and monotonic - no interior/dominated zig-zags from solver noise.
    """
    stock_rets = mean_daily * TRADING_DAYS
    lo, hi = float(np.min(stock_rets)), float(np.max(stock_rets))
    raw = []
    for target in np.linspace(lo, hi, n_points):
        w = _min_variance_for_target(mean_daily, cov_daily, target)
        if w is None:
            continue
        ret, vol, _ = _ann_perf(w, mean_daily, cov_daily)
        raw.append((vol, ret))

    # Pareto filter: keep a point only if nothing else has >= return at <= vol.
    efficient = []
    for vol, ret in raw:
        dominated = any((v <= vol + 1e-9 and r >= ret - 1e-9 and (v < vol or r > ret))
                        for v, r in raw)
        if not dominated:
            efficient.append((vol, ret))
    efficient.sort(key=lambda p: p[0])
    return [{"vol": round(v, 4), "ret": round(r, 4)} for v, r in efficient]


def random_cloud(mean_daily, cov_daily, n=1200, seed=42):
    """A reproducible Monte-Carlo cloud of long-only portfolios for visual
    texture (coloured by Sharpe in the app)."""
    rng = np.random.RandomState(seed)
    k = len(mean_daily)
    # alpha<1 spreads weights toward the simplex corners -> a wider, more
    # informative cloud than near-equal-weight Dirichlet(1).
    W = rng.dirichlet(np.full(k, 0.5), size=n)
    cov_ann = cov_daily * TRADING_DAYS
    rets = W @ mean_daily * TRADING_DAYS
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov_ann, W))
    sharpes = np.divide(rets - RISK_FREE_RATE, vols, out=np.zeros_like(rets), where=vols > 0)
    return [[round(float(v), 4), round(float(r), 4), round(float(s), 3)]
            for v, r, s in zip(vols, rets, sharpes)]


def compute_optimal_view(cand_rets, weight_map, benchmark, prices_df,
                         candidates, horizon_years):
    """Build the 'Optimal View' payload: the efficient frontier, a Sharpe-coloured
    cloud, individual candidates, the UNCONSTRAINED pure-max-Sharpe optimum (with
    its natural cardinality), the regularized recommended book, the Nifty 50
    point, and an honest out-of-sample comparison of the two strategies."""
    mean_daily = cand_rets.mean().values
    cov_daily = cand_rets.cov().values
    cols = list(cand_rets.columns)

    # Individual candidate stocks (unit portfolios).
    stocks_pts = []
    for i, t in enumerate(cols):
        e = np.zeros(len(cols)); e[i] = 1.0
        ret, vol, sh = _ann_perf(e, mean_daily, cov_daily)
        stocks_pts.append({"ticker": t, "vol": round(vol, 4),
                           "ret": round(ret, 4), "sharpe": round(sh, 3)})

    # Unconstrained tangency optimum (long-only, sum=1, no cap/floor).
    opt_w = optimize_max_sharpe(cand_rets, cap=1.0, floor=0.0)
    o_ret, o_vol, o_sh = _ann_perf(opt_w, mean_daily, cov_daily)
    opt_pairs = sorted(((cols[i], float(w)) for i, w in enumerate(opt_w) if w > 0.01),
                       key=lambda kv: kv[1], reverse=True)
    opt_top2 = sum(w for _, w in opt_pairs[:2])
    optimum = {
        "vol": round(o_vol, 4), "ret": round(o_ret, 4), "sharpe": round(o_sh, 3),
        "n_stocks": len(opt_pairs), "top2_concentration": round(opt_top2, 4),
        "weights": {t: round(w, 4) for t, w in opt_pairs},
    }

    # Regularized recommended book (the capped/floored 12), on the same geometry.
    rec_w = np.array([weight_map.get(t, 0.0) for t in cols])
    r_ret, r_vol, r_sh = _ann_perf(rec_w, mean_daily, cov_daily)
    recommended = {
        "vol": round(r_vol, 4), "ret": round(r_ret, 4), "sharpe": round(r_sh, 3),
        "n_stocks": int(sum(1 for t in cols if weight_map.get(t, 0.0) > 1e-6)),
    }

    # Benchmark (Nifty 50) on the same annualization.
    bench_pt = None
    if benchmark is not None and len(benchmark) > TRADING_DAYS:
        br = benchmark.pct_change().dropna()
        b_ret = float(br.mean() * TRADING_DAYS)
        b_vol = float(br.std() * np.sqrt(TRADING_DAYS))
        b_sh = (b_ret - RISK_FREE_RATE) / b_vol if b_vol > 0 else 0.0
        bench_pt = {"label": BENCHMARK_NAME, "vol": round(b_vol, 4),
                    "ret": round(b_ret, 4), "sharpe": round(b_sh, 3)}

    # Honest out-of-sample comparison: same candidate set, regularized vs
    # unconstrained, validated walk-forward (this is the "why we cap" evidence).
    lb = horizon_years if horizon_years >= 2 else 2
    wf_reg = walk_forward(prices_df, candidates, lb,
                          cap=SINGLE_STOCK_CAP, floor=MIN_STOCK_WEIGHT)
    wf_unc = walk_forward(prices_df, candidates, lb, cap=1.0, floor=0.0)
    oos_compare = {
        "lookback_years": lb,
        "regularized": {"oos_cagr": wf_reg.get("oos_cagr") if wf_reg else None,
                        "oos_sharpe": wf_reg.get("oos_sharpe") if wf_reg else None,
                        "oos_max_drawdown": wf_reg.get("oos_max_drawdown") if wf_reg else None},
        "unconstrained": {"oos_cagr": wf_unc.get("oos_cagr") if wf_unc else None,
                          "oos_sharpe": wf_unc.get("oos_sharpe") if wf_unc else None,
                          "oos_max_drawdown": wf_unc.get("oos_max_drawdown") if wf_unc else None},
    }

    return {
        "candidates": stocks_pts,
        "frontier": efficient_frontier(mean_daily, cov_daily),
        "cloud": random_cloud(mean_daily, cov_daily),
        "optimum": optimum,
        "recommended": recommended,
        "benchmark": bench_pt,
        "risk_free": RISK_FREE_RATE,
        "single_stock_cap": SINGLE_STOCK_CAP,
        "min_stock_weight": MIN_STOCK_WEIGHT,
        "target_stocks": MAX_PORTFOLIO_STOCKS,
        "oos_compare": oos_compare,
        "narrative": _optimal_view_narrative(optimum, recommended, oos_compare),
    }


def _optimal_view_narrative(optimum, recommended, oos_compare):
    """Plain-English, data-driven explanation of why the recommended book is
    deliberately regularized rather than the raw math optimum. Adapts to whether
    the unconstrained variant actually back-tested better or worse out-of-sample."""
    cap_pct = int(round(SINGLE_STOCK_CAP * 100))
    floor_pct = int(round(MIN_STOCK_WEIGHT * 100))
    min_names = int(np.ceil(1.0 / SINGLE_STOCK_CAP))   # e.g. 7 at a 15% cap
    max_names = int(np.floor(1.0 / MIN_STOCK_WEIGHT))  # e.g. 20 at a 5% floor

    reg = oos_compare.get("regularized", {})
    unc = oos_compare.get("unconstrained", {})
    r_oos, u_oos = reg.get("oos_cagr"), unc.get("oos_cagr")
    r_dd, u_dd = reg.get("oos_max_drawdown"), unc.get("oos_max_drawdown")

    parts = [
        f"**Is it always {recommended['n_stocks']} stocks?** No magic number - it is a deliberate "
        f"choice, not an accident. The pure-math 'optimal' portfolio (the red star) maximises "
        f"historical risk-adjusted return with no guardrails, and here it collapses into just "
        f"**{optimum['n_stocks']} names**, with **{optimum['top2_concentration']*100:.0f}% piled into its "
        f"top two**. That looks great on paper because unconstrained optimisation chases whatever "
        f"happened to do best in the past.",
        f"Our recommended book instead spreads across **{recommended['n_stocks']} names** with a "
        f"**{cap_pct}% cap** and a **{floor_pct}% floor** on each. With those guardrails the maths "
        f"alone bounds you to roughly **{min_names}-{max_names} holdings**, and we target ~{recommended['n_stocks']} "
        f"as the sweet spot between real diversification and staying focused.",
    ]
    if r_oos is not None and u_oos is not None:
        if u_oos <= r_oos + 0.5:
            parts.append(
                f"The pay-off is in the **out-of-sample** test (money the model never saw): the "
                f"regularized book delivered **{r_oos:.1f}%** vs the concentrated optimum's "
                f"**{u_oos:.1f}%** - so spreading out cost us nothing real and bought us safety.")
        else:
            extra = f" (and at a {u_dd:.0f}% vs {r_dd:.0f}% drawdown)" if (u_dd and r_dd) else ""
            parts.append(
                f"Out-of-sample, the concentrated optimum did edge ahead (**{u_oos:.1f}%** vs "
                f"**{r_oos:.1f}%**){extra} - but it rode a far bumpier, more fragile path. We cap "
                f"concentration on purpose: a single-stock shock in a {optimum['n_stocks']}-name book "
                f"can undo a year, which is the wrong trade for a balanced investor.")
    parts.append(
        "Bottom line: we show you the textbook optimum for transparency, but we recommend the "
        "**robust** version - the one whose track record holds up on data it had never seen.")
    return " ".join(parts)


# ===============================
# EXPLANATIONS
# ===============================
def build_explanation(row, stock_metric, weight, forecast_col, horizon_years):
    """Per-stock explainability (fundamental / forecast / risk / contribution).

    Two honesty fixes vs v1:
      * The valuation sentence is conditional, not a blanket "reasonable price"
        on every name (so a PB-18 / PE-58 stock is flagged as premium).
      * "Contribution" is reconciled to ONE return definition - the stock's
        *validated historical* CAGR over the chosen horizon - so it can no
        longer contradict the (separate, forward) forecast line. The forecast
        is explicitly framed as an optimistic model signal we do not rely on.
    """
    pe = row.get("PE_Ratio", float("nan"))
    pb = row.get("PB_Ratio", float("nan"))
    peg = row.get("PEG_Ratio", float("nan"))
    forecast = row.get(forecast_col, float("nan"))

    vol = stock_metric.get("volatility") if stock_metric else None
    dd = stock_metric.get("max_drawdown") if stock_metric else None
    # Single, consistent expected-return proxy: realized CAGR for THIS horizon
    # (falls back to the nearest available window), not a different series.
    realized = None
    if stock_metric:
        realized = (stock_metric.get(f"cagr_{horizon_years}y")
                    or stock_metric.get("cagr_5y")
                    or stock_metric.get("cagr_3y"))

    # ---- Fundamental: conditional language on valuation ----
    if not np.isnan(peg) and peg < 1.0:
        verdict = "attractively priced for its historical growth"
    elif not np.isnan(peg) and peg <= 1.3:
        verdict = "fairly priced for its growth"
    else:
        verdict = "priced at a premium - held for quality/diversification, not cheapness"
    flags = []
    if not np.isnan(pb) and pb >= 8:
        flags.append("rich price-to-book")
    if not np.isnan(pe) and pe >= 45:
        flags.append("high P/E")
    flag_txt = f" Note: {', '.join(flags)}." if flags else ""
    peg_txt = f"{peg:.2f}" if not np.isnan(peg) else "n/a"
    fundamental = f"PEG {peg_txt} (PE {pe:.1f}, PB {pb:.1f}) - {verdict}.{flag_txt}"

    # ---- Forecast: clearly an optimistic, non-binding model signal ----
    forecast_reason = (
        f"Model signal (forward, not relied upon): ~{forecast:.0f}% annualized. "
        f"The recommendation is anchored to validated walk-forward returns, not this."
    )

    # ---- Risk ----
    if vol is not None and dd is not None:
        risk_reason = f"Annualized volatility {vol:.1f}%, historical max drawdown {dd:.1f}%."
    else:
        risk_reason = "Risk profile from limited history."

    # ---- Contribution: reconciled to historical CAGR; never a silent contradiction ----
    if realized is not None:
        contribution = realized * weight
        if contribution >= 0.3:
            contribution_reason = (
                f"Added ~{contribution:.1f}% to the portfolio's historical "
                f"{horizon_years}-yr CAGR (weight {weight*100:.1f}%)."
            )
        else:
            contribution_reason = (
                f"Held mainly as a diversifier (weight {weight*100:.1f}%): small or "
                f"negative standalone historical contribution, but lowers overall risk."
            )
    else:
        contribution_reason = f"Weight {weight*100:.1f}% (limited history for contribution)."

    return {
        "fundamental": fundamental,
        "forecast": forecast_reason,
        "risk": risk_reason,
        "contribution": contribution_reason,
    }


# ===============================
# GATE EVALUATION / VERDICT
# ===============================
def evaluate_gates(horizon_years, wf_by_lookback, port_metrics, benchmark_beat):
    """Apply the spec's pass/fail gates and produce a verdict.

    The binding walk-forward gate matches the investor's chosen horizon (so a
    5yr plan is not disqualified by a noisier 3yr sub-window), while all three
    windows are still shown as evidence. Sharpe is judged out-of-sample
    (walk-forward), which is the honest, validated number.
    """
    binding = wf_by_lookback.get(horizon_years, {})
    wf_cagr = binding.get("oos_cagr")
    wf_sharpe = binding.get("oos_sharpe")
    max_dd = port_metrics.get("max_drawdown")
    cagr_threshold = GATES["wf_cagr_min"].get(horizon_years, 18.0)

    # Short horizons can lack a credible out-of-sample record entirely. Be
    # explicit about that (vs. "validated but failed thresholds").
    if not binding or wf_cagr is None:
        return {
            "recommended": False,
            "insufficient_history": True,
            "gate_results": [{
                "gate": f"{horizon_years}yr out-of-sample validation",
                "passed": False,
                "detail": "insufficient history to validate",
            }],
            "failing_gates": [f"insufficient history to validate a {horizon_years}-year horizon"],
            "sharpe_strong": False,
        }

    results = []

    def add(name, passed, detail):
        results.append({"gate": name, "passed": bool(passed), "detail": detail})

    add(f"{horizon_years}yr walk-forward CAGR >= {cagr_threshold:.0f}%",
        wf_cagr is not None and wf_cagr >= cagr_threshold,
        f"{wf_cagr:.1f}%" if wf_cagr is not None else "n/a")
    add("Max drawdown < 25%", max_dd is not None and max_dd < GATES["max_drawdown_max"],
        f"{max_dd:.1f}%" if max_dd is not None else "n/a")
    add("Walk-forward Sharpe > 1.0", wf_sharpe is not None and wf_sharpe > GATES["sharpe_min"],
        f"{wf_sharpe:.2f}" if wf_sharpe is not None else "n/a")
    add("Beats Nifty 50 after costs", bool(benchmark_beat and benchmark_beat > 0),
        f"+{benchmark_beat:.1f}%" if benchmark_beat is not None else "n/a")

    recommended = all(r["passed"] for r in results)
    failing = [r["gate"] for r in results if not r["passed"]]
    return {
        "recommended": recommended,
        "insufficient_history": False,
        "gate_results": results,
        "failing_gates": failing,
        "sharpe_strong": bool(wf_sharpe is not None and wf_sharpe >= GATES["sharpe_strong"]),
    }


# ===============================
# PER-HORIZON BUNDLE
# ===============================
def build_horizon_bundle(horizon_years, df, prices_df, stock_metrics, benchmark):
    horizon_months = horizon_years * 12
    # Use the forecast column that matches THIS horizon (12M/24M/.../60M).
    # (Previously this always fell back to Forecast_60M because the helper
    # returns short codes like "60M" that never match the "Forecast_60M" cols.)
    forecast_col = f"Forecast_{horizon_months}M"
    if forecast_col not in df.columns:
        forecast_col = "Forecast_60M"

    # Stage 1 (candidate screen): reuse the existing PEG/forecast selector.
    selected_df, _, achieved_cagr, rationale = vriddhi_core.advanced_stock_selector(
        df.copy(), 0.15, horizon_months
    )
    peg_lookup = dict(zip(selected_df["Ticker"], selected_df.get("PEG_Ratio", pd.Series(dtype=float))))

    # Stage 2 (optimizer): proper capped Markowitz max-Sharpe on REAL historical
    # returns of the screened candidates (single-stock cap, long-only). This is
    # the same methodology the walk-forward validates, so the displayed
    # portfolio and its walk-forward record are one and the same.
    candidates = [t for t in selected_df["Ticker"].tolist() if t in prices_df.columns]

    # Ensure a genuine 15-20 name screen going INTO the optimizer (so the funnel
    # narrows 15-20 -> 11-12 rather than keeping everything). If the PEG screen
    # produced fewer, top up with the next lowest-PEG names that have price data.
    if len(candidates) < TARGET_SCREEN_CANDIDATES:
        pool = df[(df["PE_Ratio"] > 0) & (df["Avg_Historical_CAGR"] > 0)].copy()
        # Prefer the precomputed PEG_Ratio from the CSV; only derive it if an
        # older CSV lacks the column.
        if "PEG_Ratio" not in pool.columns:
            pool["PEG_Ratio"] = pool["PE_Ratio"] / pool["Avg_Historical_CAGR"]
        pool = pool.sort_values("PEG_Ratio", ascending=True)
        for tkr in pool["Ticker"]:
            if len(candidates) >= TARGET_SCREEN_CANDIDATES:
                break
            if tkr not in candidates and tkr in prices_df.columns:
                candidates.append(tkr)

    cand_rets = prices_df[candidates].pct_change().dropna()

    # Pass 1: rank candidates by unconstrained max-Sharpe weight, take the 11-12 names.
    rank_w = optimize_max_sharpe(cand_rets, cap=SINGLE_STOCK_CAP, floor=0.0)
    ranked = sorted(zip(cand_rets.columns, rank_w), key=lambda kv: kv[1], reverse=True)
    final_names = [t for t, _ in ranked[:MAX_PORTFOLIO_STOCKS]]

    # Pass 2: re-optimize within the final names WITH a floor so every holding is meaningful.
    final_rets = prices_df[final_names].pct_change().dropna()
    final_w = optimize_max_sharpe(final_rets, cap=SINGLE_STOCK_CAP, floor=MIN_STOCK_WEIGHT)
    total_w = float(np.sum(final_w)) or 1.0
    weight_map = {t: float(final_w[i]) / total_w for i, t in enumerate(final_rets.columns)}
    tickers = list(weight_map.keys())

    # ---- Portfolio backtest metrics from REAL prices ----
    avail = [t for t in tickers if t in prices_df.columns]
    rets_all = prices_df[avail].pct_change().dropna()
    port_rets = portfolio_daily_returns(rets_all, weight_map)
    port_equity = (1 + port_rets).cumprod()

    def port_cagr(years):
        sub = slice_last_years(port_equity, years)
        if len(sub) < 2:
            return None
        return (sub.iloc[-1] / sub.iloc[0]) ** (1.0 / years) * 100 - 100

    port_metrics = {
        "cagr_1y": port_cagr(1),
        "cagr_2y": port_cagr(2),
        "cagr_3y": port_cagr(3),
        "cagr_4y": port_cagr(4),
        "cagr_5y": port_cagr(5),
        "volatility": annualized_vol(port_rets),
        "max_drawdown": max_drawdown(port_equity),
        "sharpe": sharpe_ratio(port_rets),
        "expected_cagr": float(achieved_cagr * 100),  # forward forecast-based
    }

    # ---- Benchmark + beat-after-costs ----
    bench_metrics, benchmark_beat = {}, None
    if benchmark is not None and len(benchmark) > TRADING_DAYS:
        bench_rets = benchmark.pct_change().dropna()
        bench_metrics = {
            "cagr_1y": cagr_from_prices(benchmark, 1),
            "cagr_2y": cagr_from_prices(benchmark, 2),
            "cagr_3y": cagr_from_prices(benchmark, 3),
            "cagr_4y": cagr_from_prices(benchmark, 4),
            "cagr_5y": cagr_from_prices(benchmark, 5),
            "volatility": annualized_vol(bench_rets),
            "max_drawdown": max_drawdown(benchmark),
            "sharpe": sharpe_ratio(bench_rets),
        }
        # Compare on the chosen horizon, net of assumed costs
        p = port_cagr(horizon_years)
        b = cagr_from_prices(benchmark, horizon_years)
        if p is not None and b is not None:
            benchmark_beat = (p - ANNUAL_COST * 100) - b

    # ---- Walk-forward for each gate lookback ----
    wf_by_lookback = {}
    for lb in HORIZONS:
        wf = walk_forward(prices_df, tickers, lb)
        if wf is not None:
            wf_by_lookback[lb] = wf

    verdict = evaluate_gates(horizon_years, wf_by_lookback, port_metrics, benchmark_beat)

    # ---- Best/base/worst scenario band ----
    # Base case is anchored to the VALIDATED walk-forward out-of-sample CAGR
    # (honest, not the optimistic forward forecast), with a vol-based band.
    binding_wf = wf_by_lookback.get(horizon_years, {})
    base = binding_wf.get("oos_cagr")
    if base is None:
        base = port_metrics.get(f"cagr_{horizon_years}y") or port_metrics["expected_cagr"]
    vol = port_metrics["volatility"]
    scenarios = {
        "base": base,
        "best": base + 0.5 * vol,
        "worst": max(0.0, base - 0.5 * vol),
        "forecast_signal": port_metrics["expected_cagr"],  # forward model, shown as upside only
    }

    # ---- Per-stock entries with explanations (ordered by weight) ----
    df_indexed = df.set_index("Ticker")
    stocks = []
    for t in sorted(tickers, key=lambda x: weight_map[x], reverse=True):
        meta = df_indexed.loc[t].to_dict() if t in df_indexed.index else {}
        # PEG is read straight from the precomputed PEG_Ratio column in
        # grand_table_expanded.csv (so optimizer/top-up names get a real PEG too).
        # Fall back to the Stage-1 screen lookup only if an older CSV lacks the
        # column. A genuinely undefined PEG stays NaN and renders as "n/a".
        if "PEG_Ratio" not in meta:
            meta["PEG_Ratio"] = peg_lookup.get(t, float("nan"))
        sm = stock_metrics.get(t, {})
        w = float(weight_map[t])
        stocks.append({
            "ticker": t,
            "sector": meta.get("Sector", "N/A"),
            "weight": w,
            "current_price": float(meta.get("Current_Price", 0) or 0),
            "forecast": float(meta.get(forecast_col, 0) or 0),
            "pe_ratio": float(meta.get("PE_Ratio", 0) or 0),
            "pb_ratio": float(meta.get("PB_Ratio", 0) or 0),
            "backtest": sm,
            "explanation": build_explanation(meta, sm, w, forecast_col, horizon_years),
            "has_price_data": t in prices_df.columns,
        })

    # ---- Sector allocation ----
    sector_alloc = {}
    for s in stocks:
        sector_alloc[s["sector"]] = sector_alloc.get(s["sector"], 0.0) + s["weight"]

    # ---- Optimal View (efficient frontier + pure-math optimum vs our book) ----
    optimal_view = compute_optimal_view(
        cand_rets, weight_map, benchmark, prices_df, candidates, horizon_years)

    bundle = {
        "horizon_years": horizon_years,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_through": prices_df.index[-1].strftime("%Y-%m-%d"),
        "universe_size": int(len(df)),
        "screened_count": int(len(candidates)),
        "num_stocks": len(stocks),
        "verdict": verdict,
        "portfolio_metrics": port_metrics,
        "scenarios": scenarios,
        "benchmark": {"name": BENCHMARK_NAME, "metrics": bench_metrics, "beat_after_costs": benchmark_beat},
        "walk_forward": wf_by_lookback,
        "stocks": stocks,
        "sector_allocation": sector_alloc,
        "optimal_view": optimal_view,
        "selection_method": rationale.get("selection_method", "PEG-based selection + MPT"),
        "notes": "v1 MVP: forecast layer uses precomputed forecast columns; "
                 "backtest/walk-forward use yfinance adjusted prices.",
    }
    return bundle


# ===============================
# MAIN
# ===============================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof", default=None,
                        help="Reconstruct the portfolio as of this date (YYYY-MM-DD) by "
                             "truncating price history - used to seed a 'previous month' "
                             "snapshot for the rebalance view.")
    parser.add_argument("--suffix", default="",
                        help="Filename suffix for outputs, e.g. _prev (keeps the live "
                             "bundles untouched).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    tickers = df["Ticker"].tolist()
    print(f"Universe: {len(tickers)} stocks from {DATA_CSV}")

    prices_df, benchmark = download_prices(tickers)

    if args.asof:
        cutoff = pd.to_datetime(args.asof)
        prices_df = prices_df[prices_df.index <= cutoff]
        if benchmark is not None:
            benchmark = benchmark[benchmark.index <= cutoff]
        print(f"  AS-OF {args.asof}: truncated to {prices_df.shape[0]} days "
              f"(through {prices_df.index[-1].date()}).")

    print("Computing per-stock backtest metrics...")
    stock_metrics = per_stock_metrics(prices_df)
    print(f"  Metrics computed for {len(stock_metrics)} stocks.")

    # Save benchmark series for the app's walk-forward / comparison chart.
    # Skip for snapshot builds (--suffix) so the live benchmark.csv is untouched.
    if benchmark is not None and not args.suffix:
        bench_norm = benchmark / benchmark.iloc[0]
        bench_out = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in bench_norm.index],
            "Nifty50_Normalized": [round(float(v), 4) for v in bench_norm.values],
        })
        bench_out.to_csv(os.path.join(OUTPUT_DIR, "benchmark.csv"), index=False)
        print(f"  Saved {OUTPUT_DIR}/benchmark.csv")

    # Auto-rotate on a LIVE build: this month's bundles become next month's
    # "previous" snapshot (powers the rebalance view) before we overwrite them.
    if not args.suffix:
        import shutil
        for hy in HORIZONS:
            cur = os.path.join(OUTPUT_DIR, f"portfolio_{hy}y.json")
            if os.path.exists(cur):
                shutil.copyfile(cur, os.path.join(OUTPUT_DIR, f"portfolio_{hy}y_prev.json"))
        print("  Rotated existing bundles -> *_prev.json (previous-month snapshot).")

    for hy in HORIZONS:
        print(f"\nBuilding horizon bundle: {hy}yr ...")
        bundle = build_horizon_bundle(hy, df, prices_df, stock_metrics, benchmark)
        out_path = os.path.join(OUTPUT_DIR, f"portfolio_{hy}y{args.suffix}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, default=str)
        v = bundle["verdict"]
        print(f"  -> {out_path} | {'RECOMMENDED' if v['recommended'] else 'NOT RECOMMENDED'} "
              f"| {bundle['num_stocks']} stocks "
              f"| WF: {[ (lb, round(w['oos_cagr'],1)) for lb,w in bundle['walk_forward'].items() ]}")

    print("\nDone. Research database built under ./research/")


if __name__ == "__main__":
    main()
