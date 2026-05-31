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
SINGLE_STOCK_CAP = 0.20         # 20% single-stock cap (no over-concentration)
MIN_STOCK_WEIGHT = 0.03         # 3% floor so every held name is meaningful

HORIZONS = [3, 4, 5]

# Pass/fail gates (from the investor MVP spec)
GATES = {
    "wf_cagr_3y_min": 18.0,
    "wf_cagr_4y_min": 18.0,
    "wf_cagr_5y_min": 20.0,
    "max_drawdown_max": 25.0,
    "sharpe_min": 1.0,
    "sharpe_strong": 1.2,
}


# ===============================
# TICKER MAPPING
# ===============================
def to_yahoo_symbol(ticker: str) -> str:
    """Map an NSE ticker from our CSV to a yfinance symbol."""
    return f"{ticker.strip()}.NS"


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


def walk_forward(prices_df, tickers, lookback_years):
    """Genuine walk-forward over the trailing `lookback_years`.

    Splits the lookback into an initial training window then steps forward,
    re-optimizing weights on data seen so far and measuring the next unseen
    window. Returns in-sample vs out-of-sample annualized stats plus the
    stitched out-of-sample equity curve for charting.
    """
    avail = [t for t in tickers if t in prices_df.columns]
    window_prices = slice_last_years(prices_df[avail], lookback_years).dropna(how="all")
    rets = window_prices.pct_change().dropna()
    if len(rets) < TRADING_DAYS * 2:
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

        weights = optimize_max_sharpe(train, cap=SINGLE_STOCK_CAP, floor=MIN_STOCK_WEIGHT)
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
# EXPLANATIONS
# ===============================
def build_explanation(row, stock_metric, weight, forecast_col):
    """Three-reason explainability per the spec (fundamental/forecast/risk)
    plus the stock's contribution to expected portfolio return."""
    pe = row.get("PE_Ratio", float("nan"))
    pb = row.get("PB_Ratio", float("nan"))
    peg = row.get("PEG_Ratio", float("nan"))
    forecast = row.get(forecast_col, float("nan"))
    hist_cagr = row.get("Avg_Historical_CAGR", float("nan"))

    vol = stock_metric.get("volatility") if stock_metric else None
    dd = stock_metric.get("max_drawdown") if stock_metric else None
    realized = stock_metric.get("cagr_5y") if stock_metric else None

    fundamental = (
        f"PEG {peg:.2f} (PE {pe:.1f} vs hist CAGR {hist_cagr:.1f}%), PB {pb:.1f} "
        f"- growth at a reasonable price."
    )
    forecast_reason = f"Forecast model projects {forecast:.1f}% over the horizon."
    if vol is not None and dd is not None:
        risk_reason = f"Annualized volatility {vol:.1f}%, historical max drawdown {dd:.1f}%."
    else:
        risk_reason = "Risk profile from limited history."
    contribution = (realized if realized is not None else hist_cagr) * weight
    contribution_reason = (
        f"Contributes ~{contribution:.1f}% to expected portfolio return "
        f"(weight {weight*100:.1f}%)."
    )
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
    cagr_threshold = GATES["wf_cagr_5y_min"] if horizon_years == 5 else GATES["wf_cagr_3y_min"]

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
        "gate_results": results,
        "failing_gates": failing,
        "sharpe_strong": bool(wf_sharpe is not None and wf_sharpe >= GATES["sharpe_strong"]),
    }


# ===============================
# PER-HORIZON BUNDLE
# ===============================
def build_horizon_bundle(horizon_years, df, prices_df, stock_metrics, benchmark):
    horizon_months = horizon_years * 12
    forecast_col = vriddhi_core.get_forecast_column(horizon_months)
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
            "explanation": build_explanation(meta, sm, w, forecast_col),
            "has_price_data": t in prices_df.columns,
        })

    # ---- Sector allocation ----
    sector_alloc = {}
    for s in stocks:
        sector_alloc[s["sector"]] = sector_alloc.get(s["sector"], 0.0) + s["weight"]

    bundle = {
        "horizon_years": horizon_years,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_through": prices_df.index[-1].strftime("%Y-%m-%d"),
        "universe_size": int(len(df)),
        "num_stocks": len(stocks),
        "verdict": verdict,
        "portfolio_metrics": port_metrics,
        "scenarios": scenarios,
        "benchmark": {"name": BENCHMARK_NAME, "metrics": bench_metrics, "beat_after_costs": benchmark_beat},
        "walk_forward": wf_by_lookback,
        "stocks": stocks,
        "sector_allocation": sector_alloc,
        "selection_method": rationale.get("selection_method", "PEG-based selection + MPT"),
        "notes": "v1 MVP: forecast layer uses precomputed forecast columns; "
                 "backtest/walk-forward use yfinance adjusted prices.",
    }
    return bundle


# ===============================
# MAIN
# ===============================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    tickers = df["Ticker"].tolist()
    print(f"Universe: {len(tickers)} stocks from {DATA_CSV}")

    prices_df, benchmark = download_prices(tickers)

    print("Computing per-stock backtest metrics...")
    stock_metrics = per_stock_metrics(prices_df)
    print(f"  Metrics computed for {len(stock_metrics)} stocks.")

    # Save benchmark series for the app's walk-forward / comparison chart
    if benchmark is not None:
        bench_norm = benchmark / benchmark.iloc[0]
        bench_out = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in bench_norm.index],
            "Nifty50_Normalized": [round(float(v), 4) for v in bench_norm.values],
        })
        bench_out.to_csv(os.path.join(OUTPUT_DIR, "benchmark.csv"), index=False)
        print(f"  Saved {OUTPUT_DIR}/benchmark.csv")

    for hy in HORIZONS:
        print(f"\nBuilding horizon bundle: {hy}yr ...")
        bundle = build_horizon_bundle(hy, df, prices_df, stock_metrics, benchmark)
        out_path = os.path.join(OUTPUT_DIR, f"portfolio_{hy}y.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, default=str)
        v = bundle["verdict"]
        print(f"  -> {out_path} | {'RECOMMENDED' if v['recommended'] else 'NOT RECOMMENDED'} "
              f"| {bundle['num_stocks']} stocks "
              f"| WF: {[ (lb, round(w['oos_cagr'],1)) for lb,w in bundle['walk_forward'].items() ]}")

    print("\nDone. Research database built under ./research/")


if __name__ == "__main__":
    main()
