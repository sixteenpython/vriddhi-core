"""
Vriddhi - Knowledge-Asset Builder (grand_table_expanded.csv)
============================================================

Regenerates the monthly "knowledge asset" - grand_table_expanded.csv - directly
from market data, so the pipeline no longer depends on an external forecasting
pipeline. It produces, per Nifty 50 stock:

  - Current_Price, Avg_Historical_CAGR, Risk_Adjusted_Return   (from prices)
  - Forecast_12M..60M  (annualized) and Expected_Returns_12M..60M (cumulative)
  - PE_Ratio, PB_Ratio  (from yfinance fundamentals)

Forecasting technique: damped-trend (Holt) exponential smoothing on MONTHLY
log-prices via statsmodels - a classic, transparent time-series method (no LSTM).
Damping flattens the trend at longer horizons so 5-year forecasts stay realistic
instead of exploding. A guardrail then shrinks each annualized forecast toward a
long-run market return and caps it, so we never emit an implausible number.

Honest note: multi-year forecasting of single stocks is inherently low-skill.
This is a transparent, rules-based projection - the app deliberately does NOT
rely on it for recommendations (those anchor to the walk-forward validation);
it powers the PEG screen and the displayed "model signal".

Ticker + Sector are preserved from the existing CSV (curated labels); every
numeric column is refreshed.

Build-time only (yfinance, statsmodels in requirements-dev.txt).
Run:  py build_grand_table.py            # writes grand_table_expanded.csv
      py build_grand_table.py --out x.csv  # dry-run to a staging file
Then: py build_research_db.py
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ticker_resolver import (
    load_aliases, save_aliases, symbol_for, harvest_name,
    resolve_broken, write_health_report,
)

warnings.filterwarnings("ignore")

SOURCE_CSV = "grand_table_expanded.csv"   # Ticker + Sector (+ fallback PE/PB)
PERIOD = "5y"
TRADING_DAYS = 252
HORIZON_MONTHS = [12, 24, 36, 48, 60]

# Realism guardrail for annualized forecasts (%).
MARKET_CAGR = 12.0     # long-run Nifty-ish anchor to shrink toward
SHRINK = 0.6           # weight on the model signal vs. the market anchor
FORECAST_CAP = 30.0    # never emit more than this annualized
FORECAST_FLOOR = -10.0

COLUMNS = [
    "Overall_Rank", "Ticker", "Sector", "Current_Price",
    "Expected_Returns_12M", "Expected_Returns_24M", "Expected_Returns_36M",
    "Expected_Returns_48M", "Expected_Returns_60M",
    "Forecast_12M", "Forecast_24M", "Forecast_36M", "Forecast_48M", "Forecast_60M",
    "Avg_Historical_CAGR", "Risk_Adjusted_Return", "PE_Ratio", "PB_Ratio", "PEG_Ratio",
]


def to_yahoo(ticker):
    return f"{ticker.strip()}.NS"


def trailing_cagr(prices, years):
    end = prices.index[-1]
    window = prices.loc[prices.index >= end - pd.DateOffset(years=years)].dropna()
    if len(window) < 2 or window.iloc[0] <= 0:
        return None
    return ((window.iloc[-1] / window.iloc[0]) ** (1.0 / years) - 1.0) * 100.0


def guardrail(ann):
    """Shrink an annualized forecast toward the market anchor, then cap/floor."""
    if ann is None or not np.isfinite(ann):
        return MARKET_CAGR
    shrunk = MARKET_CAGR + SHRINK * (ann - MARKET_CAGR)
    return float(np.clip(shrunk, FORECAST_FLOOR, FORECAST_CAP))


def loglinear_annualized(monthly_prices):
    """Fallback: annualized drift from a log-linear fit of monthly prices."""
    y = np.log(monthly_prices.values)
    if len(y) < 6:
        return MARKET_CAGR
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]           # mean monthly log-return
    return (np.exp(slope * 12.0) - 1.0) * 100.0


def damped_holt_forecast(series, current_price):
    """Annualized forecast per horizon via damped-trend ETS on monthly log-prices,
    with a log-linear fallback. Returns {months: annualized_cagr_%}."""
    monthly = series.resample("ME").last().dropna()
    fc_log = None
    if len(monthly) >= 24:
        try:
            model = ExponentialSmoothing(
                np.log(monthly.values), trend="add", damped_trend=True,
                initialization_method="estimated",
            )
            fc_log = model.fit().forecast(max(HORIZON_MONTHS))
        except Exception:
            fc_log = None

    fallback_ann = loglinear_annualized(monthly)
    out = {}
    for h in HORIZON_MONTHS:
        if fc_log is not None and len(fc_log) >= h and current_price > 0:
            price_fc = float(np.exp(fc_log[h - 1]))
            ann = ((price_fc / current_price) ** (12.0 / h) - 1.0) * 100.0
        else:
            ann = fallback_ann
        out[h] = guardrail(ann)
    return out


def get_fundamentals(symbol, fallback_pe, fallback_pb):
    """PE, PB and company longName from yfinance fundamentals, falling back to the
    existing CSV. The name is harvested so the resolver can find this stock's
    successor if its symbol ever changes."""
    pe, pb, name = fallback_pe, fallback_pb, None
    try:
        info = yf.Ticker(symbol).info
        if info.get("trailingPE") and info["trailingPE"] > 0:
            pe = round(float(info["trailingPE"]), 2)
        if info.get("priceToBook") and info["priceToBook"] > 0:
            pb = round(float(info["priceToBook"]), 2)
        name = info.get("longName") or info.get("shortName")
    except Exception:
        pass
    return pe, pb, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=SOURCE_CSV,
                        help="Output CSV path (use a staging name for a dry run).")
    args = parser.parse_args()

    src = pd.read_csv(SOURCE_CSV)
    src_by_ticker = src.set_index("Ticker")
    tickers = src["Ticker"].tolist()
    print(f"Refreshing {len(tickers)} stocks from {SOURCE_CSV} -> {args.out}")

    # Resolve each ticker to a Yahoo symbol (honors prior auto-healed overrides).
    aliases = load_aliases()
    resolved = {t: symbol_for(t, aliases) for t in tickers}
    symbols = list(dict.fromkeys(resolved.values()))
    print(f"Downloading {len(symbols)} symbols ({PERIOD}, adjusted)...")
    raw = yf.download(symbols, period=PERIOD, auto_adjust=True, progress=False)
    close = raw["Close"].rename(columns={sym: t for t, sym in resolved.items()})

    # Self-heal any ticker whose symbol no longer returns data (rename/demerger).
    for t in tickers:
        if t in close.columns and close[t].dropna().shape[0] >= TRADING_DAYS:
            continue
        new_sym, status = resolve_broken(t, aliases, min_days=TRADING_DAYS, period=PERIOD)
        if new_sym:
            h = yf.download(new_sym, period=PERIOD, auto_adjust=True, progress=False)
            series = h["Close"].dropna() if h is not None and not h.empty else None
            if series is not None and hasattr(series, "columns"):
                series = series.iloc[:, 0].dropna()
            if series is not None and len(series) >= TRADING_DAYS:
                close[t] = series
                resolved[t] = new_sym
                print(f"  AUTO-HEALED {t}: {symbol_for(t, {}):>14s} -> {new_sym} "
                      f"({len(series)} days)")

    rows = []
    skipped = []
    health = []
    for t in tickers:
        prev = src_by_ticker.loc[t]
        if t not in close.columns or close[t].dropna().shape[0] < TRADING_DAYS:
            # No usable price data even after resolution -> keep prior row (stale).
            skipped.append(t)
            rows.append(prev.to_dict() | {"Ticker": t})
            health.append({"ticker": t, "symbol": resolved.get(t), "status": "stale", "days": 0})
            continue

        prices = close[t].dropna()
        current = float(prices.iloc[-1])
        cagrs = [c for c in (trailing_cagr(prices, y) for y in (1, 2, 3, 4, 5)) if c is not None]
        avg_cagr = float(np.mean(cagrs)) if cagrs else MARKET_CAGR
        daily = prices.pct_change().dropna()
        vol = float(daily.std() * np.sqrt(TRADING_DAYS) * 100.0)
        rar = avg_cagr / (1.0 + vol / 100.0)

        fc = damped_holt_forecast(prices, current)
        exp_ret = {h: ((1.0 + fc[h] / 100.0) ** (h / 12.0) - 1.0) * 100.0 for h in HORIZON_MONTHS}

        pe, pb, name = get_fundamentals(resolved[t], prev.get("PE_Ratio"), prev.get("PB_Ratio"))
        harvest_name(t, name, aliases)
        health.append({
            "ticker": t, "symbol": resolved[t], "days": int(len(prices)),
            "status": "renamed" if resolved[t] != f"{t}.NS" else "ok",
        })

        rows.append({
            "Ticker": t,
            "Sector": prev["Sector"],
            "Current_Price": round(current, 2),
            **{f"Expected_Returns_{h}M": round(exp_ret[h], 2) for h in HORIZON_MONTHS},
            **{f"Forecast_{h}M": round(fc[h], 2) for h in HORIZON_MONTHS},
            "Avg_Historical_CAGR": round(avg_cagr, 2),
            "Risk_Adjusted_Return": round(rar, 2),
            "PE_Ratio": pe,
            "PB_Ratio": pb,
        })

    out_df = pd.DataFrame(rows)

    # Precompute PEG once, as a first-class column in the knowledge asset, so the
    # backend reads it instead of recomputing on the fly. Same definition the
    # screen uses: PE divided by historical CAGR ("price you pay per unit of the
    # company's past growth"). Undefined when PE<=0 or CAGR<=0 -> left blank
    # (NaN), which the app renders as "n/a". Computed for active and stale rows
    # alike since both carry PE_Ratio and Avg_Historical_CAGR.
    pe_col = pd.to_numeric(out_df["PE_Ratio"], errors="coerce")
    cagr_col = pd.to_numeric(out_df["Avg_Historical_CAGR"], errors="coerce")
    peg_col = pe_col / cagr_col
    peg_col[(pe_col <= 0) | (cagr_col <= 0)] = np.nan
    out_df["PEG_Ratio"] = peg_col.round(2)

    out_df = out_df.sort_values("Forecast_60M", ascending=False).reset_index(drop=True)
    out_df["Overall_Rank"] = out_df.index + 1
    out_df = out_df[COLUMNS]
    out_df.to_csv(args.out, index=False)

    # Persist self-healed symbol overrides + harvested names, and a health report.
    save_aliases(aliases)
    report = write_health_report(health)
    renamed = [r for r in health if r["status"] == "renamed"]
    if renamed:
        print("  AUTO-HEALED (symbol changed): "
              + ", ".join(f"{r['ticker']}->{r['symbol']}" for r in renamed))
    if skipped:
        print(f"  STALE (no data even after resolution, kept previous values): {skipped}")
    print(f"  Universe health: {report['summary']} (see research/universe_health.json)")
    print(f"  Forecast_60M range: {out_df['Forecast_60M'].min():.1f}% .. "
          f"{out_df['Forecast_60M'].max():.1f}%  (mean {out_df['Forecast_60M'].mean():.1f}%)")
    print(f"  PE coverage: {(out_df['PE_Ratio'] > 0).sum()}/{len(out_df)} | "
          f"Avg_Historical_CAGR mean {out_df['Avg_Historical_CAGR'].mean():.1f}%")
    print(f"  PEG coverage: {out_df['PEG_Ratio'].notna().sum()}/{len(out_df)} "
          f"(blank where PE<=0 or CAGR<=0)")
    print(f"Done. Wrote {args.out}")


if __name__ == "__main__":
    main()
