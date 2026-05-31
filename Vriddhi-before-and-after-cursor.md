# Vriddhi Alpha Finder — Before & After (One Cursor Session)

**App:** [vriddhi-core-beta.streamlit.app](https://vriddhi-core-beta.streamlit.app/)
**Repo:** [github.com/sixteenpython/vriddhi-core](https://github.com/sixteenpython/vriddhi-core)
**Date:** 31 May 2026 · **Context:** Covering for the lead developer (OOO) to ship a major upgrade ahead of an **investor demo the next morning**.
**Starting material:** the live beta, the GitHub repo, and a few plain-English blurbs describing the MVP vision. No design docs, no historical price data, no forecasting code.

This document captures the full journey — every big and small change — from the base version to the version live on the same link today.

---

## 1. The one-line summary

> We turned a good-looking **stock picker** into a credible, self-validating **investment decision aid** — one that only recommends a portfolio after it survives a backtest and walk-forward gate, explains itself in plain English, shows month-to-month rebalancing, and **refreshes its entire back-end with a single command** that even heals itself when a stock's ticker changes.

---

## 2. Where we started (the base version)

The beta already had a solid foundation, and we were careful to build *on* it, not replace it:

- A **Streamlit app** (`streamlit_app.py`) that read a curated dataset (`grand_table_expanded.csv`) of ~50 Nifty 50 stocks.
- A **core engine** (`vriddhi_core.py`) doing PEG-based stock selection and a Markowitz-style optimisation, plus projection charts.
- A nice **growth projection** visual and a clean layout.

**But for an investor demo, three things were missing:**

1. **No evidence.** It recommended a portfolio, but couldn't answer *"why should I trust this with real money?"* There was no backtest, no walk-forward validation, no benchmark comparison.
2. **No knowledge asset of our own.** The forecasts and fundamentals were inherited from an external pipeline we couldn't access or reproduce — and some forecasts were implausibly high (50–80% annualised).
3. **It spoke "quant," not "human."** The numbers were there, but a common investor couldn't read them.

We also resolved the **starting hygiene issues** first: live **Git merge conflicts** in `vriddhi_core.py`, a stray syntax error, and confirmed the active backend was `grand_table_expanded.csv` (not the older `grand_table.csv`).

---

## 3. The product goal we locked onto

From the developer's blurbs, we distilled a single, testable MVP goal:

> **Build a Nifty 50 portfolio decision aid that screens 15–20 stocks, validates them with backtest and walk-forward testing, optimises a final 11–12 stock portfolio, and shows _only_ portfolios that clear robust return and risk thresholds. Answer one question clearly: "Would I trust this portfolio with new monthly money?"**

And one guiding principle that shaped every decision:

> **The app is exactly as credible as the knowledge asset — no more.** So we invested heavily in making that knowledge asset honest, reproducible, and self-maintaining, and kept the app a transparent, deterministic wrapper around it.

---

## 4. The architecture we settled on

A clean two-layer split that made the demo fast *and* honest:

```
                    OFFLINE (monthly, heavy)                        LIVE (instant, thin)
  ┌─────────────────────────────────────────────┐      ┌────────────────────────────────────┐
  │  build_grand_table.py                        │      │  streamlit_app.py                    │
  │   • yfinance 5y prices + fundamentals        │      │   • loads precomputed JSON bundles   │
  │   • damped-trend (Holt) ETS forecasts        │      │   • renders 4 + 1 panels             │
  │   → grand_table_expanded.csv  (knowledge     │ ───► │   • scales weights to ₹/month        │
  │      asset)                                  │      │   • plain-English narration          │
  │                                              │      │                                      │
  │  build_research_db.py                        │      │  vriddhi_core.py                     │
  │   • screen → backtest → walk-forward → MPT   │      │   • thin loaders for the bundles     │
  │   • gates, scenarios, explanations           │      │   • allocation scaling               │
  │   → research/portfolio_{1..5}y.json          │      │                                      │
  │   → research/benchmark.csv                   │      │                                      │
  └─────────────────────────────────────────────┘      └────────────────────────────────────┘
```

Heavy quant runs **once a month, offline**; the live app just renders the result. This is why the deployed app is instant and never depends on a flaky live data pull during a demo.

---

## 5. Before → After, feature by feature

### 5.1 Credibility: from "trust me" to "here's the evidence"

| | Before | After |
|---|---|---|
| **Backtest** | None | Per-stock & portfolio CAGR, max drawdown, volatility, Sharpe, over 1–5 year windows |
| **Walk-forward** | None | Expanding-window, **out-of-sample** validation (in-sample vs OOS CAGR, Sharpe, drawdown) |
| **Benchmark** | None | Compared against **Nifty 50 (`^NSEI`)** — does it beat the index after costs? |
| **Verdict gate** | Always showed a portfolio | **Hard gate**: shows a portfolio *only* if it clears the thresholds for that horizon |

**The gate is horizon-aware and honest.** A 5-year plan is judged on its **5-year walk-forward** CAGR — not an optimistic forward forecast. Thresholds: walk-forward CAGR ≥ 18% (3Y/4Y) / ≥ 20% (5Y), max drawdown < 25%, Sharpe > 1.0, and it must beat the benchmark. Horizons with too little history (e.g. 1Y) are honestly flagged as **"Not Validated"** rather than faked.

If a portfolio **fails**, the app says **"Not Recommended"**, suggests which horizons *do* pass, and tucks the full analysis behind an expander for transparency — it never hides the bad news.

### 5.2 A genuine funnel (not just a re-label)

| Before | After |
|---|---|
| Selection produced a portfolio | **Real 18 → 12 funnel**: screen ~18 candidates by PEG, then a two-pass Markowitz optimisation picks the final 12 |

The optimiser was fixed so it no longer produces near-zero "ghost" weights: it ranks candidates by max-Sharpe, then **re-optimises the top 12 with explicit constraints** — a **15% single-stock cap** and a **5% minimum weight** — so every holding is meaningful, with no shorting.

### 5.3 Our own knowledge asset (`build_grand_table.py`)

The biggest structural win: we stopped depending on an opaque external pipeline and built a **self-contained, transparent forecaster**.

| Before | After |
|---|---|
| Inherited forecasts (50–80% annualised, not reproducible) | **Damped-trend (Holt) exponential smoothing** on monthly log-prices via `statsmodels` |
| No realism control | **Realism guardrail**: shrinks each forecast toward a ~12% market anchor and **caps at 30%** → honest 3–18% forecasts |
| Couldn't be regenerated | One command regenerates the whole CSV from `yfinance` (prices, PE, PB, sector) |

Deliberately **simple and explainable** (no LSTM), and the app intentionally does **not** rely on the forecast for recommendations — those anchor to the walk-forward evidence. The forecast is shown only as a "model signal."

### 5.4 The investor dashboard (4 panels + plain English)

| Before | After |
|---|---|
| Single view | **4 panels**: Summary · Backtest Evidence · Final Portfolio · Risk |
| Quant-only language | **"Finance doctor" narration** on every tab |

We added a balanced, plain-English narrative aimed at a common investor — e.g. *"You're investing ₹1 lakh/month; build and monitor this basket, and based on its validated track record there's high confidence in ~X% CAGR, because…"* — adapting the message to the verdict (recommended / failed / insufficient history). Then we extended the same "teach a common man" treatment to **all** tabs:

- **Backtest Evidence** — how to read in-sample vs out-of-sample, and the growth chart.
- **Final Portfolio** — the two-step selection story ("value-for-money screen + Markowitz"), plus a jargon decoder (PEG, PE, PB, volatility, max drawdown, contribution).
- **Risk** — concentration, volatility, drawdown, and a closing "golden rule."

### 5.5 Monthly rebalancing — *"what do I actually buy this month?"*

| Before | After |
|---|---|
| Showed one fresh portfolio | New **Monthly Rebalance** tab comparing last month → this month |

It diffs the two portfolios and gives concrete, plain-English actions — **PICK** (new buy), **DROP** (exit), **TOP-UP**, **TRIM**, **HOLD** — each with the ₹ and **share** change at the user's monthly amount, plus a turnover summary. Low turnover is framed as a *good* thing (lower cost/tax).

**Clever demo enabler:** we didn't have to wait until 30 June for a second data point. Because the engine is deterministic and we have 5 years of history, we **reconstructed last month's portfolio** (as of 30 April) by truncating price history — a genuine, not faked, comparison. Going forward each monthly build **auto-rotates** this month's bundles into `*_prev.json`, so rebalancing is hands-free from June onward.

### 5.6 UI/UX polish

- **Logo:** added `Vriddhi.png` via `st.logo` + sidebar image.
- **Projection chart:** a polished investment-journey chart (styled like the reference `projection.png`) anchored to the **validated** CAGR, showing year-wise growth and monthly stock allocations.
- **Controls:** a **"Generate Investment Plan"** button (driven by `st.session_state`), horizon dropdown standardised to **1, 2, 3, 4, 5 years** (matching the 12/24/36/48/60-month columns), and a monthly-investment input.
- **Honest labelling:** lump-sum CAGR vs monthly-contribution projections are clearly distinguished — we gate on lump-sum evidence but project monthly contributions and say so.

### 5.7 One-command monthly refresh (`vriddhi_monthly_refresh.py`)

So that *anyone* — the lead dev or a junior — can refresh the entire back-end at month-end:

```
py vriddhi_monthly_refresh.py --push
```

A single, safe, well-logged pipeline:

1. **Pre-flight** — checks required files and Python packages.
2. **Backup** — snapshots the CSV + `research/` into `backups/<timestamp>/` (keeps the last 6).
3. **Knowledge asset** — runs `build_grand_table.py`.
4. **Research bundles** — runs `build_research_db.py` (auto-rotates last month → `_prev`, rebuilds all horizons + benchmark).
5. **Validate + report** — checks every bundle, prints verdicts **and** the rebalance deltas to eyeball before the demo.
6. **Smoke test** — runs the actual app headless across horizons to catch breakage *before* it goes live.
7. **(`--push`)** — commits with a dated message and pushes to GitHub → live beta redeploys.

It **aborts loudly** on any failure and never tells you to publish broken bundles. Flags: `--skip-grand-table`, `--no-smoke`, `--no-backup`, `--yes`. **Proven:** it ran end-to-end in ~35 seconds.

### 5.8 Self-healing universe (`ticker_resolver.py`)

The real-world snag: during a live run, Yahoo returned **`TATAMOTORS.NS` as delisted** (Tata Motors' demerger). Instead of a manual fix every time a symbol changes, we made the universe **self-healing**:

- **Harvest** — every healthy build caches each stock's **company name** (the key to reliable search; searching a bare ticker returns junk).
- **Resolve** — when a download fails, it tries a cached override → the `.BO` listing → a **name search**, and picks the match with the **longest usable history**.
- **Report** — writes `research/universe_health.json` and flags auto-heals for a 2-second human confirm.

**Proven live:** it auto-resolved `TATAMOTORS → TMPV.NS` (1238 days of history, vs the freshly-listed `TMCV.NS`'s 136), restoring the universe to **50 stocks with zero manual editing**. Truly dead tickers degrade gracefully — flagged and excluded, never breaking the run.

---

## 6. The "honesty engineering" thread

A recurring theme: every credibility feature came with a matching dose of honesty, because for an investor tool, trust *is* the product.

- Gate on **out-of-sample** evidence, not optimistic forecasts.
- **Cap forecasts** at realistic levels; show them only as a signal.
- Show **"Not Recommended"** plainly, with passing alternatives.
- Label **lump-sum vs monthly** projections distinctly.
- An honest **MVP footer** listing what's genuine vs simplified.
- **Flag** ambiguous auto-heals (a demerger is a real human judgement) rather than pretending they're risk-free.

---

## 7. Deployment battles (and how we won them)

Shipping to Streamlit Cloud surfaced real-world infra issues, all resolved:

- **`ImportError` on the live app** — root-caused to a `numpy` ABI mismatch; hardened `vriddhi_core.py` (guarded `seaborn`, lazy `scipy` import) and pinned compatible deps.
- **`scipy` build failure (`gfortran` not found)** — Streamlit Cloud ran Python 3.13 and tried to compile `scipy` from source. Fix: **removed `scipy`/`seaborn` from runtime deps entirely** (not needed at render time) and relaxed version caps so prebuilt 3.13 wheels install cleanly.
- **Headless test crash** on missing `secrets.toml` — wrapped `st.secrets` access in a guard.
- **Git remote alignment** — the live beta deploys from **GitHub** (not the GitLab `origin`); aligned and pushed there.

---

## 8. The commit timeline (the story in git)

```
57da577  Add backtest + walk-forward credibility gate and 4-panel investor dashboard
1184da2  Fix live ImportError: pin numpy<2 deps and harden vriddhi_core imports
7cdae6b  Sharpen credibility: real funnel, honest explanations, hard gate, projection chart + logo
3bfacaf  Fix deploy build failure: drop scipy/seaborn from runtime deps, remove version caps
08037fb  Add finance-doctor summary, Generate Investment Plan button, 1-5yr horizons
493e2f5  Add build_grand_table.py: self-contained ML forecaster for the knowledge asset
81fc663  Adopt self-generated knowledge asset: refresh CSV + rebuild bundles
edd739e  Add plain-English "common man" explanations to all dashboard tabs
e1e9c6c  Add monthly rebalance view (pick/drop/top-up/trim/hold)
125ece1  Add Vriddhi monthly refresh pipeline (one-command back-end refresh)
48462ee  Add self-healing ticker resolver (auto-handle renames/demergers)
```

---

## 9. Final file inventory (what's new / changed)

**New this session**
- `build_grand_table.py` — self-contained knowledge-asset builder (forecasts + fundamentals).
- `vriddhi_monthly_refresh.py` — one-command monthly back-end refresh orchestrator.
- `ticker_resolver.py` — self-healing symbol resolver.
- `ticker_aliases.json` — name/symbol cache + audit log (self-maintaining).
- `research/` — precomputed bundles: `portfolio_{1..5}y.json`, their `_prev` snapshots, `benchmark.csv`, `universe_health.json`.

**Substantially upgraded**
- `streamlit_app.py` — 4 + 1 panel dashboard, finance-doctor narration, projection chart, logo, button, rebalance tab.
- `vriddhi_core.py` — conflict resolution, dependency hardening, bundle loaders, allocation scaling.
- `build_research_db.py` — screen→backtest→walk-forward→MPT, gates, scenarios, explanations, `--asof`/`--suffix`, auto-rotation, resolver integration.
- `requirements.txt` / `requirements-dev.txt` — runtime slimmed for clean cloud builds; `statsmodels` added for builds.

---

## 10. What's genuine vs. simplified (MVP transparency)

**Genuine today:** yfinance adjusted prices · damped-trend time-series forecasts · CAGR / drawdown / volatility / Sharpe · walk-forward validation · Markowitz optimisation · pass/fail gates · benchmark beat · month-over-month rebalancing · self-healing universe.

**Simplified for v1 (next on the list):** point-in-time fundamentals (PE/PB are current, not historical) · explicit transaction-cost & tax modelling · a richer multi-model forecast ensemble · auto-tracking Nifty 50 index membership changes.

---

## 11. Suggested next steps

1. **Confirm** the TATAMOTORS auto-heal direction (passenger vs commercial entity) — a one-line override if you prefer the other.
2. **Schedule** the monthly refresh (Task Scheduler / cron) on the last day of each month, or keep running it by hand.
3. Add **transaction-cost/tax** modelling to make turnover advice even more precise.
4. Consider **point-in-time fundamentals** to make reconstructed history exact.

---

*Built collaboratively in a single Cursor session — from base repo to a credible, self-validating, self-refreshing investor demo, live at [vriddhi-core-beta.streamlit.app](https://vriddhi-core-beta.streamlit.app/).*
