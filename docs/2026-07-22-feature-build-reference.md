# Vriddhi Feature Build Reference — 22 July 2026

## 1. Purpose and completion status

This document records the two feature streams completed and deployed on
22 July 2026:

1. **Backtest Evidence:** honest historical evidence, an illustrative SIP replay,
   and a prospective record of recommendations that Vriddhi actually published.
2. **Monthly Rebalance:** plain-English decision support followed by an executable,
   whole-share, netted monthly order sheet.

Both streams are live at
[vriddhi-core-beta.streamlit.app](https://vriddhi-core-beta.streamlit.app/).
The implementation passed the complete 14-test local suite after the final change,
and every deployment commit listed in this document passed GitHub CI.

This reference supplements, rather than replaces:

- [`architecture.md`](architecture.md)
- [`methodology.md`](methodology.md)
- [`recommendation-ledger.md`](recommendation-ledger.md)
- [`monthly-refresh-runbook.md`](monthly-refresh-runbook.md)
- [`backend-reengineering-reference.md`](backend-reengineering-reference.md)

## 2. Product principle used throughout

The UX and calculations were aligned to one question:

> Given everything we know today, which portfolio maximizes the probability of
> superior long-term, risk-adjusted returns?

The app therefore does not claim to identify tomorrow's winning stock. It explains
how each stock fits the complete portfolio, separates historical evidence from
forecasts, and converts monthly portfolio changes into explicit actions without
hiding data limitations.

## 3. Backtest Evidence feature

### 3.1 Problem addressed

The earlier Backtest Evidence view contained useful quantitative output, but it did
not adequately distinguish four different concepts:

- in-sample fit;
- walk-forward out-of-sample evidence;
- a hypothetical monthly cash-flow replay over the stored out-of-sample stream; and
- the real sequence of recommendations Vriddhi has published prospectively.

Without this separation, a user could reasonably mistake a reconstructed historical
view for a record of actual monthly PICK, DROP, TOP-UP, TRIM, and HOLD decisions.

### 3.2 Benchmark restoration

`load_benchmark_series()` in `vriddhi_core.py` now reads
`research/benchmark.csv`, parses `Date`, and returns a pandas `DataFrame`. Missing,
unreadable, or invalid files return `None` rather than crashing the dashboard.

The walk-forward chart renders:

- the solid portfolio out-of-sample equity curve;
- the dashed Nifty 50 normalized benchmark when benchmark data is valid; and
- the grey 1.0 break-even line.

A loader regression test prevents the benchmark CSV-reading block from silently
disappearing again.

### 3.3 Illustrative out-of-sample SIP replay

`build_oos_sip_replay()` translates the stored walk-forward equity curve into a
cash-flow illustration for 12, 24, 36, 48, and 60-month starting windows.

The algorithm:

1. validates and orders positive dated equity observations;
2. takes the first available out-of-sample observation in each calendar month;
3. adds one selected monthly contribution at each contribution point; and
4. grows each contribution by the remaining stored equity-curve multiple through
   the final out-of-sample date.

Unavailable periods stay unavailable when there is insufficient out-of-sample
history. The app does not fill gaps with in-sample results or forecasts.

This is explicitly labelled an **illustrative out-of-sample SIP replay**. It is not
an account ledger, a historical XIRR, or proof that the displayed recommendations
were issued on those earlier dates.

### 3.4 Prospective recommendation ledger

Vriddhi now stores actual published monthly recommendations in the append-only file
`research/recommendation_ledger.json`. The retained 4 June 2026 and 17 July 2026
releases seed the prospective record.

Each release records, for all five horizons:

- release and data-through dates;
- methodology version and provenance;
- recommendation status;
- stocks, weights, sectors, and publication-time prices; and
- derived INITIAL, PICK, DROP, TOP-UP, TRIM, and HOLD actions.

The action threshold is one percentage point. Same-date retries do not rewrite the
first published snapshot. The ledger is covered by validation and manifest hashing;
missing, malformed, stale, duplicate-date, out-of-order, or incomplete data blocks a
research release.

The **Actual Vriddhi Track Record** view remains gated until 12 validated monthly
releases exist. Before the gate, the UI shows collection progress and the releases
recorded. After the gate, `build_recommendation_ledger_replay()` can replay the actual
published sequence with these disclosed assumptions:

- one contribution immediately before each recorded rebalance;
- fractional shares;
- complete rebalancing to recorded weights;
- fully invested cash; and
- no transaction-cost deduction.

The prospective ledger does not replace walk-forward validation. The former records
what Vriddhi actually published after tracking began; the latter tests the methodology
on historical unseen periods.

### 3.5 Narrative and trust layer

The Backtest Evidence tab now explains:

- why out-of-sample evidence matters more than in-sample fit;
- how to read the portfolio, Nifty 50, and break-even lines;
- how the SIP replay is calculated;
- why the replay is historical evidence rather than a forecast;
- that future test-period prices were unavailable when weights were chosen;
- that the method re-optimizes at train/test boundaries; and
- that Vriddhi improves probabilities rather than guaranteeing returns.

### 3.6 CAGR and XIRR terminology

Ambiguous performance labels were corrected throughout the app:

- **Validated OOS CAGR** means annualized compound growth of the strategy's stored
  out-of-sample return stream.
- **Historical portfolio CAGR** and **Historical Nifty 50 CAGR** describe lump-sum
  historical compound growth over the specified window.
- **Illustrative SIP value** is a projected or replayed cash-flow value and is not
  labelled CAGR or XIRR.
- **XIRR** is reserved for a calculation using actual dated cash flows and an ending
  value. Vriddhi will use that label only when the prospective ledger supports it.

### 3.7 Runtime deployment repair

The Streamlit entry point defensively reloads `vriddhi_core` only when a long-lived
worker still holds a pre-ledger module without the new exports. This repaired the
deployment-time import failure while avoiding an unconditional reload on every app
run.

## 4. Monthly Rebalance feature

### 4.1 Decision model

The current and previous portfolio bundles are compared for the chosen investment
horizon. For each stock, the previous weight `pw` and current weight `cw` determine
the action:

| Condition | Action | User meaning |
|---|---|---|
| `pw == 0` and `cw > 0` | PICK | Begin investing |
| `cw == 0` and `pw > 0` | DROP | Exit the model position |
| `cw - pw > 0.01` | TOP-UP | Give the stock a larger role |
| `pw - cw > 0.01` | TRIM | Keep it with a smaller role |
| otherwise | HOLD | Retain it and repeat the monthly purchase |

Turnover is reported as half the sum of absolute weight changes.

The crucial product definition is:

> **HOLD means retain the stock and buy it again with the current month's SIP.**

It does not mean that the user should omit the stock from this month's contribution.

### 4.2 Comparison and evidence UX

The tab now provides:

- a summary of PICK, DROP, TOP-UP, TRIM, and HOLD counts;
- a compact before/after table with target rupee changes;
- a one-decimal **Change in shares** planning equivalent;
- the complete current and previous portfolio tables;
- prominent action cards instead of clipped long rationale cells; and
- a separate, collapsed HOLD rationale section after the actionable changes.

Each rationale uses the available portfolio and stock evidence, including:

- previous and current portfolio weights;
- optimizer return, volatility, and Sharpe inputs;
- historical horizon CAGR;
- maximum drawdown; and
- PEG when retained in the stock explanation.

The copy translates these into everyday language and explains how the action supports
the whole portfolio. It does not pretend that one metric alone caused an optimizer
decision; weights depend on the complete return/covariance relationship.

### 4.3 SIP input and model-sleeve execution

The Monthly Rebalance tab accepts a current-month SIP from ₹50,000 to ₹1,00,000 in
₹5,000 steps. This tab-specific amount scales both recorded portfolios and drives the
execution plan.

The app does not connect to a demat account and does not know the user's actual
accumulated holdings. SELL quantities therefore refer to **one previous-month model
sleeve scaled to the entered SIP**. Users must adjust those quantities to their real
holdings and verify live prices before execution.

### 4.4 Gross execution algorithm

`_build_monthly_execution_plan()` creates the transparent gross calculation:

1. Convert the previous model allocation to whole shares using the previous stored
   price.
2. For a DROP, sell every whole share in the previous model sleeve.
3. For a TRIM, compare previous model shares with current target shares and sell the
   positive difference, using the current price when available.
4. Add estimated sale proceeds to the entered SIP to obtain gross buying power.
5. Allocate that buying power across all current holdings according to current target
   weights.
6. Start each holding at the whole-share floor.
7. Repeatedly place another affordable share where it causes the smallest incremental
   absolute deviation from the target rupee allocation.
8. Stop when no additional share fits.

The BUY table labels intent as:

- `BUY · new PICK`;
- `BUY MORE · TOP-UP`;
- `BUY AGAIN · HOLD`; or
- `BUY LESS · TRIM`.

This means a TRIM can still receive part of the new monthly SIP. TRIM describes a
smaller portfolio weight, not necessarily a permanently negative share count as the
portfolio receives new contributions.

### 4.5 Combined net execution sheet

The gross detail is useful for explaining the rebalance, but it can create redundant
same-stock legs. `_build_net_execution_ledger()` therefore generates the primary,
algorithm-friendly order sheet:

1. group gross BUY and SELL legs by ticker;
2. calculate `net shares = buy shares - sell shares`;
3. emit one BUY when the result is positive;
4. emit one SELL when the result is negative;
5. remove zero-net instructions;
6. sequence genuine SELL orders before BUY orders; and
7. show the estimated cash movement and cash remaining after every step.

For the 5-year, ₹1,00,000 example deployed on 22 July 2026, gross NTPC instructions
were SELL 8 and BUY 17. The executable sheet correctly nets them to **BUY 9 NTPC**.

The distinction between gross and net cash is intentional:

- Gross accounting temporarily reports ₹1,11,199 of buying power because it includes
  the NTPC sale and subsequent repurchase.
- The net order ledger rises only to approximately ₹1,08,464 because it removes that
  redundant round trip.
- Both calculations reach the same approximately ₹156 closing cash balance, subject
  to stored-price precision.

### 4.6 Fractional versus whole-share values

The comparison table's **Change in shares** is:

`target rupee change / applicable stored stock price`

It is rounded to one decimal and communicates portfolio intent. The execution sheet
uses whole shares and may therefore show a different executable number. Fractional
planning values do not imply that an Indian broker supports fractional equity trades.

## 5. Adjacent UX completed in the same feature session

The work also made the rest of the app consistent with these two primary features:

- The landing page now states the evidence-led vision and explains monthly review.
- Final Portfolio stock rationales are visible by default as readable cards.
- Valuation terms such as PEG, PE, and PB are translated into plain English.
- Stock cards explain growth/contribution, risk, and whole-portfolio fit.
- The Risk tab now contains a one-sentence assessment, a six-row scorecard,
  concentration and sector exposure, historical loss translated to rupees, stock-level
  risk hotspots, diversification evidence, and an investor playbook.
- Historical drawdown and volatility remain explicitly described as observations, not
  ceilings on future loss.

## 6. Implementation map

| File | Responsibility added or materially changed |
|---|---|
| `streamlit_app.py` | Backtest narratives, OOS replay UI, actual ledger gate, CAGR/XIRR labels, portfolio/risk cards, rebalance rationales, SIP execution plan, gross and net order sheets |
| `vriddhi_core.py` | Benchmark loader, recommendation-ledger loader/replay, out-of-sample SIP replay |
| `vriddhi_ledger.py` | Append-only snapshot construction, action derivation, idempotent ledger update |
| `vriddhi_validation.py` | Ledger schema, chronology, completeness, and policy validation |
| `vriddhi_monthly_refresh.py` | Ledger update inside the transactional monthly refresh |
| `research/recommendation_ledger.json` | Published prospective evidence for every horizon |
| `research/manifest.json` | Ledger included in the published artifact integrity contract |
| `tests/test_core_loaders.py` | Benchmark, manifest, ledger, and helper regression coverage |
| `tests/test_validation.py` | Ledger release-validation and same-date behavior |
| `tests/test_streamlit_app.py` | Cross-horizon production-style UI regression coverage |

## 7. Regression contract

The Streamlit integration test renders every horizon in an isolated process and checks
the presence and semantics of the completed UX. The final suite specifically verifies:

- benchmark and release evidence can load;
- all five horizons render without Streamlit exceptions;
- the Backtest Evidence narrative and prospective evidence status appear;
- the current and previous portfolio tables render;
- action tables use the intended labels and fractional share equivalents;
- rationale cards expose quantitative evidence and philosophy alignment;
- the SIP input accepts the ₹50,000 lower bound and defaults from the selected plan;
- gross SELL and BUY tables are present;
- HOLD rows are labelled `BUY AGAIN · HOLD`;
- the 5-year gross SELL plan includes ADANIPORTS 2, HEROMOTOCO 1, and NTPC 8;
- the net sheet contains `BUY 9 shares of NTPC` and no NTPC SELL instruction;
- the net ledger closes at approximately ₹156 for the recorded ₹1,00,000 fixture;
- CAGR/XIRR labels remain unambiguous; and
- the enriched Final Portfolio and Risk views remain present.

The final local result was **14 passed**.

## 8. Deployment record

The work was delivered incrementally so every semantic change had a reviewable commit:

| Commit | Purpose |
|---|---|
| `d1551aa` | Add honest OOS wealth replay narrative |
| `a1fc50a` | Add prospective recommendation ledger |
| `28f7fbf` | Recover stale Streamlit core imports |
| `d283ae3` | Explain Vriddhi vision and monthly rebalancing |
| `34be4c7` | Show compared portfolios in monthly rebalance |
| `9d37d6d` | Explain monthly rebalance actions with evidence |
| `80f3a85` | Redesign monthly rebalance decision experience |
| `8aa0b2b` | Enrich portfolio rationales and risk experience |
| `8c5c210` | Make portfolio rationales visible and intuitive |
| `4ca9481` | Clarify CAGR and SIP return labels |
| `b899819` | Show fractional rebalance share changes |
| `e87188b` | Recommend whole-share use of rebalance surplus |
| `d579733` | Add SIP-based monthly execution plan |
| `80e7d25` | Add netted rebalance execution sheet |

The last deployed application commit before this documentation was `80e7d25`.

## 9. Known limitations and engineering guardrails

1. **Not brokerage-aware.** Execution quantities are model-sleeve instructions, not a
   calculation from actual account holdings, taxes, or cost basis.
2. **Stored prices, not live quotes.** The order sheet uses release snapshot prices.
   Users must verify current quotes; actual cash remaining will differ.
3. **Whole-share approximation.** Greedy allocation minimizes target distortion but
   does not claim a globally optimal integer solution under every possible price set.
4. **No execution friction.** Brokerage, securities transaction tax, bid/ask spreads,
   slippage, capital-gains tax, and liquidity are not included.
5. **DROP price fallback.** A dropped stock is absent from the current bundle, so its
   previous recorded price is used when no current price is available.
6. **Evidence remains historical.** Neither walk-forward evidence nor a future unlocked
   recommendation replay guarantees future returns.
7. **Ledger maturity gate.** Actual track-record performance must stay locked until at
   least 12 valid releases exist.
8. **CAGR is not XIRR.** Do not relabel strategy CAGR or an illustrative SIP value as
   XIRR without actual dated cash flows.
9. **Gross and net views serve different purposes.** Gross tables explain how target
   changes were funded; the net sheet is the practical order list. They must reconcile
   to the same closing cash under the same stored prices.

## 10. Recommended next steps

The next evolution should be driven by real product requirements rather than adding
more labels to the current model:

1. accept an optional user holdings ledger and produce account-specific delta orders;
2. introduce a licensed or independently reconciled live-price feed for execution;
3. add explicit cost, tax, and slippage assumptions to the order ledger;
4. consider an integer optimizer if exact whole-share target tracking becomes material;
5. preserve every future monthly release through the transactional refresh; and
6. add XIRR only after sufficient dated, prospective cash-flow evidence exists.

## 11. Definition of done

The two feature streams are complete when all of the following remain true:

- historical walk-forward evidence and actual published evidence are visibly separate;
- the Nifty 50 benchmark renders when its validated CSV is available;
- no track-record claim is unlocked before the 12-release gate;
- CAGR and XIRR labels describe the calculation actually performed;
- every rebalance action has an intuitive, quantitative rationale;
- HOLD means retain and repeat the monthly purchase;
- the entered SIP drives a whole-share model-sleeve plan;
- gross and net executions reconcile;
- opposing same-stock legs are netted in the primary order sheet;
- disclosures remain visible; and
- the full automated suite and deployment CI pass.
