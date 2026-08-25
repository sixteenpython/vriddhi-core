# BTI Reuse Assessment

**Status:** Phase 0 architecture decision
**Assessed:** 25 August 2026
**Vriddhi release inspected:** `refresh-2026-08-14`, methodology 2.0

## Decision

BTI will be an isolated product boundary inside the Vriddhi repository. It will consume promoted
Vriddhi artifacts through a server-side `ReferenceDecisionProvider`; it will not import Streamlit,
run the monthly optimiser in a request, or copy Vriddhi formulas into Android.

The Android application is a native Kotlin/Jetpack Compose client targeting Android API 36. The
backend is a Python modular monolith with an HTTP API and SQLite MVP persistence. The backend can
later move to managed Postgres without changing game contracts.

## Repository evidence

| Existing capability | Evidence | Decision |
|---|---|---|
| 50-stock Golden Knowledge Asset | `grand_table_expanded.csv`, 50 rows, 19 decision columns | Reuse with read-only adapter |
| Current price, sector, PE, PB, PEG | GKA columns | Reuse as market-board inputs |
| Forecast horizons 12–60 months | GKA forecast and expected-return columns | Reuse as bounded scenario signals |
| Nifty benchmark | `research/benchmark.csv`, 1,238 observations through 2026-08-14 | Reuse for release provenance and benchmark calibration |
| Five horizon portfolios | `research/portfolio_1y.json` through `portfolio_5y.json` | Reuse server-side; 3/4/5 year bundles are BTI references |
| Portfolio weights and sectors | Bundle `stocks` and `sector_allocation` | Keep private from Android; expose only evaluation feedback |
| Portfolio metrics | Bundle `portfolio_metrics`, scenarios and benchmark | Reuse server-side |
| Per-stock CAGR, volatility, drawdown, Sharpe | Bundle stock metrics for selected names | Reuse where present; derive simulated metrics for the full game board |
| Walk-forward OOS evidence | Bundle `walk_forward` and equity curves | Reuse for methodology provenance, not as a month-by-month stock tape |
| Efficient frontier and optimisation comparison | Bundle `optimal_view` | Keep private/server-side |
| Recommendation gates | Bundle `verdict` | Reuse as release authority; do not show “Vriddhi says buy” |
| Monthly recommendation history | `research/recommendation_ledger.json` | Reuse later for prospective puzzles and live-season validation |
| Provenance and hashes | `research/manifest.json`, `validation_report.json` | Reuse in every BTI game identity |
| Transactional monthly refresh | `vriddhi_monthly_refresh.py` | Reuse operational pattern; BTI game-data export is a downstream candidate |
| Streamlit CI and tests | `.github/workflows/ci.yml`, `tests/` | Extend without changing production behavior |

## Reuse classification

### Reusable as-is

- promoted GKA and research JSON/CSV artifacts;
- artifact manifest, methodology version and data-through date;
- current 3/4/5-year reference portfolios;
- Nifty benchmark release;
- validation and transactional-promotion philosophy; and
- existing Vriddhi regression suite.

### Reusable with an adapter

- GKA rows become versioned `StockSnapshot` objects;
- research bundles become opaque `ReferenceDecision` inputs;
- portfolio metrics become game feedback baselines;
- release hashes become `knowledge_asset_version`; and
- recommendation ledger snapshots later become puzzle seeds.

### Must be newly built

- game, market, move, evaluation, result, puzzle, lesson and rating contracts;
- deterministic seeded simulator;
- temporal-integrity policy and game-data release;
- immutable move store and audit chain;
- authenticated HTTP API;
- player profile/history/rating;
- Android application and Google sign-in;
- operational telemetry without unnecessary personal data; and
- Play Store build, declarations and release documentation.

### Must remain server-side

- reference holdings and target weights;
- frontier/optimisation payloads;
- reference alignment calculations;
- move-scoring component weights and anti-gaming controls;
- service credentials, session-signing secret and OAuth verification; and
- any future licensed market data.

The client receives public market facts, its own portfolio, the final move class and bounded
explanations. It never receives the reference portfolio or hidden component vector.

### Must not be duplicated

- the Vriddhi GKA;
- security selection and portfolio optimisation;
- walk-forward research;
- release provenance; or
- monthly market-data acquisition.

## Critical data limitation

Vriddhi currently retains current fundamentals, current/previous research bundles, a benchmark
series and walk-forward portfolio curves. It does **not** retain the complete point-in-time monthly
fundamental/forecast snapshot and historical Nifty constituent set required for an honest 36–60
move historical game.

Using the current PE/PB/PEG values at a 2021 game date would be look-ahead bias. Therefore the first
release is an explicitly labelled deterministic simulation beginning at the promoted knowledge
cutoff. A later historical mode requires immutable monthly stock snapshots, contemporaneous
fundamentals, historical constituents, corporate actions and news timestamps.

## Toolchain finding

The inspected workstation has Python 3.12 and Node.js but no Java, Gradle, Android SDK, `adb` or
Flutter. The Android source can be implemented immediately; local `.aab` production requires an
isolated JDK 17 + Android API 36 toolchain or an approved CI runner. Google sign-in additionally
requires an OAuth client registered for the final application ID and upload certificate.

## Stability boundary

No existing Vriddhi production module, artifact or Streamlit behavior will be modified merely to
make BTI convenient. BTI additions live below `bti/`; shared integration is through read-only
adapters and additive CI jobs.
