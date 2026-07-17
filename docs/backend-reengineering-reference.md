# Vriddhi Backend Reengineering Reference

**Audience:** Vriddhi engineers, maintainers, reviewers, and monthly operators  
**Status:** Production reference  
**Engineering release:** `04bad7a` — *Harden monthly refresh and release workflow*  
**Research release at migration:** `e1a23aa` — data through 2026-07-17  
**Methodology version:** 2.0  
**Python baseline:** 3.12

## 1. Executive summary

Vriddhi began this reengineering exercise with a fundamentally sound product
architecture: expensive market research was calculated offline and committed as
static artifacts, while the live Streamlit application only rendered those
artifacts. That separation kept the app inexpensive, responsive, and insulated
from market-data outages during user sessions.

The reengineering did **not** replace that architecture. It made the monthly
research release process reproducible, transactional, testable, auditable, and
safe to automate.

The principal change is this:

```text
Before
------
Download directly into published files
  -> rebuild published bundles in place
  -> smoke test
  -> optionally push master

After
-----
Create isolated candidate workspace
  -> build candidate knowledge asset
  -> build candidate research bundles
  -> validate every artifact and cross-artifact invariant
  -> generate provenance, hashes, and change report
  -> smoke-test all five horizons against the candidate
  -> promote only after every gate passes
  -> create a reviewed GitHub candidate release
  -> deploy from master
```

As a result, a failed data download, missing dependency, malformed bundle,
unexpected turnover event, UI regression, Git error, or same-date retry can no
longer silently become a production research release.

## 2. Goals and non-goals

### 2.1 Goals

The reengineering was designed to achieve the following operational properties:

1. A clean checkout can recreate the approved Python environment.
2. Monthly research is built away from published artifacts.
3. A failed build leaves the current published release unchanged.
4. Every portfolio bundle is validated before publication.
5. Month-over-month turnover and security changes are explicit.
6. Every research release records its inputs, environment, source commit, and
   artifact hashes.
7. The UI is tested against the candidate research rather than the previous
   production files.
8. GitHub provides a browser-triggered monthly candidate workflow.
9. Production changes pass the same locked test environment locally and in CI.
10. Operators have a documented recovery and rollback path.

### 2.2 Non-goals

The exercise deliberately avoided architecture for architecture's sake. It did
not introduce:

- A database for static monthly artifacts
- A FastAPI or microservice layer
- Live portfolio optimization in Streamlit sessions
- Live market-data calls from the deployed app
- An LLM dependency
- A new frontend framework
- A change to the July production portfolio merely to accompany engineering
  changes

These additions would have increased operational surface area without solving
the observed reliability problems.

## 3. Original architecture and its strengths

The pre-existing backend already had four useful layers:

| Layer | Canonical file | Responsibility |
|---|---|---|
| Knowledge asset | `build_grand_table.py` | Refresh stock prices, current fundamentals, sectors, ticker health, and transparent damped-Holt forecast signals |
| Research engine | `build_research_db.py` | Screen candidates, optimize portfolios, calculate risk/backtests, run walk-forward analysis, apply gates, and write horizon bundles |
| Domain/runtime helpers | `vriddhi_core.py` | Portfolio logic plus loaders used by Streamlit |
| Presentation | `streamlit_app.py` | Render precomputed research and investor-facing explanations |

The live application did not require `yfinance`, SciPy optimization, or
statsmodels. It loaded committed JSON and CSV files. This was retained as a core
design constraint.

## 4. Why reengineering was necessary

The July 2026 refresh exposed several concrete operational weaknesses.

### 4.1 Environment drift

The existing workspace virtual environment could not start Python because its
standard-library `encodings` module was unavailable. The repository did not
have a complete, portable environment contract that could recreate the monthly
builder on demand.

The repository also contained conflicting runtime expectations:

- `runtime.txt` specified Python 3.9.
- Documentation discussed newer Streamlit Cloud Python versions.
- The successful July build ultimately ran on Python 3.12.
- Requirements expressed minimum versions rather than a complete resolution.

### 4.2 Incomplete preflight dependency contract

The original refresh preflight checked:

```python
pandas, numpy, yfinance, statsmodels, scipy
```

It did not check `matplotlib`. However, `build_research_db.py` imports
`vriddhi_core.py`, which imports matplotlib. The first July refresh therefore:

1. Passed preflight.
2. Backed up the June release.
3. Successfully replaced `grand_table_expanded.csv` with July data.
4. Failed while starting the research build because matplotlib was missing.

The live GitHub application remained safe because nothing had been pushed, but
the local published workspace temporarily contained a July knowledge asset and
June research bundles.

### 4.3 In-place artifact generation

Both builders wrote directly to canonical published paths. Research bundles
were rotated and overwritten sequentially. A failure could therefore leave a
mixed local release.

### 4.4 Weak publication error semantics

The earlier `git_push()` implementation treated every non-zero commit result as
"nothing to commit." A real commit error could therefore be misreported.
Publication errors were logged but did not consistently make the complete
refresh fail.

### 4.5 Limited validation and testing

The earlier process checked that files existed, portfolios contained stocks,
and selected Streamlit horizons rendered. It did not enforce a complete artifact
contract such as:

- Weight totals
- Duplicate holdings
- Non-finite JSON numbers
- Benchmark/bundle date agreement
- Strictly increasing benchmark dates
- Knowledge-asset numeric validity
- Stale data limits
- Ticker membership in the universe
- Excessive turnover approval
- All five UI horizons

### 4.6 Same-date retry behavior

The research builder rotates the current bundle into `*_prev.json` before
writing a new current bundle. Retrying a refresh on the same market date would
make "previous" and "current" share the same date, destroying the genuine
month-over-month comparison.

### 4.7 GitHub operational gaps

The repository had no CI workflow, monthly workflow, dependency automation,
canonical `README.md`, standard license file, security guidance, or engineering
runbook.

## 5. Reengineered architecture

### 5.1 System view

```text
                           MONTHLY BUILD PLANE

                     +---------------------------+
                     | Locked Python 3.12 env    |
                     | pyproject.toml + uv.lock  |
                     +-------------+-------------+
                                   |
                                   v
 +----------------+      +------------------------+      +------------------+
 | Existing live  | copy | .refresh_staging/<run> |      | Market provider  |
 | artifacts      +----->| isolated candidate     |<-----+ Yahoo Finance    |
 +----------------+      +-----------+------------+      +------------------+
                                   |
                      +------------+-------------+
                      |                          |
                      v                          v
           build_grand_table.py       build_research_db.py
                      |                          |
                      +------------+-------------+
                                   v
                     +---------------------------+
                     | vriddhi_validation.py     |
                     | schema + cross-file gates |
                     +-------------+-------------+
                                   |
                      +------------+-------------+
                      |                          |
                      v                          v
           manifest + hashes + report      Streamlit AppTest
                                              horizons 1-5
                      |                          |
                      +------------+-------------+
                                   v
                         validated promotion
                                   |
                                   v
                     +---------------------------+
                     | Git candidate / PR / CI   |
                     +-------------+-------------+
                                   |
                                   v
                           GitHub master
                                   |
                                   v
                         Streamlit deployment


                              LIVE PLANE

                 committed CSV/JSON -> Streamlit rendering
                 no live research or market-data dependency
```

### 5.2 Source files introduced or materially changed

| File | Reengineering role |
|---|---|
| `vriddhi_monthly_refresh.py` | Transactional orchestrator, backup/restore, staging, validation, smoke testing, promotion, and strict Git publication |
| `vriddhi_validation.py` | Network-free artifact and cross-artifact validation contract |
| `build_grand_table.py` | Adds an optional price-history `--as-of` cutoff and continues to support staged output |
| `vriddhi_core.py` | Supports `VRIDDHI_RESEARCH_DIR` candidate injection and release-manifest loading |
| `streamlit_app.py` | Displays release provenance, uses current Streamlit width APIs, and states methodology limitations more accurately |
| `pyproject.toml` | Canonical Python 3.12 project, dependency groups, pytest configuration, and Ruff configuration |
| `uv.lock` | Cross-platform lock for the complete 72-package environment resolution |
| `.github/workflows/ci.yml` | Frozen-environment compilation, lint, and test gate |
| `.github/workflows/monthly-refresh.yml` | Browser/scheduled candidate build that opens a pull request |
| `.github/dependabot.yml` | Monthly Python and GitHub Actions dependency review |
| `tests/` | Unit, artifact, and isolated all-horizon Streamlit tests |
| `research/manifest.json` | Release provenance and SHA-256 hashes |
| `research/validation_report.json` | Machine-readable validation evidence |
| `research/refresh_summary.md` | Human-readable operator and pull-request report |

## 6. Transactional monthly refresh

The orchestrator now runs seven conceptual stages.

### Stage 1: Preflight

Preflight verifies required source files and imports the actual dependency set:

```text
pandas
numpy
yfinance
statsmodels
scipy
matplotlib
streamlit
```

It records exact package versions for the release manifest and reports the
interpreter path and Python version.

### Stage 2: Backup

Unless disabled or running candidate-only mode, the process copies:

- `grand_table_expanded.csv`
- `ticker_aliases.json`
- the complete `research/` directory

into:

```text
backups/YYYY-MM-DD_HHMMSS/
```

Backups are pruned only **after successful promotion**. The most recent six are
retained by default. A failed build does not cause an old recovery point to be
deleted.

### Stage 3: Candidate workspace

The process creates:

```text
.refresh_staging/YYYY-MM-DD_HHMMSS/
```

and copies the current knowledge asset, aliases, and research bundle set into
it. All builder subprocesses run with this directory as their working directory.
The original builders can therefore use their familiar relative paths while
writing only inside staging.

The candidate workspace also saves a private copy of the genuine previous-month
snapshots under `.previous_baseline/` for same-date retry recovery.

### Stage 4: Candidate build

The knowledge builder runs from the production source tree but writes its output
inside staging. It updates the staged alias cache and staged universe health
report.

The research builder then downloads price history, rotates the staged current
bundles, and writes all five candidate horizon bundles plus the benchmark.

No canonical published data file is modified during these steps.

### Same-date idempotency

After the research build, the orchestrator compares the current and rotated
previous dates. If the refresh is a retry on the same data date, it restores the
older baseline `*_prev.json` files. This preserves the genuine month-over-month
rebalance comparison.

### Stage 5: Candidate validation

`vriddhi_validation.py` validates the complete candidate. Details are specified
in Section 7.

### Stage 6: Release evidence and UI smoke test

After structural validation passes, the process writes validation evidence,
artifact hashes, and a change summary. It then launches Streamlit AppTest in an
isolated subprocess with:

```text
VRIDDHI_RESEARCH_DIR=<candidate research directory>
```

This ensures the UI is tested against the candidate, not the old production
release. Horizons 1, 2, 3, 4, and 5 are all rendered and interacted with.

The process boundary also prevents Streamlit/matplotlib native-resource teardown
on Windows from corrupting the parent test or orchestration process.

### Stage 7: Promotion

Only a completely validated candidate is promoted. Each destination is written
to a sibling `*.promoting` temporary file and replaced using `os.replace`.

If promotion fails after any files have moved, the orchestrator restores the
pre-run backup and exits unsuccessfully.

Candidate-only mode stops before this stage and retains staging for inspection.

## 7. Validation contract

The validator is deliberately network-free so the same rules can run locally,
in CI, during monthly automation, and against already committed artifacts.

### 7.1 Knowledge-asset validation

`grand_table_expanded.csv` must satisfy all of the following:

- At least 40 securities
- Required columns present:
  - `Ticker`
  - `Sector`
  - `Current_Price`
  - `Forecast_12M`
  - `Forecast_24M`
  - `Forecast_36M`
  - `Forecast_48M`
  - `Forecast_60M`
  - `PE_Ratio`
  - `Avg_Historical_CAGR`
- Unique tickers
- Parseable and finite required numeric fields
- Positive current prices

### 7.2 Benchmark validation

`research/benchmark.csv` must have:

- At least 250 observations
- A `Date` column
- Strictly increasing and unique dates
- Parseable, positive, finite normalized values
- A final date identical to every current portfolio bundle's `data_through`

### 7.3 Universe-health validation

`research/universe_health.json` is parsed and reviewed. More than five
`unresolved` or `stale` tickers aborts the release. A smaller number is surfaced
as a warning for human review.

Ticker renames remain visible in `needs_review`. For example, the July process
confirmed the corporate-action mapping:

```text
TATAMOTORS -> TMPV.NS
```

### 7.4 JSON correctness

Each bundle must be valid JSON. Non-standard numeric values such as `NaN`,
`Infinity`, and `-Infinity` are rejected rather than being tolerated by Python's
default JSON parser.

Every nested numeric field is checked for finiteness.

### 7.5 Portfolio invariants

For every horizon from one through five years:

- The file exists.
- `horizon_years` matches its filename.
- `stocks` is a non-empty list.
- `num_stocks` equals the list length.
- Each stock contains `ticker`, `weight`, `current_price`, and `sector`.
- Tickers are non-empty and unique.
- Every ticker is present in the knowledge-asset universe.
- Weights and prices are positive.
- Weights total 1.0 within an absolute tolerance of 0.002.

### 7.6 Cross-horizon and temporal invariants

- All current bundles use one `data_through` date.
- The benchmark ends on that same date.
- Previous snapshots must be strictly older than current snapshots.
- A requested `--as-of` date is treated as a maximum price date.
- Candidate data cannot be in the future.
- Candidate data is no more than ten days old unless the validation contract is
  deliberately changed.

The `--as-of` option controls price-history cutoff. Current PE/PB values remain
build-time fundamentals rather than historical point-in-time fundamentals; this
limitation is explicit in documentation and the app.

### 7.7 Turnover approval gate

For each horizon, turnover is calculated as:

```text
0.5 * sum(abs(current_weight - previous_weight)) * 100
```

The default publication threshold is 30%. A candidate above the threshold fails
unless the operator explicitly uses `--allow-high-turnover`. An override changes
the result to a recorded warning; it does not hide the event.

The release report records:

- Turnover by horizon
- New picks
- Dropped securities
- Recommendation status
- Number of stocks
- Weight total

## 8. Provenance and release evidence

### 8.1 `research/manifest.json`

Every promoted research release records:

- Manifest schema version
- Release ID
- Build timestamp with timezone
- Requested as-of behavior
- Actual market-data date
- Source Git commit and branch
- Data provider
- Benchmark identification and review status
- Methodology version
- Python version
- Direct dependency versions
- Validation status
- SHA-256 hashes of generated artifacts

The manifest intentionally distinguishes the **research source commit** from the
later **application engineering commit**. At migration:

- `e1a23aa` produced the July research data.
- `04bad7a` added the transactional release system and deployed that same
  research without silently changing recommendations.

### 8.2 `research/validation_report.json`

The validation report is designed for machines, CI, and future monitoring. It
contains:

- Status and validation timestamp
- Data date and age
- Universe size
- Benchmark observation count
- Unresolved/stale count
- Maximum turnover and configured limit
- Warnings
- Per-horizon recommendation, stock count, weight total, turnover, picks, and
  drops

### 8.3 `research/refresh_summary.md`

The Markdown summary is optimized for human review and is used directly as the
monthly candidate pull-request body.

## 9. Reproducible environment

### 9.1 Python baseline

Python 3.12 is now consistent across:

- `runtime.txt`
- `pyproject.toml`
- local locked environments
- GitHub CI
- monthly GitHub automation

### 9.2 Dependency groups

`pyproject.toml` defines:

- Runtime dependencies: Streamlit, pandas, NumPy, matplotlib
- Refresh dependencies: yfinance, SciPy, statsmodels
- Test dependencies: pytest, Ruff

Direct dependencies are exactly versioned. `uv.lock` stores the complete
cross-platform 72-package resolution.

The canonical setup command is:

```bash
uv sync --frozen --all-extras
```

`--frozen` prevents an operator or CI runner from silently changing the lockfile
during installation.

## 10. GitHub CI/CD

### 10.1 CI workflow

`.github/workflows/ci.yml` runs for pull requests, `master`, and `codex/**`
branches. It:

1. Checks out the exact commit.
2. Installs Python 3.12.
3. Installs uv with caching.
4. Runs `uv sync --frozen --all-extras`.
5. Compiles production modules.
6. Runs Ruff against the hardened pipeline and tests.
7. Runs the full pytest suite.

The production hardening commit passed both local Windows tests and hosted Linux
CI before being fast-forwarded to `master`.

### 10.2 Monthly research candidate workflow

`.github/workflows/monthly-refresh.yml` supports both:

- Manual **Run workflow** execution
- A scheduled run on the first day of each month

Inputs:

- Optional `as_of` date
- Explicit `allow_high_turnover` approval

The workflow:

1. Installs the frozen environment.
2. Runs the transactional refresh.
3. Runs the regression suite against promoted workspace artifacts.
4. Uploads manifest, validation report, and summary as workflow evidence.
5. Creates or updates `automation/monthly-refresh`.
6. Opens a pull request using the generated refresh summary.

It does not directly push research to production. Merging the reviewed candidate
into `master` is the deployment decision.

### 10.3 Dependabot

Dependabot performs monthly reviews for:

- Python dependencies
- GitHub Actions dependencies

Dependency changes still need the same CI and methodology review as application
changes.

### 10.4 Branch protection

The workflows support a protected-branch review model, but GitHub branch
protection is a repository-owner administrative setting and is not encoded in
the repository. Owners should require the `test` check and pull-request review
for `master`.

## 11. Test architecture

### 11.1 Validation tests

`tests/test_validation.py` constructs a complete synthetic 50-stock candidate
and verifies:

- A valid candidate passes.
- Excessive turnover fails without approval.
- Approved excessive turnover remains visible as a warning.
- Non-finite JSON is rejected.

### 11.2 Runtime helper tests

`tests/test_core_loaders.py` verifies:

- Release-manifest loading
- Monthly allocation totals and whole-share calculations

### 11.3 Published artifact tests

`tests/test_published_artifacts.py` runs the validator against the actual
committed research release and checks cross-horizon date consistency.

### 11.4 Streamlit integration test

`tests/test_streamlit_app.py` renders, selects, and generates plans for horizons
1 through 5 in an isolated process.

The final hardening gate produced:

```text
8 passed
Ruff: all checks passed
GitHub hosted CI: success
```

## 12. Streamlit changes

The live app remains a static-artifact consumer. Reengineering changes include:

- `VRIDDHI_RESEARCH_DIR` support for candidate testing
- `load_release_manifest()` for provenance display
- A footer showing release ID and source commit
- Replacement of deprecated `use_container_width` arguments with current width
  APIs
- Removal of the conflicting repository-level CORS override
- More accurate description of the actual damped-Holt production methodology
- Explicit disclosure of point-in-time fundamentals, costs, and benchmark review
  limitations

The app remains private on Streamlit Community Cloud. Unauthenticated probes
correctly receive a redirect to Streamlit authentication.

## 13. Operating procedures

### 13.1 Preferred monthly operation

1. Open GitHub **Actions**.
2. Select **Monthly research candidate**.
3. Select **Run workflow**.
4. Leave `as_of` blank for the latest complete market date, or supply an explicit
   date.
5. Do not approve high turnover unless the change has been investigated.
6. Wait for the candidate build, validation, all-horizon smoke test, and pytest.
7. Review the generated pull request and evidence files.
8. Confirm ticker renames and recommendation-status changes.
9. Merge the candidate only after required checks pass.
10. Confirm the authenticated app footer shows the new release.

### 13.2 Local candidate build

```bash
uv sync --frozen --all-extras
uv run python vriddhi_monthly_refresh.py --candidate-only --yes
```

This leaves the candidate under `.refresh_staging/` and does not touch published
artifacts.

### 13.3 Local transactional promotion

```bash
uv run python vriddhi_monthly_refresh.py --yes
```

Review the Git diff before committing or pushing.

### 13.4 Explicit price cutoff

```bash
uv run python vriddhi_monthly_refresh.py --as-of 2026-07-17 --yes
```

This applies a price-history cutoff. It does not transform current fundamentals
into point-in-time historical fundamentals.

### 13.5 Restore the latest local backup

```bash
uv run python vriddhi_monthly_refresh.py --restore-last
```

Restoration changes local artifacts. Review and commit those changes if a remote
rollback is required.

## 14. Failure behavior and recovery

| Failure | Expected behavior | Operator action |
|---|---|---|
| Missing package | Abort during preflight before build | Run `uv sync --frozen --all-extras` |
| Market download failure | Candidate build fails; published files unchanged | Retry after provider recovery; inspect retained staging |
| More than five stale/unresolved tickers | Validation fails | Treat as provider outage or investigate universe changes |
| Invalid weights/schema/JSON | Validation fails | Fix builder or data; never hand-edit production JSON |
| Benchmark date mismatch | Validation fails | Investigate download alignment and provider completeness |
| Data older than ten days | Validation fails | Confirm market calendar/provider status or intentionally revise policy |
| Turnover above threshold | Validation fails unless explicitly approved | Review holdings, aliases, data, and methodology before override |
| Streamlit candidate regression | Smoke test fails; published files unchanged | Fix UI/bundle compatibility and rerun |
| Promotion filesystem failure | Backup restoration is attempted; process exits unsuccessfully | Verify restored artifacts and rerun after correcting filesystem issue |
| Git commit failure | Publication fails explicitly | Inspect Git state/identity; do not claim deployment |
| Git push rejection | Publication fails explicitly | Resolve remote/branch controls and push the validated commit |
| Deployment inaccessible without login | Expected for private app | Verify through an authenticated Streamlit session |

## 15. July 2026 migration evidence

The July refresh and subsequent hardening provided a real production rehearsal.

### 15.1 July research refresh

- Data through: 2026-07-17
- Universe: 50 stocks
- Ticker health: 49 unchanged, 1 confirmed rename
- Maximum published turnover: approximately 13.24%
- Published recommended horizons: 4 years and 5 years
- New published picks versus June: BPCL and EICHERMOT
- Published drops versus June: ADANIPORTS and HEROMOTOCO
- Initial all-horizon smoke checks: passed
- Research commit: `e1a23aa`

### 15.2 Hardening rehearsal

The new candidate-only pipeline was then run end to end:

- Locked dependency preflight: passed
- Candidate research build: passed
- Same-date previous-snapshot preservation: passed for all five horizons
- Artifact validation: passed
- Candidate manifest and hashes: generated
- Candidate UI smoke test: passed for all five horizons
- Candidate retained without changing published artifacts
- Local frozen-environment tests: 8 passed
- Hosted Linux CI: passed
- Engineering commit: `04bad7a`

The rehearsal observed that upstream provider values could produce a slightly
different same-date candidate. The engineering release intentionally preserved
the already-published July portfolios and added provenance evidence rather than
silently issuing a second portfolio recommendation on the same day. This is an
important example of separating an engineering deployment from an investment
methodology/data release.

## 16. Security and repository hygiene

The reengineering added:

- `SECURITY.md`
- `CONTRIBUTING.md`
- `.gitignore` entries for staging, backups, test caches, coverage, and secrets
- A canonical `README.md`
- MIT `LICENSE`
- A clear declaration that `streamlit_app.py` is the production UI
- Documentation identifying older Streamlit variants as historical prototypes

Git operations in the refresh stage only generated paths:

```text
grand_table_expanded.csv
ticker_aliases.json
research/
```

Unrelated working files are not included in monthly research commits.

## 17. Quantitative limitations that remain

Engineering hardening does not remove financial-model risk. The following items
remain explicit priorities:

1. **Benchmark total return:** `^NSEI` must be independently reconciled with, or
   replaced by, an authoritative Nifty 50 Total Return Index series before
   institutional-grade excess-return claims.
2. **Point-in-time fundamentals:** PE and PB are current build-time values, not
   historical values aligned to each backtest date.
3. **Survivorship bias:** the maintained current universe needs historical
   constituent data for unbiased long-period claims.
4. **Transaction modeling:** the annual cost drag is simplified and does not
   model investor-specific tax, spread, slippage, or partial-share constraints.
5. **Single provider:** Yahoo Finance is convenient but is not an exchange-grade
   licensed feed. Material production use should add reconciliation or a licensed
   source.
6. **Methodology governance:** changes to gates, optimizer constraints, forecasts,
   or universe construction require a new methodology version and regression
   comparison.

These are documented rather than hidden. They should be addressed through data
governance and quantitative research, not by adding application infrastructure.

## 18. Recommended next engineering steps

### Priority 1: Repository controls

- Enable GitHub branch protection for `master`.
- Require pull requests and the `test` check.
- Restrict force pushes and branch deletion.
- Pin third-party GitHub Actions to reviewed commit SHAs if organizational policy
  requires supply-chain immutability.

### Priority 2: Deployment verification

- Add an authenticated or non-sensitive build-version health mechanism.
- Verify that the deployed footer release ID equals the merged manifest.
- Record deployment completion separately from Git push completion.

### Priority 3: Data governance

- Select and license an authoritative benchmark/market-data source.
- Add cross-provider reconciliation thresholds.
- Store immutable raw monthly input snapshots where licensing permits.
- Add a historical constituent and point-in-time fundamentals dataset.

### Priority 4: Quantitative regression suite

- Add fixed-fixture golden portfolios.
- Add leakage tests for every train/test boundary.
- Add regime and alternate-start-date sensitivity reports.
- Add champion/challenger comparisons for methodology changes.
- Track recommendation stability and gate transitions over time.

## 19. Engineering acceptance criteria

The reengineered backend is considered operationally successful when:

- `uv sync --frozen --all-extras` succeeds from a clean checkout.
- Ruff and pytest pass.
- Candidate-only refresh leaves canonical artifacts unchanged.
- A failed candidate cannot be promoted.
- Validation evidence is generated for every release.
- All five horizons render using candidate artifacts.
- A same-date retry preserves the older monthly snapshot.
- High turnover cannot pass without explicit approval.
- Git publication failure produces a failing result.
- GitHub CI passes before master deployment.
- The remote manifest and live app report the intended release.

At the time of this reference, all locally enforceable criteria above passed for
the production hardening release. Branch protection, authenticated post-deploy
inspection, and authoritative benchmark/data-provider selection remain owner or
organization-level controls.

## 20. Quick command reference

```bash
# Recreate the exact environment
uv sync --frozen --all-extras

# Lint and test
uv run ruff check vriddhi_validation.py vriddhi_monthly_refresh.py tests
uv run pytest -q

# Safe candidate rehearsal
uv run python vriddhi_monthly_refresh.py --candidate-only --yes

# Full local transactional refresh
uv run python vriddhi_monthly_refresh.py --yes

# Explicit price cutoff
uv run python vriddhi_monthly_refresh.py --as-of YYYY-MM-DD --yes

# Explicitly reviewed high-turnover release
uv run python vriddhi_monthly_refresh.py --allow-high-turnover --yes

# Restore newest local backup
uv run python vriddhi_monthly_refresh.py --restore-last
```

## 21. Closing design principle

Vriddhi's backend should remain simple in production and rigorous before
production. The live app should continue to read small, validated, versioned
artifacts. Complexity belongs in the offline release plane, where it can be
tested, audited, rejected, and reproduced without affecting users.

That principle guided the reengineering and should guide future work.
