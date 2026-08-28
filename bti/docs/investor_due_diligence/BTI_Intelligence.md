# BTI Intelligence

## Capital-market simulation, decision scoring and investor assurance report

- **Product:** BTI — Beat the Index
- **Release assessed:** v0.16.0
- **Simulation engine:** `bti-capital-market-2026-08-v5`
- **Game engine:** `bti-game-v4`
- **Decision scoring:** `bti-score-v2`
- **Report date:** 29 August 2026
- **Live application:** <https://beat-the-index.onrender.com/>

---

## 1. Executive conclusion

BTI is not a random-return game and it is not a disguised live trading terminal. It is a
versioned, auditable capital-market simulation whose starting state is automatically anchored to
the latest successfully promoted Vriddhi research release. From that common state, a seeded market
engine generates a sealed future containing market regimes, correlated equity moves, sector and
security-specific shocks, interest-rate effects, inflation effects, liquidity stress, cross-asset
behaviour, a broad-market benchmark, changing ratios, risk measures, news and daily OHLC paths.

The objective is **behavioural fidelity rather than price prophecy**. BTI does not claim to predict
the future closing price of BEL, Nifty 50, gold or another asset. It aims to preserve relationships
that make real capital allocation difficult:

- attractive growth can be offset by expensive valuation;
- concentration can produce exceptional gain or severe loss;
- equities, gold and bonds react differently to inflation, rates, liquidity and risk appetite;
- a strong process can lose over one realised path;
- a weak decision can occasionally win; and
- diversification changes the distribution of outcomes rather than guaranteeing success.

Every campaign is generated independently of the player's later holdings. The hidden Vriddhi
reference does not control the market against the player. It evaluates the decision using the
information available before the market advances; realised outcome is reported separately. That
separation is the foundation of BTI's fairness.

The current implementation is a strong, internally coherent educational simulator. It is not yet
a statistically certified digital twin of Indian capital markets. External validation should test
distributions, correlations, drawdowns, regime behaviour, strategy dominance and point-in-time
data integrity before any stronger realism claim is made. This report defines those tests.

---

## 2. Product claim and boundary

### 2.1 Supported claim

> BTI provides a realistic, internally coherent and reproducible simulated market in which players
> can practise portfolio construction, risk management and rebalancing against a simulated Nifty
> benchmark.

“Realistic” means that prices, ratios, risk, forecasts, news, asset-class behaviour and benchmark
returns move together under explicit economic rules. It does **not** mean that a simulated future
forecasts what the actual market will do.

### 2.2 Claims BTI must not make

BTI must not claim that:

- displayed prices are live quotes;
- a simulated future predicts actual security prices;
- a high score recommends buying or selling a security;
- a player who wins is ready to trade real money;
- the hidden reference is unbeatable;
- a `BLUNDER` must lose or an `EXCELLENT` move must win; or
- historical-looking realism itself proves statistical calibration.

The application therefore persistently identifies itself as **SIMULATION MODE** and introduces the
game through a mandatory educational-use consent screen.

---

## 3. Intelligence architecture

```text
Latest promoted Vriddhi release
        ↓ verified manifest and hashes
50-stock research universe + horizon portfolios
        ↓ campaign creation freezes the baseline
Seeded regime schedule + market state
        ↓ sealed path independent of player holdings
Equity / bond / gold / benchmark evolution
        ↓ coherent derived state
Prices + OHLC + ratios + risk + forecasts + news
        ↓ player observes only public intelligence
Complete portfolio decision
        ↓ pre-outcome assessment
Hidden Vriddhi-derived reference + deterministic scoring
        ↓ market advances
Realised wealth, alpha, drawdown and XIRR
```

Four authorities are deliberately separated:

1. **Vriddhi research** owns the promoted starting universe, metrics and horizon portfolios.
2. **The simulation engine** owns the future market path and generated market state.
3. **The scoring engine** owns decision quality, regret, classification and position evaluation.
4. **The game engine** owns cash, holdings, accounting, progression, results and rating.

The React/PWA client presents information and maintains a reversible draft move. It does not own
financial truth, scoring or campaign settlement.

---

## 4. Data provenance and automatic synchronisation

BTI does not permanently encode “14 August 2026” or another fixed snapshot. Its baseline is:

> **the latest successfully promoted and verified Vriddhi release available to the deployed build.**

At startup, the artifact adapter:

- reads the Vriddhi release manifest;
- requires passed/promoted release status;
- identifies release ID and as-of date;
- verifies the expanded 50-stock table;
- verifies 2-year, 3-year, 4-year and 5-year portfolio artifacts;
- checks hashes, allowing only harmless line-ending normalisation;
- requires exactly 50 securities; and
- fails closed if a release is missing, incomplete or corrupt.

Production exposes the active release at:

<https://beat-the-index.onrender.com/api/v1/health>

When the monthly Vriddhi workflow promotes a new release and that commit reaches the deployment
branch, Render rebuilds BTI. The next process start validates and adopts the new release without a
manual data-copy step. An existing campaign retains its frozen release, seed and market path; a new
campaign receives the latest baseline. Thus a refresh cannot rewrite the past of a rated game.

This is **monthly research freshness**, not a real-time quote feed.

See [Capital Market Intelligence Assurance and Refresh](./CAPITAL_MARKET_INTELLIGENCE_ASSURANCE_AND_REFRESH.md)
for the full operating contract.

---

## 5. Reproducibility and fairness

Each campaign receives a seed. Random streams are namespaced by hashing the seed together with the
operation and its identifiers. Therefore:

- the same engine version, artifacts, seed and actions reproduce the same result;
- different seeds can produce different futures;
- stock-order iteration does not change unrelated random streams;
- the entire regime schedule is generated when the campaign is created; and
- player holdings cannot cause the engine to retaliate or reward the player.

This creates a chess-like fairness contract: the position is difficult because of the market model,
not because the system changes the future after seeing the move.

A campaign freezes its source release and date, engine versions, seed, mode, horizon, initial state,
reference inputs and regime schedule. An auditor can distinguish a changed model from a changed
seed and a changed decision.

---

## 6. Market-regime engine

BTI models six recurring regimes. Each provides market drift, volatility and narrative context;
month-specific inflation, rates, risk appetite and liquidity add variation.

| Regime | Bias | Volatility multiplier | Intended behaviour |
|---|---:|---:|---|
| Selective growth | +0.10 | 0.95 | Quality growth is rewarded, but valuation matters. |
| Sector rotation | 0.00 | 1.08 | Leadership changes; concentration becomes costly. |
| Earnings dispersion | +0.02 | 1.15 | Security selection matters more than broad beta. |
| Risk off | -0.22 | 1.25 | Equity risk rises; liquidity and resilience matter. |
| Valuation reset | -0.12 | 1.18 | Expensive expectations compress. |
| Recovery | +0.18 | 1.05 | Risk appetite improves without erasing prior damage. |

Difficulty rises gradually with campaign progress. The path is never selected because the player is
winning or losing.

**Current boundary:** regimes and transitions are designed rules, not yet an empirically fitted
hidden Markov model. Statistical certification requires validating their frequency, duration and
transition probabilities against point-in-time Indian-market history.

---

## 7. Equity simulation

Every equity starts from promoted Vriddhi data: price, sector, PE, PB, PEG, risk-adjusted return,
historical growth, rank and horizon forecasts.

For security *i* in month *t*:

```text
shock(i,t) = 0.58 × market shock(t)
           + 0.25 × sector shock(sector(i),t)
           + 0.57 × idiosyncratic shock(i,t)

return(i,t) = forecast drift(i,t) / 12
            + annual volatility(i,t) / √12 × shock(i,t)
            - reversal(i,t) × previous return(i,t)
```

These coefficients are factor loadings, not weights and need not total one. Monthly returns are
capped at ±24% to prevent numerical pathologies. Reversal becomes stronger following an extreme
prior move, reducing runaway momentum while preserving trends.

The model therefore contains common market movement, sector co-movement, company dispersion,
forecast-linked drift, volatility-scaled uncertainty and partial mean reversion. It does not decree
that low PEG or high Sharpe must win. Regimes, concentration, shocks and changing forecasts can
defeat a static screen.

---

## 8. Cross-asset simulation

Rapid and Blitz support government bonds, corporate bonds and gold alongside equities.

### Government bonds

Returns combine yield carry, duration loss when rates rise, low-volatility idiosyncratic movement
and a modest risk-appetite effect. Monthly returns are capped between -5.5% and +5.5%.

### Corporate bonds

Returns combine higher carry, duration sensitivity, moderate idiosyncratic movement and credit
stress under weak liquidity or risk appetite. Monthly returns are capped from -7.5% to +6.5%.

### Gold

Gold combines base carry, a positive inflation response, negative rate and risk-appetite responses,
and substantial independent uncertainty. Monthly returns are capped at ±14%.

Gold can hedge some conditions and disappoint in others. It is not hard-coded to lose. An all-Gold
bet loses diversification and growth exposure, so it has a weak long-run process even though a
favourable realised gold regime may occasionally outperform.

---

## 9. Benchmark construction

The simulated Nifty participates in the same market as the securities; it is not a fixed CAGR line.
Each month the engine:

1. sorts the 50 equity returns;
2. removes the five highest and five lowest observations;
3. computes the broad trimmed-market mean;
4. creates a macro return from long-run drift and the common shock; and
5. combines 85% broad market with 15% macro return.

Benchmark return is capped at ±18% monthly. Player holdings and Nifty share the same regime, while
concentrated and cross-asset portfolios retain distinct exposures.

**Current boundary:** this is a simulated broad-market proxy, not exact replication of point-in-time
Nifty 50 TRI membership and free-float weights.

---

## 10. A coherent market screen

BTI does not generate a price and decorate it with unrelated numbers. After each market move, common
state updates:

- current price and return;
- a 21-session OHLC path reconciling exactly to the monthly close;
- simulated volume, momentum, RSI and sentiment;
- PE, PB, PEG and forecast curve;
- realised volatility, Sharpe and maximum drawdown;
- 95% VaR and expected shortfall; and
- market, sector, risk and security news.

The initial view includes a deterministic 252-session lookback ending at the starting price. Each
new month contains 21 daily candles whose log returns are adjusted so the final candle closes at the
authoritative monthly price. The chart therefore reconciles with the portfolio ledger.

PE and PB move with price. PEG responds more gradually; forecasts decay and react to realised
returns. This is screen-level coherence, not a complete three-statement company model: earnings,
book value and analyst revisions are not yet independently simulated.

Volatility and drawdown come from the realised path. VaR and expected shortfall are parametric
volatility-based approximations, appropriate as game signals rather than certified tail estimates.

---

## 11. News and narrative

Newswire stories derive from public state: momentum, forecast, valuation, Sharpe, sentiment,
sector, holdings, regime and events. Headline and table therefore describe one market.

The design creates deliberate but fair distraction. A visible winner may be expensive; an alarming
headline may concern a holding whose long-term health remains intact. BTI teaches:

> **A headline is information, not a complete investment thesis.**

Current news mostly interprets generated state. It is not yet a fully causal independent event
engine in which a surprise first changes revenue, margins or supply and then propagates to price and
ratios. This is a future enhancement and a current model-risk disclosure.

---

## 12. Hidden reference portfolio

The hidden reference begins with Vriddhi's promoted portfolio for the selected horizon and adapts
modestly to public state. Its equity signal combines forecast growth, inverse-PEG valuation and
Sharpe/volatility resilience.

Targets are approximately 72% anchored to promoted Vriddhi weights and 28% responsive to the
current signal. Rapid and Blitz may reserve regime-aware weights for gold, government bonds and
corporate bonds. A whole-share solver converts targets into executable holdings and uses residual
cash where feasible.

The reference is hidden to preserve gameplay. It cannot read the next realised move and is a
decision benchmark—not a claim of omniscience.

---

## 13. Move scoring: the Stockfish-like layer

BTI evaluates a complete portfolio decision. It calculates weighted forecast, weighted marginal
volatility, stock and sector Herfindahl concentration, inverse-PEG valuation quality and a modest
cross-asset bonus.

```text
utility = 2.25 × forecast
        + 8.00 × valuation
        - 0.38 × risk
        - 30.0 × stock concentration
        - 12.0 × sector concentration
        + cross-asset bonus

regret = max(0, reference utility - player utility)
decision score = max(0, 100 - 3 × regret)
```

| Score | Classification |
|---:|---|
| 98–100 | Brilliant |
| 90–97.99 | Excellent |
| 80–89.99 | Good |
| 65–79.99 | Inaccuracy |
| 45–64.99 | Mistake |
| Below 45 | Blunder |

The chess-like position evaluation maps score around a neutral process level and caps its display at
approximately ±3.

**Current limitation:** risk utility uses weighted marginal volatility, not a full covariance
matrix. It teaches diversification and concentration but is not yet covariance-aware portfolio VaR.

---

## 14. Process and outcome are separate

BTI first freezes public information, assesses the completed portfolio, records the classification,
then advances the sealed market and reports realised wealth. Consequently:

- a blunder does not guarantee failure;
- a win does not prove the move was good;
- a good move does not guarantee success; and
- repeated good decisions should improve probability, not guarantee outcome.

This is the essential educational contract: grade the position chosen without hindsight, then allow
the market to realise one possible continuation.

---

## 15. The paired all-Gold Blitz experiment

Playtesting submitted the same deliberately weak strategy in two independent two-year Blitz
campaigns: invest the full ₹1 lakh in gold and make no further decision.

| Session | Player | Simulated Nifty | Result |
|---|---:|---:|---|
| Mobile | about 4.9% CAGR | about 23.8% CAGR | Lost materially |
| Web | ₹1.1 lakh; 5.41% CAGR | ₹1.7 lakh; 29.69% CAGR | Lost materially |

Numbers differed because seeds differed. The lesson remained similar because both paths shared
long-horizon equity growth, regime-sensitive gold and the risk of a single-asset bet.

This supports internal coherence: BTI is not returning a memorised outcome; independent paths can
differ while preserving a structural lesson. It does **not** prove empirical fidelity, validate tail
behaviour or imply that gold must always lose. If gold could never win under inflation shock or
equity stress, the simulator would be rigged. Validation must ask whether outcomes occur with
plausible frequencies across thousands of seeds.

---

## 16. Strengths

1. New campaigns use a current, verified research baseline.
2. The sealed future is independent of player holdings.
3. Version, artifacts, seed and actions make campaigns reproducible.
4. Market, sector and company factors coexist.
5. Bonds and gold behave differently from equities.
6. Nifty participates in the same market.
7. OHLC, ratios, risk, forecasts and news derive from common state.
8. Scoring is point-in-time and cannot use the next outcome.
9. Decision quality and luck remain separate.
10. Invalid artifacts fail closed.
11. Refreshes cannot rewrite active campaigns.
12. Simulation disclosure is a first-class product boundary.

---

## 17. Model risks and maturity gaps

| Area | Current implementation | Required maturity step |
|---|---|---|
| Regimes | Six designed regimes | Fit and validate transition/duration distributions. |
| Loadings | Hand-calibrated factors | Estimate rolling betas and residual correlation. |
| Return caps | Fixed monthly caps | Validate tails and disclose clipping frequency. |
| Fundamentals | Ratios evolve with price/forecast | Model earnings, book and margins independently. |
| Portfolio risk | Marginal vol + concentration | Add covariance-aware risk and stress tests. |
| VaR / ES | Parametric approximation | Calibrate non-Gaussian tails. |
| Benchmark | Trimmed broad-market proxy | Add point-in-time Nifty TRI methodology if licensed. |
| Cross-assets | Designed sensitivities | Calibrate rates, spreads and commodity regimes. |
| News | Mostly state-derived | Add causal event-to-fundamental propagation. |
| Data rights | Prototype/public-source pipeline | Establish production licensing and provenance. |
| Freshness | Monthly promoted baseline | Never describe it as intraday/live market data. |
| Validation | Correctness tests and playtests | Add large-seed and independent model validation. |

These are not hidden weaknesses. They define the boundary between a credible MVP and an
institutionally validated market model.

---

## 18. External validation programme

### 18.1 Deterministic correctness

- Replay identical seeds and actions for byte-equivalent transitions.
- Change only the seed and confirm paths differ.
- Change holdings while retaining the seed and confirm the market path does not.
- Reconcile cash, holdings and value after every move.
- Reconcile daily OHLC closes with monthly prices.
- Independently recompute benchmark and risk measures.

### 18.2 Point-in-time integrity

- Confirm scoring reads only pre-move state.
- Confirm future shocks and future releases are unavailable to the reference.
- Confirm active campaigns remain frozen across monthly refreshes.
- Independently verify artifact hashes.

### 18.3 Distributional realism

Run at least 10,000 seeds per horizon and compare with point-in-time Indian-market history:

- monthly and annual return distributions;
- volatility clustering and autocorrelation;
- skew, kurtosis and extreme loss;
- drawdown depth and recovery duration;
- cross-sectional and sector dispersion;
- equity/bond/gold correlations by regime;
- benchmark tracking behaviour; and
- regime frequency, duration and transition.

### 18.4 Strategy tournament

Pre-register and run all-cash, all-gold, all-bond, equal-weight, minimum-PEG, maximum-Sharpe,
momentum, prior-winner, concentrated, random and diversified Vriddhi-style strategies through the
same seeds.

The goal is not that Vriddhi wins every path. It is that no trivial heuristic dominates all regimes
and disciplined, diversified, valuation-aware methods improve the distribution of long-horizon
outcomes.

### 18.5 Sensitivity and anti-exploit testing

- Perturb coefficients and measure result stability.
- Count how often caps bind.
- search automatically for dominant strategies and scoring exploits;
- test residual cash and whole-share edge cases;
- verify good-score/bad-outcome and bad-score/good-outcome cases; and
- verify web and mobile operate on the same backend truth.

---

## 19. Proposed investor acceptance gates

1. **Replay:** identical version, artifacts, seed and actions reproduce exactly.
2. **Independence:** player holdings cannot change the sealed future.
3. **Accounting:** cash and holdings reconcile throughout every mode.
4. **Point in time:** no future information enters scoring.
5. **Distribution:** return, volatility, correlation and drawdown sit within pre-agreed tolerances.
6. **Tail:** crash frequency and severity are plausible and not dominated by caps.
7. **Strategy:** no simple heuristic wins implausibly across nearly all regimes.
8. **Process:** stronger scores improve distributions without guaranteeing wins.
9. **Refresh:** each promoted Vriddhi release is adopted, exposed in health and frozen per campaign.
10. **Licence:** production data and derived use are legally cleared.

Statistical tolerances should be agreed with the independent reviewer before results are examined.

---

## 20. Version and change governance

Any outcome-changing modification must increment the relevant simulation, game or scoring version.
Every campaign also records its Vriddhi release. Rated campaigns must never migrate silently.

Model changes should require a written hypothesis, regression and statistical validation, a
strategy-tournament comparison, reviewed assumptions, release notes and the ability to replay old
campaigns with their original versions.

---

## 21. Recommended 30-day diligence roadmap

### Week 1 — Reproducibility and audit pack

Export canonical fixtures, document every coefficient, prove path independence and scoring timing,
and add complete accounting/OHLC reconciliation reports.

### Week 2 — Statistical harness

Run multi-seed simulations across all modes and horizons; produce distribution, correlation, regime,
tail and drawdown diagnostics.

### Week 3 — Strategy tournament and calibration

Pre-register strategies, find exploits and calibrate only against declared targets while preserving
an untouched validation period.

### Week 4 — Independent review

Deliver code, formulas, data provenance and results to the independent quant reviewer, resolve
material findings, freeze the reviewed version and publish only validated assurance claims.

---

## 22. Source traceability

| Responsibility | Canonical implementation |
|---|---|
| Regimes, paths, assets, benchmark, OHLC and market state | [`simulation.py`](../../game_engine/simulation.py) |
| Campaigns, reference, accounting and settlement | [`engine.py`](../../game_engine/engine.py) |
| Utility, regret, classifications and health | [`scoring.py`](../../game_engine/scoring.py) |
| Newswire interpretation | [`Newswire.tsx`](../../frontend/src/Newswire.tsx) |
| Research identity and hashes | [`research/manifest.json`](../../../research/manifest.json) |
| Refresh and assurance | [`CAPITAL_MARKET_INTELLIGENCE_ASSURANCE_AND_REFRESH.md`](./CAPITAL_MARKET_INTELLIGENCE_ASSURANCE_AND_REFRESH.md) |

Source code is the final authority if prose and implementation differ.

---

## 23. Final assessment

BTI combines four uncommon properties: a current verified research anchor, a coherent reproducible
multi-asset simulation, a hidden point-in-time decision benchmark and an explicit separation of
skill from luck.

The paired all-Gold experiment is a useful testimonial to internal coherence, not the end of due
diligence. The strongest supportable investor conclusion today is:

> **BTI is a serious, auditable capital-allocation training simulator whose market engine behaves
> coherently enough to warrant formal independent validation. It teaches that beating the index is
> difficult but possible through disciplined methods; it does not pretend that any method can
> eliminate uncertainty.**

That is both a compelling product proposition and a defensible technical claim.
