# Intelligence and simulation design

## Vriddhi as the hidden analytical engine

`VriddhiArtifacts` reads the promoted Vriddhi research manifest, the expanded 50-stock knowledge
table and horizon-specific portfolio bundles. These artifacts provide the governed information from
which BTI constructs its playable decision universe.

The browser may see public company metrics. It must never receive:

- horizon reference weights or holdings;
- reference utility or regret;
- optimiser output or efficient-frontier coordinates;
- coefficients and scoring thresholds;
- scenario seed or future market states.

## Initial information set

For each security, the engine constructs an initial generated market containing:

- OHLC and price history;
- sector;
- PE, PB and PEG;
- forecast percentage;
- volatility and Sharpe;
- running drawdown;
- 95% VaR and expected shortfall.

Real ticker identities are retained. The displayed numbers are gameplay data and are explicitly
labelled simulated.

## Hidden feasible reference

For the chosen horizon, BTI starts from Vriddhi's governed portfolio weights. Each month it blends
those weights with current public signals for growth, valuation and resilience, normalises them,
then solves a whole-share portfolio feasible for the campaign's actual capital.

Residual cash is assigned iteratively to the security that most reduces target underweight among
affordable shares. This matters because a ₹10,000 campaign and a ₹1,00,000 campaign do not share the
same feasible whole-share solution.

The reference is an oracle for evaluation—not a recommendation exposed to the player.

## Portfolio quality

The current `bti-score-v1` utility considers:

- weighted simulated forecast;
- valuation efficiency based on PEG;
- weighted volatility;
- company concentration;
- sector concentration.

The player's portfolio is compared with the hidden feasible reference. The utility gap becomes
regret; regret becomes a bounded score. Multiple economically equivalent portfolios can score well,
so BTI evaluates quality rather than exact weight imitation.

## Decision versus outcome

The sequence is deliberately ordered:

```text
current information
  -> execute player move
  -> construct hidden feasible reference
  -> score decision quality
  -> advance market
  -> calculate portfolio and Nifty outcome
```

This prevents hindsight. A high-quality decision can experience a poor next month, and a weak
decision can get lucky. BTI explains both without confusing them.

## Deterministic forward market

`bti-forward-2026-08-v1` derives an independent pseudo-random stream from the campaign seed, month,
sector and ticker. Each monthly return combines:

- a common market shock;
- a sector shock;
- an idiosyncratic company shock;
- bounded forecast-derived drift;
- bounded annualised volatility.

Returns are capped to control unusable scenario extremes. The engine then updates OHLC, price
history, PE, PB, PEG, forecast, realised volatility, Sharpe, drawdown, VaR and expected shortfall.
The simulated Nifty uses the common factor with its own drift and volatility assumptions.

At campaign creation the public projection also builds an 18-period deterministic pre-campaign
OHLC lookback from the opening quote and artifact volatility. It exposes a 12/24/36/48/60-month
forecast term structure derived from the governed Vriddhi columns. These are generated research
surfaces for charting—not historical/live exchange observations—and contain no hidden reference
weights or future realised prices.

Identical seed, release, horizon and inputs produce identical results. The simulator cannot inspect
the player's portfolio and therefore cannot reward or punish a chosen stock.

## Versioning

The engine stores `engine_version`, the Vriddhi `release_id`, simulation version in public market
data and scoring version in move results. A saved campaign that requires an incompatible engine or
research release is rejected instead of being silently reinterpreted.
