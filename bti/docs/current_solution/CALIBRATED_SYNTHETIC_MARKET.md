# Calibrated synthetic market

## Product position

BTI uses real listed-company names and tickers inside a newly generated market. It does not replay,
perturb or conceal an identifiable historical trading day and it does not present delayed exchange
data as fictional. Every playable price, volume proxy, ratio state, forecast, event, news item and
outcome remains part of the simulation.

The design objective is **behavioural realism without factual impersonation**.

## Generation model

The governed Vriddhi knowledge asset supplies the cross-sectional starting relationships. The BTI
simulation then produces:

- 252 deterministic pre-campaign daily OHLC candles per security;
- 21 daily candles bridging every investment month;
- market, sector and idiosyncratic shocks;
- bounded drift derived from the public forecast state;
- cross-sectional volatility, Sharpe, drawdown, VaR and expected shortfall;
- volume activity, momentum, RSI, beta and sentiment signals;
- internally related PE, PB, PEG, ROE and earnings-growth measures;
- company/sector/risk news generated from the current simulated state;
- a deterministic benchmark independent of the player's holdings.

The same campaign seed, engine version, research release and player instructions reproduce the same
world. Player holdings cannot influence the generated market.

## Coherence rules

- OHLC obeys `low <= open/close <= high`.
- Monthly daily candles bridge exactly from the previous close to the new simulated close.
- ROE is kept consistent with the PE/PB relationship used in this release.
- Earnings growth, valuation and forecast signals remain bounded and interpretable.
- Sentiment is a signal derived from market state; it is not a buy/sell instruction.
- News describes the generated state and cannot silently overwrite deterministic calculations.
- Every chart and story retains visible simulation language.

## Provenance boundary

The public payload exposes `simulation_version` and the generated information required by the
client. It never exposes the campaign seed, future realised states, hidden reference weights or
scoring coefficients.

The current version is `bti-calibrated-synthetic-2026-08-v2`.

## Future calibration

Future releases may calibrate distributions from appropriately licensed or public aggregate market
research. Calibration must store source/licence metadata and statistical parameters—not copy an
undisclosed historical path and relabel it as hypothetical.
