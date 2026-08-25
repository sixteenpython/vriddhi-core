# BTI simulation engine

## Purpose

The first web game uses a clearly labelled deterministic simulated market derived from the governed
Vriddhi knowledge release. The repository does not yet contain monthly point-in-time stock fundamentals
for an honest historical replay. Simulated forward play avoids presenting today's evidence as if it had
been known in the past.

## Independence rule

The market generator does not read the hidden reference portfolio or player holdings. It produces a
world independently; the player and Vriddhi reference engine separately decide inside that world. This
prevents the simulator from rewarding the reference simply because it generated the scenario.

## Determinism and causality

Every campaign freezes knowledge release, scenario version, simulation version and seed. Monthly returns
combine seeded market, sector and idiosyncratic factors with bounded drift and volatility calibrated from
the released knowledge asset. OHLC and event cards are derived from the same factors. Identical inputs
produce identical snapshots.

The engine exposes only month `m` before the move at `m`. Snapshot `m+1` is generated/revealed after the
move commits. No scorer or public endpoint can request a later snapshot. Every snapshot has a content hash.

## Derived market measures

PE, PB, PEG, forecast and uncertainty evolve under bounded, documented rules. Trailing volatility,
Sharpe, maximum drawdown, VaR and expected shortfall use only returns observed through the current month.
Events are simulated and labelled; they are not real news.

## Future historical mode

Historical play requires immutable monthly constituent lists, prices adjusted for corporate actions,
fundamentals, forecasts and promoted knowledge releases. It will implement the same scenario interface
without changing game rules or scoring contracts.
