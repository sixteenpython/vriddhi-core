# Immersive web experience

## Experience objective

During active decision-making, BTI should resemble a professional portfolio intelligence desk. An
observer should see a serious research and execution environment rather than a casual game screen.
The persistent **SIMULATION MODE** label remains non-negotiable so realism never becomes deception.

## Implemented shell

- persistent dark left navigation rail;
- campaign identity and move counter in the top control bar;
- repeated simulation disclosure strip;
- desktop-first, responsive layout;
- professional contextual navigation during a move:
  - Overview;
  - Market Monitor;
  - Order Workbench;
  - Performance;
  - Scenario Lab;
  - Research Academy;
  - Analyst Rankings;
  - Newswire.

## Landing and campaign setup

The landing page explains the core challenge and provides campaign/puzzle entry. Setup exposes
monthly contribution and 24/36/48/60-month horizon, previews total contribution and move count, and
states the immutable campaign rules.

## Market workstation

The market surface is a dense, terminal-style information environment containing:

- generated Nifty, Bank Nifty, VIX, currency, crude and gold tape;
- market breadth, median PEG, mean forecast, cross-sectional volatility, cash and alpha;
- search plus value, quality, low-risk and held-security filters;
- Excel-like column sorting across ticker, price, monthly move, valuation, risk and forecast;
- simultaneous per-column filters for sector, maximum PE/PB/PEG/volatility/drawdown and minimum
  Sharpe/forecast, alongside the global search and strategy lenses;
- 50-security grid with price, move, sparkline, PE, PB, PEG, Sharpe, volatility, drawdown, VaR and forecast;
- ranked top-five monthly gainers and losers beside the security grid;
- sector monitor;
- public-signal quant radar;
- generated newswire summaries.

Selecting a security opens a dedicated research workstation with an 18-period generated OHLC
candlestick chart, five-horizon forecast curve, public rank, valuation/risk matrix, relevant
generated catalysts, sector peers and explicit portfolio questions. The public board never marks
the hidden best move.

## Newswire

The simulated newsroom provides a macro lead story, sector and earnings cards, market pulse,
trending securities and generated event context. Every article and tape is locally marked as
generated/simulated. News changes context; it does not directly dictate the move.

The Newswire also carries separate ranked top-gainer and top-loser tables for the current simulated
month. Positive and negative lists are directionally strict; before the first market advance the UI
shows an awaiting-first-move state rather than mislabelling flat securities.

## Order workbench

The order-management screen includes:

- buying power, order count, deployment, post-trade cash, positions and concentration;
- quick security ticket with price, PEG, Sharpe and forecast;
- whole-share BUY/SELL blotter;
- collision-free quantity steppers with separately aligned simulated price and notional columns;
- estimated notional and portfolio effect;
- projected holdings and allocation bars;
- projected sector exposure;
- weighted PE, PEG, Sharpe, forecast and volatility;
- concentration and scenario lenses;
- cash reconciliation and 90% deployment gate.

The UI estimates are advisory previews. The server performs the authoritative validation and
commit. The player should net related actions into one final security instruction rather than sell
and repurchase the same security unnecessarily.

## Analysis and learning surfaces

The release includes move-result analysis, campaign progress, puzzles, lessons, rankings/profile
shells and final-result concepts. Their product depth is still being expanded through current user
testing. No screen should claim live social, brokerage or real-news functionality.

## Responsive principle

Desktop is the canonical experience during this phase. At narrower widths, panels stack, tapes and
tables scroll deliberately, and the execution blotter remains usable. Native Android design begins
only after the web interaction model and rules are stable.
