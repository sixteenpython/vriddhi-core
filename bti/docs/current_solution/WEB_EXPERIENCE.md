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
- exactly two active-campaign destinations: **Market Monitor** and **Game Board**.

## Landing and campaign setup

The landing page explains the core challenge and provides campaign entry. Setup exposes
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
- a sticky, portfolio-aware construction ribbon showing buying power, net orders, deployment,
  residual cash, projected positions/sectors and top-weight concentration;
- inline share-delta controls, HOLD and EXIT actions, and projected post-move holdings for every
  security;
- a configurable intelligence deck with fundamental, technical, quantitative and sentiment
  presets; tiles can be hidden, widened, reordered and restored;
- full-screen generated OHLC exploration over 30D/90D/6M/1Y windows and daily/weekly/monthly views,
  with volume, crosshair values and explicit simulation provenance.

Selecting a security opens a dedicated research workstation with a 252-trading-day generated OHLC
candlestick chart, five-horizon forecast curve, public rank, valuation/risk matrix, relevant
generated catalysts, sector peers and explicit portfolio questions. Fundamental, technical and
sentiment signals include ROE, earnings growth, margin, leverage, yield, beta, RSI, momentum,
activity and sentiment. The public board never marks the hidden best move.

## Embedded news intelligence

News and sentiment are signals inside Market Monitor and stock research, not a separate destination.
They provide generated macro, sector, earnings, mover and security context. News changes context; it
does not dictate the move.

## Rated Game Board

The Game Board combines final order review, irreversible execution, move analysis and progress:

- buying power, order count, deployment, post-trade cash, positions and concentration;
- sell-first then buy execution sheet with running cash;
- cash reconciliation and 90% deployment gate.
- deterministic 0–100 move quality and chess-like signed position evaluation;
- separate decision-quality and next-market-outcome explanations;
- a cricket-chase chart where Nifty supplies the projected target and the player advances one
  completed over per committed monthly move;
- immutable move cards and read-only historical reconstruction with chart rewind;
- exact repeat-last-move from Market Monitor, subject to the new month’s holdings, prices, cash and
  deployment rules.

The UI estimates are advisory previews. The server performs the authoritative validation and
commit. The player should net related actions into one final security instruction rather than sell
and repurchase the same security unnecessarily.

## Release boundary

Puzzles, lessons, rankings, leaderboards, profile shells, scenario labs and standalone Newswire are
not part of this web release. Learning occurs through the market signals, permanent decisions,
deterministic feedback and historical review in the core game. No screen claims live social,
brokerage or real-news functionality.

## Responsive principle

Desktop is the canonical experience during this phase. At narrower widths, panels stack, tapes and
tables scroll deliberately, and the execution blotter remains usable. Native Android design begins
only after the web interaction model and rules are stable.
