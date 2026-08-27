# Playtest 05 — Rapid/Blitz Market Replay and Full Newswire

## Release

BTI Market Replay v0.13.0 promotes Rapid and Blitz from a close-value animation into a synchronized simulated market journey. The accounting engine and final campaign outcome remain authoritative and unchanged.

## Gameplay correction

- Rapid and Blitz now progressively reveal only the months already reached in the replay.
- The player portfolio is represented as deterministic NAV OHLC candles.
- Nifty is represented by its synchronized simulated OHLC close path.
- Hover on desktop and tap on mobile reveal month, OHLC, alpha, drawdown, regime and event context.
- Pause, resume, 1×/2× speed and skip-to-stop controls affect presentation only.
- The result, move score and future months remain sealed until the replay reaches its stop.
- Rapid pauses after each annual segment with **Care to rebalance?** and a 120-second decision window for the next move.
- Blitz remains a single irreversible allocation with no intervention during its full replay.

## Newswire correction

The compact Market Monitor feed remains the trailer. The dedicated Newswire is the full simulated newsroom, including:

- market breadth and sentiment;
- portfolio-specific coverage;
- macro, risk, earnings, sector and quantitative desks;
- top gainers and losers;
- valuation, quality, low-risk and momentum signal breadth;
- current-month evidence boundaries and ticker drill-down.

## Integrity boundary

Monthly closes continue to come from the deterministic capital-market simulation. Presentation OHLC excursions are stable derivatives of campaign seed, month and regime and cannot change accounting, score, rating or final result. News events are synchronized to the revealed month and provide no future information.

## Acceptance gates

- Frontend unit suite and production TypeScript/Vite build.
- Full backend/API/contract suite.
- Complete Rapid and Blitz browser playthroughs.
- Rapid stop/rebalance prompt and no-future-leak inspection.
- Desktop hover and mobile tap/scroll validation.
- Newswire density, filtering, ticker drill-down and responsive overflow checks.
