# Playtest 07 — Release-Candidate UX and Gameplay

**Release:** BTI Release Candidate v0.14.0

**Date:** 27 August 2026
**Purpose:** Freeze the cleanest shippable web and portrait-mobile experience for the final investor playtest window

## Release decision

This increment deliberately avoids expanding game scope. It corrects the experience around the already-validated capital-market intelligence: loading, replay control, responsive rendering, completed-journey evidence and desktop news density.

The capital-market engine remains unchanged in authority. Classic rewards repeated decision discipline; concentrated Blitz allocations and poorly planned Rapid interventions retain their realistic exposure to volatility, drawdown and regime risk.

## 1. Faster perceived loading

- Removed a duplicate campaign-list API call from every authenticated page refresh.
- Campaigns and player profile now load concurrently.
- The preferred market begins loading as soon as the campaign list identifies the resumable game; no redundant campaign-state request sits in front of it.
- A committed move is revealed immediately; market and profile enrichment refresh in the background instead of holding the player behind a second spinner.
- Replaced the generic spinning wheel with a market-desk loading scan so cold-start waiting is informative rather than visually broken.
- Local warm-browser refresh reached the usable Home experience in approximately 1.3 seconds during release testing.

Render's free-instance cold-start delay remains an infrastructure boundary. This release minimizes redundant application work after the instance responds.

## 2. Replay is now a controllable experience

Rapid and Blitz market travel now provide:

- pause and resume;
- one-month rewind;
- replay from the beginning;
- 1× and 2× speed;
- skip to the next stop;
- an explicit **Open rebalance stop** or **Open final result** action when travel completes.

The result no longer replaces the replay automatically. The player may inspect and replay the completed travel before opening the verdict. Future months and final outcomes remain sealed until the path reaches its end.

## 3. Mobile chart containment

The previous mobile replay intentionally used a 760-pixel horizontal canvas. That made detailed candles available but allowed the chart to travel beyond the portrait screen and encouraged accidental sideways navigation.

The release candidate instead:

- fits the complete replay SVG inside the mobile content width;
- preserves the full horizon through the SVG view box;
- keeps tap-based candle inspection;
- places the OHLC evidence panel after the chart rather than over it;
- wraps compact replay controls for narrow screens;
- prevents the document itself from developing horizontal overflow.

Measured at a 412 × 915 portrait viewport:

- application width: 397 pixels;
- replay container: 339 pixels;
- replay SVG: 337 pixels;
- OHLC readout top: exactly the replay SVG bottom;
- no page-level horizontal overflow.

## 4. Completed campaigns show completed market evidence

The former final summary reused the cricket-chase chart. Its projected Nifty path and sparse annual Rapid points could appear as flat purple and green lines, which was not an adequate record of the simulation the player had just watched.

The final report now uses a dedicated **Completed Market Journey**:

- portfolio NAV OHLC candles for every simulated month;
- synchronized Nifty close path;
- month-level hover on desktop and tap on mobile;
- open, high, low, close, Nifty value and alpha readout;
- responsive full-width rendering;
- no projected future line after campaign completion.

The backend now exposes a canonical `journey_series` assembled from the immutable move results. Presentation fallback candles support older stored campaigns, but all new v0.14 campaigns use the deterministic OHLC values produced by the engine.

## 5. Desktop Newswire is a workstation

Mobile Newswire remains the accepted scroll-first version. Desktop adds a dense intelligence desk above the full story stream:

- Market Pulse with regime narrative and breadth;
- Quant Snapshot with average Sharpe and VaR;
- Forward Radar with forecast and PEG context;
- Quality Board with ROE and PE context.

These panels supplement the existing ticker tape, breadth dashboard, desk filters, lead story, 28-story stream, top gainers/losers and signal matrix. The desktop page remains within a 1280-pixel viewport without horizontal overflow.

## 6. Authority and integrity

- Rewind and replay affect presentation only; they are not move takebacks.
- Completed rated moves remain immutable.
- Engine monthly closes remain authoritative.
- OHLC high/low excursions remain deterministic display evidence and cannot alter cash, holdings, score, rating, benchmark or verdict.
- The visible **SIMULATION MODE** boundary remains unchanged.
- No live-market claim or investment recommendation is introduced.

## 7. Quality gates

### Automated

- Backend/API/contract suite: **34/34 passed**.
- Frontend unit suite: **9/9 passed**.
- TypeScript compilation: passed.
- Vite production build: passed.

### Browser

- Warm desktop refresh completed without an error toast.
- Desktop Newswire rendered four intelligence panels and 28 story cards at 1280 pixels.
- Fresh 24-month mobile Blitz campaign was constructed and permanently executed.
- The live replay rendered inside the portrait viewport with the OHLC panel below it.
- Pause and one-month rewind were exercised; the visible month moved from 23 to 22.
- Replay reached its stop without leaking the final verdict.
- Final report was opened explicitly and rendered 24 completed OHLC candles.
- The obsolete mobile and desktop chase charts were absent from the completed campaign report.

## 8. Release boundary

The release candidate is intentionally a product freeze rather than a new feature branch. Any further investor feedback should first be classified as:

1. release-blocking correctness;
2. release-blocking UX;
3. post-decision product backlog.

Only the first two categories should change this candidate before the investor decision.
