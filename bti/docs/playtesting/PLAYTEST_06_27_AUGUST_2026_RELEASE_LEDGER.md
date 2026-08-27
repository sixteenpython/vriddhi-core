# Playtest 06 — 27 August 2026 Release Ledger

**Date:** 27 August 2026  
**Final production release:** BTI Market Replay v0.13.0  
**Production URL:** <https://beat-the-index.onrender.com/>  
**Product boundary:** Rated simulated investment-strategy game; not live market data or investment advice

## Purpose of this record

This document consolidates the complete BTI build performed on 27 August 2026. The focused records in Playtests 02–05 remain the authoritative deep dives for individual changes. This ledger explains how those changes combine into the production system delivered for investor playtesting.

The day's work transformed BTI from one monthly Classic campaign into a multi-tempo simulated capital-market game with an immersive Rapid/Blitz journey, a dedicated Newswire, hardened API response handling, and distinct desktop and mobile experiences.

## 1. Mobile investor-preview stabilization

The first release boundary was a usable installable PWA for investors without a Play Store dependency.

### Changes

- Rebuilt portrait mobile as a scroll-first experience instead of compressing the desktop terminal.
- Added mobile-native campaign discovery, Market, Game and News navigation.
- Corrected viewport sizing and bottom-navigation height on Android-class screens.
- Preserved the full deterministic game and financial engine behind a simpler presentation hierarchy.
- Retained the persistent **SIMULATION MODE** boundary.
- Kept the production URL as the common entry point: desktop renders the workstation and mobile renders the responsive PWA.
- Preserved same-device campaign continuity through the durable application/session design.

### Related commits

- `a7ec896` — restore mobile viewport usability.
- `7eb1eae` — enforce mobile tab-bar height.
- `f1cf4c7` — rebuild mobile campaign experience.
- `a78541e` — record investor PWA and mobile UX release.

## 2. Three rated gameplay modes

BTI now treats investment tempo as part of the player's strategic choice while preserving one overall BTI rating.

### Classic

- ₹10,000–₹1,00,000 monthly SIP.
- 24, 36, 48 or 60 permanent monthly decisions.
- Market Monitor is the principal research surface.
- Player and Nifty are evaluated using SIP XIRR.

### Rapid

- Fixed capital over a two-to-five-year horizon.
- Initial allocation followed by one annual decision stop.
- Later stops are limited to 120 seconds; expiry records HOLD and advances the simulation.
- Uses both Market Monitor and Newswire for intervention decisions.
- Player and Nifty are compared using CAGR.

### Blitz

- One fixed-capital allocation over a two-to-five-year horizon.
- At least ₹1 lakh must be deployed, subject to whole-share residual handling.
- The market then runs without intervention.
- Player and Nifty are compared using CAGR.

### Rating integrity

- All modes contribute to one player identity, BTI rating and Beat-the-Index record.
- Mode-normalized weights prevent one Blitz allocation from carrying the learning weight of a long Classic campaign.
- Move quality remains distinct from market outcome: a player may beat Nifty with a fragile decision and still receive a poor process classification.

## 3. Unified simulated capital-market engine

Rapid and Blitz required a broader engine than a stock-only SIP campaign. The release introduced one hidden, reproducible economic path per campaign.

### Decision universe

- Nifty 50 equity securities.
- Simulated government bonds.
- Simulated investment-grade corporate bonds.
- Simulated INR gold.
- Cash.

Crypto remains deliberately deferred until volatility calibration and product-suitability rules are strong enough.

### Market mechanics

- Growth, inflation, rates, liquidity and risk appetite form the campaign's hidden macro path.
- Equities respond probabilistically to health, valuation, regime and idiosyncratic effects.
- Bonds respond to carry, duration, interest-rate and credit conditions.
- Gold responds to inflation, rates, risk appetite and independent shocks.
- Forecasts tilt probabilities; they do not dictate outcomes.
- The campaign seed makes the path reproducible for testing without making it visible to the player.
- Deterministic engines remain authoritative for accounting, benchmark, score, rating and verdict.

## 4. First-class Newswire

The small Market Monitor news ribbon remains the trailer. The dedicated Newswire is now the complete simulated information environment.

### Information desks

- Market and macro.
- Risk.
- Earnings.
- Sectors.
- Quantitative signals.
- Portfolio-specific coverage.

### Decision aids and deliberate distractions

- Current-month lead story and simulated ticker tape.
- Top gainers and losers.
- Market breadth and news sentiment.
- Average VaR and portfolio-position summary.
- Valuation, quality, low-risk and momentum signal breadth.
- Ticker drill-down into stock research.

The learning tension is intentional. News can be signal, context or distraction; it never becomes an automatic instruction to trade.

## 5. API and rendering hardening

Investor playtesting exposed an HTML platform response being parsed as JSON, which surfaced as:

> Unexpected token '<', "<!DOCTYPE "... is not valid JSON

### Corrections

- Validate that API responses are non-empty JSON before parsing.
- Convert HTML, empty and malformed responses into stable BTI errors instead of raw parser exceptions.
- Keep profile refresh non-critical so an already-committed move cannot be falsely reported as failed.
- Return JSON for unknown API routes instead of falling through to the SPA shell.
- Isolate Newswire header/filter styles from broad application-shell selectors.
- Apply CAGR labels to lump-sum Rapid/Blitz and retain SIP XIRR for Classic.
- Advance the service-worker cache so installed clients replace stale bundles.

### Regression evidence at v0.12.1

- Frontend suite: 9/9.
- API/content/immersive contracts: 17/17.
- Engine non-horizon suite: 12/12.
- Completed Classic simulations at 24, 36, 48 and 60 months.
- Complete desktop Rapid and Blitz campaigns.
- Portrait checks at 412 × 915 without page-level horizontal overflow.

## 6. Immersive Rapid and Blitz market replay

The v0.13.0 release replaced a basic two-line close-value animation with an inspectable simulated market journey.

### Replay surface

- Portfolio NAV is rendered as deterministic OHLC candles.
- Nifty follows a synchronized simulated benchmark path.
- Only reached months are rendered; the future path and final verdict remain sealed.
- Desktop hover and mobile tap expose month, OHLC, alpha, drawdown, regime and event context.
- Pause/resume, 1×/2× speed and skip-to-stop controls alter presentation only.
- Synchronized event cards travel with the revealed portion of the path.
- Rapid stops after each annual segment and asks **Care to rebalance?**.
- A visible 120-second Rapid decision clock defaults to HOLD at expiry.
- Blitz remains one irreversible allocation followed by one uninterrupted full-horizon market answer.

### Authority boundary

The engine's monthly close remains authoritative. OHLC high/low excursions are deterministic presentation derivatives of campaign seed, month and regime. They cannot change cash, holdings, score, rating, benchmark or final campaign result.

## 7. Desktop and mobile product relationship

### Desktop

Desktop remains the dense professional workstation:

- full Market Monitor;
- sortable decision universe;
- portfolio construction;
- dedicated Newswire;
- interactive replay and post-move analysis.

### Mobile/PWA

Mobile is the investor acquisition and play entry point:

- portrait, scroll-first disclosure;
- compact campaign controls;
- readable cards and mobile navigation;
- the same deterministic campaign, scoring and persistence contracts;
- touch-oriented replay inspection rather than a squeezed desktop chart.

Both experiences are served by the same production URL and application version. The responsive shell changes presentation, not game authority.

## 8. Final v0.13.0 quality gates

### Automated

- Backend, API and contract suite: **34/34 passed**.
- Frontend unit suite: **9/9 passed**.
- TypeScript compilation: passed.
- Vite production build: passed.

### Browser playtesting

- Fresh Rapid campaign played through allocation, permanent execution, progressive replay and annual stop.
- Rapid replay paused after two revealed months; no final-result leakage was present.
- Rapid stop rendered **Care to rebalance?** and a live `02:00` clock.
- Fresh Blitz campaign completed through its one-decision replay and endgame.
- Expanded Newswire rendered without raw JSON errors or error toasts.
- Desktop application checked at 1280 px.
- Mobile application checked at 412 × 915.
- Both responsive shells stayed within the viewport width.

### Production verification

- Release commit: `150972c`.
- Health endpoint: `status: ok`.
- Reported release: `0.13.0`.
- Storage: healthy, durable PostgreSQL.
- Production shell displayed `BTI MARKET REPLAY · v0.13.0` on desktop and mobile.

## 9. Release chronology

| Commit | Purpose |
|---|---|
| `a7ec896` | Restore mobile viewport usability |
| `7eb1eae` | Enforce mobile tab-bar height |
| `f1cf4c7` | Rebuild the mobile campaign experience |
| `a78541e` | Document investor PWA and mobile UX release |
| `2ad49b4` | Launch Rapid, Blitz and the simulated capital-market modes |
| `1b844d9` | Harden response contracts and multi-mode rendering |
| `150972c` | Ship immersive Rapid/Blitz market replay and full Newswire |

## 10. Final state at close of day

BTI now offers three forms of the same core challenge:

> Construct a decision under uncertainty, let a simulated market answer, understand the quality of the process, and determine whether the result beat the index.

Classic teaches repeated portfolio discipline. Rapid teaches time-constrained intervention. Blitz teaches the consequences of one conviction allocation. Market Monitor is the analytical workbench; Newswire is the attention and sentiment environment; the replay is the moment when the market makes its move against the player.

This is the production baseline for the next investor feedback cycle.
