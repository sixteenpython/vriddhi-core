# BTI Playtest 03 — Multi-Mode Capital Market Release

**Release:** v0.12.0  
**Release date:** 27 August 2026  
**Boundary:** Rated simulated investment strategy game; no live prices or investment advice

## Product decision

Investor playtesting established that the monthly campaign is BTI's equivalent of classical chess,
but it is not the only legitimate way to test investment decision-making. This release adds two faster
campaign tempos without fragmenting player identity or weakening the deterministic market engine.

## Rated modes

### Classic

- Monthly SIP of ₹10,000–₹1,00,000 over 24, 36, 48 or 60 months.
- One permanent portfolio decision per month.
- Market Monitor is the primary thinking surface.
- Player and Nifty are compared using SIP XIRR.

### Rapid

- Fixed starting capital and a two-to-five-year horizon.
- Initial allocation plus one stop after every completed year.
- Later stops have a 120-second clock; expiry records HOLD and advances the market.
- Newswire and Market Monitor are equally important at each stop.
- Player and Nifty are compared using CAGR.

### Blitz

- Fixed starting capital and a two-to-five-year horizon.
- One initial allocation, with at least ₹1 lakh deployed.
- No intervention after RUN; the complete monthly path unfolds as an animated simulation.
- Newswire is the natural entry surface.
- Player and Nifty are compared using CAGR.

All modes contribute to one BTI rating and Beat-the-Index record. Mode-normalized weights prevent a
one-click Blitz result from carrying the learning weight of a completed 60-move Classic campaign.

## Unified simulated capital market

Each campaign precomputes one hidden, reproducible macro path containing growth, inflation, rates,
liquidity and risk appetite. That economy drives Nifty 50 equities, simulated government bonds,
investment-grade corporate bonds, INR gold and cash. Equity forecasts tilt probabilities rather than
dictating outcomes. Bonds respond to carry, duration, rates and credit conditions; gold responds to
inflation, rates, risk appetite and idiosyncratic shocks. Crypto is deferred pending stronger calibration
and suitability rules.

## First-class Newswire

Newswire now supplies a simulated ticker tape, lead macro story, sector/momentum/risk stories,
portfolio-relevant alerts, movers, filters and security-research click-through. Its learning tension is
deliberate: headlines attract attention, while the supporting valuation and risk evidence asks whether
the story is signal, context or distraction.

## Desktop and mobile

Desktop remains the full professional workstation: dense sortable universe, portfolio construction,
Newswire and post-move analysis. Mobile is a scroll-first product with a horizontal tape, sticky news
filters, one-column stories, lens-based discovery, compact Rapid/Blitz controls and a four-destination
bottom bar. It is not a squeezed desktop clone.

## Compatibility and authority

- Existing saved campaigns default safely to Classic.
- Deterministic engines own prices, accounting, score, rating, benchmark and verdict.
- Public payloads never expose the hidden Vriddhi reference portfolio.
- The persistent Simulation Mode boundary remains visible.

## Quality gates

- 15 game-engine tests pass, including Classic determinism and Rapid/Blitz contracts.
- 6 API tests pass, including persistence, idempotency, ownership and unified profile.
- 6 frontend tests pass; TypeScript checking and the production build pass.
- Production health contract is v0.12.0.

