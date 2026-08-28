# BTI current solution baseline

> **Current release:** v0.15.1 — pre-live handover plus an unobstructed desktop market-run chart.
> See [Playtest 03](../playtesting/PLAYTEST_03_MULTI_MODE_CAPITAL_MARKET_RELEASE.md) for the current
> mode contracts, Newswire, mobile boundary and verification gates.

- **Product:** Beat the Index (BTI)
- **Baseline date:** 27 August 2026
- **Release:** BTI Pre-Live Handover v0.15.1
- **Live application:** <https://beat-the-index.onrender.com/>
- **Status:** deployed investor-preview web/PWA release under active desktop and mobile playtesting

This folder documents the BTI solution that exists today. It complements the forward-looking files
in `bti/docs/` by separating implemented behaviour from planned architecture.

Active findings are documented separately under `bti/docs/playtesting/` so proposed improvements
are never confused with shipped behaviour.

The implementation chronology and production validation of the current PWA/mobile surface are
recorded in
[`PLAYTEST_02_INVESTOR_PREVIEW_PWA_AND_MOBILE_UX.md`](../playtesting/PLAYTEST_02_INVESTOR_PREVIEW_PWA_AND_MOBILE_UX.md).

## The product in one sentence

BTI is Lichess for investment decision-making: the player reads a realistic but explicitly
simulated financial market, constructs one complete whole-share portfolio move per investment
month, receives deterministic decision-quality analysis against a hidden Vriddhi-derived reference,
and tries to beat the simulated Nifty benchmark over a fixed 24–60 month campaign.

> **Vriddhi is the engine. BTI is the game.**

## Document map

1. [PRODUCT_AND_GAME.md](PRODUCT_AND_GAME.md) — product promise, campaign loop and rules.
2. [SOLUTION_ARCHITECTURE.md](SOLUTION_ARCHITECTURE.md) — deployed modular-monolith architecture and authority boundaries.
3. [INTELLIGENCE_AND_SIMULATION.md](INTELLIGENCE_AND_SIMULATION.md) — Vriddhi adapter, hidden reference, scoring and forward market.
4. [WEB_EXPERIENCE.md](WEB_EXPERIENCE.md) — immersive workstation UX and implemented surfaces.
5. [API_STATE_AND_PERSISTENCE.md](API_STATE_AND_PERSISTENCE.md) — HTTP contract, campaign state, accounting and session recovery.
6. [SAFETY_PRIVACY_AND_INTEGRITY.md](SAFETY_PRIVACY_AND_INTEGRITY.md) — simulation disclosure, private-oracle boundary and security posture.
7. [TESTING_DEPLOYMENT_AND_OPERATIONS.md](TESTING_DEPLOYMENT_AND_OPERATIONS.md) — automated gates, Docker/Render release and runbooks.
8. [CURRENT_SCOPE_AND_ANDROID_HANDOFF.md](CURRENT_SCOPE_AND_ANDROID_HANDOFF.md) — implemented/deferred matrix and conditions for mobile replication.
9. [CODEBASE_TRACEABILITY.md](CODEBASE_TRACEABILITY.md) — source-file ownership and change-impact guide.
10. [CALIBRATED_SYNTHETIC_MARKET.md](CALIBRATED_SYNTHETIC_MARKET.md) — realism, coherence and market-data provenance boundary.
11. [RATED_GAMEPLAY.md](RATED_GAMEPLAY.md) — two-screen loop, permanent moves, repeat, chase and historical review.
12. [INVESTOR_PREVIEW_PWA.md](INVESTOR_PREVIEW_PWA.md) — installable portrait PWA, durable guest persistence and investor-release runbook.

## Canonical principles

- One move means one complete monthly portfolio decision, not one stock trade.
- Campaign amount and horizon are fixed at creation; 60 moves is the ceiling, not the default.
- The deterministic Python engine owns money, holdings, market progression, score and rating.
- The React client owns presentation and a reversible draft order sheet only.
- Decision quality is judged using information available before the market advances.
- Market outcome never retroactively changes whether the decision itself was good.
- Hidden Vriddhi/reference intelligence never reaches the browser.
- All gameplay data except listed-company names and tickers is generated for the simulation.
- Every playable financial surface must visibly say **SIMULATION MODE**.
- The launched web surface contains only Market Monitor and Rated Game Board during an active campaign.
- The web experience is the product reference for Android; Android must not fork the rules.

## Current release evidence

The release has been verified through two committed moves: create a campaign, build directly from
the 50-security Market Monitor, review and permanently execute on the Rated Game Board, receive a
position evaluation and outcome, advance the market, repeat the prior instruction exactly, block an
under-deployed repeat, correct and execute it, then rewind the Nifty/player chase and complete move
analysis to Move 1 without changing the live state. The simulation boundary remains visible
throughout. Puzzles, lessons, rankings, profiles and standalone Newswire are intentionally absent
from this gameplay-only web release.

The 26 August investor web increment adds a canonical after-every-move match scoreboard and an
explicit campaign-complete experience. Final wealth determines `YOU BEAT THE INDEX`, `NIFTY WON`
or a 0.5%-band `PHOTO FINISH`; decision quality remains a separate process verdict. Completed
campaigns reopen on their immutable result rather than presenting a fictitious next move, and the
player can download the match summary.

The investor preview uses a new anonymous guest identity on every browser launch. Earlier campaign
history is deliberately not restored in v0.15.0, keeping the launch path fast and giving each
playtest a clean slate. Google identity, resume and cross-device account recovery remain deferred.

The v0.11.0 phone client is now a dedicated scroll-first composition rather than a compressed
desktop workstation. Its Market and Game surfaces were verified against production at a 360 × 800
viewport after Render reported healthy PostgreSQL storage. Physical Samsung M30s installation and
full-campaign acceptance remain part of investor playtesting rather than automated release evidence.

The v0.15.0 pre-live handover adds a mandatory, non-persistent consent gate before any anonymous
session or gameplay is created. It explains all three modes, the educational-only boundary and why
the Capital Market Intelligence feels realistic: recent-pattern calibration, coherent cross-asset
behaviour and sealed generated futures. It also states plainly that the simulation is not real-time
information, advice or a prediction. The final Rapid replay now leads to the completed result rather
than incorrectly offering another rebalance stop.
