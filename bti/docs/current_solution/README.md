# BTI current solution baseline

- **Product:** Beat the Index (BTI)
- **Baseline date:** 25 August 2026
- **Release:** GitHub `master` (see repository history for the immutable commit)
- **Live application:** <https://beat-the-index.onrender.com/>
- **Status:** playable immersive web MVP under active product testing

This folder documents the BTI solution that exists today. It complements the forward-looking files
in `bti/docs/` by separating implemented behaviour from planned architecture.

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
- The web experience is the product reference for Android; Android must not fork the rules.

## Current release evidence

The release has been verified to create a fresh 36-move campaign, recover an expired anonymous
session, analyse the 50-security universe with column filters, open a chart-rich stock workstation,
construct/review/commit a whole-share move, reveal its score, advance the simulated market and show
directionally correct monthly movers. Opening breadth remains `0 : 0` before the first move and the
simulation boundary remains visible throughout.
