# BTI immersive web release plan

## Release objective

Ship a web experience an investor can open and play end to end:

> Start campaign -> investigate market -> construct whole-share move -> commit -> receive analysis
> -> watch the simulated month advance -> see portfolio versus Nifty -> continue.

The release is one Python service serving `/api/v1/*` and the compiled React application. Streamlit
is not part of the target runtime.

## Non-negotiable simulation boundary

- `SIMULATION MODE` persists across onboarding, market, move, analysis and result surfaces.
- Onboarding states that the securities are real but the playable market is not.
- Setup states that decisions affect the game rather than real money.
- Generated prices, metrics, forecasts, news and outcomes cannot be mistaken for a live feed.
- Educational/entertainment purpose and no-advice language is available before the first move.

Failure of any item above blocks release.

## Vertical slices

### Slice 1 — immersive shell plus real engine

- React/TypeScript/Vite shell, routes, design tokens and accessible components;
- Python ASGI adapter around the existing `BTIGame` domain;
- anonymous showcase session and SQLite campaign repository;
- campaign creation and recovery;
- real current market board and stock panel;
- complete whole-share move workshop;
- server validation, atomic commit, score reveal and next-month progression.

**Exit:** the complete ten-minute test passes with no mocked financial response.

### Slice 2 — analysis depth and learning

- campaign timeline and immutable move records;
- decision-quality/outcome separation;
- portfolio-versus-Nifty chart;
- lessons and puzzles from governed content;
- skill evidence shown only after sufficient decisions;
- campaign completion, resignation and export.

**Exit:** a 24-move automated campaign completes and a human wants to play another move.

### Slice 3 — release hardening

- production headers, rate limits, redacted structured logging and request limits;
- keyboard, screen-reader, contrast, reduced-motion and responsive verification;
- browser E2E tests at desktop and mobile widths;
- reproducibility, idempotency, stale-version and private-field tests;
- asset optimisation, error recovery, health/readiness and deployment runbook.

**Exit:** all gates below pass against the deployed artifact.

Google identity, public leaderboard and Android follow as separate slices; they do not block the
anonymous investor showcase and must not be presented as current capabilities.

## Build and deploy

The production image performs two stages:

1. run `pnpm install --frozen-lockfile`, tests and `pnpm build` in `bti/frontend`;
2. install the pinned Python server dependencies and copy `bti/frontend/dist` into the runtime image.

The runtime starts one ASGI process. It serves `/api/v1/*` before static routing, immutable hashed
files under `/assets`, `index.html` with `no-cache`, and SPA fallback only for non-API GET routes.
Missing `dist/index.html`, weak session secret, debug authentication or insecure production origin
must fail startup. Database and secret paths are runtime configuration, never baked into the image.

Single-instance SQLite is acceptable for the private showcase with a persistent volume and backup.
Horizontal or multi-region deployment is prohibited until the PostgreSQL repository is enabled.

## CI gates

Every change must pass:

- Python unit, contract and integration tests;
- deterministic 24/36/48/60-move campaign simulations;
- accounting, temporal-integrity and no-private-field checks;
- frontend type-check, lint and unit tests;
- Playwright happy path, refresh recovery, duplicate click and error path;
- accessibility scan with zero serious/critical violations plus manual keyboard test;
- responsive screenshots at 360, 390, 768, 1024 and 1440 px;
- production frontend build and Python static-serving smoke test;
- dependency and secret scan;
- clean deployment smoke test against `/health` and a complete first move.

The build must also prove that two identical scenario inputs produce identical public outcomes and
that changing monthly capital alone does not change the normalised move classification for an
economically equivalent allocation.

## Deployment verification

After deployment:

1. confirm HTTPS, CSP and secure cookie flags;
2. confirm `/health` without authentication and API ownership everywhere else;
3. create a fresh 24-move campaign;
4. inspect stock metrics and simulation label;
5. submit one valid move and deliberately retry the identical request;
6. verify only one move exists;
7. refresh and resume at move 2;
8. inspect network responses for forbidden private keys;
9. complete the ten-minute product test on Chrome and one non-Chromium browser;
10. archive build identifiers, scenario/release versions and test evidence.

Rollback means restoring the previous immutable application image. Persistence migrations must be
backward compatible for the showcase; destructive migrations require a verified backup and a
separate maintenance window.

## Observability

Record request ID, route template, response status, duration, anonymous player ID hash, campaign ID,
campaign version and engine versions. Never log cookies, full request bodies, draft rationales,
scenario seed, reference state or raw artefacts. Product telemetry is limited to funnel steps,
validation code, time to first move, move completion and voluntary continuation; it must not feed the
financial engine.

Alerts cover elevated 5xx, failed move commits, accounting invariant failures, database write
failure and health/readiness failure. A scoring or accounting invariant pages the release owner and
blocks further move submission until investigated.

## Investor-showcase release checklist

- [ ] The domain engine, not a fixture, produces every market, execution and score response.
- [ ] The landing page and HUD say **SIMULATED MARKET** and educational game.
- [ ] Current and future features are labelled truthfully.
- [ ] A whole-share move reconciles to the paise and commits once.
- [ ] Hidden Vriddhi/reference/scoring state never reaches the browser.
- [ ] Decision quality and market outcome are visually and verbally separate.
- [ ] Refresh resumes the committed campaign.
- [ ] Keyboard-only and reduced-motion flows complete.
- [ ] 360 px and 1440 px flows have no clipped action or accidental page overflow.
- [ ] The deployed ten-minute acceptance test passes.
- [ ] A release owner has recorded rollback image and backup location.

## Definition of done

The immersive web release is done only when a user can complete the real loop on the deployed URL,
understand the classification in plain English, continue to the next month and recover after refresh.
A beautiful shell around stubbed evaluation is not done; a correct engine hidden behind a dull or
confusing interface is also not done.
