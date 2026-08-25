# BTI immersive web architecture

## Decision

The investor showcase is a real web application, not a Streamlit restyle. The browser client is a
React + TypeScript single-page application built with Vite. A single Python ASGI process serves both
the versioned JSON API and the compiled static assets. The existing deterministic Python game engine
remains authoritative.

```text
Browser
  React + TypeScript + Vite
  presentation, routing, charts, draft execution sheet
                    |
                    | HTTPS /api/v1/*
                    v
Python modular monolith (one deployable service)
  ASGI web adapter -> application service -> BTI domain engine
                                      |-> Vriddhi read-only adapter (private)
                                      |-> campaign repository
                                      |-> content repository
  /assets/* and SPA fallback serve web/dist
```

This is a deployment decision, not a permanent coupling. The HTTP contract is client-neutral so the
later Android application can consume it without changing the game engine.

## Repository target

```text
bti/
  game_engine/                 # existing deterministic authority
  server/
    api/                       # route adapters and public schemas
    application/               # commands, queries, ownership and transactions
    persistence/               # local SQLite; production repository adapter
    auth/                      # showcase identity now; Google verification later
    main.py                    # ASGI entry point and static-file fallback
  web/
    package.json
    src/
      app/                     # shell, routes, providers, error boundary
      features/campaign/
      features/market/
      features/portfolio/
      features/analysis/
      features/learning/
      components/              # visual primitives, charts, state components
      api/                     # generated/typed public client only
      styles/
    dist/                      # generated in CI; not hand-edited
  content/
  tests/
```

Existing `bti/web/*.py` is a disposable Streamlit showcase. It is not a dependency of the immersive
client, the ASGI server or the domain engine. Removal can follow feature parity; it must never sit on
the production request path.

## Dependency and authority rules

1. React owns rendering, navigation, accessibility, animation and an uncommitted move draft.
2. Python owns identity, campaign lifecycle, prices, holdings, cash, month progression, validation,
   simulation, score, classification and rating.
3. `bti.game_engine` imports neither React artifacts, ASGI frameworks nor persistence adapters.
4. The client never recalculates an authoritative price, portfolio value, score or rating. It may
   derive display-only formatting and draft estimates labelled **Estimate**.
5. Move submission is atomic. The server checks campaign ownership, expected version and an
   idempotency key before committing exactly once.
6. A move is evaluated against the current information set before the market advances. Outcome is
   returned separately from decision quality.
7. The simulator cannot read player/reference holdings. The evaluator cannot read future states.

## Public HTTP contract

All JSON responses contain `schema_version`. Campaign resources also contain `campaign_version`,
`simulation_version`, `scoring_version`, `engine_version`, `release_id` and `market_label`.

Minimum showcase routes:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness; contains no game data |
| `POST` | `/api/v1/session/showcase` | Create an anonymous showcase session |
| `GET` | `/api/v1/profile` | Player summary and campaigns |
| `POST` | `/api/v1/campaigns` | Freeze amount, horizon, scenario and versions |
| `GET` | `/api/v1/campaigns/{id}` | Public campaign projection |
| `GET` | `/api/v1/campaigns/{id}/market` | Current public information set |
| `GET` | `/api/v1/campaigns/{id}/stocks/{ticker}` | Current stock evidence and chart series |
| `POST` | `/api/v1/campaigns/{id}/drafts/validate` | Server reconciliation without committing |
| `POST` | `/api/v1/campaigns/{id}/moves` | Atomically commit one complete move |
| `GET` | `/api/v1/campaigns/{id}/moves/{move}` | Immutable public move analysis |
| `POST` | `/api/v1/campaigns/{id}/resign` | Explicitly record a loss |
| `GET` | `/api/v1/lessons` | Published learning content |
| `GET` | `/api/v1/puzzles` | Published puzzle catalogue |

Mutation headers are `Idempotency-Key` and `If-Match: <campaign_version>`. Stable conflicts include
`STALE_GAME_VERSION`, `MOVE_ALREADY_COMMITTED`, `INSUFFICIENT_CASH`, `INSUFFICIENT_HOLDINGS`,
`INVALID_QUANTITY` and `UNDER_DEPLOYED`. Errors use `{schema_version, error: {code, message,
field_errors?, retryable}}`; Python tracebacks never reach the browser.

## Public/private intelligence boundary

The public market contract may expose ticker, sector, current OHLC, historical price points through
the current move, PE, PB, PEG, forecast, volatility, Sharpe, drawdown, VaR, expected shortfall and
clearly labelled simulated events. Public campaign data may expose the player's holdings, cash,
contributions, benchmark, alpha, committed execution, classification and bounded teaching text.

The following are server-private and forbidden in API payloads, HTML bootstrap data, source maps,
browser logs and analytics:

- Vriddhi target holdings, weights and transactions;
- reference utility and component scores;
- optimiser output and efficient-frontier coordinates;
- feature weights, coefficients and scoring thresholds;
- scenario seed, random state and future snapshots;
- alternative hidden reference trajectories;
- filesystem paths and raw Vriddhi artefacts.

Public projections use allow-lists; they are not produced by deleting known secret fields from a raw
domain object. Contract tests recursively reject private key fragments such as `reference`,
`optimum`, `seed`, `coefficient`, `frontier`, `target_weight` and `future`.

## State, persistence and integrity

- The browser keeps only the current route, UI preferences and move draft. A refresh reloads the
  authoritative campaign from the API.
- The repository stores a versioned campaign snapshot plus append-only moves. Each move records the
  pre-move information-set hash, instructions, result, engine versions and previous-move hash.
- SQLite is acceptable for the single-instance showcase. A multi-instance deployment requires the
  PostgreSQL adapter before horizontal scaling.
- Amount and horizon are immutable after creation. Inactivity has no penalty. Only explicit
  resignation records a loss.
- The server uses integer paise and integer shares for accounting. Floating point is limited to
  analytical values and never decides cash reconciliation.

## Performance and resilience budgets

On a warm showcase deployment, the target is p95 below 250 ms for reads and below 750 ms for move
submission, excluding score-reveal animation. Initial JavaScript should remain below 300 KiB gzip;
chart code is route-split. The market board virtualises or paginates 50 instruments. Every route has
loading, empty, error and offline/retry states. A failed mutation is safely repeatable with the same
idempotency key.

## Security and privacy

The showcase session uses an opaque, secure, HTTP-only, `SameSite=Lax` cookie; campaign identifiers
are not credentials. Production adds Google ID-token verification server-side without changing game
routes. HTTPS, restrictive CORS, CSP, frame denial, secure headers, request-size limits, rate limits
and structured redacted logs are release gates. No real brokerage connection, real-money execution
or personalised investment instruction exists in the showcase.

## Explicitly deferred

Google sign-in, social leaderboard, multiplayer, live quotations/news, durable cloud identity,
push notifications, Android, brokerage execution and real historical replay are future capabilities.
The interface must not render them as working actions. A small labelled **Coming later** area is
acceptable; deceptive enabled controls are not.
