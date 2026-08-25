# Current solution architecture

## Architectural style

BTI is a web-first modular monolith. A React single-page application and a deterministic Python
domain engine are deployed in one Docker image behind one origin.

```text
Browser
  React 19 + TypeScript + Vite
  shell, terminal UX, draft orders, display-only estimates
                  |
                  | HTTPS JSON /api/v1/*
                  v
Starlette ASGI application
  routes -> BTIService -> JSONRepository
                      -> BTIGame
                           |-> VriddhiArtifacts (read-only)
                           |-> portfolio scoring
                           |-> deterministic market simulation
```

Streamlit is not part of the BTI runtime.

## Responsibility and authority

| Layer | Owns | Must not own |
| --- | --- | --- |
| React client | navigation, responsive rendering, draft order editing, local formatting, display estimates | committed cash, canonical holdings, score, market advancement, hidden reference |
| Starlette adapter | HTTP parsing, authentication boundary, error envelope, static assets | financial rules or score calculation |
| Application service | commands, campaign ownership, input validation, idempotency, repository transactions | UI behaviour |
| `BTIGame` | lifecycle, accounting, execution ordering, move progression, final result | HTTP or React concerns |
| Scoring module | portfolio utility, regret-to-score mapping, classifications, bounded teaching text | future outcomes |
| Simulation module | generated OHLC/metrics/returns from deterministic scenario state | player or hidden-reference holdings |
| Vriddhi adapter | governed universe and horizon portfolio bundles | game state or public serialization |
| Repository | opaque sessions, owner isolation, atomic versioned campaign envelopes | domain calculation |

## Key dependency rules

1. `bti.game_engine` has no dependency on React, Starlette or persistence.
2. The simulator receives market state, scenario seed and month—not player holdings.
3. The evaluator receives the current information set—not the next market state.
4. The client receives allow-listed public projections, never raw domain state.
5. All accounting uses integer paise and integer shares.
6. Browser estimates are explicitly non-authoritative and reconciled again by the server.

## Build and runtime topology

The multi-stage `bti/Dockerfile`:

1. installs the locked pnpm dependency graph;
2. compiles the Vite frontend;
3. installs Python and server dependencies;
4. copies repository code and the compiled frontend into one runtime image;
5. starts `python -m bti.server` on Render's assigned port.

The Starlette app serves versioned API routes first, hashed assets under `/assets`, and the SPA
fallback for non-API paths. Unknown API paths remain JSON errors.

## Why this architecture fits the MVP

- One deployable keeps shoestring operations simple.
- Python preserves and reuses Vriddhi's quantitative engine.
- A real React application provides the immersive desktop experience Streamlit could not.
- The HTTP boundary is already suitable for a later Android client.
- Deterministic state makes defects reproducible and investor demonstrations stable.

## Scaling boundary

The current JSON repository and Render free instance are appropriate for an anonymous showcase,
not a horizontally scaled consumer service. Durable identity, PostgreSQL/object storage, rate
limiting, telemetry and production secrets must precede multi-instance or public-scale operation.
