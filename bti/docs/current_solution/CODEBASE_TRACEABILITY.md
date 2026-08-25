# Codebase traceability and change-impact guide

## Source ownership

| Path | Responsibility | Typical change impact |
| --- | --- | --- |
| `bti/game_engine/artifacts.py` | read-only Vriddhi artifact adapter | universe/release compatibility, hidden reference inputs |
| `bti/game_engine/engine.py` | canonical campaign lifecycle and accounting | every client and saved campaign |
| `bti/game_engine/scoring.py` | portfolio quality, score and narrative | classifications, rating and teaching text |
| `bti/game_engine/simulation.py` | generated market and public metrics | every future month and benchmark outcome |
| `bti/server/service.py` | application commands, validation and idempotency | API semantics and transactional behaviour |
| `bti/server/repository.py` | anonymous sessions and JSON campaign envelopes | recovery, ownership and persistence |
| `bti/server/app.py` | Starlette routes, errors and SPA serving | HTTP contract and deployment health |
| `bti/frontend/src/api.ts` | typed client and session recovery | all browser/server integration |
| `bti/frontend/src/economics.ts` | display-only draft economics | pre-trade estimates, never canonical accounting |
| `bti/frontend/src/App.tsx` | shell, campaign flow and non-terminal views | navigation and end-to-end UX |
| `bti/frontend/src/Cockpit.tsx` | configurable market tiles, charts and portfolio ribbon | primary research cockpit and saved desk layout |
| `bti/frontend/src/portfolioDraft.ts` | reversible client-side order netting and projected portfolio | Market Monitor/Workbench reconciliation |
| `bti/frontend/src/Terminal.tsx` | market terminal, OMS and newsroom | immersive decision surfaces |
| `bti/frontend/src/StockResearch.tsx` | ticker OHLC, forecasts, ratios, catalysts and peers | security-level research UX |
| `bti/frontend/src/styles.css` | full responsive visual system | product look, density and accessibility |
| `bti/content/*.json` | governed lessons and puzzles | learning catalogue |
| `bti/tests/` | domain, API, content and contract gates | release confidence |
| `bti/Dockerfile` | frontend/runtime production image | deploy reproducibility |
| `render.yaml` | Render service and health configuration | production topology |

## Change rules

### When changing game rules

Update the engine, API validation, UI copy, domain/API tests and this baseline. Consider whether the
engine version must change and whether existing saved campaigns remain compatible.

### When changing scoring or simulation

Version the affected engine, add reproducibility and distribution tests, verify that no future or
private state enters public payloads, and rerun complete campaigns for all four horizons.

### When changing public API payloads

Update `api.ts`, contract tests and the Android handoff specification. Prefer additive changes until
mobile clients support explicit contract negotiation.

### When changing the terminal UX

Preserve `SIMULATION MODE`, no-real-order language, responsive escape routes, keyboard usability and
the distinction between public evidence and hidden engine conclusions.

### When changing deployment or persistence

Test session/campaign recovery, idempotency and ownership across process restart. Never assume
ephemeral Render disk provides durable consumer data.

## Release identity

The documentation baseline corresponds to the live chain:

- `0f0a3b0` — immersive market terminal, newsroom and order workbench;
- `b10e1e9` — expired showcase-session recovery;
- `41b284d` — signed tape and opening breadth polish.

The next release adds the analysis-grade Market Monitor, stock research workstation, strict monthly
movers, collision-free order controls and command-safe session recovery documented here.
