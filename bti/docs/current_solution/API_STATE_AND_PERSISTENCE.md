# API, state and persistence

## Public API envelope

Successful responses use:

```json
{"api_version":"1.0","data":{}}
```

Failures use a stable code and safe message; Python tracebacks are not returned to the browser.

## Implemented routes

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/v1/health` | liveness |
| POST | `/api/v1/showcase/session` | anonymous bearer session |
| GET/POST | `/api/v1/campaigns` | list or create owned campaigns |
| GET | `/api/v1/campaigns/{id}` | current public campaign state |
| GET | `/api/v1/campaigns/{id}/market` | current public information set |
| POST | `/api/v1/campaigns/{id}/moves/validate` | non-committing server validation |
| POST | `/api/v1/campaigns/{id}/moves` | atomic move commit |
| GET | `/api/v1/campaigns/{id}/result` | latest committed result |
| GET | `/api/v1/campaigns/{id}/history` | immutable move summaries |
| GET | `/api/v1/campaigns/{id}/history/{move}` | reconstructed move, market and chase state |
| POST | `/api/v1/campaigns/{id}/resign` | explicit campaign resignation |
| POST | `/api/v1/campaigns/{id}/abort` | permanently abort an active campaign while retaining history |

Campaign creation includes `initial_market` in the creation response so the client can open the
Market Monitor without issuing a redundant second market request. Campaign listing remains the
canonical source for the active, completed and aborted campaign rail.
| GET | `/api/v1/lessons` | governed lesson content |
| GET | `/api/v1/puzzles` | governed puzzle content |

Move commit requires an `Idempotency-Key`. A stale expected month returns a conflict instead of
silently applying the move to the wrong information set.

## Canonical campaign state

The domain state contains campaign identity/status, versions, research release, fixed amount and
horizon, current month, integer cash, whole-share holdings, current simulated market, precomputed
portfolio-independent regime schedule, benchmark, contributions, immutable move records and rating.

The public projection intentionally omits seed, hidden reference and private evaluation internals.

## Cash model

At the start of a move, buying power equals retained cash plus the current monthly contribution.
SELL proceeds add to cash before BUY orders consume it. After execution:

```text
post-trade cash + post-trade holdings at current prices
= pre-trade cash + pre-trade holdings at current prices + contribution
```

This equality is a runtime invariant. Amounts are integer paise; prices and share counts cannot
introduce floating-point accounting drift.

## Showcase persistence

The current `JSONRepository` stores:

- SHA-256-keyed anonymous session files;
- campaign envelopes containing owner hash, revision, timestamps, canonical game JSON and bounded
  idempotency responses;
- writes through a temporary file followed by atomic replace;
- in-process locking around reads and mutations.

Campaign ownership is checked without revealing whether another owner has a requested identifier.

Render currently points `BTI_SAVE_DIR` at `/tmp/bti-saves`; this is ephemeral showcase storage. A
redeploy can invalidate anonymous sessions and remove campaigns. The client now detects a rejected
stored token, removes it and creates a new anonymous session automatically.

Safe reads are retried once after renewal. A move commit is deliberately never replayed after a
session reset because doing so could duplicate a financial command. Instead the client returns the
stable `SESSION_RESET` condition, clears stale local campaign state and explains that a fresh
showcase campaign is required. Ordinary active-session commits continue normally.

## Production migration requirement

Before durable public accounts, replace showcase storage with a durable repository—normally
PostgreSQL plus object storage/backup—and add verified Google identity. The domain and HTTP command
contracts should remain stable during that migration.
