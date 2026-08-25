# BTI implementation plan

## Objective

Deliver a playable, auditable investment strategy game while keeping Vriddhi's research engine
authoritative and private. The implementation is an additive modular monolith. Product validation is
web-first; Android packaging follows after the core loop is tested on a desktop screen.

## Release slice

The investor release supports Google-authenticated users, game configuration, a public 50-stock
market board, total-portfolio allocation, immutable monthly moves, deterministic market simulation,
move classification, progress, a final benchmark comparison and a capital-neutral player rating.

The first game is labelled **Simulated market — knowledge cutoff 14 August 2026**. Historical replay
is not enabled until genuine point-in-time monthly knowledge assets exist.

## Architecture

- Android: native Kotlin, Jetpack Compose, target/compile API 36, minimum API 26.
- API: versioned Python modular monolith using Starlette and SQLite behind repository interfaces.
- Intelligence: read-only Vriddhi adapter plus deterministic simulation, portfolio, scoring and
  rating engines. Reference allocations never cross the server boundary.
- Identity: Google ID token verification in production, exchanged for a short-lived BTI session.
- Delivery: CI executes Python gates and Android unit/lint/bundle tasks. Signing material remains
  outside Git.

## Ordered stories

1. **BTI-001–004 — foundations:** reuse assessment, architecture, contracts, threat model.
2. **BTI-005–010 — intelligence:** governed artifact adapter, deterministic simulation, allocation,
   scoring, rating and temporal-integrity tests.
3. **BTI-011–016 — API:** auth, profiles, games, moves, history, puzzles/lessons and audit trail.
4. **BTI-017–024 — Android:** sign-in, home, setup, board, stock detail, portfolio, result, progress.
5. **BTI-025–030 — release:** observability, smoke simulation, CI, deployment, signing and Play
   submission checklist.

## Quality gates

- same seed and engine version produce byte-equivalent market paths and scores;
- a move can be committed exactly once per month and is hash chained;
- public API responses cannot expose reference weights or private optimisation artifacts;
- investment allocation is validated, numerical values are finite and horizon is 36/48/60 moves;
- rating is independent of chosen rupee capital;
- a 60-move simulated game completes and reconciles cash, holdings and benchmark value;
- production builds reject localhost, cleartext transport, fake auth and absent release configuration.

## External release prerequisites

The codebase cannot invent account-owned credentials. Publishing requires the owner's Google OAuth
client, Play Console organisation account, upload key/Play App Signing enrolment and a reachable TLS
API URL. These are release inputs, not business-logic dependencies.
