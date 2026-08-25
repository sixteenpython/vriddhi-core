# BTI architecture

## Shape

BTI is a web-first modular monolith with three boundaries:

1. a Python web UI owns presentation and transient draft execution sheets;
2. an application-service facade owns identity, game lifecycle, persistence and public projections;
3. server-private intelligence owns the Vriddhi adapter, simulation, reference decision, move
   scoring and rating.

The client is never authoritative for prices, month progression, scores or rating.

## Dependency rule

`Web UI -> application service -> domain engines -> ports`; SQLite, identity and Vriddhi files implement
ports at the outside. Domain engines do not import web, database or future Android code. The facade can
later be exposed as a versioned HTTP API without rewriting the game.

## Temporal model

Every campaign freezes `knowledge_release`, `scenario_version`, `simulation_version`, `scoring_version`
and `rating_version`. A move sees the current snapshot, commits a whole-share execution sheet, and only
then does the server reveal the next snapshot. The independent seeded simulator cannot read player or
reference holdings. A forward simulation is used until genuine point-in-time historical data is available.

## Trust boundary

Public responses may include stock facts, prices, player holdings and evaluation summaries. They may
not include reference weights, component coefficients, efficient frontiers, optimiser output or
future scenario values. Configuration secrets and signing material remain outside Git.

## Evolution

SQLite is the single-instance MVP adapter. The repository interface permits PostgreSQL without changing
domain behavior. A future historical-game provider implements the same scenario port. A future native
Android client consumes the same facade/API after the desktop game loop has been tested and calibrated.
