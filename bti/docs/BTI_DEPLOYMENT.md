# BTI web deployment

## First showcase

Ship the web version before Android so the investment loop can be played, observed and calibrated on a
computer. The deployable unit is one Python modular monolith plus immutable Vriddhi release artefacts and
BTI content. Production configuration supplies a persistent database, session secret and HTTPS origin.

## Release sequence

1. Freeze knowledge, scenario, simulation, scoring, rating and schema versions.
2. Run unit, integration, content and 24/36/48/60 campaign smoke tests.
3. Build from a clean checkout with locked dependencies.
4. Migrate/backup the database and deploy to staging.
5. Complete one full first move: create, inspect, trade, score, advance and resume.
6. Promote the same immutable build; verify health and static assets.
7. Monitor errors, latency, move failures and reconciliation mismatches.

Rollback changes application code, never rewrites committed game history. Frozen engine versions remain
available to replay existing campaigns. Database migrations are additive during MVP.

## Environment

Required values include environment, data root, database URL, public origin and strong session secret.
Development defaults may use SQLite and anonymous showcase identity. Production must reject debug auth,
weak secrets, ephemeral game storage and non-HTTPS public origins.
