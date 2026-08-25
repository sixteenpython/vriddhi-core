# BTI intelligence API and web facade

The Python web application is the first client. UI code calls an application-service facade; it does
not open Vriddhi artefacts, simulate prices or calculate scores. The same facade becomes a versioned
HTTP API when Android is introduced.

Operations cover player creation, campaign create/list/read/resume/resign, current market and stock
detail, draft validation, atomic move submission, analysis, final result, history, rating, leaderboard,
puzzles and lessons. Every response includes a schema version; campaign responses include frozen engine
versions and the simulation label. Mutations require an idempotency key and expected campaign version.

Stable errors include `STALE_GAME_VERSION`, `INSUFFICIENT_CASH`, `INSUFFICIENT_HOLDINGS`,
`INVALID_QUANTITY`, `UNDER_DEPLOYED` and `MOVE_ALREADY_COMMITTED`.

Serialization uses explicit public projections. Reference trades, target weights, optimiser output,
coefficients, future states and seeds are forbidden. Explanations contain public evidence and teaching,
not the hidden answer.
