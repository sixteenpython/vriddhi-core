# Safety, privacy and integrity

## Simulation boundary

BTI's professional appearance makes disclosure more important, not less.

The canonical statement is:

> BTI is a simulated investment strategy game for educational and entertainment purposes. Except
> for actual listed-security names and tickers, prices, metrics, forecasts, news, events, portfolio
> outcomes and other gameplay information are generated for the game. They are not real-time market
> information, investment advice or a recommendation to buy or sell a security.

Implemented safeguards include:

- persistent purple **SIMULATION MODE** badges;
- the hook: **The stocks are real. The market you are about to play in is not.**;
- generated-feed labels on the terminal and newsroom;
- explicit “no real orders” language in the order workbench;
- no-advice language on financial surfaces and final outputs.

No future redesign may visually merge generated data with a purported live feed without explicit
local simulation labelling.

## Private-oracle boundary

Vriddhi's hidden reference is BTI's Stockfish equivalent. Protecting it preserves both intellectual
property and game integrity.

Public DTOs are built from allow-listed fields. Tests recursively reject private concepts such as
reference holdings, target weights, optimiser/frontier data, regret, utility and scenario seed.
Private values must not appear in API responses, HTML, client bundles, source maps, browser logs or
analytics.

## Temporal integrity

- The decision is scored before the next market is generated.
- The simulator cannot access player or reference holdings.
- The evaluator cannot access future market states.
- A committed move records its result and cannot be rewritten by a later outcome.
- Identical deterministic inputs reproduce identical state.

## Current authentication and privacy

The showcase creates an opaque bearer token and stores only its hash server-side. Campaign IDs are
not credentials. Sessions and campaigns are anonymous; the current release does not request or
durably store personal identity.

The release does not connect to a broker, accept real trades, move money or send screenplay/user
content to an LLM. BTI's current engine has no LLM dependency.

## Security controls implemented in code

- request bodies limited to 1 MB;
- safe structured JSON errors;
- campaign ownership isolation;
- opaque high-entropy session tokens;
- idempotent move commits;
- atomic repository file replacement;
- API paths excluded from SPA fallback;
- Starlette debug disabled;
- gzip for suitable responses.

## Required hardening before public scale

- HTTPS/security header and CSP audit;
- server-side rate limiting and abuse controls;
- durable secret management and token expiry/rotation;
- Google identity verification;
- durable transactional database and backups;
- structured redacted logs and alerts;
- dependency, secret and container scanning;
- privacy policy and data-retention controls.
