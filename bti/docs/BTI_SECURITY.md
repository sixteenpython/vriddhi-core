# BTI security

## Trust boundaries

The server owns prices, holdings, cash, time progression, simulation, scores and ratings. Browser fields
are untrusted. Hidden Vriddhi intelligence and future scenario states never cross the public-projection
boundary. Secrets, databases and signing material remain outside Git.

## Web MVP controls

- opaque server-side session identity and secure cookies in production;
- CSRF protection for state-changing HTTP requests;
- strict input, ticker, quantity, money and campaign-ownership validation;
- whole-share, no-short-selling and cash reconciliation enforced server-side;
- idempotency plus optimistic concurrency for submissions;
- append-only committed moves and tamper-evident snapshot/move hashes;
- rate limits on authentication and mutations;
- escaped output, restrictive content security policy and no secrets in logs;
- dependency scanning, locked production versions and HTTPS-only deployment.

Showcase identity may be anonymous, but production fake authentication must fail closed. Later Google
authentication must verify issuer, audience, signature and expiry server-side before creating a BTI
session.

BTI stores the minimum profile and game data. It must provide retention, export and deletion paths and
state clearly that it is simulated education—not brokerage, personalised advice or a return promise.
