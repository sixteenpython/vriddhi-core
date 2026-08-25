# BTI test strategy

## Deterministic engine

- identical release/version/seed produces identical 24/36/48/60-month campaigns;
- simulator output is unchanged when player/reference portfolios change;
- month `m` cannot access any value from `m+1`;
- accounting reconciles to the paise and quantities remain whole;
- overspending, short selling, invalid tickers and under-deployment fail;
- several near-optimal feasible portfolios receive equivalent-good recognition;
- a lucky next-month outcome cannot dominate process quality;
- capital-scaled equivalent decisions produce equivalent skill evidence.

## Lifecycle and integration

Test activation locks amount/horizon, commits are immutable, duplicate idempotency keys return one move,
stale versions fail, explicit resignation records a loss and inactivity does not. Restart tests reload the
same campaign. Public-payload tests reject hidden reference and future-state fields.

## Content and UX

Schema tests require at least 10 unique puzzles and 8 unique lessons, valid answer indices, recognised
difficulty/skill values and resolvable lesson links. A ten-minute novice walkthrough validates that the
next action, whole-share execution, process/outcome split and campaign progress are understandable.

Release gates include unit/integration tests, static checks, dependency audit, reproducibility fixture,
24/36/48/60 smoke simulations, accessibility checks and a clean deployment smoke test.
