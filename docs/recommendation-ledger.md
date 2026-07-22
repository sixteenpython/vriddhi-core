# Prospective recommendation ledger

Vriddhi preserves an append-only record of published monthly recommendations in
`research/recommendation_ledger.json`. The ledger began with the retained
2026-06-04 and 2026-07-17 releases and is extended by every successful
transactional monthly refresh.

Each snapshot records the release date, methodology version, recommendation
status, holdings, weights, publication-time prices, and derived PICK, DROP,
TOP-UP, TRIM, and HOLD actions for all five horizons. Same-date refresh retries
do not rewrite the first published entry for that date.

The live track-record view remains gated until 12 validated releases exist.
Before that point, the app shows collection progress rather than a performance
claim. After the gate is reached, the app can replay one contribution per
recorded release using fractional shares and full rebalancing to the published
weights. Transaction costs and residual cash are not modeled and must remain
disclosed.

The prospective ledger does not replace walk-forward validation. They answer
different questions:

- Walk-forward evidence tests the methodology on historical unseen periods.
- The ledger records what happened after Vriddhi began preserving actual
  published recommendations.

The ledger is part of the validation and manifest-hash contract. A missing,
malformed, duplicate-date, out-of-order, stale, or incomplete ledger blocks a
monthly research release.
