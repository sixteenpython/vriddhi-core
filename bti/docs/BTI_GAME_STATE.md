# BTI game state

## Aggregate

A campaign is the authoritative aggregate. It freezes `campaign_id`, player, monthly contribution,
horizon (24/36/48/60), release and engine versions, scenario seed, status, current month and version.
It also holds cash, whole-share positions, contribution ledger, benchmark units and immutable move IDs.

Statuses are `DRAFT`, `ACTIVE`, `COMPLETED` and `RESIGNED`. Inactivity never changes status. Amount and
horizon cannot change after activation. Completion requires every scheduled monthly move; explicit
resignation settles the campaign as a loss.

## Monthly transition

1. Add the fixed monthly contribution to cash.
2. Expose the current public market snapshot.
3. Accept a draft execution sheet of whole-share BUY/SELL instructions.
4. Validate holdings, prices, cash and deployment.
5. Commit the complete sheet atomically and append its hash to the move chain.
6. Evaluate decision quality from the pre-move information set.
7. Advance the independent simulator exactly one month.
8. Mark holdings to market; update benchmark, outcome and provisional performance.
9. Reveal analysis and unlock the next month, or settle the final result.

Drafts may change; committed moves may not. An idempotency key returns the original response. An
`expected_version` prevents two browser tabs from committing competing moves.

## Public and private projections

The browser receives current facts, holdings, cash, draft validation, classification and bounded
explanation. It never receives reference trades, target weights, frontier data, coefficients, future
snapshots or scenario seeds. Persistence stores money as integer paise and quantities as integers.
