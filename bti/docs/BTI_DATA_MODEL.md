# BTI data model

| Entity | Important fields | Invariants |
|---|---|---|
| Player | id, identity subject, display name, created time | Capital is not a rating input |
| Campaign | player, amount, horizon, month, status, versions, seed reference | Horizon is 24/36/48/60 and immutable after start |
| MarketSnapshot | campaign, month, public facts, content hash | One snapshot per month; future values are private |
| Position | campaign, ticker, whole shares, cost basis | Quantity is a non-negative integer |
| Move | campaign, month, idempotency key, prior hash, score/result ids | Unique and immutable per campaign-month |
| Instruction | move, order, BUY/SELL, ticker, shares, price, cash after | Sells cannot exceed holdings; buys cannot exceed cash |
| MoveEvaluation | policy version, process score, class, bounded explanation | Hidden features are not publicly serialised |
| Outcome | next snapshot, player value, benchmark value, alpha | Separate from process score |
| RatingEvent | player, campaign, before/after rating and deviation | Permanent only on completion/resignation |
| PuzzleAttempt | player, puzzle/content version, choice, score | Frozen content is auditable |
| LessonProgress | player, lesson/content version, status | Content version retained |

Money is integer paise; percentage/return values use decimal representation at persistence boundaries.
Timestamps are UTC. Foreign keys enforce ownership. Unique constraints protect campaign-month and
idempotency. Append-only moves, evaluations and rating events preserve audit history.
