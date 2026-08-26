# Product and game specification

## Product positioning

BTI turns governed investment intelligence into repeatable play. It is neither a brokerage terminal
nor a stock-tip product. Its job is to develop portfolio judgment through a full campaign of
decisions, feedback and consequence.

The player experience combines:

- Bloomberg-like information density;
- Yahoo Finance/Moneycontrol-like market discovery;
- Lichess-like moves, classifications, analysis and progression;
- Vriddhi-like quantitative portfolio discipline and plain-English explanation.

## Campaign contract

Players may keep multiple rated campaigns active and switch between them from the campaign rail.
An explicit abort is irreversible: it records the campaign as `ABORTED`, preserves every completed
move for read-only analysis and does not permit a takeback or resumption.

At campaign creation the player fixes two independent inputs:

| Input | Allowed values |
| --- | --- |
| Monthly contribution | Whole rupees from ₹10,000 to ₹1,00,000 |
| Horizon | 24, 36, 48 or 60 months |

One campaign month equals one complete portfolio move. The campaign therefore contains exactly 24,
36, 48 or 60 required moves. Amount and horizon do not change after creation.

The active lifecycle is:

```text
CREATE CAMPAIGN
  -> READ CURRENT PUBLIC INFORMATION SET
  -> CONSTRUCT BUY/SELL EXECUTION
  -> RECONCILE CASH AND DEPLOYMENT
  -> COMMIT IMMUTABLE MOVE
  -> SCORE DECISION
  -> ADVANCE SIMULATED MARKET
  -> EXPLAIN DECISION AND OUTCOME
  -> CONTINUE OR COMPLETE
```

## Move rules implemented today

- Instructions are BUY or SELL orders in positive whole shares.
- The player cannot short a security or sell more than the current holding.
- Duplicate instructions for the same ticker and side must be combined.
- SELL instructions execute before BUY instructions.
- The player must deploy at least 90% of available buying power for that month.
- Cash cannot become negative.
- The portfolio accounting equation must reconcile exactly in integer paise.
- A committed move cannot be edited.
- A stale month cannot be committed after another move has advanced the campaign.
- Retrying the same idempotent submission does not create a second move.
- Resignation is explicit, closes the campaign and records a loss.

## What constitutes a decision

The unit of judgment is the post-trade portfolio. BTI does not reward a player merely for selecting
one attractive stock. It assesses the combined growth, valuation, risk, company concentration,
sector concentration and opportunity cost of the entire feasible portfolio.

This is important because BTI teaches portfolio construction rather than ticker prediction.

## Move classifications

The current scoring version maps a 0–100 decision score to:

| Score | Classification |
| ---: | --- |
| 98–100 | BEST MOVE |
| 90–97.9 | EXCELLENT |
| 80–89.9 | GOOD |
| 65–79.9 | INACCURACY |
| 45–64.9 | MISS |
| Below 45 | BLUNDER |

The result explains what the player did well and the largest available improvement. The hidden
reference portfolio and private optimiser calculations are never shown.

## Campaign result

At completion the engine reports:

- total contribution;
- final portfolio and benchmark values;
- wealth alpha in rupees;
- portfolio and benchmark money-weighted annual return estimates;
- average move score;
- final BTI rating.

BTI does not promise or predict real investment performance. Campaign results belong only to the
generated scenario.
