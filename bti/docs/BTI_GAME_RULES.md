# BTI game rules

- Choose monthly capital: ₹10,000, ₹25,000, ₹50,000, ₹75,000 or ₹1,00,000.
- Choose 2, 3, 4 or 5 years: 24, 36, 48 or 60 monthly moves. The horizon is fixed at activation.
- At each move, inspect only the current public market board and submit one complete execution sheet.
- Every BUY/SELL quantity is a positive whole number. Short selling and overspending are prohibited.
- One contribution is added before each rebalance. Sells release cash before buys. Purchases must use
  at least 90% of available buying power after the contribution, carried cash and sale proceeds.
- Confirmation commits the move permanently. The web facade rejects stale-month resubmission; the
  later HTTP facade will add durable idempotency keys for network retry.
- The market advances only after commitment. Feedback is then revealed.
- Scoring compares the player with the best feasible whole-share opportunity for the same capital and
  holdings. Near-equivalent strong portfolios are recognised rather than punished for different weights.
- Decision quality is shown separately from the next-month market outcome.
- Final results compare correctly labelled time-weighted and money-weighted player results with the
  simulated Nifty proxy.
- Player rating measures decision quality and uncertainty; monthly capital cannot affect it.
- Inactivity has no penalty. Explicit resignation records a loss; permanent rating settles at completion
  or resignation.

The product always labels prices and outcomes as simulated and educational.
