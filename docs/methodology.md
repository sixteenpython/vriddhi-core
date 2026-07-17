# Methodology and limitations

Vriddhi combines a valuation/growth screen with capped, long-only Markowitz
optimization. Historical evidence uses adjusted daily prices, rolling metrics,
and train-then-test walk-forward evaluation. A horizon is recommended only when
all configured return, drawdown, Sharpe, and benchmark gates pass.

Forecast columns use damped Holt exponential smoothing on monthly log prices,
shrunk toward a long-run market anchor and bounded by explicit guardrails. They
are transparent scenario signals, not guaranteed price targets.

## Known limitations

- PE and PB are current values, not point-in-time historical fundamentals.
- The universe is a maintained Nifty 50 list and needs a formal survivorship-bias
  audit for historical claims.
- Costs use a simplified annual drag; taxes, bid/ask spread, and execution
  slippage are not modeled account by account.
- The benchmark is currently sourced as `^NSEI`. Before treating relative-return
  results as institutional-grade, replace or independently validate it against a
  Nifty 50 Total Return Index series with matching dividend treatment.
- Yahoo Finance is a convenient single data source, not an exchange-grade market
  data service. Material production decisions should use a licensed source or an
  independent reconciliation feed.

These limitations must remain visible in the app and release reports. Changes to
methodology require versioning, fixed-fixture regression tests, and a comparison
against the prior production method.
