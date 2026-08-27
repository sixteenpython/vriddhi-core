# Playtest 08 — Persisted-campaign compatibility hotfix

**Release:** BTI Release Candidate v0.14.1

**Date:** 27 August 2026

## Incident

After v0.14.0 deployment, existing desktop and mobile sessions could load their profile and campaign history but repeatedly displayed an outdated-page/server-restart warning. Cache and service-worker resets did not resolve it.

The failure was isolated to the automatic Market request for an older active campaign. Some durable campaign markets predated quantitative fields introduced by later workstation views. Campaign summaries remained valid, but Market serialization could fail while deriving the Risk Watch card.

## Repair

- Older saved markets now receive additive defaults for missing VaR, Expected Shortfall, drawdown, history, OHLC and forecast collections at the public-view boundary.
- Existing prices, holdings, cash, executed moves, scores and outcomes are not changed.
- Home no longer requests a complete 50-security Market merely to display campaign history.
- Market data loads only when Market, Newswire, Stock Research or Game Board requires it.
- A failed Market load now produces a bounded reconnect screen with an explicit retry instead of a persistent global toast.
- Historical campaign review reuses the immutable market snapshot returned with the selected move.

## Verification

- Complete backend/API/contract suite: 35 tests passed.
- Added a regression that removes modern risk fields from a serialized campaign, restores it and verifies that the 50-stock Market and Risk Watch render successfully.
- Frontend TypeScript, production build and unit tests remain release gates.

## Data integrity

This is a read-compatibility migration at presentation time. It does not rewrite persisted campaign documents and does not alter rated gameplay authority.
