# Architecture

Vriddhi deliberately uses a static research-release architecture.

1. The knowledge builder downloads adjusted price history and current
   fundamentals, resolves ticker changes, and produces the stock universe CSV.
2. The research builder screens candidates, optimizes capped long-only
   portfolios, runs walk-forward evaluation, calculates risk metrics, and writes
   one bundle per investment horizon.
3. The transactional refresh validates the complete candidate in staging,
   appends the prospective recommendation ledger, smoke-tests the UI, records
   provenance, and only then promotes it.
4. Streamlit reads committed artifacts and performs no network research work.

This keeps production small and deterministic. A database or API service should
only be introduced if concurrent writes or a larger product surface require it.

The append-only ledger is documented in
[`recommendation-ledger.md`](recommendation-ledger.md). It accumulates actual
monthly published recommendations prospectively and remains separate from the
historical walk-forward evidence.

The completed Backtest Evidence and Monthly Rebalance feature design, calculation
semantics, regression contract, and deployment history are documented in
[`2026-07-22-feature-build-reference.md`](2026-07-22-feature-build-reference.md).

Vriddhi's positioning against Zerodha, Groww, Moneycontrol, smallcase, Dezerv, and
PowerUp Money is assessed in
[`competitor-analysis-2026-07-22.md`](competitor-analysis-2026-07-22.md).
