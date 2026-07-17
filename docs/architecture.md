# Architecture

Vriddhi deliberately uses a static research-release architecture.

1. The knowledge builder downloads adjusted price history and current
   fundamentals, resolves ticker changes, and produces the stock universe CSV.
2. The research builder screens candidates, optimizes capped long-only
   portfolios, runs walk-forward evaluation, calculates risk metrics, and writes
   one bundle per investment horizon.
3. The transactional refresh validates the complete candidate in staging,
   smoke-tests the UI, records provenance, and only then promotes it.
4. Streamlit reads committed artifacts and performs no network research work.

This keeps production small and deterministic. A database or API service should
only be introduced if concurrent writes or a larger product surface require it.
