# Contributing

1. Create a branch from `master`.
2. Keep market-data access out of the live Streamlit request path.
3. Add or update tests for methodology and artifact-schema changes.
4. Run `pytest` and `ruff check .`.
5. Do not commit secrets, local backups, or staging directories.
6. Describe any impact on portfolio gates, turnover, benchmark methodology, or
   historical comparability in the pull request.

Generated monthly research should be produced through
`vriddhi_monthly_refresh.py` or the GitHub candidate workflow, never by manually
editing JSON bundles.
