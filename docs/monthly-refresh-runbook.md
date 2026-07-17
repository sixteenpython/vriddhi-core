# Monthly refresh runbook

## Preferred: GitHub Actions

1. Open **Actions -> Monthly research candidate -> Run workflow**.
2. Leave the as-of field blank to use the latest complete market date.
3. Review the workflow summary and downloadable candidate report.
4. Review the generated pull request: data date, unresolved tickers, turnover,
   picks/drops, gate changes, and artifact hashes.
5. Merge only after required checks pass. Streamlit redeploys from `master`.
6. Confirm the app footer shows the new release ID and data date.

## Local fallback

```bash
uv sync --frozen --all-extras
uv run python vriddhi_monthly_refresh.py --candidate-only --yes
uv run python vriddhi_monthly_refresh.py --yes
```

Use `--allow-high-turnover` only after explicitly reviewing turnover above the
configured threshold. Use `--restore-last` to restore the latest local snapshot.

## Never publish when

- Data dates differ across bundles or benchmark.
- More than five tickers are unresolved/stale.
- A corporate-action mapping has not been confirmed.
- Weights, schema, finite-number, or smoke tests fail.
- Turnover is unexpectedly high or recommendation gates change without review.
