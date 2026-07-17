# Vriddhi Core

Vriddhi is a Nifty 50 portfolio decision aid. Heavy market-data retrieval,
forecasting, walk-forward validation, and portfolio optimization happen in an
offline monthly release. The Streamlit application only reads the resulting
versioned research bundles, keeping the live experience fast and predictable.

> Educational decision aid. Past performance and model outputs do not
> guarantee future results. This repository does not provide investment advice.

## Current architecture

```text
Yahoo Finance -> knowledge asset -> research bundles -> validation -> approval
                                                                  -> Streamlit
```

- `build_grand_table.py` refreshes prices, fundamentals, sectors, and transparent
  damped-Holt forecast signals.
- `build_research_db.py` builds portfolios, walk-forward evidence, risk metrics,
  benchmark comparisons, and recommendation gates for 1-5 year horizons.
- `vriddhi_monthly_refresh.py` performs a staged, transactional monthly release.
- `streamlit_app.py` renders only precomputed, committed artifacts.

Start with the detailed [backend reengineering reference](docs/backend-reengineering-reference.md).
Supporting documents cover the [architecture](docs/architecture.md),
[methodology](docs/methodology.md), and
[monthly operations runbook](docs/monthly-refresh-runbook.md).

## Local application

Python 3.12 is required.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

## Monthly refresh

Install the locked environment, then run:

```bash
uv sync --frozen --all-extras
uv run python vriddhi_monthly_refresh.py
```

The refresh builds in `.refresh_staging/`; published files are unchanged until
validation and all-horizon Streamlit smoke tests pass. Useful modes:

```bash
uv run python vriddhi_monthly_refresh.py --candidate-only --yes
uv run python vriddhi_monthly_refresh.py --as-of 2026-07-17 --yes
uv run python vriddhi_monthly_refresh.py --restore-last
```

The recommended production path is the **Monthly research candidate** workflow
under GitHub Actions. It builds a candidate and opens a pull request for review;
merging the validated pull request triggers Streamlit deployment.

## Tests

```bash
pytest
ruff check .
```

Generated releases include `research/manifest.json`, a validation report, hashes,
dependency versions, source commit, data date, turnover, picks, and drops.

## Canonical files

`streamlit_app.py` is the production UI. `streamlit_app_clean.py` and
`streamlit_app_fixed.py` are retained only as historical prototypes and are not
deployment entry points.

## License

MIT. See [LICENSE](LICENSE).
