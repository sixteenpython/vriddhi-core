"""Validation and provenance helpers for Vriddhi research artifacts.

This module deliberately has no network dependencies.  It is used by the
monthly refresh, CI, and tests to decide whether a candidate build is safe to
publish.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

HORIZONS = (1, 2, 3, 4, 5)
REQUIRED_STOCK_FIELDS = {"ticker", "weight", "current_price", "sector"}


class ValidationError(RuntimeError):
    """Raised when candidate artifacts are not safe to publish."""


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValidationError(f"invalid JSON artifact {path}: {exc}") from exc


def _assert_finite(value: Any, location: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValidationError(f"non-finite number at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{location}.{key}")


def _parse_date(value: str, location: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"invalid date at {location}: {value!r}") from exc


def _portfolio_weights(bundle: dict[str, Any], horizon: int) -> dict[str, float]:
    stocks = bundle.get("stocks")
    if not isinstance(stocks, list) or not stocks:
        raise ValidationError(f"{horizon}y bundle has no stocks")
    if bundle.get("num_stocks") != len(stocks):
        raise ValidationError(f"{horizon}y num_stocks does not match stock list")

    weights: dict[str, float] = {}
    for index, stock in enumerate(stocks):
        missing = REQUIRED_STOCK_FIELDS - set(stock)
        if missing:
            raise ValidationError(f"{horizon}y stock {index} missing fields: {sorted(missing)}")
        ticker = str(stock["ticker"]).strip()
        if not ticker or ticker in weights:
            raise ValidationError(f"{horizon}y contains duplicate/blank ticker {ticker!r}")
        weight = float(stock["weight"])
        price = float(stock["current_price"])
        if weight <= 0 or price <= 0:
            raise ValidationError(f"{horizon}y {ticker} has non-positive weight or price")
        weights[ticker] = weight

    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=0.002):
        raise ValidationError(f"{horizon}y weights total {total:.6f}, expected 1.0")
    return weights


def validate_candidate(
    data_csv: os.PathLike[str] | str,
    research_dir: os.PathLike[str] | str,
    *,
    turnover_limit: float = 30.0,
    allow_high_turnover: bool = False,
    expected_as_of: str | None = None,
    max_data_age_days: int = 10,
) -> dict[str, Any]:
    """Validate a complete candidate build and return a publishable report."""
    data_path = Path(data_csv)
    research = Path(research_dir)
    errors: list[str] = []
    warnings: list[str] = []

    if not data_path.exists():
        raise ValidationError(f"missing knowledge asset: {data_path}")
    try:
        with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValidationError(f"cannot read knowledge asset: {exc}") from exc
    if len(rows) < 40:
        raise ValidationError(f"knowledge asset has only {len(rows)} rows")
    required_columns = {
        "Ticker", "Sector", "Current_Price", "Forecast_12M", "Forecast_24M",
        "Forecast_36M", "Forecast_48M", "Forecast_60M", "PE_Ratio",
        "Avg_Historical_CAGR",
    }
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise ValidationError(f"knowledge asset missing columns: {sorted(missing_columns)}")
    tickers = [row["Ticker"].strip() for row in rows]
    if len(tickers) != len(set(tickers)):
        raise ValidationError("knowledge asset contains duplicate tickers")
    numeric_columns = [
        "Current_Price", "Forecast_12M", "Forecast_24M", "Forecast_36M",
        "Forecast_48M", "Forecast_60M", "PE_Ratio", "Avg_Historical_CAGR",
    ]
    for row_index, row in enumerate(rows, start=2):
        for column in numeric_columns:
            try:
                value = float(row[column])
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"knowledge asset row {row_index} has invalid {column}"
                ) from exc
            if not math.isfinite(value):
                raise ValidationError(
                    f"knowledge asset row {row_index} has non-finite {column}"
                )
        if float(row["Current_Price"]) <= 0:
            raise ValidationError(f"knowledge asset row {row_index} has non-positive price")

    benchmark_path = research / "benchmark.csv"
    if not benchmark_path.exists():
        raise ValidationError("missing research/benchmark.csv")
    with benchmark_path.open("r", encoding="utf-8-sig", newline="") as handle:
        benchmark_rows = list(csv.DictReader(handle))
    if len(benchmark_rows) < 250 or "Date" not in benchmark_rows[0]:
        raise ValidationError("benchmark.csv is missing or has insufficient history")
    benchmark_dates = [
        _parse_date(row["Date"], f"benchmark.csv row {index}")
        for index, row in enumerate(benchmark_rows, start=2)
    ]
    if benchmark_dates != sorted(set(benchmark_dates)):
        raise ValidationError("benchmark dates are duplicated or not strictly increasing")
    for index, row in enumerate(benchmark_rows, start=2):
        try:
            value = float(row["Nifty50_Normalized"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"invalid benchmark value at row {index}") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValidationError(f"non-positive/non-finite benchmark value at row {index}")
    benchmark_last = _parse_date(benchmark_rows[-1]["Date"], "benchmark.csv last row")

    health = _load_json(research / "universe_health.json")
    unresolved = [
        item for item in health.get("needs_review", [])
        if item.get("status") in {"unresolved", "stale"}
    ]
    if len(unresolved) > 5:
        raise ValidationError(f"{len(unresolved)} unresolved/stale tickers")
    if unresolved:
        warnings.append(f"{len(unresolved)} unresolved/stale tickers require review")

    through_dates: set[date] = set()
    horizons: list[dict[str, Any]] = []
    max_turnover = 0.0
    for horizon in HORIZONS:
        current_path = research / f"portfolio_{horizon}y.json"
        previous_path = research / f"portfolio_{horizon}y_prev.json"
        current = _load_json(current_path)
        _assert_finite(current, f"portfolio_{horizon}y")
        if int(current.get("horizon_years", -1)) != horizon:
            raise ValidationError(f"{horizon}y bundle reports the wrong horizon")
        current_date = _parse_date(current.get("data_through"), f"portfolio_{horizon}y.data_through")
        through_dates.add(current_date)
        current_weights = _portfolio_weights(current, horizon)
        if not set(current_weights).issubset(set(tickers)):
            unknown = sorted(set(current_weights) - set(tickers))
            raise ValidationError(f"{horizon}y contains tickers outside the universe: {unknown}")

        turnover = None
        picks: list[str] = []
        drops: list[str] = []
        if previous_path.exists():
            previous = _load_json(previous_path)
            _assert_finite(previous, f"portfolio_{horizon}y_prev")
            previous_date = _parse_date(
                previous.get("data_through"), f"portfolio_{horizon}y_prev.data_through"
            )
            if previous_date >= current_date:
                raise ValidationError(f"{horizon}y previous snapshot is not older than current")
            previous_weights = _portfolio_weights(previous, horizon)
            universe = set(current_weights) | set(previous_weights)
            turnover = sum(
                abs(current_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0))
                for ticker in universe
            ) / 2.0 * 100.0
            max_turnover = max(max_turnover, turnover)
            picks = sorted(set(current_weights) - set(previous_weights))
            drops = sorted(set(previous_weights) - set(current_weights))
            if turnover > turnover_limit:
                message = (
                    f"{horizon}y turnover {turnover:.1f}% exceeds "
                    f"the {turnover_limit:.1f}% approval threshold"
                )
                if allow_high_turnover:
                    warnings.append(message)
                else:
                    errors.append(message)

        horizons.append({
            "horizon_years": horizon,
            "recommended": bool(current.get("verdict", {}).get("recommended")),
            "data_through": current_date.isoformat(),
            "num_stocks": len(current_weights),
            "weight_total": round(sum(current_weights.values()), 8),
            "turnover_pct": None if turnover is None else round(turnover, 2),
            "picks": picks,
            "drops": drops,
        })

    if len(through_dates) != 1:
        raise ValidationError(f"portfolio bundles use inconsistent dates: {sorted(through_dates)}")
    data_through = next(iter(through_dates))
    if benchmark_last != data_through:
        raise ValidationError(
            f"benchmark ends {benchmark_last}, bundles end {data_through}"
        )
    if expected_as_of and data_through > _parse_date(expected_as_of, "expected_as_of"):
        raise ValidationError(f"candidate data {data_through} exceeds requested as-of {expected_as_of}")
    reference_date = _parse_date(expected_as_of, "expected_as_of") if expected_as_of else date.today()
    age_days = (reference_date - data_through).days
    if age_days < 0:
        raise ValidationError(f"candidate data date {data_through} is in the future")
    if age_days > max_data_age_days:
        raise ValidationError(
            f"candidate data is {age_days} days old (maximum {max_data_age_days})"
        )
    if errors:
        raise ValidationError("; ".join(errors))

    return {
        "schema_version": 1,
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed",
        "data_through": data_through.isoformat(),
        "universe_size": len(rows),
        "benchmark_rows": len(benchmark_rows),
        "data_age_days": age_days,
        "unresolved_or_stale": len(unresolved),
        "max_turnover_pct": round(max_turnover, 2),
        "turnover_limit_pct": turnover_limit,
        "warnings": warnings,
        "horizons": horizons,
    }


def artifact_hashes(root: os.PathLike[str] | str, relative_paths: list[str]) -> dict[str, str]:
    base = Path(root)
    return {path.replace("\\", "/"): sha256_file(base / path) for path in relative_paths}
