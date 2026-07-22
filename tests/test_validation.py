import csv
import json
from datetime import date, timedelta

import pytest

from vriddhi_ledger import update_recommendation_ledger
from vriddhi_validation import ValidationError, validate_candidate


def _write_candidate(tmp_path, *, turnover=False):
    data = tmp_path / "grand_table_expanded.csv"
    fields = [
        "Ticker", "Sector", "Current_Price", "Forecast_12M", "Forecast_24M",
        "Forecast_36M", "Forecast_48M", "Forecast_60M", "PE_Ratio",
        "Avg_Historical_CAGR",
    ]
    with data.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(50):
            writer.writerow({
                "Ticker": f"STOCK{index}", "Sector": "Test", "Current_Price": 100 + index,
                "Forecast_12M": 10, "Forecast_24M": 10, "Forecast_36M": 10,
                "Forecast_48M": 10, "Forecast_60M": 10, "PE_Ratio": 20,
                "Avg_Historical_CAGR": 12,
            })

    research = tmp_path / "research"
    research.mkdir()
    start = date.today() - timedelta(days=399)
    with (research / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Date", "Nifty50_Normalized"])
        writer.writeheader()
        for index in range(400):
            writer.writerow({
                "Date": (start + timedelta(days=index)).isoformat(),
                "Nifty50_Normalized": 1 + index / 1000,
            })
    through = (start + timedelta(days=399)).isoformat()
    health = {
        "summary": {"ok": 50}, "needs_review": [],
        "tickers": [{"ticker": f"STOCK{i}", "status": "ok"} for i in range(50)],
    }
    (research / "universe_health.json").write_text(json.dumps(health), encoding="utf-8")

    current_names = [f"STOCK{i}" for i in range(10)]
    previous_names = [f"STOCK{i}" for i in range(10, 20)] if turnover else current_names
    for horizon in range(1, 6):
        base = {
            "horizon_years": horizon,
            "num_stocks": 10,
            "data_through": through,
            "verdict": {"recommended": horizon >= 4},
            "stocks": [
                {"ticker": ticker, "weight": 0.1, "current_price": 100, "sector": "Test"}
                for ticker in current_names
            ],
        }
        previous = dict(base)
        previous["data_through"] = (start + timedelta(days=350)).isoformat()
        previous["stocks"] = [
            {"ticker": ticker, "weight": 0.1, "current_price": 100, "sector": "Test"}
            for ticker in previous_names
        ]
        (research / f"portfolio_{horizon}y.json").write_text(json.dumps(base), encoding="utf-8")
        (research / f"portfolio_{horizon}y_prev.json").write_text(
            json.dumps(previous), encoding="utf-8"
        )
    update_recommendation_ledger(research, data)
    return data, research


def test_valid_candidate_passes(tmp_path):
    data, research = _write_candidate(tmp_path)
    report = validate_candidate(data, research)
    assert report["status"] == "passed"
    assert report["universe_size"] == 50
    assert [item["horizon_years"] for item in report["horizons"]] == [1, 2, 3, 4, 5]


def test_high_turnover_requires_explicit_approval(tmp_path):
    data, research = _write_candidate(tmp_path, turnover=True)
    with pytest.raises(ValidationError, match="turnover"):
        validate_candidate(data, research, turnover_limit=30)
    report = validate_candidate(
        data, research, turnover_limit=30, allow_high_turnover=True
    )
    assert report["max_turnover_pct"] == 100
    assert report["warnings"]


def test_non_finite_json_is_rejected(tmp_path):
    data, research = _write_candidate(tmp_path)
    path = research / "portfolio_1y.json"
    payload = path.read_text(encoding="utf-8").replace('"weight": 0.1', '"weight": NaN', 1)
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValidationError, match="non-finite"):
        validate_candidate(data, research)


def test_recommendation_ledger_is_required(tmp_path):
    data, research = _write_candidate(tmp_path)
    (research / "recommendation_ledger.json").unlink()
    with pytest.raises(ValidationError, match="recommendation_ledger"):
        validate_candidate(data, research)


def test_recommendation_ledger_does_not_duplicate_same_date(tmp_path):
    data, research = _write_candidate(tmp_path)
    ledger, changed = update_recommendation_ledger(research, data)
    assert changed is False
    assert len(ledger["snapshots"]) == 2
