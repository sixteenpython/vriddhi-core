"""Append-only recommendation ledger for prospective Vriddhi track records."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

HORIZONS = (1, 2, 3, 4, 5)
LEDGER_FILENAME = "recommendation_ledger.json"
ACTION_THRESHOLD = 0.01
MINIMUM_RELEASES = 12

EXECUTION_POLICY = {
    "minimum_releases_for_performance": MINIMUM_RELEASES,
    "contribution_timing": "One contribution immediately before each recorded rebalance",
    "rebalance_timing": "At each published monthly recommendation snapshot",
    "share_model": "Fractional shares for normalized evidence",
    "residual_cash": "Fully invested; residual cash is not modeled",
    "transaction_costs": "Not applied; the live ledger view must disclose this limitation",
    "non_recommended_horizon": "Performance remains unavailable if any release is not recommended",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _bundle_set(research_dir: Path, suffix: str = "") -> tuple[str, dict[str, dict]]:
    bundles: dict[str, dict] = {}
    dates: set[str] = set()
    for horizon in HORIZONS:
        path = research_dir / f"portfolio_{horizon}y{suffix}.json"
        if not path.exists():
            return "", {}
        bundle = _read_json(path)
        bundles[str(horizon)] = bundle
        dates.add(str(bundle.get("data_through", "")))
    if len(dates) != 1 or not next(iter(dates)):
        raise ValueError(f"bundle set {suffix or 'current'} has inconsistent dates: {dates}")
    return next(iter(dates)), bundles


def _stock_rows(bundle: dict) -> list[dict[str, Any]]:
    return [
        {
            "ticker": str(stock["ticker"]),
            "weight": round(float(stock["weight"]), 10),
            "current_price": round(float(stock["current_price"]), 4),
            "sector": str(stock.get("sector", "Unknown")),
        }
        for stock in bundle.get("stocks", [])
    ]


def _weights(horizon_snapshot: dict | None) -> dict[str, float]:
    if not horizon_snapshot:
        return {}
    return {
        str(stock["ticker"]): float(stock["weight"])
        for stock in horizon_snapshot.get("stocks", [])
    }


def _actions(stocks: list[dict], previous: dict | None) -> list[dict[str, Any]]:
    current_weights = {str(stock["ticker"]): float(stock["weight"]) for stock in stocks}
    previous_weights = _weights(previous)
    actions = []
    for ticker in sorted(set(current_weights) | set(previous_weights)):
        old = previous_weights.get(ticker, 0.0)
        new = current_weights.get(ticker, 0.0)
        if not previous_weights:
            action = "INITIAL"
        elif old == 0 and new > 0:
            action = "PICK"
        elif new == 0 and old > 0:
            action = "DROP"
        elif new - old > ACTION_THRESHOLD:
            action = "TOP-UP"
        elif old - new > ACTION_THRESHOLD:
            action = "TRIM"
        else:
            action = "HOLD"
        actions.append({
            "ticker": ticker,
            "action": action,
            "previous_weight": round(old, 10),
            "current_weight": round(new, 10),
        })
    return actions


def _market_prices(data_csv: Path | None, bundles: dict[str, dict]) -> dict[str, float]:
    prices: dict[str, float] = {}
    if data_csv and data_csv.exists():
        with data_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    price = float(row["Current_Price"])
                except (KeyError, TypeError, ValueError):
                    continue
                if price > 0:
                    prices[str(row["Ticker"])] = round(price, 4)
    for bundle in bundles.values():
        for stock in bundle.get("stocks", []):
            price = float(stock.get("current_price", 0) or 0)
            if price > 0:
                prices.setdefault(str(stock["ticker"]), round(price, 4))
    return dict(sorted(prices.items()))


def _snapshot(
    data_through: str,
    bundles: dict[str, dict],
    *,
    market_prices: dict[str, float],
    previous: dict | None,
    provenance: str,
) -> dict[str, Any]:
    first_bundle = bundles["1"]
    horizons: dict[str, dict] = {}
    for horizon in HORIZONS:
        key = str(horizon)
        bundle = bundles[key]
        stocks = _stock_rows(bundle)
        previous_horizon = previous.get("horizons", {}).get(key) if previous else None
        horizons[key] = {
            "recommended": bool(bundle.get("verdict", {}).get("recommended")),
            "num_stocks": len(stocks),
            "stocks": stocks,
            "actions": _actions(stocks, previous_horizon),
        }
    return {
        "release_id": f"refresh-{data_through}",
        "data_through": data_through,
        "captured_at": str(first_bundle.get("generated_at") or data_through),
        "methodology_version": "2.0",
        "provenance": provenance,
        "market_prices": market_prices,
        "horizons": horizons,
    }


def update_recommendation_ledger(research_dir: Path, data_csv: Path) -> tuple[dict, bool]:
    """Seed retained history if needed and append the current release once.

    An existing date is never rewritten. This makes same-date refresh retries
    preserve the recommendation that was actually published first.
    """
    ledger_path = research_dir / LEDGER_FILENAME
    if ledger_path.exists():
        ledger = _read_json(ledger_path)
    else:
        ledger = {
            "schema_version": 1,
            "tracking_started": None,
            "execution_policy": EXECUTION_POLICY,
            "snapshots": [],
        }

    snapshots = ledger.setdefault("snapshots", [])
    existing_dates = {str(item.get("data_through")) for item in snapshots}
    changed = False

    previous_date, previous_bundles = _bundle_set(research_dir, "_prev")
    if previous_bundles and previous_date not in existing_dates:
        prior_snapshot = snapshots[-1] if snapshots else None
        snapshots.append(_snapshot(
            previous_date,
            previous_bundles,
            market_prices=_market_prices(None, previous_bundles),
            previous=prior_snapshot,
            provenance="seeded_from_retained_previous_bundle",
        ))
        existing_dates.add(previous_date)
        changed = True

    current_date, current_bundles = _bundle_set(research_dir)
    if not current_bundles:
        raise ValueError("current portfolio bundle set is incomplete")
    if current_date not in existing_dates:
        prior_snapshot = snapshots[-1] if snapshots else None
        snapshots.append(_snapshot(
            current_date,
            current_bundles,
            market_prices=_market_prices(data_csv, current_bundles),
            previous=prior_snapshot,
            provenance="monthly_refresh" if ledger_path.exists() else "seeded_from_current_bundle",
        ))
        changed = True

    snapshots.sort(key=lambda item: item["data_through"])
    ledger["tracking_started"] = snapshots[0]["data_through"] if snapshots else None
    ledger["execution_policy"] = EXECUTION_POLICY
    if changed:
        ledger["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        temporary = ledger_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, ledger_path)
    return ledger, changed


def add_ledger_hash_to_manifest(research_dir: Path) -> None:
    manifest_path = research_dir / "manifest.json"
    ledger_path = research_dir / LEDGER_FILENAME
    if not manifest_path.exists() or not ledger_path.exists():
        return
    manifest = _read_json(manifest_path)
    digest = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    manifest.setdefault("artifacts_sha256", {})[
        f"research/{LEDGER_FILENAME}"
    ] = digest
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    research = root / "research"
    ledger, appended = update_recommendation_ledger(
        research, root / "grand_table_expanded.csv"
    )
    add_ledger_hash_to_manifest(research)
    print(
        f"Recommendation ledger contains {len(ledger['snapshots'])} release(s); "
        f"{'updated' if appended else 'unchanged'}."
    )
