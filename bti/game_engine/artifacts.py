"""Read-only adapter for promoted Vriddhi research artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class VriddhiArtifacts:
    def __init__(self, repository_root: str | Path | None = None) -> None:
        self.root = Path(repository_root or Path(__file__).resolve().parents[2])
        self.manifest = json.loads(
            (self.root / "research" / "manifest.json").read_text(encoding="utf-8")
        )
        with (self.root / "grand_table_expanded.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            self.stocks = {row["Ticker"]: self._normalise(row) for row in csv.DictReader(handle)}
        self._bundles: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _normalise(row: dict[str, str]) -> dict[str, Any]:
        result: dict[str, Any] = {"ticker": row["Ticker"], "sector": row["Sector"]}
        for key, value in row.items():
            if key not in {"Ticker", "Sector"}:
                result[key] = float(value) if value not in {None, ""} else 0.0
        return result

    @property
    def release_id(self) -> str:
        return str(self.manifest["release_id"])

    @property
    def data_through(self) -> str:
        return str(self.manifest["data_through"])

    def bundle(self, months: int) -> dict[str, Any]:
        if months not in {24, 36, 48, 60}:
            raise ValueError("Campaign horizon must be 24, 36, 48 or 60 months")
        if months not in self._bundles:
            path = self.root / "research" / f"portfolio_{months // 12}y.json"
            self._bundles[months] = json.loads(path.read_text(encoding="utf-8"))
        return self._bundles[months]

    def public_universe(self) -> list[dict[str, Any]]:
        return [dict(stock) for stock in self.stocks.values()]

    def _reference_weights(self, months: int) -> dict[str, float]:
        """Server-private reference. Never include this result in client payloads."""
        return {item["ticker"]: float(item["weight"]) for item in self.bundle(months)["stocks"]}
