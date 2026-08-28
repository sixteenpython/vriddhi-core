"""Read-only adapter for promoted Vriddhi research artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


class ArtifactIntegrityError(RuntimeError):
    """A promoted Vriddhi release is incomplete, stale or internally inconsistent."""


@lru_cache(maxsize=64)
def _verified_digest(path_text: str, expected_hash: str) -> str:
    """Hash each immutable deployment artifact only once per service process."""
    path = Path(path_text)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        raise ArtifactIntegrityError(f"Promoted artifact hash mismatch: {path.name}")
    return actual_hash


class VriddhiArtifacts:
    REQUIRED_ARTIFACTS = (
        "grand_table_expanded.csv",
        "research/portfolio_2y.json",
        "research/portfolio_3y.json",
        "research/portfolio_4y.json",
        "research/portfolio_5y.json",
    )

    def __init__(self, repository_root: str | Path | None = None) -> None:
        self.root = Path(repository_root or Path(__file__).resolve().parents[2])
        self.manifest = json.loads(
            (self.root / "research" / "manifest.json").read_text(encoding="utf-8")
        )
        self._verification = self._verify_promoted_release()
        with (self.root / "grand_table_expanded.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            self.stocks = {row["Ticker"]: self._normalise(row) for row in csv.DictReader(handle)}
        if len(self.stocks) != 50:
            raise ArtifactIntegrityError(
                f"Promoted Vriddhi universe must contain 50 stocks; found {len(self.stocks)}"
            )
        self._bundles: dict[int, dict[str, Any]] = {}

    def _verify_promoted_release(self) -> dict[str, Any]:
        release_id = str(self.manifest.get("release_id", "")).strip()
        data_through = str(self.manifest.get("data_through", "")).strip()
        if not release_id or not data_through:
            raise ArtifactIntegrityError("Vriddhi manifest is missing release_id or data_through")
        if self.manifest.get("validation_status") != "passed":
            raise ArtifactIntegrityError("Vriddhi manifest is not a validated promoted release")
        expected = self.manifest.get("artifacts_sha256")
        if not isinstance(expected, dict):
            raise ArtifactIntegrityError("Vriddhi manifest is missing artifact hashes")

        verified: dict[str, str] = {}
        for relative in self.REQUIRED_ARTIFACTS:
            path = self.root / relative
            expected_hash = str(expected.get(relative, ""))
            if not path.is_file() or not expected_hash:
                raise ArtifactIntegrityError(f"Required promoted artifact is missing: {relative}")
            actual_hash = _verified_digest(str(path.resolve()), expected_hash)
            verified[relative] = actual_hash
        return {
            "ready": True,
            "release_id": release_id,
            "data_through": data_through,
            "validation_status": "passed",
            "verified_artifacts": len(verified),
            "built_at": self.manifest.get("built_at"),
            "data_provider": self.manifest.get("data_provider"),
        }

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

    def status(self) -> dict[str, Any]:
        """Deployment-safe evidence that BTI is aligned to the promoted Vriddhi release."""
        return dict(self._verification)

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
