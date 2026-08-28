"""Vriddhi-to-BTI promotion and frozen-campaign release contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bti.game_engine.artifacts import (
    ArtifactIntegrityError,
    VriddhiArtifacts,
    _verified_digest,
)
from bti.game_engine.engine import BTIGame


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(root: Path) -> None:
    research = root / "research"
    research.mkdir(parents=True)
    table = root / "grand_table_expanded.csv"
    table.write_text(
        "Ticker,Sector\n"
        + "".join(f"TEST{index:02d},Testing\n" for index in range(50)),
        encoding="utf-8",
    )
    paths = ["grand_table_expanded.csv"]
    for years in (2, 3, 4, 5):
        relative = f"research/portfolio_{years}y.json"
        (root / relative).write_text(
            json.dumps({"data_through": "2026-09-30", "stocks": []}),
            encoding="utf-8",
        )
        paths.append(relative)
    manifest = {
        "release_id": "refresh-2026-09-30",
        "data_through": "2026-09-30",
        "validation_status": "passed",
        "built_at": "2026-10-01T00:00:00+05:30",
        "data_provider": "test fixture",
        "artifacts_sha256": {relative: _sha256(root / relative) for relative in paths},
    }
    (research / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_promoted_release_is_hash_verified_before_bti_uses_it(tmp_path: Path) -> None:
    _candidate(tmp_path)
    adapter = VriddhiArtifacts(tmp_path)

    assert adapter.status()["ready"] is True
    assert adapter.release_id == "refresh-2026-09-30"
    assert adapter.status()["verified_artifacts"] == 5

    (tmp_path / "grand_table_expanded.csv").write_text("tampered", encoding="utf-8")
    _verified_digest.cache_clear()
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        VriddhiArtifacts(tmp_path)


def test_promoted_release_hash_accepts_git_line_ending_normalisation(tmp_path: Path) -> None:
    _candidate(tmp_path)
    table = tmp_path / "grand_table_expanded.csv"
    table.write_bytes(table.read_bytes().replace(b"\r\n", b"\n"))
    _verified_digest.cache_clear()

    assert VriddhiArtifacts(tmp_path).status()["ready"] is True


def test_campaign_carries_its_vriddhi_baseline_across_future_refreshes() -> None:
    game = BTIGame.create(25_000, 24, "frozen-vriddhi-release")
    payload = json.loads(game.to_json())
    payload["release_id"] = "refresh-older-than-current"
    payload["data_through"] = "2026-07-17"

    restored = BTIGame.from_json(json.dumps(payload))

    assert restored.state["release_id"] == "refresh-older-than-current"
    assert restored.state["data_through"] == "2026-07-17"
    assert restored.state["baseline_stocks"] == payload["baseline_stocks"]
    assert restored.state["reference_weights"] == payload["reference_weights"]
