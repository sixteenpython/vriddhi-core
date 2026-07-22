import json

import pandas as pd

import vriddhi_core


def test_benchmark_series_loader_returns_dataframe(tmp_path, monkeypatch):
    (tmp_path / "benchmark.csv").write_text(
        "Date,Nifty50_Normalized\n2026-07-16,1.0\n2026-07-17,1.01\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(vriddhi_core, "RESEARCH_DIR", str(tmp_path))

    frame = vriddhi_core.load_benchmark_series()

    assert isinstance(frame, pd.DataFrame)
    assert frame["Date"].tolist() == [pd.Timestamp("2026-07-16"), pd.Timestamp("2026-07-17")]
    assert frame["Nifty50_Normalized"].tolist() == [1.0, 1.01]


def test_release_manifest_loader(tmp_path, monkeypatch):
    manifest = {"release_id": "refresh-2026-07-17", "data_through": "2026-07-17"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(vriddhi_core, "RESEARCH_DIR", str(tmp_path))
    assert vriddhi_core.load_release_manifest() == manifest


def test_oos_sip_replay_uses_stored_growth_factors():
    curve = {
        "dates": ["2026-01-02", "2026-02-02"],
        "portfolio": [1.0, 2.0],
    }

    replay = vriddhi_core.build_oos_sip_replay(curve, 1_000, periods=(2, 3))

    assert replay.loc[0, "Contributions"] == 2
    assert replay.loc[0, "Total invested"] == 2_000
    assert replay.loc[0, "Replay value"] == 3_000
    assert pd.isna(replay.loc[1, "Replay value"])


def test_scale_allocations_preserves_total():
    bundle = {
        "stocks": [
            {"ticker": "AAA", "sector": "A", "weight": 0.6, "current_price": 100},
            {"ticker": "BBB", "sector": "B", "weight": 0.4, "current_price": 200},
        ]
    }
    frame = vriddhi_core.scale_allocations(bundle, 50_000)
    assert frame["Monthly Allocation (INR)"].sum() == 50_000
    assert frame["Whole_Shares"].tolist() == [300, 100]
