import json

import vriddhi_core


def test_release_manifest_loader(tmp_path, monkeypatch):
    manifest = {"release_id": "refresh-2026-07-17", "data_through": "2026-07-17"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(vriddhi_core, "RESEARCH_DIR", str(tmp_path))
    assert vriddhi_core.load_release_manifest() == manifest


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
