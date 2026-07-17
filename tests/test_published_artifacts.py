from pathlib import Path

from vriddhi_validation import validate_candidate

ROOT = Path(__file__).resolve().parents[1]


def test_committed_research_artifacts_are_valid():
    report = validate_candidate(
        ROOT / "grand_table_expanded.csv",
        ROOT / "research",
        turnover_limit=100,
        allow_high_turnover=True,
    )
    assert report["status"] == "passed"


def test_all_horizons_share_one_data_date():
    report = validate_candidate(
        ROOT / "grand_table_expanded.csv",
        ROOT / "research",
        turnover_limit=100,
        allow_high_turnover=True,
    )
    assert len({item["data_through"] for item in report["horizons"]}) == 1
