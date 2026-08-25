import json
from pathlib import Path

CONTENT = Path(__file__).resolve().parents[1] / "content"


def _load(name: str) -> dict:
    with (CONTENT / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_puzzle_catalog_is_valid_and_links_to_lessons() -> None:
    puzzles = _load("puzzles.json")
    lessons = _load("lessons.json")
    items = puzzles["puzzles"]
    lesson_ids = {lesson["id"] for lesson in lessons["lessons"]}

    assert puzzles["schema_version"] == "1.0"
    assert len(items) >= 10
    assert len({item["id"] for item in items}) == len(items)
    for item in items:
        assert item["difficulty"] in {"beginner", "intermediate", "advanced"}
        assert len(item["choices"]) >= 2
        assert 0 <= item["correct_choice"] < len(item["choices"])
        assert item["choices"][item["correct_choice"]]
        assert len(item["explanation"]) >= 40
        assert item["lesson_id"] in lesson_ids


def test_lesson_catalog_is_plain_english_and_complete() -> None:
    catalog = _load("lessons.json")
    items = catalog["lessons"]

    assert catalog["schema_version"] == "1.0"
    assert len(items) >= 8
    assert len({item["id"] for item in items}) == len(items)
    for item in items:
        assert 1 <= item["duration_minutes"] <= 10
        assert len(item["plain_english"]) >= 80
        assert len(item["decision_rule"]) >= 40
        assert item["example"]
        assert item["check_question"].endswith("?")
        assert item["check_answer"]


def test_content_is_investment_education_not_return_promises() -> None:
    text = (CONTENT / "puzzles.json").read_text(encoding="utf-8")
    text += (CONTENT / "lessons.json").read_text(encoding="utf-8")
    forbidden = ("guaranteed return", "risk-free profit", "cannot lose money")

    assert not any(term in text.lower() for term in forbidden)
