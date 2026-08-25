from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _open_campaign() -> AppTest:
    app = AppTest.from_file("bti_web_app.py", default_timeout=30).run()
    app.text_input[0].set_value("BTI Tester")
    app.checkbox[0].check()
    app.button[0].click()
    app.run()
    assert not app.exception
    app.button[0].click()
    app.run()
    assert not app.exception
    return app


def test_first_time_journey_reaches_playable_market() -> None:
    app = _open_campaign()
    assert app.session_state["screen"] == "game"
    assert len(app.tabs) == 4
    assert len(app.dataframe) >= 1
    assert any(button.label == "MAKE YOUR MOVE" for button in app.button)


def test_player_can_commit_a_move_and_receive_analysis() -> None:
    app = _open_campaign()
    app.number_input[0].set_value(8)
    next(button for button in app.button if button.label == "Add step").click()
    app.run()
    app.checkbox[0].check()
    app.run()
    submit = next(button for button in app.button if button.label == "MAKE YOUR MOVE")
    assert not submit.disabled
    submit.click()
    app.run()
    assert not app.exception
    assert app.session_state["screen"] == "result"
    assert any(button.label == "Continue to move 2" for button in app.button)
