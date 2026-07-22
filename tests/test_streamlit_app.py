import os
import subprocess
import sys

import pytest


@pytest.mark.app
def test_streamlit_renders_every_horizon_in_isolated_process():
    # Streamlit/matplotlib can retain native resources between AppTest instances
    # on Windows.  The production smoke test intentionally uses the same process
    # boundary so a UI teardown cannot corrupt the orchestrator.
    code = (
        "from streamlit.testing.v1 import AppTest\n"
        "for horizon in (1, 2, 3, 4, 5):\n"
        "    app = AppTest.from_file('streamlit_app.py', default_timeout=90).run()\n"
        "    assert not app.exception, app.exception\n"
        "    landing_text = ' '.join(x.value for group in (app.markdown, app.success) for x in group)\n"
        "    assert 'Invest with evidence, not emotion.' in landing_text\n"
        "    assert 'AI-assisted portfolio decision engine' in landing_text\n"
        "    assert 'What Vriddhi does' in landing_text\n"
        "    app.sidebar.selectbox[0].set_value(horizon).run()\n"
        "    app.sidebar.button[0].click().run()\n"
        "    assert not app.exception, (horizon, app.exception)\n"
        "    text_groups = (app.markdown, app.info, app.success, app.warning, app.caption)\n"
        "    page_text = ' '.join(x.value for group in text_groups for x in group)\n"
        "    assert 'What if you had started earlier?' in page_text, horizon\n"
        "    assert 'Actual Vriddhi Track Record' in page_text, horizon\n"
        "    assert 'Prospective evidence collection is active' in page_text, horizon\n"
        "    assert 'What should I do this month?' in page_text, horizon\n"
        "    assert 'health check-up for your investments' in page_text, horizon\n"
        "    assert all(action in page_text for action in ('PICK', 'DROP', 'TOP-UP', 'TRIM', 'HOLD')), horizon\n"
        "    assert '₹50,000 every month' in page_text, horizon\n"
        "    assert 'Why rebalance every month?' in page_text, horizon\n"
        "    assert 'Which stock will go up tomorrow?' in page_text, horizon\n"
        "    assert 'The two portfolios being compared' in page_text, horizon\n"
        "    assert 'Current portfolio —' in page_text, horizon\n"
        "    assert 'Previous portfolio —' in page_text, horizon\n"
        "    portfolio_tables = [x.value for x in app.dataframe if 'Whole Shares' in x.value.columns]\n"
        "    assert len(portfolio_tables) >= 3, (horizon, len(portfolio_tables))\n"
        "    action_tables = [x.value for x in app.dataframe if 'Action' in x.value.columns]\n"
        "    assert action_tables and 'Rationale' in action_tables[0].columns, horizon\n"
        "    assert action_tables[0]['Rationale'].str.len().gt(80).all(), horizon\n"
        "    assert action_tables[0]['Rationale'].str.contains('long-term').all(), horizon\n"
        "    assert action_tables[0]['Rationale'].str.contains('The numbers behind it:').all(), horizon\n"
        "    if horizon >= 2:\n"
        "        assert 'Why should I trust these results?' in page_text, horizon\n"
        "        assert 'How Vriddhi thinks' in page_text, horizon\n"
    )
    env = os.environ.copy()
    env["STREAMLIT_SERVER_ENABLE_CORS"] = "true"
    env["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "true"
    env["STREAMLIT_LOGGER_LEVEL"] = "error"
    result = subprocess.run([sys.executable, "-c", code], timeout=300, env=env)
    assert result.returncode == 0
