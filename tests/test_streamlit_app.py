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
        "    app.sidebar.selectbox[0].set_value(horizon).run()\n"
        "    app.sidebar.button[0].click().run()\n"
        "    assert not app.exception, (horizon, app.exception)\n"
    )
    env = os.environ.copy()
    env["STREAMLIT_SERVER_ENABLE_CORS"] = "true"
    env["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "true"
    env["STREAMLIT_LOGGER_LEVEL"] = "error"
    result = subprocess.run([sys.executable, "-c", code], timeout=300, env=env)
    assert result.returncode == 0
