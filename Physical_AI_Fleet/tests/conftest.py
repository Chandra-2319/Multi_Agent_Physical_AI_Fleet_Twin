import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _never_touch_real_inputs(monkeypatch, tmp_path_factory):
    """v0.5.0: every test gets its own inputs/ folder (tests that do not import isolation.py too)."""
    import os
    if not os.environ.get("FLEET_INPUTS_DIR"):
        monkeypatch.setenv("FLEET_INPUTS_DIR", str(tmp_path_factory.mktemp("inputs")))
