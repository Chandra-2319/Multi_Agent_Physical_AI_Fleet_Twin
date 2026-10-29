"""Keeps the tests away from your real ~/.physical_ai_fleet files and NEBIUS_* / FLEET_* variables."""
import pytest


@pytest.fixture(autouse=True)
def isolated_user_files(monkeypatch, tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for name in ("NEBIUS_API_KEY", "NEBIUS_BASE_URL", "NEMOTRON_MODEL_ID", "COSMOS_API_KEY", "COSMOS_BASE_URL", "COSMOS_MODEL_ID",
                 "FLEET_SECRETS_FILE", "FLEET_LOCATION_FILE", "FLEET_ASSETS_FILE", "FLEET_LOG_DIR", "FLEET_SITE_MAP",
                 "FLEET_ASSET_LAYOUT", "FLEET_ASSET_TYPES", "FLEET_ACTIONS_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FLEET_INPUTS_DIR", str(home / "ext_inputs"))  # v0.5.0: never touch the real inputs/ folder
    return home
