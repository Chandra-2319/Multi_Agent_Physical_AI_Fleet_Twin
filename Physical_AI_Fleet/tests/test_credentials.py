"""Nemotron / Nebius setup: where the key comes from, where it is saved, how the location is remembered, live apply, test."""
import json
import os
import stat
import urllib.error

import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.fleet_config import secrets_store as ss
from Physical_AI_Fleet_python.fleet_config.settings import DEFAULT_BASE_URL, DEFAULT_NEMOTRON_MODEL, Settings
from Physical_AI_Fleet_python.reasoning import token_factory_client as tfc
from Physical_AI_Fleet_python.reasoning.token_factory_client import LLMError, TokenFactoryClient
from helpers_v4 import FakeNemotron, FakeRobot, make_cfg, make_coordinator, pump_until

SECRET = "sk-test-1234567890abcdef"


def test_default_model_is_set_so_the_key_alone_enables_nemotron():
    s = Settings()
    assert s.nemotron_model_id == DEFAULT_NEMOTRON_MODEL == "nvidia/nemotron-3-super-120b-a12b"
    assert s.nebius_base_url == DEFAULT_BASE_URL and s.nemotron_enabled is False and "no API key" in s.nemotron_status()
    s.apply_nebius(api_key=SECRET)
    assert s.nemotron_enabled is True and "ON nvidia/nemotron-3-super-120b-a12b" in s.nemotron_status()


def test_environment_variables_win_over_the_saved_file_at_start_up(monkeypatch):
    ss.save_secrets({"nebius_api_key": "from-file", "nemotron_model_id": "file/model"})
    s = Settings()
    assert s.nebius_api_key == "from-file" and s.nemotron_model_id == "file/model" and s.nebius_key_source.startswith("saved file ")
    monkeypatch.setenv("NEBIUS_API_KEY", "from-env")
    monkeypatch.setenv("NEMOTRON_MODEL_ID", "env/model")
    s = Settings()
    assert (s.nebius_api_key, s.nemotron_model_id) == ("from-env", "env/model")
    assert "environment" in s.nebius_key_source and s.cosmos_api_key == "from-env"


def test_status_never_shows_the_key():
    s = Settings()
    s.apply_nebius(api_key=SECRET)
    text = s.nemotron_status()
    assert SECRET not in text and "cdef" in text and "****" in text


def test_a_picked_folder_becomes_folder_slash_secrets_json_and_a_json_file_is_used_as_is(tmp_path):
    assert ss.normalize_secrets_path(str(tmp_path / "keys")) == tmp_path / "keys" / "secrets.json"
    assert ss.normalize_secrets_path(f'"{tmp_path}/keys/"') == tmp_path / "keys" / "secrets.json"
    assert ss.normalize_secrets_path(str(tmp_path / "mine.json")) == tmp_path / "mine.json"
    assert ss.normalize_secrets_path(str(tmp_path / "MINE.JSON")) == tmp_path / "MINE.JSON"
    assert ss.normalize_secrets_path("~/keys").is_absolute()


@pytest.mark.parametrize("text,match", [("", "empty"), ("   ", "empty"), ("keys/secrets.json", "full path"),
                                        ("omniverse://server/key.json", "on this PC"), ("https://x/y.json", "on this PC")])
def test_bad_key_file_input_gets_a_clear_message(text, match):
    with pytest.raises(ValueError, match=match):
        ss.normalize_secrets_path(text)


def test_an_existing_non_json_file_is_refused(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="folder or a .json file"):
        ss.normalize_secrets_path(str(f))


def test_saved_key_in_a_chosen_folder_is_found_again_by_a_brand_new_session(tmp_path):
    chosen = tmp_path / "my keys" / "secrets.json"
    written = ss.save_secrets({"nebius_api_key": SECRET, "nemotron_model_id": "nvidia/other"}, chosen)
    ss.remember_location(chosen)
    assert written == str(chosen) and chosen.is_file()
    s = Settings()
    assert s.nebius_api_key == SECRET and s.nemotron_model_id == "nvidia/other" and s.nemotron_enabled
    assert s.nebius_key_source == f"saved file {chosen}" and ss.secrets_path() == chosen and ss.secrets_path_source() == "last used location"
    assert not ss.default_secrets_path().exists()


def test_without_remembering_the_default_location_is_used(tmp_path):
    ss.save_secrets({"nebius_api_key": "in-default"}, tmp_path / "elsewhere.json")
    assert Settings().nebius_api_key == ""
    ss.save_secrets({"nebius_api_key": "in-default"})
    assert Settings().nebius_api_key == "in-default" and ss.secrets_path() == ss.default_secrets_path()


def test_environment_variable_file_has_priority_then_remembered_then_default(monkeypatch, tmp_path):
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", ss.default_secrets_path()
    ss.save_secrets({"nebius_api_key": "A"}, a)
    ss.save_secrets({"nebius_api_key": "B"}, b)
    ss.save_secrets({"nebius_api_key": "C"}, c)
    ss.remember_location(b)
    assert Settings().nebius_api_key == "B"
    monkeypatch.setenv("FLEET_SECRETS_FILE", str(a))
    assert Settings().nebius_api_key == "A" and ss.secrets_path_source() == "environment variable FLEET_SECRETS_FILE"
    monkeypatch.setenv("FLEET_SECRETS_FILE", str(tmp_path / "missing.json"))
    assert Settings().nebius_api_key == "B"
    b.unlink()
    assert Settings().nebius_api_key == "C"


def test_a_broken_location_file_or_key_file_never_stops_the_extension(tmp_path):
    ss.pointer_path().parent.mkdir(parents=True, exist_ok=True)
    ss.pointer_path().write_text("{not json")
    assert ss.remembered_path() is None and Settings().nebius_api_key == ""
    ss.pointer_path().write_text(json.dumps({"secrets_file": 5}))
    assert ss.remembered_path() is None
    bad = tmp_path / "k.json"
    bad.write_text("[1, 2]")
    ss.remember_location(bad)
    assert Settings().nebius_api_key == "" and ss.load_secrets(bad) == {}


def test_key_file_is_private_merged_and_unknown_keys_refused(tmp_path):
    p = tmp_path / "k" / "secrets.json"
    ss.save_secrets({"nebius_api_key": SECRET}, p)
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    ss.save_secrets({"nemotron_model_id": "x/y"}, p)
    assert ss.load_secrets(p) == {"nebius_api_key": SECRET, "nemotron_model_id": "x/y"}
    with pytest.raises(ValueError):
        ss.save_secrets({"password": "no"}, p)


def test_describe_and_forget(tmp_path):
    p = tmp_path / "secrets.json"
    assert ss.describe_secrets(p) == {"path": str(p), "exists": False, "has_key": False, "key": "(none)", "model": "", "base_url": "", "remembered": False}
    ss.save_secrets({"nebius_api_key": SECRET, "nemotron_model_id": "m/x"}, p)
    ss.remember_location(p)
    d = ss.describe_secrets(p)
    assert d["exists"] and d["has_key"] and d["key"].endswith("cdef") and SECRET not in json.dumps(d) and d["remembered"] and d["model"] == "m/x"
    assert ss.forget_secrets(p) is True and ss.forget_secrets(p) is False and not p.exists()


def test_mask_key():
    assert ss.mask_key("") == "(none)" and ss.mask_key("short") == "****"
    assert ss.mask_key(SECRET).endswith("cdef") and SECRET not in ss.mask_key(SECRET)


def test_apply_saved_file_uses_everything_in_the_file_and_refuses_a_file_without_a_key(tmp_path):
    p = tmp_path / "s.json"
    ss.save_secrets({"nebius_api_key": SECRET, "nebius_base_url": "https://other.example/v1", "nemotron_model_id": "a/b", "cosmos_model_id": "nvidia/cosmos"}, p)
    s = Settings()
    changed = s.apply_saved_file(p)
    assert {"API key", "base URL", "Nemotron model", "Cosmos model"} <= set(changed)
    assert (s.nebius_api_key, s.nebius_base_url, s.nemotron_model_id, s.cosmos_model_id) == (SECRET, "https://other.example/v1", "a/b", "nvidia/cosmos")
    assert s.cosmos_api_key == SECRET and s.cosmos_base_url == "https://other.example/v1" and s.nebius_key_source == f"saved file {p}"
    empty = tmp_path / "e.json"
    ss.save_secrets({"nemotron_model_id": "only/model"}, empty)
    with pytest.raises(ValueError, match="no API key"):
        s.apply_saved_file(empty)


def test_apply_updates_cosmos_only_when_it_follows_the_nebius_values(monkeypatch):
    s = Settings()
    s.apply_nebius(api_key="k1", base_url="https://other.example/v1/", model_id="a/b")
    assert s.cosmos_api_key == "k1" and s.cosmos_base_url == "https://other.example/v1" and s.nebius_base_url == "https://other.example/v1"
    monkeypatch.setenv("COSMOS_API_KEY", "cosmos-own")
    s2 = Settings()
    s2.apply_nebius(api_key="k2")
    assert s2.nebius_api_key == "k2" and s2.cosmos_api_key == "cosmos-own"


def test_apply_returns_what_changed_and_keeps_blank_fields():
    s = Settings()
    assert s.apply_nebius(api_key="", base_url="", model_id="", cosmos_model_id=None) == []
    assert s.apply_nebius(api_key="k", cosmos_model_id="nvidia/cosmos-reason") == ["API key", "Cosmos model"]
    assert s.cosmos_enabled is True and s.nemotron_model_id == DEFAULT_NEMOTRON_MODEL


@pytest.mark.parametrize("kw,match", [({"base_url": "ftp://x"}, "https://"), ({"base_url": "api.nebius.com"}, "https://"),
                                      ({"model_id": "bad model"}, "spaces"), ({"cosmos_model_id": "bad model"}, "spaces")])
def test_apply_rejects_bad_input_and_changes_nothing(kw, match):
    s = Settings()
    before = (s.nebius_base_url, s.nemotron_model_id, s.cosmos_model_id)
    with pytest.raises(ValueError, match=match):
        s.apply_nebius(api_key="k", **kw)
    assert (s.nebius_base_url, s.nemotron_model_id, s.cosmos_model_id) == before and s.nebius_api_key == ""


def test_running_coordinator_picks_up_a_key_entered_after_load(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path)
    coord.brain.client = coord.nebius
    s = coord.s
    s.nebius_api_key = ""
    coord.nebius.api_key = ""
    assert coord.brain.nemotron_available is False
    s.nebius_api_key, s.cosmos_api_key, s.nemotron_model_id = SECRET, SECRET, "nvidia/new-model"
    s.nebius_base_url = "https://changed.example/v1/"
    coord.refresh_credentials()
    assert coord.nebius.api_key == SECRET and coord.nebius.base_url == "https://changed.example/v1"
    assert coord.brain.model_id == "nvidia/new-model" and coord.assigner.model_id == "nvidia/new-model"
    assert coord.cosmos_client.api_key == SECRET and coord.brain.nemotron_available is True
    assert coord.audit.last("credentials_updated")["nemotron"] is True and SECRET not in json.dumps(coord.audit.records, default=str)


def test_config_file_model_override_still_wins(monkeypatch, tmp_path):
    cfg = make_cfg(lambda r: r["nemotron"].update(model_id="pinned/model"))
    coord, _ = make_coordinator(monkeypatch, tmp_path, cfg=cfg)
    coord.s.nemotron_model_id = "ui/model"
    coord.refresh_credentials()
    assert coord.brain.model_id == "pinned/model"


class Resp:
    def __init__(self, payload): self.p = payload
    def read(self): return json.dumps(self.p).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_ping_success_reports_latency_and_uses_no_retries(monkeypatch):
    seen = {}
    def urlopen(req, timeout):
        seen["body"], seen["timeout"] = json.loads(req.data.decode()), timeout
        return Resp({"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}], "usage": {}})
    monkeypatch.setattr(tfc.urllib.request, "urlopen", urlopen)
    r = TokenFactoryClient("https://h/v1", SECRET).ping("nvidia/m")
    assert r["ok"] and r["status"] == 200 and "Connected" in r["message"] and SECRET not in r["message"]
    assert seen["body"]["model"] == "nvidia/m" and seen["body"]["max_tokens"] == 32 and seen["timeout"] == 15.0


def test_ping_is_ok_even_if_a_reasoning_model_runs_out_of_its_tiny_budget(monkeypatch):
    monkeypatch.setattr(tfc.urllib.request, "urlopen", lambda req, timeout: Resp({"choices": [{"message": {"content": None}, "finish_reason": "length"}], "usage": {}}))
    assert TokenFactoryClient("https://h/v1", "k").ping("m")["ok"] is True


@pytest.mark.parametrize("code,expect", [(401, "rejected the API key"), (403, "rejected the API key"), (404, "model id or the base URL"), (429, "Rate limited"), (500, "HTTP 500")])
def test_ping_explains_http_errors(monkeypatch, code, expect):
    calls = {"n": 0}
    def urlopen(req, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", code, "x", {}, None)
    monkeypatch.setattr(tfc.urllib.request, "urlopen", urlopen)
    r = TokenFactoryClient("https://h/v1", SECRET).ping("m")
    assert r["ok"] is False and r["status"] == code and expect in r["message"] and calls["n"] == 1 and SECRET not in r["message"]


def test_ping_explains_network_problems_and_missing_settings(monkeypatch):
    def urlopen(req, timeout):
        raise urllib.error.URLError("Name or service not known")
    monkeypatch.setattr(tfc.urllib.request, "urlopen", urlopen)
    r = TokenFactoryClient("https://api.tokenfactory.nebius.com/v1", "k").ping("m")
    assert not r["ok"] and r["status"] is None and "Cannot reach api.tokenfactory.nebius.com" in r["message"]
    assert "No API key" in TokenFactoryClient("https://h/v1", "").ping("m")["message"]
    assert "No model id" in TokenFactoryClient("https://h/v1", "k").ping("")["message"]


def test_llm_error_keeps_the_status_for_callers():
    assert LLMError("x", 401).status == 401 and LLMError("x").status is None


def test_brain_health_counts_nemotron_answers_and_fallbacks(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path, nemotron=FakeNemotron())
    assert coord.brain.health_line() == "Brain: no decision yet"
    coord.fleet.add(FakeRobot("go2-1", "go2"))
    assert pump_until(coord, clock, lambda: coord.executor.started)
    assert coord.brain.ok == 1 and coord.brain.failed == 0 and "answered 1/1" in coord.brain.health_line() and coord.brain.health_line() in coord.status_lines()
    coord2, clock2 = make_coordinator(monkeypatch, tmp_path / "b", nemotron=FakeNemotron("garbage"))
    coord2.fleet.add(FakeRobot("go2-1", "go2"))
    assert pump_until(coord2, clock2, lambda: coord2.executor.started)
    line = coord2.brain.health_line()
    assert "answered 0/1" in line and "last rule_based" in line and "validation" in line


def test_no_key_is_reported_as_the_reason_for_the_fallback(monkeypatch, tmp_path):
    nem = FakeNemotron()
    nem.api_key = ""
    coord, clock = make_coordinator(monkeypatch, tmp_path, nemotron=nem)
    coord.fleet.add(FakeRobot("go2-1", "go2"))
    assert pump_until(coord, clock, lambda: coord.executor.started)
    assert "no API key" in coord.brain.last_error and nem.calls == []
