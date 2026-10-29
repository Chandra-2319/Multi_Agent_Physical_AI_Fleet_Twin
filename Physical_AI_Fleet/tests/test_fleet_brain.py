import json
import urllib.error

import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.reasoning import token_factory_client as tfc
from Physical_AI_Fleet_python.reasoning.fleet_brain import SYSTEM_PROMPT, FleetBrain, rule_based_decide
from Physical_AI_Fleet_python.reasoning.token_factory_client import LLMError, TokenFactoryClient
from helpers_v4 import FakeAudit, FakeNemotron, make_cfg
from test_snapshot_and_decision import snapshot


def brain(tmp_path, client, **kw):
    return FleetBrain(client, "nvidia/nemotron-test", make_cfg(), FakeAudit(str(tmp_path)), out_dir=tmp_path / "n", **kw)


def reply(s, **over):
    d = {"snapshot_id": s["snapshot_id"], "decision": "assign", "task_id": "inspect:pump-01", "robot_id": "go2-1", "priority": "low", "rationale": "ok"}
    d.update(over)
    return json.dumps(d)


def test_the_prompt_tells_the_model_that_a_robot_walking_home_is_available():
    assert "returning_home=true is available" in SYSTEM_PROMPT and "high > medium > low" in SYSTEM_PROMPT


def test_valid_reply_uses_nemotron_and_saves_both_files(tmp_path):
    s = snapshot(tmp_path)
    client = FakeNemotron(lambda m, rf: reply(s))
    d = brain(tmp_path, client).decide(s)
    assert d.source == "nemotron" and d.robot_id == "go2-1" and len(client.calls) == 1
    assert client.formats[0]["type"] == "json_schema" and client.formats[0]["json_schema"]["strict"] is True
    assert (tmp_path / "n" / f"{s['snapshot_id']}.json").exists()
    rec = json.loads((tmp_path / "n" / f"{s['snapshot_id']}_decision.json").read_text())
    assert rec["decision"]["source"] == "nemotron" and rec["attempts"][0]["structured"] is True
    assert "created_ts" not in json.loads(client.calls[0][1]["content"])


def test_one_corrective_retry_then_success(tmp_path):
    s = snapshot(tmp_path)
    answers = iter([reply(s, robot_id="go2-9"), reply(s)])
    client = FakeNemotron(lambda m, rf: next(answers))
    d = brain(tmp_path, client).decide(s)
    assert d.source == "nemotron" and len(client.calls) == 2 and "go2-9" in client.calls[1][-1]["content"]


def test_two_bad_replies_fall_back_to_rules(tmp_path):
    s = snapshot(tmp_path)
    audit = FakeAudit(str(tmp_path))
    d = FleetBrain(FakeNemotron("garbage"), "m", make_cfg(), audit, out_dir=tmp_path / "n").decide(s)
    assert d.source == "rule_based" and d.robot_id == "go2-1" and "assigner_fallback" in audit.kinds()


def test_api_failure_falls_back_without_a_retry_storm(tmp_path):
    s = snapshot(tmp_path)
    def boom(m, rf): raise LLMError("HTTP 503: down")
    client = FakeNemotron(boom)
    assert brain(tmp_path, client).decide(s).source == "rule_based" and len(client.calls) == 1


def test_endpoint_rejecting_response_format_switches_to_prompt_only(tmp_path):
    s = snapshot(tmp_path)
    def fn(m, rf):
        if rf is not None:
            raise LLMError("HTTP 400: response_format not supported")
        return reply(s)
    client = FakeNemotron(fn)
    b = brain(tmp_path, client)
    assert b.decide(s).source == "nemotron" and b.structured is False
    b.decide(s)
    assert [f is not None for f in client.formats] == [True, False, False]


def test_disabled_in_ui_or_unconfigured_uses_rules(tmp_path):
    s = snapshot(tmp_path)
    client = FakeNemotron(lambda m, rf: reply(s))
    flag = {"on": False}
    b = brain(tmp_path, client, enabled_fn=lambda: flag["on"])
    assert b.decide(s).source == "rule_based" and client.calls == []
    flag["on"] = True
    assert b.decide(s).source == "nemotron"
    nokey = FakeNemotron("first")
    nokey.api_key = ""
    assert brain(tmp_path, nokey).decide(s).source == "rule_based"


def test_rule_based_prefers_priority_then_waiting_and_handles_empty(tmp_path):
    s = snapshot(tmp_path)
    s["due_tasks"] = [
        {"task_id": "a", "action": "inspect", "target_id": "x", "for_asset": "x", "reason": "", "priority": "low", "waiting_s": 900, "candidates": [{"robot_id": "r1", "distance_m": 1}]},
        {"task_id": "b", "action": "reinspect", "target_id": "y", "for_asset": "y", "reason": "", "priority": "high", "waiting_s": 1, "candidates": [{"robot_id": "r2", "distance_m": 9}]}]
    d = rule_based_decide(s)
    assert (d.task_id, d.robot_id, d.action) == ("b", "r2", "reinspect")
    s["due_tasks"] = []
    assert rule_based_decide(s).decision == "no_action"


class FakeResp:
    def __init__(self, payload): self.payload = payload
    def read(self): return json.dumps(self.payload).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_client_sends_response_format_and_checks_truncation(monkeypatch):
    sent = {}
    def fake_urlopen(req, timeout):
        sent["body"] = json.loads(req.data.decode()); sent["auth"] = req.headers["Authorization"]
        return FakeResp({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {}})
    monkeypatch.setattr(tfc.urllib.request, "urlopen", fake_urlopen)
    c = TokenFactoryClient("https://h/v1/", "KEY")
    rf = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {}}}
    assert c.chat("m", [{"role": "user", "content": "hi"}], response_format=rf) == "{}"
    assert sent["body"]["response_format"] == rf and sent["auth"] == "Bearer KEY"
    monkeypatch.setattr(tfc.urllib.request, "urlopen", lambda req, timeout: FakeResp({"choices": [{"message": {"content": "{\"a\":"}, "finish_reason": "length"}]}))
    with pytest.raises(LLMError, match="truncated"):
        c.chat("m", [])


def test_client_retries_retriable_http_errors_then_fails_clearly(monkeypatch):
    calls = {"n": 0}
    def fail(req, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 429, "rate", {}, None)
    monkeypatch.setattr(tfc.urllib.request, "urlopen", fail)
    monkeypatch.setattr(tfc.time, "sleep", lambda s: None)
    with pytest.raises(LLMError, match="HTTP 429"):
        TokenFactoryClient("https://h/v1", "K", max_retries=2).chat("m", [])
    assert calls["n"] == 3
