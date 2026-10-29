import copy
import json

import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.fleet_config.action_config import ConfigError, FleetActionConfig, load_action_config
from helpers_v4 import make_cfg, raw_config


def test_shipped_config_loads_and_matches_the_catalogue():
    cfg = load_action_config()
    assert set(cfg.actions) == {"inspect", "reinspect", "pickup_tool"}
    assert cfg.actions["inspect"].executor_skill == "read_gauge" and cfg.actions["pickup_tool"].target_id == "tool_station"
    assert cfg.actions["pickup_tool"].approval == "operator" and cfg.actions["pickup_tool"].real_robot_allowed is False


def test_shipped_defaults_are_quiet_and_quadrupeds_go_home_after_a_reading():
    cfg = load_action_config()
    assert cfg.schedule["routine_inspections"] is False and cfg.schedule["initial_due"] is False
    assert cfg.nemotron["snapshot_valid_for_s"] == 90  # a reasoning model needs 10-40 s per decision
    assert cfg.actions["inspect"].return_home_after and cfg.actions["reinspect"].return_home_after
    assert not cfg.actions["pickup_tool"].return_home_after
    assert cfg.home_after_skills() == {"read_gauge", "reinspect_asset"}


def test_return_home_after_is_a_config_edit():
    cfg = make_cfg(lambda r: r["actions"]["inspect"].update(return_home_after=False))
    assert cfg.home_after_skills() == {"reinspect_asset"}
    with pytest.raises(ConfigError, match="return_home_after"):
        make_cfg(lambda r: r["actions"]["inspect"].update(return_home_after="yes"))


def test_routine_flag_must_be_boolean():
    with pytest.raises(ConfigError, match="routine_inspections"):
        make_cfg(lambda r: r["schedule"].update(routine_inspections="on"))


def test_g1_is_configured_but_unavailable_until_you_enable_it():
    cfg = load_action_config()
    assert cfg.models["g1"].actions == ("pickup_tool",) and cfg.models["g1"].available is False
    assert cfg.can("g1", "pickup_tool") is False and cfg.actions_for_model("g1") == ()
    with pytest.raises(ConfigError, match="pick-up policy in training"):
        cfg.check_model_available("g1")
    cfg.check_model_available("go2")


def test_quadruped_has_exactly_inspect_and_reinspect():
    cfg = load_action_config()
    assert cfg.actions_for_model("go2") == ("inspect", "reinspect") and cfg.can("h1", "inspect") is False


def test_enabling_g1_is_a_config_edit_only():
    def enable(raw):
        raw["robot_models"]["g1"]["available"] = True
        raw["robot_models"]["g1"]["locomotion"] = {"policy_path": "/p/g1.pt", "env_yaml_path": "/p/g1_env.yaml"}
    cfg = make_cfg(enable)
    assert cfg.can("g1", "pickup_tool") and cfg.policy_override("g1") == ("/p/g1.pt", "/p/g1_env.yaml") and cfg.policy_override("go2") == ("", "")


def test_policy_without_env_yaml_is_rejected():
    def bad(raw):
        raw["robot_models"]["go2"]["locomotion"] = {"policy_path": "/p/new.pt", "env_yaml_path": ""}
    with pytest.raises(ConfigError, match="must be set together"):
        make_cfg(bad)


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r["actions"]["inspect"].update(bogus=1), "unknown keys"),
    (lambda r: r["actions"]["inspect"].update(executor_skill="fly"), "not an executor skill"),
    (lambda r: r["actions"]["inspect"].update(risk="extreme"), "not in"),
    (lambda r: r["actions"]["pickup_tool"]["target"].update(id="nowhere"), "unknown location"),
    (lambda r: r["robot_models"]["go2"].update(actions=["inspect", "fly"]), "unknown action"),
    (lambda r: r["robot_models"]["go2"].update(actions=["pickup_tool"]), "not allowed for robot type"),
    (lambda r: r["robot_models"]["go2"].update(robot_type="humanoid"), "catalogue says"),
    (lambda r: r["robot_models"].update(rover={"robot_type": "amr", "actions": []}), "unknown model key"),
    (lambda r: r["policy"].update(real_robot_autonomy="yolo"), "not in"),
    (lambda r: r["nemotron"]["triggers"].update(periodic_s=0), "positive"),
    (lambda r: r["follow_ups"]["amr_tool_delivery"].update(after="tea_time"), "not in"),
    (lambda r: r["locations"]["tool_station"].update(pose=[1, 2]), "pose"),
    (lambda r: r.update(schema_version="2.0"), "schema_version"),
    (lambda r: r.update(surprise={}), "unknown keys"),
    (lambda r: r["policy"].update(max_unreadable_before_operator=0), ">= 1"),
])
def test_invalid_configs_are_rejected_with_a_clear_message(mutate, match):
    with pytest.raises(ConfigError, match=match):
        make_cfg(mutate)


def test_comment_keys_are_ignored_and_bad_json_reports_the_line(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{\n  \"a\": 1,\n  oops\n}")
    with pytest.raises(ConfigError, match="invalid JSON at line 3"):
        load_action_config(p)
    with pytest.raises(ConfigError, match="not found"):
        load_action_config(tmp_path / "missing.json")
    raw = copy.deepcopy(raw_config())
    raw["_note"] = "hello"
    FleetActionConfig(raw)


def test_prompt_catalogue_lists_exactly_the_three_actions():
    names = [a["action"] for a in make_cfg().catalogue_for_prompt()]
    assert names == ["inspect", "reinspect", "pickup_tool"] and json.dumps(make_cfg().catalogue_for_prompt())
