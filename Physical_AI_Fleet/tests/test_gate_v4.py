import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.coordinator.fleet_state import FleetState
from Physical_AI_Fleet_python.safety.gate import APPROVED, PENDING, REJECTED, SafetyGate
from Physical_AI_Fleet_python.schemas.task_assignment import TaskAssignment
from helpers_v4 import FakeAsset, FakeRobot, FakeSettings, make_cfg


def setup(tmp_path, robot, cfg=None, assets=None):
    s = FakeSettings(tmp_path, assets)
    f = FleetState(s)
    f.add(robot)
    cfg = cfg or make_cfg()
    return SafetyGate(s, cfg), f, cfg


def ta(robot_id, skill, asset="pump-01", source="nemotron"):
    return TaskAssignment("FD-1", robot_id, skill, asset, "medium", "r", source=source)


def enable_g1(raw):
    raw["robot_models"]["g1"].update(available=True)
    raw["robot_models"]["g1"]["locomotion"] = {"policy_path": "/p.pt", "env_yaml_path": "/e.yaml"}


def test_inspect_on_a_simulated_go2_is_approved(tmp_path):
    g, f, _ = setup(tmp_path, FakeRobot("go2-1", "go2"))
    assert g.check(ta("go2-1", "read_gauge"), None, f, action="inspect").status == APPROVED


def test_a_robot_walking_home_counts_as_idle_for_the_gate(tmp_path):
    r = FakeRobot("go2-1", "go2")
    r.status = "busy"
    g, f, _ = setup(tmp_path, r)
    d = g.check(ta("go2-1", "read_gauge"), None, f, action="inspect")
    assert d.status == REJECTED and any("is busy" in x for x in d.reasons)  # genuinely busy: refused
    f.homing.add("go2-1")
    assert g.check(ta("go2-1", "read_gauge"), None, f, action="inspect").status == APPROVED
    r.status = "error"
    d = g.check(ta("go2-1", "read_gauge"), None, f, action="inspect")
    assert d.status == REJECTED and any("is error" in x for x in d.reasons)  # an error is never 'available'


def test_action_and_skill_must_match(tmp_path):
    g, f, _ = setup(tmp_path, FakeRobot("go2-1", "go2"))
    assert g.check(ta("go2-1", "reinspect_asset"), None, f, action="inspect").status == REJECTED
    assert g.check(ta("go2-1", "read_gauge"), None, f, action="made_up").status == REJECTED


def test_model_must_be_configured_for_the_action(tmp_path):
    g, f, _ = setup(tmp_path, FakeRobot("h1-1", "h1"))
    d = g.check(ta("h1-1", "read_gauge"), None, f, action="inspect")
    assert d.status == REJECTED and any("not configured" in r for r in d.reasons)


def test_unavailable_g1_cannot_pick_up(tmp_path):
    g, f, _ = setup(tmp_path, FakeRobot("g1-1", "g1"))
    assert g.check(ta("g1-1", "pick_tool"), None, f, action="pickup_tool").status == REJECTED


def test_pickup_always_needs_operator_approval_and_checks_the_tool_station(tmp_path):
    g, f, _ = setup(tmp_path, FakeRobot("g1-1", "g1"), make_cfg(enable_g1))
    d = g.check(ta("g1-1", "pick_tool"), None, f, action="pickup_tool")
    assert d.status == PENDING and "operator approval" in d.reasons[0]

    def far(raw):
        enable_g1(raw)
        raw["locations"]["tool_station"]["pose"] = [99.0, 0.0, 0.0]
    g2, f2, _ = setup(tmp_path, FakeRobot("g1-1", "g1"), make_cfg(far))
    d = g2.check(ta("g1-1", "pick_tool"), None, f2, action="pickup_tool")
    assert d.status == REJECTED and any("outside geofence" in r for r in d.reasons)


def test_real_robot_policy(tmp_path):
    g, f, _ = setup(tmp_path, FakeRobot("real-go2", "go2", mode="digital_twin"))
    d = g.check(ta("real-go2", "read_gauge"), None, f, action="inspect")
    assert d.status == PENDING and "real robot" in d.reasons[0]
    auto = make_cfg(lambda r: r["policy"].update(real_robot_autonomy="auto_low_risk"))
    g, f, _ = setup(tmp_path, FakeRobot("real-go2", "go2", mode="digital_twin"), auto)
    assert g.check(ta("real-go2", "read_gauge"), None, f, action="inspect").status == APPROVED
    off = make_cfg(lambda r: r["policy"].update(real_robot_autonomy="off"))
    g, f, _ = setup(tmp_path, FakeRobot("real-go2", "go2", mode="digital_twin"), off)
    d = g.check(ta("real-go2", "read_gauge"), None, f, action="inspect")
    assert d.status == REJECTED and "autonomy is off" in d.reasons[-1]
    g, f, _ = setup(tmp_path, FakeRobot("g1-real", "g1", mode="digital_twin"), make_cfg(enable_g1))
    d = g.check(ta("g1-real", "pick_tool"), None, f, action="pickup_tool")
    assert d.status == REJECTED and any("not allowed on real robots" in r for r in d.reasons)


def test_availability_freshness_and_geofence(tmp_path):
    r = FakeRobot("go2-1", "go2")
    r.status = "busy"
    g, f, _ = setup(tmp_path, r)
    assert g.check(ta("go2-1", "read_gauge"), None, f, action="inspect").status == REJECTED
    r.status, r.age = "idle", 99.0
    d = g.check(ta("go2-1", "read_gauge"), None, f, action="inspect")
    assert d.status == REJECTED and any("stale" in x for x in d.reasons)
    g2, f2, _ = setup(tmp_path, FakeRobot("go2-1", "go2"), assets={"pump-01": FakeAsset("pump-01", (50, 0), (49, 0, 0))})
    d = g2.check(ta("go2-1", "read_gauge"), None, f2, action="inspect")
    assert d.status == REJECTED and any("outside geofence" in x for x in d.reasons)
    d = g2.check(ta("go2-1", "read_gauge", asset="nope"), None, f2, action="inspect")
    assert d.status == REJECTED and any("unknown asset" in x for x in d.reasons)
    assert g.check(ta("ghost", "read_gauge"), None, f, action="inspect").status == REJECTED


def test_amr_follow_up_is_automatic_for_the_carrier_role_only(tmp_path):
    cfg = make_cfg()
    fu = cfg.follow_ups["amr_tool_delivery"]
    g, f, _ = setup(tmp_path, FakeRobot("amr-1", "amr"), cfg)
    assert g.check(ta("amr-1", "deliver_tools", source="rule_based"), None, f, follow_up=fu).status == APPROVED
    g, f, _ = setup(tmp_path, FakeRobot("go2-1", "go2"), cfg)
    d = g.check(ta("go2-1", "deliver_tools", source="rule_based"), None, f, follow_up=fu)
    assert d.status == REJECTED and any("role" in x for x in d.reasons)
    g, f, _ = setup(tmp_path, FakeRobot("amr-1", "amr"), cfg)
    assert g.check(ta("amr-1", "pick_tool", source="rule_based"), None, f, follow_up=fu).status == REJECTED


def test_legacy_behaviour_without_an_action_is_unchanged(tmp_path):
    g, f, _ = setup(tmp_path, FakeRobot("amr-1", "amr"))
    d = g.check(ta("amr-1", "deliver_tools"), None, f)
    assert d.status == PENDING and "high-risk" in d.reasons[0]
    assert g.check(ta("amr-1", "deliver_tools", source="operator"), None, f).status == APPROVED
    g, f, _ = setup(tmp_path, FakeRobot("go2-1", "go2"))
    assert g.check(ta("go2-1", "deliver_tools"), None, f).status == REJECTED
    assert g.check(ta("go2-1", "read_gauge"), None, f).status == APPROVED
