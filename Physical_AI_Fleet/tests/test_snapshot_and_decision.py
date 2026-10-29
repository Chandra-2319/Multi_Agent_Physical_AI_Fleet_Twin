import json

import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.coordinator.fleet_state import CONFIRMED, FleetState, OPEN, VERIFYING
from Physical_AI_Fleet_python.reasoning.snapshot_builder import (Scheduler, attach_candidates, build_snapshot,
                                                                 compute_due_tasks, explain_no_candidates)
from Physical_AI_Fleet_python.schemas.fleet_decision import output_json_schema, validate_decision
from Physical_AI_Fleet_python.schemas.validate import SchemaError
from helpers_v4 import FakeRobot, FakeSettings, make_cfg


def fleet_with(tmp_path, robots, assets=None):
    f = FleetState(FakeSettings(tmp_path, assets))
    for r in robots:
        f.add(r)
    return f


def tasks_for(cfg, fleet, now=1000.0, busy_assets=(), busy_robots=(), sched=None):
    sched = sched or Scheduler(cfg, list(fleet.s.assets), now)
    ts = compute_due_tasks(cfg, fleet, sched, set(busy_assets), now)
    attach_candidates(cfg, fleet, ts, set(busy_robots))
    return ts, sched


def test_routine_inspection_is_due_at_start_and_only_quadrupeds_are_candidates(tmp_path):
    f = fleet_with(tmp_path, [FakeRobot("go2-1", "go2", (0, 0)), FakeRobot("go2-2", "go2", (9, 0)), FakeRobot("h1-1", "h1"), FakeRobot("amr-1", "amr")])
    ts, _ = tasks_for(make_cfg(), f)
    assert [t.task_id for t in ts] == ["inspect:pump-01"]
    assert [c[1] for c in ts[0].candidates] == ["go2-1", "go2-2"] and [c[0] for c in ts[0].candidates] == [4.0, 5.0]


def test_candidates_sorted_by_distance(tmp_path):
    f = fleet_with(tmp_path, [FakeRobot("go2-1", "go2", (0, 0)), FakeRobot("go2-2", "go2", (3.5, 0))])
    assert [c[1] for c in tasks_for(make_cfg(), f)[0][0].candidates] == ["go2-2", "go2-1"]


def test_not_due_when_schedule_says_so(tmp_path):
    cfg = make_cfg(lambda r: r["schedule"].update(initial_due=False))
    f = fleet_with(tmp_path, [FakeRobot("go2-1", "go2")])
    ts, sched = tasks_for(cfg, f, now=1000.0)
    assert ts == []
    sched.last["pump-01"] = 1000.0 - 901
    assert [t.task_id for t in tasks_for(cfg, f, now=1000.0, sched=sched)[0]] == ["inspect:pump-01"]
    sched.mark_inspected("pump-01", 999.0)
    assert tasks_for(cfg, f, now=1000.0, sched=sched)[0] == []


def test_routine_tasks_exist_only_while_routine_inspections_are_on(tmp_path):
    f = fleet_with(tmp_path, [FakeRobot("go2-1", "go2")])
    cfg = make_cfg(lambda r: r["schedule"].update(routine_inspections=False))
    ts, sched = tasks_for(cfg, f)
    assert ts == [] and sched.enabled is False
    sched.set_routine(True, now=1000.0)  # switching ON with initial_due true: everything is due
    assert [t.task_id for t in tasks_for(cfg, f, sched=sched)[0]] == ["inspect:pump-01"]


def test_switching_routine_on_starts_a_fresh_interval(tmp_path):
    cfg = make_cfg(lambda r: r["schedule"].update(routine_inspections=False, initial_due=False))
    f = fleet_with(tmp_path, [FakeRobot("go2-1", "go2")])
    sched = Scheduler(cfg, ["pump-01"], 0.0)
    sched.set_routine(True, 600.0, now=5000.0)  # hours later: must not be 'overdue' the moment it is switched on
    assert sched.last["pump-01"] == 5000.0 and sched.interval_s == 600.0
    assert tasks_for(cfg, f, now=5300.0, sched=sched)[0] == []
    assert [t.task_id for t in tasks_for(cfg, f, now=5601.0, sched=sched)[0]] == ["inspect:pump-01"]
    sched.restart(["pump-01"], 7000.0)
    assert sched.last == {"pump-01": 7000.0}


def test_scheduler_uses_its_own_clock_and_per_asset_intervals(tmp_path):
    t = {"now": 100.0}
    cfg = make_cfg(lambda r: r["schedule"].update(initial_due=False, per_asset_interval_s={"b": 50}))
    s = Scheduler(cfg, ["a", "b"], clock=lambda: t["now"])
    assert s.due_in("a") == 900.0 and s.due_in("b") == 50.0
    t["now"] = 160.0
    assert s.due_in("a") == 840.0 and s.due_in("b") == -10.0


def test_ineligible_robots_are_never_candidates(tmp_path):
    stale, busy, err, ok = (FakeRobot(n, "go2") for n in ("stale", "busy", "err", "ok"))
    stale.age, busy.status, err.status = 99.0, "busy", "error"
    assert [c[1] for c in tasks_for(make_cfg(), fleet_with(tmp_path, [stale, busy, err, ok]))[0][0].candidates] == ["ok"]
    assert tasks_for(make_cfg(), fleet_with(tmp_path, [FakeRobot("go2-1", "go2")]), busy_robots={"go2-1"})[0][0].candidates == []


def test_a_robot_walking_home_is_a_candidate_and_shown_as_idle_in_the_snapshot(tmp_path):
    walker, idle = FakeRobot("go2-1", "go2", (0, 0)), FakeRobot("go2-2", "go2", (9, 0))
    walker.status = "busy"
    f = fleet_with(tmp_path, [walker, idle])
    assert [c[1] for c in tasks_for(make_cfg(), f)[0][0].candidates] == ["go2-2"]  # busy for real
    f.homing.add("go2-1")
    ts, sched = tasks_for(make_cfg(), f)
    assert [c[1] for c in ts[0].candidates] == ["go2-1", "go2-2"]
    snap = build_snapshot(make_cfg(), f, ts, sched, {}, {"kind": "event", "reasons": []}, [], 1000.0)
    by = {r["robot_id"]: r for r in snap["robots"]}
    assert by["go2-1"]["status"] == "idle" and by["go2-1"]["returning_home"] is True and by["go2-2"]["returning_home"] is False
    walker.status = "error"
    assert [c[1] for c in tasks_for(make_cfg(), f)[0][0].candidates] == ["go2-2"]  # an error is not 'walking home'


def test_real_robot_rules(tmp_path):
    f = fleet_with(tmp_path, [FakeRobot("real-go2", "go2", mode="digital_twin")])
    assert [c[1] for c in tasks_for(make_cfg(), f)[0][0].candidates] == ["real-go2"]
    assert tasks_for(make_cfg(lambda r: r["policy"].update(real_robot_autonomy="off")), f)[0][0].candidates == []


def test_incident_states_create_the_right_tasks(tmp_path):
    cfg = make_cfg()
    f = fleet_with(tmp_path, [FakeRobot("go2-1", "go2"), FakeRobot("go2-2", "go2", (8, 0)), FakeRobot("h1-1", "h1")])
    inc = f.open_incident("pump-01")
    inc.status = CONFIRMED
    ts, _ = tasks_for(cfg, f)
    assert [(t.task_id, t.priority) for t in ts] == [("pickup_tool:pump-01", "high")] and ts[0].candidates == []
    assert "no available robot model is configured for pickup_tool" in explain_no_candidates(cfg, f, ts[0], set())
    inc.status, inc.last_inspector = VERIFYING, "go2-1"
    ts, _ = tasks_for(cfg, f)
    assert ts[0].task_id == "reinspect:pump-01" and [c[1] for c in ts[0].candidates] == ["go2-2", "go2-1"]
    inc.status, inc.needs_reinspect = OPEN, True
    ts, _ = tasks_for(cfg, f)
    assert ts[0].task_id == "reinspect:pump-01" and ts[0].priority == "medium"
    inc.needs_reinspect = False
    assert tasks_for(cfg, f)[0] == [] and tasks_for(cfg, f, busy_assets={"pump-01"})[0] == []


def test_enabled_g1_becomes_the_pickup_candidate(tmp_path):
    def enable(raw):
        raw["robot_models"]["g1"].update(available=True)
        raw["robot_models"]["g1"]["locomotion"] = {"policy_path": "/p.pt", "env_yaml_path": "/e.yaml"}
    cfg = make_cfg(enable)
    f = fleet_with(tmp_path, [FakeRobot("g1-1", "g1", (2, 2))])
    f.open_incident("pump-01").status = CONFIRMED
    assert [c[1] for c in tasks_for(cfg, f)[0][0].candidates] == ["g1-1"]
    f2 = fleet_with(tmp_path, [FakeRobot("g1-real", "g1", mode="digital_twin")])
    f2.open_incident("pump-01").status = CONFIRMED
    assert tasks_for(cfg, f2)[0][0].candidates == []


def snapshot(tmp_path, robots=None, cfg=None):
    cfg = cfg or make_cfg()
    f = fleet_with(tmp_path, robots or [FakeRobot("go2-1", "go2"), FakeRobot("go2-2", "go2", (8, 0))])
    ts, sched = tasks_for(cfg, f)
    return build_snapshot(cfg, f, ts, sched, {}, {"kind": "periodic", "reasons": ["periodic"]}, [], 1000.0)


def test_snapshot_contents_and_size(tmp_path):
    s = snapshot(tmp_path)
    assert {"snapshot_id", "robots", "assets", "due_tasks", "actions", "policy", "locations", "recent_decisions", "trigger", "active_tasks", "valid_for_s"} <= set(s)
    assert s["policy"]["assignments_per_call"] == 1 and s["policy"]["routine_inspections"] is True and s["valid_for_s"] == 90.0
    assert s["assets"][0]["gauge_limits"] == [2.0, 8.0, "bar"] and s["assets"][0]["inspection_due_in_s"] == 0
    assert s["due_tasks"][0]["candidates"][0]["robot_id"] == "go2-1" and len(json.dumps(s)) < 6000


def test_snapshot_says_nothing_about_routine_timing_when_routine_is_off(tmp_path):
    s = snapshot(tmp_path, cfg=make_cfg(lambda r: r["schedule"].update(routine_inspections=False)))
    assert s["policy"]["routine_inspections"] is False and s["assets"][0]["inspection_due_in_s"] is None and s["due_tasks"] == []


def good(s, **over):
    d = {"snapshot_id": s["snapshot_id"], "decision": "assign", "task_id": "inspect:pump-01", "robot_id": "go2-1", "priority": "low", "rationale": "nearest idle go2"}
    d.update(over)
    return d


def test_valid_decision_is_accepted_and_enriched_from_the_task(tmp_path):
    s = snapshot(tmp_path)
    d = validate_decision(good(s), s, "nemotron")
    assert (d.action, d.target_id, d.for_asset, d.source) == ("inspect", "pump-01", "pump-01", "nemotron")
    assert validate_decision(good(s, decision="no_action", task_id=None, robot_id=None), s, "nemotron").decision == "no_action"


@pytest.mark.parametrize("over,match", [
    ({"snapshot_id": "SNAP-old"}, "does not match"), ({"task_id": "inspect:pump-99"}, "not one of the due tasks"),
    ({"robot_id": "go2-9"}, "not a candidate"), ({"robot_id": "amr-1"}, "not a candidate"),
    ({"decision": "dance"}, "not in"), ({"priority": "urgent"}, "not in"),
    ({"decision": "no_action"}, "must have task_id and robot_id set to null"), ({"velocity": 1.0}, "forbidden"),
    ({"extra": 1}, "unexpected fields"), ({"rationale": "x" * 301}, "longer than"),
])
def test_bad_decisions_are_rejected(tmp_path, over, match):
    s = snapshot(tmp_path)
    with pytest.raises(SchemaError, match=match):
        validate_decision(good(s, **over), s, "nemotron")


def test_json_schema_only_allows_ids_from_the_snapshot(tmp_path):
    s = snapshot(tmp_path)
    sch = output_json_schema(s)
    assert sch["properties"]["task_id"]["anyOf"][0]["enum"] == ["inspect:pump-01"]
    assert sch["properties"]["robot_id"]["anyOf"][0]["enum"] == ["go2-1", "go2-2"]
    assert sch["additionalProperties"] is False and set(sch["required"]) == set(sch["properties"])
