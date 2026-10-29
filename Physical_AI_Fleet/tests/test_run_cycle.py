"""Walk home after a task, a clean start for every run, routine inspections in running time, and a replay of the log."""
import json
import math
import threading
import time

import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.common.audit_log import AuditLog
from Physical_AI_Fleet_python.coordinator.fleet_state import AWAITING_REPAIR, CONFIRMED, TOOLS_LOADED
from helpers_v4 import (FakeAsset, FakeNemotron, FakeRobot, FakeSettings, make_cfg, make_coordinator, out_of_limits, pump,
                        pump_until, within_limits)
from test_alarm_flow import Harness, WalkingRobot


def alarm_only(raw):
    raw["schedule"].update(routine_inspections=False, initial_due=False)


def enable_g1(raw):
    alarm_only(raw)
    raw["robot_models"]["g1"].update(available=True)
    raw["robot_models"]["g1"]["locomotion"] = {"policy_path": "/p/g1.pt", "env_yaml_path": "/p/g1_env.yaml"}


def two_assets(tmp_path):
    return FakeSettings(tmp_path, {"pump-01": FakeAsset("pump-01", (5, 0), (4.0, 0.0, 0.0)),
                                   "pump-02": FakeAsset("pump-02", (5, 6), (4.0, 6.0, 0.0))})


def inspected(monkeypatch, tmp_path, cfg=None, robot=None, error=None, pose=(4.0, 0.0), nemotron=None):
    """One quadruped is sent to pump-01, 'walks' to (pose), reads the gauge and finishes."""
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=cfg or make_cfg(alarm_only), settings=two_assets(tmp_path),
                                    nemotron=nemotron or FakeNemotron())
    robot = robot or FakeRobot("go2-1", "go2", (0, -3))
    coord.fleet.add(robot)
    coord.report_alarm("pump-01", "overpressure", "high")
    assert pump_until(coord, clock, lambda: coord.executor.started)
    robot.pose = [pose[0], pose[1], 0.0]
    coord.executor.finish(robot.robot_id, error=error, event=None if error else out_of_limits("pump-01", robot.robot_id))
    return coord, clock, robot


# ================================================================== quadruped walks home and stays available
def test_quadruped_walks_home_after_the_reading_and_is_available_while_walking(monkeypatch, tmp_path):
    nem = FakeNemotron()
    coord, clock, robot = inspected(monkeypatch, tmp_path, nemotron=nem)
    assert robot.go_to_calls == ["home"] and robot.goal[:2] == (0.0, -3.0) and robot.status == "busy"
    assert coord.fleet.homing == {"go2-1"} and coord.fleet.effective_status(robot) == "idle"
    assert coord.audit.last("returning_home") == {"robot_id": "go2-1", "after": "read_gauge"}
    assert "RETURNING HOME (available for a new task): go2-1" in coord.status_lines()
    # a second alarm: the robot that is still walking home gets the job
    coord.report_alarm("pump-02", "leak", "high")
    n = len(coord.executor.started)
    assert pump_until(coord, clock, lambda: len(coord.executor.started) > n)
    assert coord.executor.started[-1] == ("read_gauge", "go2-1", "pump-02", "")
    robots = {r["robot_id"]: r for r in json.loads(nem.calls[-1][1]["content"])["robots"]}
    assert robots["go2-1"]["returning_home"] is True and robots["go2-1"]["status"] == "idle"
    assert coord.fleet.homing == set()  # the new task replaced the walk home


def test_reinspect_also_walks_home(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(alarm_only), settings=two_assets(tmp_path))
    robot = FakeRobot("go2-1", "go2", (0, -3))
    coord.fleet.add(robot)
    coord.fleet.open_incident("pump-01").status = "verifying"
    assert pump_until(coord, clock, lambda: coord.executor.started)
    robot.pose = [4.0, 0.0, 0.0]
    coord.executor.finish("go2-1", event=within_limits("pump-01", "go2-1"))
    assert coord.executor.started[-1][0] == "reinspect_asset" and robot.go_to_calls == ["home"] and coord.fleet.homing == {"go2-1"}


def test_homing_ends_when_the_robot_arrives_or_fails(monkeypatch, tmp_path):
    coord, clock, robot = inspected(monkeypatch, tmp_path)
    robot.status, robot.goal = "idle", None  # arrived
    pump(coord, clock)
    assert coord.fleet.homing == set() and "RETURNING HOME" not in "\n".join(coord.status_lines())
    coord2, clock2, robot2 = inspected(monkeypatch, tmp_path / "b")
    robot2.status = "error"  # fell on the way
    pump(coord2, clock2)
    assert coord2.fleet.homing == set() and coord2.fleet.effective_status(robot2) == "error"


def test_stop_all_and_reset_robot_end_the_walk(monkeypatch, tmp_path):
    coord, clock, robot = inspected(monkeypatch, tmp_path)
    coord.stop_all()
    assert coord.fleet.homing == set()
    coord2, clock2, robot2 = inspected(monkeypatch, tmp_path / "b")
    coord2.reset_robot("go2-1")
    assert coord2.fleet.homing == set()


@pytest.mark.parametrize("case", ["already home", "config says no"])  # v0.5.5: a failed task DOES walk home
def test_no_walk_home_when_there_is_nothing_to_walk_back_from(monkeypatch, tmp_path, case):
    cfg = None
    kwargs = {}
    if case == "failed task":
        kwargs["error"] = "robot fell"
    elif case == "already home":
        kwargs["pose"] = (0.0, -3.0)
    else:
        cfg = make_cfg(lambda r: (alarm_only(r), r["actions"]["inspect"].update(return_home_after=False),
                                  r["actions"]["reinspect"].update(return_home_after=False)))
    coord, clock, robot = inspected(monkeypatch, tmp_path, cfg=cfg, **kwargs)
    assert robot.go_to_calls == [] and coord.fleet.homing == set() and "returning_home" not in coord.audit.kinds()


def test_a_robot_without_a_path_home_is_reported_not_marked_as_walking(monkeypatch, tmp_path):
    class NoPath(FakeRobot):
        def go_to(self, x, y, yaw, name=""):
            self.status, self.note = "error", "no path to home"
    coord, clock, robot = inspected(monkeypatch, tmp_path, robot=NoPath("go2-1", "go2", (0, -3)))
    assert coord.fleet.homing == set() and coord.audit.last("return_home_failed")["note"] == "no path to home"


def test_a_crash_while_sending_the_robot_home_never_breaks_the_mission(monkeypatch, tmp_path):
    class Boom(FakeRobot):
        def go_to(self, x, y, yaw, name=""):
            raise RuntimeError("planner exploded")
    coord, clock, robot = inspected(monkeypatch, tmp_path, robot=Boom("go2-1", "go2", (0, -3)))
    pump(coord, clock, steps=2)
    assert "planner exploded" in coord.audit.last("return_home_failed")["error"]
    assert coord.fleet.incident_for("pump-01").status == CONFIRMED  # the reading was still processed


def test_the_amr_stays_at_the_asset_after_delivering_the_tools(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(alarm_only), settings=two_assets(tmp_path))
    amr = FakeRobot("amr-1", "amr", (1, 1))
    coord.fleet.add(amr)
    coord.fleet.open_incident("pump-01").status = TOOLS_LOADED
    assert pump_until(coord, clock, lambda: coord.executor.started)
    amr.pose = [3.5, 0.0, 0.0]
    coord.executor.finish("amr-1")
    assert coord.fleet.incident_for("pump-01").status == AWAITING_REPAIR and amr.go_to_calls == [] and coord.fleet.homing == set()


def test_inspect_walk_home_tools_repair_reinspect_chain_with_the_real_executor(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, [WalkingRobot("go2-1", pose=(0, -3)), WalkingRobot("go2-2", pose=(12, -3)),
                                        WalkingRobot("amr-1", "amr", pose=(1, 1))], n_assets=1, cfg=make_cfg(alarm_only))
    h.inject("motor-01", "overpressure", "high")
    assert h.until(lambda: h.coord.fleet.incident_for("motor-01").status == CONFIRMED)
    first = h.started()[0][0]
    r1 = h.robots[first]
    assert r1.visited[:2] == ["motor-01", "home"]  # inspected, then walked home
    assert h.until(lambda: not h.coord.fleet.homing and r1.status == "idle")
    assert r1.pose[:2] == pytest.approx(list(r1.home_pose[:2]))  # back home BEFORE the AMR comes
    assert h.coord.mark_tools_loaded("motor-01").startswith("tools marked")
    assert h.until(lambda: any(r == "amr-1" for r, _t in h.started()))
    motor = h.s.assets["motor-01"].position[:2]
    assert all(math.dist(h.robots[rid].pose[:2], motor) > 2.0 for rid in ("go2-1", "go2-2"))  # nobody stands at the motor
    assert h.until(lambda: h.coord.fleet.incident_for("motor-01").status == AWAITING_REPAIR)
    h.faults.pop("motor-01")
    h.coord.mark_repaired("motor-01")  # the 'Technician Done' button
    assert h.until(lambda: h.coord.fleet.incident_for("motor-01") is None, max_steps=200)
    order = [r for r, t in h.started() if t == "motor-01"]
    assert order[0] == first and order[1] == "amr-1" and order[2] in ("go2-1", "go2-2") and order[2] != first  # a different quadruped re-inspects
    assert "incident_closed" in h.kinds()


def test_a_new_task_replaces_the_walk_home_with_the_real_executor(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, [WalkingRobot("go2-1", pose=(0, -3), ticks=30)], n_assets=2, cfg=make_cfg(alarm_only))
    robot = h.robots["go2-1"]
    h.inject("motor-01", "overpressure", "high")
    assert h.until(lambda: h.coord.fleet.incident_for("motor-01").status == CONFIRMED)
    assert robot.visited == ["motor-01", "home"] and robot.status == "busy" and h.coord.fleet.homing == {"go2-1"}
    h.inject("motor-02", "overpressure", "high")  # while it is still walking home
    assert h.until(lambda: ("go2-1", "motor-02") in h.started())
    h.step()
    assert robot.visited == ["motor-01", "home", "motor-02"] and h.coord.fleet.homing == set()
    assert json.loads(h.model.calls[-1][1]["content"])["robots"][0]["returning_home"] is True
    assert robot.pose[:2] != list(robot.home_pose[:2]) or robot.goal[:2] != tuple(robot.home_pose[:2])  # it did not finish the walk home first


# ================================================================== a clean start for every run
def test_reset_tasks_clears_everything_left_over_from_the_previous_run(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(enable_g1), settings=two_assets(tmp_path))
    coord.fleet.add(FakeRobot("go2-1", "go2", (0, -3)))
    coord.fleet.add(FakeRobot("g1-1", "g1", (2, 2)))
    coord.report_alarm("pump-01", "overpressure", "high")
    assert pump_until(coord, clock, lambda: coord.executor.started)
    coord.executor.finish("go2-1", event=out_of_limits("pump-01", "go2-1"))
    assert pump_until(coord, clock, lambda: any(m.get("state") == "pending" for m in coord._meta.values()))  # pick-up waits for approval
    coord.submit_event(within_limits("pump-01", "go2-1"))  # still queued
    coord._blocked[("x", "y")] = clock() + 100
    coord.fleet.homing.add("go2-1")
    assert coord.fleet.incidents and coord.approvals.pending() and coord._meta and not coord._inbox.empty()
    had = coord.reset_tasks("new run")
    assert (had["incidents"], had["approvals"], had["queued_events"]) == (1, 1, 1) and had["reserved"] >= 1
    assert coord.fleet.incidents == {} and coord.approvals.pending() == [] and coord._meta == {} and coord._blocked == {}
    assert coord._recent == [] and coord._inbox.empty() and coord.fleet.homing == set() and coord._last_decision is None
    assert coord._run_s == 0.0 and coord.audit.last("tasks_reset")["reason"] == "new run"
    assert any(k == "approval_resolved" and d["approved"] is False and d["by"] == "new run" for k, d in coord.audit.records)
    n = len(coord.executor.started)
    pump(coord, clock, steps=300)  # nothing of the old run comes back, however long it runs
    assert len(coord.executor.started) == n and coord.fleet.incidents == {} and coord.approvals.pending() == []


def test_running_skills_are_cancelled_by_a_reset(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(alarm_only), settings=two_assets(tmp_path))
    robot = FakeRobot("go2-1", "go2", (0, -3))
    coord.fleet.add(robot)
    coord.report_alarm("pump-01", "overpressure", "medium")
    assert pump_until(coord, clock, lambda: "go2-1" in coord.executor.runs)
    assert coord.reset_tasks("world reset")["running"] == 1 and coord.executor.runs == {} and robot.status == "idle"


def test_an_answer_still_in_flight_when_the_run_is_reset_is_never_executed(monkeypatch, tmp_path):
    gate, seen = threading.Event(), []
    def model(messages, rf):
        snap = json.loads(messages[1]["content"])
        seen.append(snap["snapshot_id"])
        if len(seen) == 1:
            gate.wait(5)  # the first call is 'slow'
        t = snap["due_tasks"][0]
        return json.dumps({"snapshot_id": snap["snapshot_id"], "decision": "assign", "task_id": t["task_id"],
                           "robot_id": t["candidates"][0]["robot_id"], "priority": t["priority"], "rationale": "x"})
    coord, clock = make_coordinator(monkeypatch, tmp_path, nemotron=FakeNemotron(model))
    coord.fleet.add(FakeRobot("go2-1", "go2", (0, 0)))
    clock.advance(1)
    coord.tick(0.005)
    assert coord._brain_future is not None
    for _ in range(300):
        if seen:
            break
        time.sleep(0.01)
    old = seen[0]
    assert coord.reset_tasks("world reset")["decision_in_flight"] is True and coord._brain_future is None
    gate.set()
    time.sleep(0.2)
    pump(coord, clock, steps=5)
    decisions = [d["snapshot_id"] for k, d in coord.audit.records if k == "fleet_decision"]
    assert decisions and old not in decisions and len(seen) >= 2  # a NEW question was asked for the new run


def test_world_reset_forgets_tasks_and_sends_robots_home(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, [WalkingRobot("go2-1", pose=(0, -3), ticks=30)], n_assets=2, cfg=make_cfg(alarm_only))
    h.inject("motor-01", "overpressure", "high")
    assert h.until(lambda: h.started())
    h.step(5)  # the robot is on its way
    h.coord.on_world_reset()  # what the toolbar's Stop does
    h.faults.clear()
    assert h.coord.fleet.incidents == {} and h.robots["go2-1"].pose[:2] == list(h.robots["go2-1"].home_pose[:2])
    n = len(h.started())
    h.step(300)  # run again: the old incident must not come back
    assert len(h.started()) == n and h.coord.fleet.incidents == {} and h.coord.executor.runs == {}


def test_new_run_restarts_the_routine_timers(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(lambda r: r["schedule"].update(initial_due=False)))
    coord.fleet.add(FakeRobot("go2-1", "go2"))
    pump(coord, clock, steps=50)
    assert coord._run_s > 20 and coord.scheduler.last["pump-01"] == 0.0
    coord._run_s = 500.0
    coord.reset_tasks("new run")
    assert coord._run_s == 0.0 and coord.scheduler.last == {"pump-01": 0.0}


# ================================================================== routine inspections: opt-in, running time only
def test_routine_inspections_are_off_by_default_so_nothing_happens_on_its_own(monkeypatch, tmp_path):
    from Physical_AI_Fleet_python.fleet_config.action_config import load_action_config
    nem = FakeNemotron()
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=load_action_config(), nemotron=nem)  # the shipped defaults
    coord.fleet.add(FakeRobot("go2-1", "go2"))
    assert coord.scheduler.enabled is False
    pump(coord, clock, steps=900, step_s=4.0)  # an hour of running time
    assert nem.calls == [] and coord.executor.started == []


def test_routine_inspections_count_running_time_only(monkeypatch, tmp_path):
    nem = FakeNemotron()
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(alarm_only), nemotron=nem)
    coord.fleet.add(FakeRobot("go2-1", "go2"))
    coord.set_routine(True, 900)
    assert coord.audit.last("routine_changed") == {"enabled": True, "interval_s": 900.0}
    pump(coord, clock, steps=200, step_s=4.0)  # 800 s of running
    assert nem.calls == [] and 795 < coord._run_s < 801  # 199 gaps of 4 s
    before = coord._run_s
    clock.advance(3600)  # an hour paused / stopped: no ticks
    pump(coord, clock, steps=1, step_s=4.0)
    assert nem.calls == [] and coord._run_s == before  # the pause did not count
    pump(coord, clock, steps=30, step_s=4.0)  # 120 s more
    assert nem.calls and coord.executor.started and coord.executor.started[0][0] == "read_gauge"


def test_switching_routine_on_mid_run_does_not_make_everything_overdue(monkeypatch, tmp_path):
    nem = FakeNemotron()
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(alarm_only), nemotron=nem)
    coord.fleet.add(FakeRobot("go2-1", "go2"))
    pump(coord, clock, steps=500, step_s=4.0)  # 2000 s of running with routine off
    coord.set_routine(True, 600)
    pump(coord, clock, steps=100, step_s=4.0)  # 400 s
    assert nem.calls == []
    pump(coord, clock, steps=60, step_s=4.0)  # 240 s more -> 640 s since it was switched on
    assert nem.calls


# ================================================================== audit log
def test_audit_log_keeps_its_own_kind_and_timestamp(tmp_path):
    log = AuditLog(tmp_path)
    e = log.record("alarm_raised", {"kind": "overpressure", "ts": "t0", "asset_id": "m"})
    assert e["kind"] == "alarm_raised" and e["payload_kind"] == "overpressure" and e["payload_ts"] == "t0" and e["ts"] != "t0"
    assert json.loads(log.path.read_text().splitlines()[-1])["kind"] == "alarm_raised"
    assert AuditLog.one_line(e).split()[1] == "alarm_raised"
    assert log.record("plain", {"a": 1})["a"] == 1


def test_the_alarm_record_is_named_alarm_raised_in_the_real_log(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, [WalkingRobot("go2-1", pose=(0, -3))], n_assets=1)
    h.inject("motor-01", "overpressure", "high")
    e = next(r for r in h.coord.audit.recent(50) if r["kind"] == "alarm_raised")
    assert e["alarm_kind"] == "overpressure" and e["priority"] == "high" and e["asset_id"] == "motor-01"
    assert not any(r["kind"] == "overpressure" for r in h.coord.audit.recent(50))


# ================================================================== replay of the reported session
def reported_fleet(coord):
    for r in (FakeRobot("spot-1", "spot", (0, -3)), FakeRobot("spot-2", "spot", (2, -3)), FakeRobot("h1-1", "h1", (4, -3)),
              FakeRobot("amr-1", "amr", (6, -3))):
        coord.fleet.add(r)


def four_assets(tmp_path):
    return FakeSettings(tmp_path, {f"motor-0{i}": FakeAsset(f"motor-0{i}", (4.0 * i, 2.0), (4.0 * i - 1.5, 2.0, 0.0)) for i in range(1, 5)})


def test_replay_motors_3_and_4_are_not_inspected_unless_someone_asks(monkeypatch, tmp_path):
    nem = FakeNemotron()
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(alarm_only), settings=four_assets(tmp_path), nemotron=nem)
    reported_fleet(coord)
    pump(coord, clock, steps=1500, step_s=4.0)  # 100 minutes of running: in the log this produced inspections of all four motors
    assert nem.calls == [] and coord.executor.started == []
    coord.report_alarm("motor-01", "overpressure", "medium")  # the only thing the operator did
    assert pump_until(coord, clock, lambda: coord.executor.started)
    assert [s[2] for s in coord.executor.started] == ["motor-01"] and coord.executor.started[0][1] in ("spot-1", "spot-2")
    pump(coord, clock, steps=300, step_s=4.0)
    assert [s[2] for s in coord.executor.started] == ["motor-01"]  # motors 2, 3 and 4 were never touched


@pytest.mark.parametrize("valid_for,outcome", [(30, "stale"), (90, "dispatched")])
def test_replay_a_slow_reasoning_model_is_not_wasted_with_the_new_window(monkeypatch, tmp_path, valid_for, outcome):
    holder = {}
    def slow(messages, rf):
        holder["clock"].advance(37)  # the log shows 37 s for one decision
        snap = json.loads(messages[1]["content"])
        t = snap["due_tasks"][0]
        return json.dumps({"snapshot_id": snap["snapshot_id"], "decision": "assign", "task_id": t["task_id"],
                           "robot_id": t["candidates"][0]["robot_id"], "priority": t["priority"], "rationale": "x"})
    cfg = make_cfg(lambda r: (alarm_only(r), r["nemotron"].update(snapshot_valid_for_s=valid_for)))
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=cfg, settings=four_assets(tmp_path), nemotron=FakeNemotron(slow))
    holder["clock"] = clock
    reported_fleet(coord)
    coord.report_alarm("motor-01", "overpressure", "medium")
    pump(coord, clock, steps=3)
    assert coord._recent[0]["outcome"] == outcome
    assert bool(coord.executor.started) == (outcome == "dispatched")
