"""v0.5.5: park only at home with an upright watch and a clean wake; simulation-time limits; never left on a spot;
cameras stay on for the whole task and are prepared again after STOP / RESET; high_value 1.55."""
import importlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from isolation import isolated_user_files  # noqa: F401
from helpers_v4 import FakeRobot, make_cfg, make_coordinator, pump, pump_until
from Physical_AI_Fleet_python.robots.legged_robot import MAX_DRIFT_WAKES, NO_PROGRESS_S, PARK_AFTER_S, LeggedSimRobot
from Physical_AI_Fleet_python.robots.locomotion.bodies import UnicycleBody
from test_planner_gate_camera import planner_for
from test_site_and_assets import site

DT = 1 / 50
SQUARE = [[-10, -10], [10, -10], [10, 10], [-10, 10]]


class Body(UnicycleBody):
    def __init__(self, *a):
        super().__init__(*a)
        self.steps = self.holds = 0
        self.holding, self.ready, self.error, self.status_note, self.last_wake = False, True, "", "", ""
        self.sag_after = None

    def step(self, dt, cmd):
        self.steps += 1
        self.holding = False
        super().step(dt, cmd)

    def hold(self, dt):
        self.holds += 1
        self.holding = True
        if self.sag_after is not None and self.holds > self.sag_after:
            self.last_wake = "sank 6 cm, tilt 2 deg while parked"
            self.step(dt, [0, 0, 0])


def robot(home=(0.0, 0.0, 0.0), body=None):
    body = body or Body(*home)
    return LeggedSimRobot("spot-1", "quadruped", home, body, planner_for(site(SQUARE)), model="spot"), body


def run(r, seconds):
    for _ in range(int(round(seconds / DT))):
        r.update(DT)


def test_parks_only_at_home_after_three_seconds_and_wakes_for_a_task():
    r, body = robot()
    run(r, PARK_AFTER_S - 0.2)
    assert body.holds == 0
    run(r, 0.5)
    assert body.holds > 0 and r.holding
    r.go_to(3.0, 0.0, 0.0, "x")
    n = body.steps
    run(r, 0.2)
    assert body.steps == n + 10 and not r.holding


def test_never_parks_away_from_home_or_during_a_task():
    r, body = robot()
    r.pose = [5.0, 5.0, 0.0]
    body.reset_to(5.0, 5.0, 0.0)
    run(r, 10)
    assert body.holds == 0
    r2, body2 = robot()
    r2.current_skill = "read_gauge"
    run(r2, 10)
    assert body2.holds == 0


def test_sagging_while_parked_wakes_the_robot_logs_it_and_stops_parking_after_repeats():
    r, body = robot()
    body.sag_after = 5
    run(r, 3 * (PARK_AFTER_S + 0.5))
    ev = r.pop_events()
    assert [e[0] for e in ev] == ["robot_woke"] * MAX_DRIFT_WAKES and "sank" in ev[0][1]["reason"]
    assert ev[-1][1]["parking_disabled"] is True and r.status == "idle"
    h = body.holds
    run(r, 10)
    assert body.holds == h
    r.reset_to_home()
    run(r, PARK_AFTER_S + 0.5)
    assert body.holds > h


def test_policy_body_upright_watch_clean_wake_and_no_fall_reports_while_parked(monkeypatch):
    from fakes import Fake6Go2, install_fake_torch_warp, install_policy_module
    from Physical_AI_Fleet_python.fleet_config.catalog import MODELS
    from Physical_AI_Fleet_python.robots.locomotion import policy_body as pb
    install_fake_torch_warp(monkeypatch)
    install_policy_module(monkeypatch, Fake6Go2)
    Fake6Go2.fail_first, Fake6Go2.accepts = 1, "any"
    body = pb.PolicyBody(MODELS["go2"], "/World/Fleet/go2_1", 0, 0, 0, robot_name="go2-1")
    for _ in range(3):
        body.step(0.005, [0, 0, 0])
    body._settle = 0.0
    body.pose()
    body.hold(0.005)
    assert body.holding and body._park_z == pytest.approx(MODELS["go2"].spawn_z)
    n = len(body.policy.forwards)
    art = body.policy.robot
    art.pos[0, 2] -= 0.02
    body.pose(); body.hold(0.005)
    assert body.holding and len(body.policy.forwards) == n and body.last_wake == ""
    art.pos[0, 2] -= 0.30
    body.pose()
    assert body.is_fallen() is False
    body.hold(0.005)
    assert not body.holding and "sank" in body.last_wake
    assert body._settle > 0 and art.pos[0, 2] == pytest.approx(MODELS["go2"].spawn_z)
    assert np.allclose(art.targets, np.linspace(-0.5, 0.5, 12)) and np.allclose(art.lin, 0)
    body._settle = 0.0
    body.pose(); body.hold(0.005)
    art.quat[0] = [math.cos(math.radians(6)), math.sin(math.radians(6)), 0, 0]
    body.pose(); body.hold(0.005)
    assert "tilt 12 deg" in body.last_wake and not body.holding


class StuckBody(Body):
    def step(self, dt, cmd):
        self.steps += 1
        self.holding = False


def test_no_progress_is_counted_in_simulation_time_not_wall_time(monkeypatch):
    import Physical_AI_Fleet_python.robots.legged_robot as L
    monkeypatch.setattr(L.time, "time", lambda: 1e9)
    r, body = robot(body=StuckBody(0, 0, 0))
    r.go_to(5.0, 0.0, 0.0, "x")
    run(r, NO_PROGRESS_S - 0.5)
    assert r.status == "busy" and "replanned" not in r.note
    run(r, 1.0)
    assert "replanned: no progress" in r.note


def test_turning_on_the_spot_counts_as_progress():
    r, body = robot()
    r.go_to(0.0, 0.0, math.pi * 0.95, "turn")
    run(r, NO_PROGRESS_S * 3)
    assert r.status == "idle" and "no progress" not in r.note and r.goal is None


def test_executor_limits_use_simulation_time(monkeypatch):
    from Physical_AI_Fleet_python.skills import executor as E
    from Physical_AI_Fleet_python.schemas.task_assignment import TaskAssignment
    audit = SimpleNamespace(rows=[], record=lambda k, p=None, **e: audit.rows.append(k))
    done = []
    s = SimpleNamespace(assets={"m": SimpleNamespace(asset_id="m", position=(10.0, 0.0, 0.0), approach_pose=(8.0, 0.0, 0.0),
                                                     half_extents=(0.4, 0.25))})
    ex = E.SkillExecutor(s, None, None, lambda r: None, lambda a: None, audit, done.append)
    r = FakeRobot("spot-1", "spot", (0, 0))
    r.locomotion, r.current_skill = "rl_policy", None
    ex.start(TaskAssignment("x", "spot-1", "read_gauge", "m", "low", "r", source="operator"), r)
    ex.tick(0.0)
    cap = ex.runs["spot-1"].nav_cap_s
    assert cap == pytest.approx(E.NAV_BASE_S + E.NAV_FACTOR * math.dist((0, 0), r.goal[:2]) / r.model.max_vx)
    monkeypatch.setattr(E.time, "time", lambda: 1e12)
    ex.tick(0.005)
    assert "spot-1" in ex.runs
    for _ in range(int(cap / 0.5) + 2):
        ex.tick(0.5)
    assert done and done[0].error.startswith("navigation timeout") and E.SKILL_TIMEOUT_S == 900.0


class StuckRobot(FakeRobot):
    def clear_error(self):
        if self.status != "error" or not self.note.startswith("stuck"):
            return False
        self.status, self.note, self.goal = "idle", "", None
        return True


def coordinator(monkeypatch, tmp_path, *robots):
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(lambda r: r["schedule"].update(routine_inspections=False)))
    for r in robots:
        coord.fleet.add(r)
    return coord, clock


def test_a_failed_task_walks_the_robot_home(monkeypatch, tmp_path):
    r = StuckRobot("spot-2", "spot", (0, -3))
    coord, clock = coordinator(monkeypatch, tmp_path, r)
    coord.report_alarm("pump-01", "overpressure", "high")
    assert pump_until(coord, clock, lambda: coord.executor.started)
    r.pose = [3.0, 0.0, 0.0]
    coord.executor.finish("spot-2", error="timeout")
    assert r.go_to_calls == ["home"] and coord.audit.last("returning_home")["after"] == "read_gauge failed"


def test_a_stuck_robot_is_released_once_then_waits_for_the_operator(monkeypatch, tmp_path):
    r = StuckRobot("spot-1", "spot", (0, -3))
    coord, clock = coordinator(monkeypatch, tmp_path, r)
    coord.report_alarm("pump-01", "overpressure", "high")
    assert pump_until(coord, clock, lambda: coord.executor.started)
    run = coord.executor.runs.pop("spot-1")
    r.pose, r.status, r.note = [3.0, 0.0, 0.0], "error", "stuck (no progress)"
    run.error = "robot error: stuck (no progress)"
    coord._on_skill_done(run)
    assert r.status == "busy" and r.go_to_calls == ["home"] and coord.audit.last("robot_recovering")["was"].startswith("stuck")
    r.status, r.note = "error", "stuck (no progress)"
    pump(coord, clock)
    assert r.status == "error" and r.go_to_calls == ["home"]
    coord.reset_robot("spot-1")
    assert "spot-1" not in coord._home_retry


def test_a_fallen_robot_is_never_auto_released(monkeypatch, tmp_path):
    r = StuckRobot("spot-1", "spot", (0, -3))
    coord, clock = coordinator(monkeypatch, tmp_path, r)
    r.pose, r.status, r.note = [3.0, 0.0, 0.0], "error", "fallen - press RESET ROBOT"
    pump(coord, clock)
    assert r.status == "error" and r.go_to_calls == [] and "robot_recovering" not in coord.audit.kinds()


def test_stuck_while_walking_home_without_a_task_is_released_once(monkeypatch, tmp_path):
    r = StuckRobot("spot-1", "spot", (0, -3))
    coord, clock = coordinator(monkeypatch, tmp_path, r)
    r.pose, r.status, r.note = [3.0, 3.0, 0.0], "error", "stuck (blocked by another robot)"
    pump(coord, clock)
    assert r.go_to_calls == ["home"] and "spot-1" in coord._home_retry
    r.status, r.goal = "idle", None
    pump(coord, clock)
    assert "spot-1" not in coord._home_retry


def test_an_idle_or_stuck_robot_on_the_target_spot_is_sent_home_first(monkeypatch, tmp_path):
    blocker = StuckRobot("spot-2", "spot", (-5, -5))
    worker = StuckRobot("spot-1", "spot", (0, -3))
    coord, clock = coordinator(monkeypatch, tmp_path, blocker, worker)
    blocker.status, blocker.note = "error", "stuck (no progress)"
    coord._error_reported.add("spot-2")
    goal = (4.0, 0.2)
    blocker.pose = [goal[0], goal[1], 0.0]
    coord._clear_spot(goal, "spot-1")
    assert blocker.go_to_calls == ["home"] and coord.audit.last("spot_cleared") == {"robot_id": "spot-2", "for": "spot-1",
                                                                                   "spot": [4.0, 0.2]}
    far = StuckRobot("spot-3", "spot", (5, 5))
    coord.fleet.add(far)
    far.pose = [8.0, 8.0, 0.0]
    coord._clear_spot(goal, "spot-1")
    assert far.go_to_calls == []


def test_the_executor_clears_the_spot_before_sending_a_robot(monkeypatch, tmp_path):
    from Physical_AI_Fleet_python.skills import executor as E
    from Physical_AI_Fleet_python.schemas.task_assignment import TaskAssignment
    called = []
    s = SimpleNamespace(assets={"m": SimpleNamespace(asset_id="m", position=(10.0, 0.0, 0.0), approach_pose=(8.0, 0.0, 0.0),
                                                     half_extents=(0.4, 0.25))})
    ex = E.SkillExecutor(s, None, None, lambda r: None, lambda a: None, SimpleNamespace(record=lambda *a, **k: None), lambda r: None)
    ex.clear_spot_fn = lambda xy, rid: called.append((tuple(round(v, 2) for v in xy), rid))
    r = FakeRobot("spot-1", "spot", (0, 0))
    r.locomotion, r.current_skill = "rl_policy", None
    ex.start(TaskAssignment("x", "spot-1", "read_gauge", "m", "low", "r", source="operator"), r)
    ex.tick(0.005)
    assert called and called[0][1] == "spot-1" and called[0][0] == tuple(round(v, 2) for v in r.goal[:2])
    ex.start(TaskAssignment("y", "spot-1", "return_home", None, "low", "r", source="operator"), r)
    ex.tick(0.005)
    assert len(called) == 1


def test_robot_events_reach_the_audit_log(monkeypatch, tmp_path):
    r = FakeRobot("spot-1", "spot", (0, -3))
    r.pop_events = lambda: [("robot_woke", {"robot_id": "spot-1", "reason": "tilt 9 deg while parked"})]
    coord, clock = coordinator(monkeypatch, tmp_path, r)
    pump(coord, clock)
    assert coord.audit.last("robot_woke")["reason"].startswith("tilt")


def test_shipped_high_value_is_inside_the_red_band():
    from Physical_AI_Fleet_python.fleet_config import paths
    m = json.load(open(paths.SHIPPED_DIR / "gauges.json"))["models"]["digiten_1p6mpa"]
    assert m["high_value"] == 1.55 and m["red_from"] + 0.1 <= m["high_value"] < m["scale_max"]


pytest.importorskip("pxr")
from test_scenario_v044 import env  # noqa: E402,F401


def test_camera_stays_on_for_the_whole_task_and_is_prepared_again_after_stop(env):
    sc = importlib.import_module("Physical_AI_Fleet_python.scenario").FleetScenario()
    env.made.append(sc)
    sc.setup_scene()
    sc.setup_post_load()
    from Physical_AI_Fleet_python.fleet_config.catalog import MODELS
    rb = SimpleNamespace(robot_id="go2-1", model=MODELS["go2"], robot_type="quadruped", pose=[0, -3, 0], home_pose=(0, -3, 0),
                         radius=0.4, status="idle", note="", mode="simulation", locomotion="rl_policy", prim_path="",
                         neighbors_fn=None, goal=None, telemetry_age=lambda: 0.0, shutdown=lambda: None, stop=lambda: None,
                         update=lambda dt: None, summary=lambda: "go2-1", idle_hold=True, holding=False,
                         reset_to_home=lambda: None)
    sc.coordinator.fleet.add(rb)
    sc._cam_paths["go2-1"] = "/World/Fleet/go2_1/base/inspection_cam"
    sc._prepare_camera("go2-1", rb.model)
    cam = sc._cameras["go2-1"]
    sc.coordinator.executor.tick = lambda *a: None
    run = SimpleNamespace(phase="navigate", purpose="read_gauge", robot=SimpleNamespace(pose=[4.0, 0, 0], goal=(5.0, 0.0, 0.0)))
    sc.coordinator.executor.runs["go2-1"] = run
    sc.update(0.005)
    assert cam.active
    run.robot.pose = [0.0, 0.0, 0.0]
    for _ in range(5):
        sc.update(0.005)
    assert cam.active and [r["kind"] for r in sc.coordinator.audit.recent(50)].count("camera_on") == 1
    del sc.coordinator.executor.runs["go2-1"]
    sc.update(0.005)
    assert not cam.active and "camera_off" in [r["kind"] for r in sc.coordinator.audit.recent(50)]
    old = cam._cam
    sc.on_timeline_stop()
    assert cam._cam is not None and cam._cam is not old and cam.active is False
    sc.reset()
    assert cam._cam is not None and sc._cam_task == {}
