"""v0.5.4: thinking always on, the AMR obeys the site map and goes home after the repair, idle robots cost less,
the floor layout holds robots (with yaw) as well as assets."""
import importlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.fleet_config import asset_catalog as ac
from Physical_AI_Fleet_python.fleet_config.geometry import distance_to_polygon_edge, point_in_polygon
from Physical_AI_Fleet_python.robots.sim_robot import SimRobot
from test_planner_gate_camera import planner_for, sampled
from test_site_and_assets import site

DT = 1 / 50


def drive(robot, seconds=120.0):
    trace = [tuple(robot.pose[:2])]
    t = 0.0
    while robot.goal is not None and t < seconds:
        robot.update(DT)
        trace.append(tuple(robot.pose[:2]))
        t += DT
    return trace


# ------------------------------------------------------------------ AMR follows the site map
def test_amr_drives_around_a_no_go_zone_inside_the_walkable_area():
    zone = [[8, 0.5], [12, 0.5], [12, 4.0], [8, 4.0]]
    s = site(zones=[zone])
    p = planner_for(s)
    amr = SimRobot("amr-1", "amr", "simulation", (3.0, 2.0, 0.0), model="amr", planner=p)
    amr.go_to(17.0, 2.0, 0.0, "pump")
    assert amr.status == "busy" and len(amr.path) >= 2  # not a straight line
    trace = drive(amr)
    assert amr.status == "idle" and math.dist(amr.pose[:2], (17.0, 2.0)) < 0.06
    poly = s.no_go[0].polygon
    for x, y in trace:
        assert not point_in_polygon(x, y, poly) and distance_to_polygon_edge(x, y, poly) >= amr.radius - 0.1
        assert point_in_polygon(x, y, s.walkable)


def test_amr_refuses_an_unreachable_goal_instead_of_driving_through():
    p = planner_for(site(zones=[[[10, 0.0], [14, 0.0], [14, 6.0], [10, 6.0]]]))  # cuts the corridor completely
    amr = SimRobot("amr-1", "amr", "simulation", (3.0, 2.0, 0.0), model="amr", planner=p)
    amr.go_to(18.0, 2.0, 0.0, "pump-01")
    assert amr.status == "error" and amr.note == "no path to pump-01" and amr.goal is None
    old = SimRobot("amr-2", "amr", "simulation", (3.0, 2.0, 0.0), model="amr")  # no planner: old behaviour
    old.go_to(5.0, 2.0, 0.0)
    drive(old)
    assert old.status == "idle" and math.dist(old.pose[:2], (5.0, 2.0)) < 0.06


def test_the_robot_factory_gives_the_amr_the_planner(monkeypatch):
    import sys
    for n in ("Physical_AI_Fleet_python.robots.robot_factory", "Physical_AI_Fleet_python.robots.sim_robot"):
        monkeypatch.delitem(sys.modules, n, raising=False)  # other tests stub these modules
    from Physical_AI_Fleet_python.fleet_config.settings import Settings
    from Physical_AI_Fleet_python.robots.robot_factory import create_robot, make_planner
    pl = make_planner(Settings())
    assert create_robot("amr-1", "amr", "simulation", (0, -3, 0), planner=pl).planner is pl


def test_amr_goes_home_when_the_repair_is_done(monkeypatch, tmp_path):
    from helpers_v4 import FakeRobot, make_cfg, make_coordinator, pump_until
    from Physical_AI_Fleet_python.coordinator.fleet_state import AWAITING_REPAIR, TOOLS_LOADED, VERIFYING
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(lambda r: r["schedule"].update(routine_inspections=False)))
    amr = FakeRobot("amr-1", "amr", (1, 1))
    coord.fleet.add(amr)
    coord.fleet.open_incident("pump-01").status = TOOLS_LOADED
    assert pump_until(coord, clock, lambda: coord.executor.started)
    amr.pose = [3.5, 0.0, 0.0]
    coord.executor.finish("amr-1")
    assert coord.fleet.incident_for("pump-01").status == AWAITING_REPAIR and amr.go_to_calls == []  # waits for the technician
    assert coord.mark_repaired("pump-01").startswith("repair recorded")
    assert amr.go_to_calls == ["home"] and coord.fleet.homing == {"amr-1"} and coord.fleet.incident_for("pump-01").status == VERIFYING
    assert coord.audit.last("returning_home") == {"robot_id": "amr-1", "after": "repair_done"}
    assert coord.fleet.effective_status(amr) == "idle"  # available for the next delivery while it drives home
    coord.reset_tasks("new run")
    assert coord._delivered == {}


# ------------------------------------------------------------------ idle robots cost less
class CountingBody:
    def __init__(self):
        from Physical_AI_Fleet_python.robots.locomotion.bodies import UnicycleBody
        self.inner, self.steps, self.holds, self.holding = UnicycleBody(0, 0, 0), 0, 0, False
        self.ready, self.error, self.status_note = True, "", ""

    def step(self, dt, cmd):
        self.steps += 1
        self.holding = False
        self.inner.step(dt, cmd)

    def hold(self, dt):
        self.holds += 1
        self.holding = True

    def pose(self): return self.inner.pose()
    def is_fallen(self): return False
    def reset_to(self, *a): self.inner.reset_to(*a)


def test_an_idle_walking_robot_stops_running_its_policy_and_resumes_for_a_task():
    from Physical_AI_Fleet_python.robots.legged_robot import IDLE_HOLD_AFTER_S, LeggedSimRobot
    body = CountingBody()
    r = LeggedSimRobot("go2-1", "quadruped", (0, 0, 0), body, planner_for(site([[-5, -5], [5, -5], [5, 5], [-5, 5]])), model="go2")
    n = int(IDLE_HOLD_AFTER_S / DT)
    for _ in range(n + 50):
        r.update(DT)
    assert body.steps <= n + 1 and body.holds >= 49 and r.holding
    r.go_to(2.0, 0.0, 0.0, "x")
    before = body.steps
    for _ in range(20):
        r.update(DT)
    assert body.steps == before + 20 and not r.holding  # walking again at once
    r.idle_hold = False
    r.stop()
    s0 = body.steps
    for _ in range(200):
        r.update(DT)
    assert body.steps == s0 + 200  # option off: the policy runs every step


def test_policy_body_hold_skips_the_network_and_resumes_cleanly(monkeypatch):
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
    body.step(0.005, [0, 0, 0])
    n = len(body.policy.forwards)
    for _ in range(100):
        body.hold(0.005)
    assert body.holding and len(body.policy.forwards) == n  # no network call while holding
    assert np.allclose(body.policy.robot.targets, np.linspace(-0.5, 0.5, 12))  # PD targets at the standing pose
    body.step(0.005, [0.3, 0, 0])
    assert not body.holding and len(body.policy.forwards) == n + 1 and not body.error


def test_camera_is_paused_until_a_capture_is_near():
    from Physical_AI_Fleet_python.perception.frame_capture import SimCameraCapture
    calls = []
    cam = SimCameraCapture("/x", warmup_s=0.1, discard_frames=0)
    cam._cam = SimpleNamespace(pause=lambda: calls.append("pause"), resume=lambda: calls.append("resume"),
                               get_rgba=lambda: np.full((4, 6, 4), 90, np.uint8))
    assert cam.set_active(False) and calls == ["pause"] and not cam.active
    cam.advance(1.0)
    assert cam._rendered_s == 0.0  # no warm-up while paused
    assert cam.grab_rgb() is None and calls == ["pause", "resume"] and cam.active  # asked for a frame: wakes up, warms up
    cam.advance(0.2)
    assert cam.grab_rgb() is not None
    assert cam.set_active(True) is False  # already on


# ------------------------------------------------------------------ the floor layout
def test_floor_layout_robots_are_parsed_and_checked(tmp_path):
    p = tmp_path / "floor.json"
    p.write_text(json.dumps({"schema_version": "1.0", "assets": [], "robots": [
        {"id": "spot-1", "model": "spot", "mode": "simulation", "position": [1, -3], "yaw_deg": 90},
        {"id": "real-go2", "model": "go2", "mode": "digital_twin", "position": [0, 0]},
        {"id": "x", "model": "rover", "position": [0, 0]},
        {"id": "spot-1", "model": "spot", "position": [2, 2]},
        {"id": "amr-1", "model": "amr", "position": [4, -3], "connection": "", "walking": False}]}))
    robots, problems = ac.load_robots(p)
    assert [r["id"] for r in robots] == ["spot-1", "amr-1"]
    assert robots[0] == {"id": "spot-1", "model": "spot", "mode": "simulation", "position": [1.0, -3.0], "yaw_deg": 90.0, "walking": True}
    assert len(problems) == 3 and "needs a connection" in problems[0] and "unknown model 'rover'" in problems[1] and "used twice" in problems[2]
    assert ac.load_robots(tmp_path / "missing.json") == ([], [])


def test_save_layout_keeps_robots_unless_given_and_the_old_asset_layout_is_migrated(tmp_path, monkeypatch):
    p = tmp_path / "floor.json"
    ac.save_layout({}, p, robots=[{"id": "go2-1", "model": "go2", "mode": "simulation", "position": [0, -3], "yaw_deg": 0}])
    ac.save_layout({}, p)  # e.g. ADD ASSET before the scene is loaded: robots stay
    assert json.loads(p.read_text())["robots"][0]["id"] == "go2-1"
    ac.save_layout({}, p, robots=[])
    assert json.loads(p.read_text())["robots"] == []
    from Physical_AI_Fleet_python.fleet_config import paths
    d = tmp_path / "inputs"
    d.mkdir()
    (d / "asset_layout.json").write_text(json.dumps({"schema_version": "1.0", "assets": [{"id": "motor-09", "type": "electric_motor", "position": [3, 3]}]}))
    monkeypatch.setenv("FLEET_INPUTS_DIR", str(d))
    monkeypatch.delenv("FLEET_ASSET_LAYOUT", raising=False)
    created = paths.ensure_inputs()
    assert any("from your asset_layout.json" in c for c in created)
    floor = json.loads((d / "floor_layout.json").read_text())
    assert floor["assets"][0]["id"] == "motor-09" and floor["robots"] == []
    assert ac.layout_path() == d / "floor_layout.json"


pxr = pytest.importorskip("pxr")
from test_scenario_v044 import STATE, env  # noqa: E402,F401


class FakeBody:
    """What PolicyBody looks like to the scenario (the env fixture stubs the real one)."""
    def __init__(self, model, prim_path, x, y, yaw, robot_name="", **k):
        from pxr import UsdGeom
        UsdGeom.Xform.Define(STATE["stage"], f"{prim_path}/{model.base_link}")
        self.spawn, self.policy_path, self.env_path, self.policy_name, self.policy_dt = (x, y, yaw), "/p.pt", "/e.yaml", "Fake", 0.005


def test_robots_with_yaw_are_saved_to_and_restored_from_the_floor_layout(env, monkeypatch, tmp_path):
    from Physical_AI_Fleet_python.fleet_config import asset_catalog
    sc_mod = importlib.import_module("Physical_AI_Fleet_python.scenario")
    made = []

    def body(*a, **k):
        b = FakeBody(*a, **k)
        made.append(b)
        return b
    monkeypatch.setattr(sc_mod, "PolicyBody", body)
    monkeypatch.setattr(sc_mod, "create_robot", lambda rid, rtype, mode, home, prim, *a, **k: SimpleNamespace(
        robot_id=rid, model=sc_mod.MODELS[k.get("model", "go2")], robot_type=rtype, mode=mode, pose=list(home), home_pose=home,
        radius=0.4, status="idle", note="", locomotion="rl_policy", prim_path=prim, neighbors_fn=None, provenance="simulated",
        telemetry_age=lambda: 0.0, shutdown=lambda: None, stop=lambda: None, update=lambda dt: None, summary=lambda: rid))
    sc = sc_mod.FleetScenario()
    env.made.append(sc)
    sc.setup_scene()
    sc.setup_post_load()
    msg, _ = sc.add_robot("spot-1", "spot", "simulation", "", "", "", (1.0, -3.0, 90.0))
    assert msg.startswith("Added") and "heading 90 deg" in msg and "Floor layout saved" in msg
    assert made[-1].spawn == pytest.approx((1.0, -3.0, math.pi / 2))
    floor = json.load(open(sc.settings.layout_path))
    assert floor["robots"] == [{"id": "spot-1", "model": "spot", "mode": "simulation", "position": [1.0, -3.0], "yaw_deg": 90.0,
                                "walking": True}] and {a["id"] for a in floor["assets"]} == {"motor-01", "pump-01", "valve-01"}
    assert "Home Yaw must be" in sc.add_robot("go2-1", "go2", "simulation", "", "", "", (0.0, -3.0, "north"))[0]
    sc.add_asset("ball_valve", "valve-02", -6.0, -8.0, 0)  # asset changes keep the robots in the file
    assert json.load(open(sc.settings.layout_path))["robots"][0]["id"] == "spot-1"
    # a new LOAD of the scene brings the robot back, with its heading
    sc.setup_scene()
    sc.setup_post_load()
    assert list(sc.coordinator.fleet.robots) == ["spot-1"] and made[-1].spawn == pytest.approx((1.0, -3.0, math.pi / 2))
    assert "spot-1 added" in sc.layout_robots_note
    assert sc.remove_robot("spot-1").startswith("Removed spot-1. Floor layout saved")
    assert json.load(open(sc.settings.layout_path))["robots"] == []
    assert sc.save_layout().startswith("Saved 4 asset(s) and 0 robot(s)")


def test_scenario_pauses_cameras_and_reports_performance(env):
    sc_mod = importlib.import_module("Physical_AI_Fleet_python.scenario")
    sc = sc_mod.FleetScenario()
    env.made.append(sc)
    sc.setup_scene()
    sc.setup_post_load()
    from Physical_AI_Fleet_python.fleet_config.catalog import MODELS
    robot = SimpleNamespace(robot_id="go2-1", model=MODELS["go2"], robot_type="quadruped", pose=[0, -3, 0], home_pose=(0, -3, 0),
                            radius=0.4, status="idle", note="", mode="simulation", locomotion="rl_policy", prim_path="",
                            neighbors_fn=None, goal=None, telemetry_age=lambda: 0.0, shutdown=lambda: None, stop=lambda: None,
                            update=lambda dt: None, summary=lambda: "go2-1", idle_hold=True, holding=True)
    sc.coordinator.fleet.add(robot)
    sc._cam_paths["go2-1"] = "/World/Fleet/go2_1/base/inspection_cam"
    sc._prepare_camera("go2-1", robot.model)
    cam = sc._cameras["go2-1"]
    assert cam.active is False  # paused right after the robot is added
    sc.update(0.005)
    assert cam.active is False and robot.idle_hold is True
    run = SimpleNamespace(phase="navigate", purpose="read_gauge", robot=SimpleNamespace(pose=[0, -3, 0], goal=(5.0, 0.0, 0.0)),
                          assignment=SimpleNamespace(target_asset_id="motor-01"), error="", future=None)
    sc.coordinator.executor.tick = lambda *a: None  # only the camera switching is tested here
    sc.coordinator.executor.runs["go2-1"] = run
    sc.update(0.005)
    assert cam.active is False  # still far away
    run.robot.pose = [4.0, 0.0, 0.0]
    sc.update(0.005)
    assert cam.active is True and "camera_on" in [r["kind"] for r in sc.coordinator.audit.recent(20)]
    del sc.coordinator.executor.runs["go2-1"]
    sc.update(0.005)
    assert cam.active is False
    assert "Skip" not in sc.set_idle_hold(False) and robot.idle_hold is False
    sc._perf.update(steps=0, fps=180.0, ext_ms=0.8)
    line = sc.perf_status()
    assert "180 physics steps/s" in line and "0.90x real time" in line and "cameras rendering 0/1" in line


def test_ui_has_no_thinking_option_and_has_home_yaw_and_idle_option(env):
    ui_mod = importlib.import_module("Physical_AI_Fleet_python.ui_builder")
    ui = ui_mod.UIBuilder()
    env.made.append(ui._scenario)
    ui.build_ui()
    assert not hasattr(ui, "_vlm_think_cb") and "thinking" not in ui._ai_fields()["vlm"]
    assert ui._yaw_field.get_value() == 0.0 and ui._idle_cb.v is True
    ui._idle_cb.click(False)
    assert "runs its policy on every step" in ui._run_msg.text
    assert "thinking always on" in ui._scenario.cosmos_status()
