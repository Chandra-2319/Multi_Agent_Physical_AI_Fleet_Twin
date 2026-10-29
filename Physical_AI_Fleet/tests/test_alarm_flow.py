"""End to end with the REAL executor, perception and coordinator; robots 'walk' instantly; Nemotron is a fake model.

fault injected -> alarm -> Nemotron picks robot by priority -> Go2 walks to the asset -> camera frame captured ->
perception reads the gauge -> reading confirms the fault -> the Go2 walks home -> repair chain.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.coordinator import coordinator as C
from Physical_AI_Fleet_python.coordinator.fleet_state import CONFIRMED, OPEN, VERIFYING
from Physical_AI_Fleet_python.fleet_config.catalog import MODELS
from Physical_AI_Fleet_python.fleet_config.settings import AssetConfig
from Physical_AI_Fleet_python.skills import executor as E
from helpers_v4 import Clock, FakeNemotron, make_cfg

PRIO = {"high": 0, "medium": 1, "low": 2}


class WalkingRobot:
    """Stand-in for LeggedSimRobot / SimRobot: go_to() 'walks' for a few ticks, then idles at the goal."""
    def __init__(self, robot_id, model_key="go2", pose=(0.0, -3.0), ticks=3):
        self.robot_id, self.model = robot_id, MODELS[model_key]
        self.robot_type, self.mode, self.provenance = self.model.robot_type, "simulation", "simulated"
        self.status, self.note, self.pose, self.goal = "idle", "", [pose[0], pose[1], 0.0], None
        self.radius, self.home_pose, self.locomotion, self.current_skill = 0.4, (pose[0], pose[1], 0.0), "unicycle", None
        self.neighbors_fn, self._ticks, self._left, self.visited = None, ticks, 0, []
    def telemetry_age(self): return 0.0
    def go_to(self, x, y, yaw, name=""):
        self.goal, self.status, self._left = (x, y, yaw), "busy", self._ticks
        self.visited.append(name)
    def update(self, dt):
        if self.goal is not None:
            self._left -= 1
            if self._left <= 0:
                self.pose = [self.goal[0], self.goal[1], self.goal[2]]
                self.goal, self.status = None, "idle"
    def stop(self): self.goal = None; self.status = "idle"
    def reset_to_home(self): self.stop(); self.pose = list(self.home_pose)
    def on_world_reset(self): self.reset_reset = True; self.reset_to_home()
    def shutdown(self): pass
    def summary(self): return f"{self.robot_id} {self.status}"
    def snapshot(self, ttl): return {"robot_id": self.robot_id}


class RecordingEvidence:
    def __init__(self): self.saved = []
    def save(self, robot_id, jpeg, tag="frame", asset_id=""):
        self.saved.append((robot_id, tag, len(jpeg)))
        return f"/evidence/{robot_id}_{tag}_{len(self.saved)}.jpg"


def settings(tmp_path, n=3):
    s = SimpleNamespace(log_dir=Path(tmp_path), nebius_base_url="https://example.invalid/v1", nebius_api_key="k", request_timeout_s=5,
                        cosmos_base_url="https://example.invalid/v1", cosmos_api_key="k", cosmos_model_id="", nemotron_model_id="m",
                        nemotron_enabled=True, cosmos_enabled=False, min_confidence=0.6, event_ttl_s=300.0, telemetry_ttl_s=5.0,
                        geofence=(-15.0, -15.0, 15.0, 15.0), gauge_min_roi_px=48, gauge_upscale=4)
    s.assets = {f"motor-0{i}": AssetConfig(f"motor-0{i}", (4.0 * i, 2.0, 0.4), (4.0 * i - 1.5, 2.0, 0.0)) for i in range(1, n + 1)}
    return s


class Harness:
    def __init__(self, monkeypatch, tmp_path, robots, n_assets=3, nemotron="smart", capture=None, cfg=None):
        self.clock = Clock()
        self.s = settings(tmp_path, n_assets)
        self.faults, self.evidence = {}, RecordingEvidence()
        self.capture_calls = []
        monkeypatch.setattr(E, "time", SimpleNamespace(time=self.clock))
        monkeypatch.setattr(C, "EvidenceStore", lambda *a, **k: self.evidence)
        monkeypatch.setattr(C, "CosmosReason", lambda client, model: SimpleNamespace(available=False, model_id=model))
        cap = capture or (lambda robot: b"JPEGBYTES-" + robot.robot_id.encode())
        def capture_fn(robot):
            self.capture_calls.append(robot.robot_id)
            return cap(robot)
        self.coord = C.FleetCoordinator(self.s, capture_fn=capture_fn, fault_fn=self.faults.get, action_config=cfg or make_cfg())
        self.coord._clock = self.clock
        self.model = FakeNemotron(self._smart if nemotron == "smart" else nemotron)
        self.coord.brain.client = self.model
        self.robots = {r.robot_id: r for r in robots}
        for r in robots:
            self.coord.fleet.add(r)
    @staticmethod
    def _smart(messages, rf):
        snap = json.loads(messages[1]["content"])
        tasks = [t for t in snap["due_tasks"] if t["candidates"]]
        tasks.sort(key=lambda t: (PRIO[t["priority"]], -t["waiting_s"]))
        t = tasks[0]
        return json.dumps({"snapshot_id": snap["snapshot_id"], "decision": "assign", "task_id": t["task_id"],
                           "robot_id": t["candidates"][0]["robot_id"], "priority": t["priority"], "rationale": "priority, then nearest"})
    def inject(self, asset, kind="overpressure", severity="medium"):
        a = self.s.assets[asset]
        self.faults[asset] = {"kind": kind, "gauge_value": a.normal_high + 2.5}
        return self.coord.report_alarm(asset, kind, severity)
    def step(self, n=1):
        for _ in range(n):
            self.clock.advance(0.6)
            self.coord.tick(0.005)
            fut = self.coord._brain_future
            if fut is not None:
                fut.result(timeout=10)
            for run in list(self.coord.executor.runs.values()):
                if run.future is not None:
                    run.future.result(timeout=10)
    def until(self, cond, max_steps=80):
        for _ in range(max_steps):
            if cond():
                return True
            self.step()
        return cond()
    def started(self):
        return [(r["robot_id"], r["target"]) for r in self.coord.audit.recent(400) if r["kind"] == "skill_started"]
    def kinds(self):
        return [r["kind"] for r in self.coord.audit.recent(400)]


@pytest.fixture
def two_go2(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path, [WalkingRobot("go2-1", pose=(0, -3)), WalkingRobot("go2-2", pose=(12, -3))])


def test_injected_fault_sends_a_quadruped_to_inspect_and_the_camera_frame_is_captured(two_go2):
    h = two_go2
    assert "inspection will be assigned" in h.inject("motor-01", "overpressure", "high")
    assert h.coord.fleet.incident_for("motor-01").status == OPEN and h.coord.fleet.incident_for("motor-01").needs_inspection is True
    assert h.until(lambda: h.coord.fleet.incident_for("motor-01").status == CONFIRMED)
    assert h.started()[0] == ("go2-1", "motor-01") and h.robots["go2-1"].visited[0] == "motor-01"
    assert h.capture_calls == ["go2-1"] and h.evidence.saved[0][:2] == ("go2-1", "read_gauge")
    ev = [r for r in h.coord.audit.recent(400) if r["kind"] == "fleet_event"][-1]["event"]
    assert ev["robot_id"] == "go2-1" and ev["event_type"] == "anomaly" and ev["evidence"] == ["/evidence/go2-1_read_gauge_1.jpg"]
    assert ev["gauge"]["in_range"] is False and ev["gauge"]["value"] == pytest.approx(8.5)
    inc = h.coord.fleet.incident_for("motor-01")
    assert inc.needs_inspection is False and inc.priority == "high" and any("confirmed by ['go2-1']" in n for n in inc.history)


def test_the_alarm_alone_does_not_confirm_the_fault(two_go2):
    h = two_go2
    h.inject("motor-01")
    h.coord.tick(0.005)
    inc = h.coord.fleet.incident_for("motor-01")
    assert inc.status == OPEN and inc.problem_reports == 0 and "incident_confirmed" not in h.kinds()


def test_priority_decides_which_asset_is_served_first_and_nearest_robot_goes(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, [WalkingRobot("go2-1", pose=(0, -3), ticks=60), WalkingRobot("go2-2", pose=(12, -3), ticks=60)])
    h.inject("motor-01", severity="low")
    h.inject("motor-03", severity="high")
    h.inject("motor-02", severity="medium")
    assert h.until(lambda: len(h.started()) >= 2)
    assert h.started()[:2] == [("go2-2", "motor-03"), ("go2-1", "motor-02")]
    snap = json.loads(h.model.calls[0][1]["content"])
    assert {t["task_id"]: t["priority"] for t in snap["due_tasks"]} == {"inspect:motor-01": "low", "inspect:motor-03": "high", "inspect:motor-02": "medium"}
    assets = {a["asset_id"]: a for a in snap["assets"]}
    assert assets["motor-03"]["incident_priority"] == "high" and assets["motor-03"]["alarm"] == "overpressure"


def test_low_priority_waits_until_a_robot_is_free_then_goes_last(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, [WalkingRobot("go2-1", pose=(0, -3), ticks=30), WalkingRobot("go2-2", pose=(12, -3), ticks=30)])
    h.inject("motor-01", severity="low")
    h.inject("motor-02", severity="high")
    h.inject("motor-03", severity="medium")
    assert h.until(lambda: len(h.started()) >= 3, max_steps=200)
    assert [t for _r, t in h.started()][:3] == ["motor-02", "motor-03", "motor-01"]


def test_rule_based_fallback_also_serves_by_priority(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, [WalkingRobot("go2-1", pose=(0, -3))])
    h.coord.assigner.use_nemotron = False
    h.inject("motor-01", severity="low")
    h.inject("motor-03", severity="high")
    assert h.until(lambda: h.started())
    assert h.started()[0] == ("go2-1", "motor-03") and h.model.calls == [] and h.coord.brain.last.source == "rule_based"


def test_a_leak_alarm_is_confirmed_by_the_robot_reading(two_go2):
    h = two_go2
    h.inject("motor-02", "leak", "medium")
    assert h.until(lambda: h.coord.fleet.incident_for("motor-02").status == CONFIRMED)
    ev = [r for r in h.coord.audit.recent(400) if r["kind"] == "fleet_event"][-1]["event"]
    assert ev["event_type"] == "leak_suspected" and ev["robot_id"] in ("go2-1", "go2-2")


def test_alarm_with_a_healthy_gauge_is_closed_as_a_false_alarm(two_go2):
    h = two_go2
    h.coord.report_alarm("motor-01", "overpressure", "medium")
    assert h.until(lambda: h.coord.fleet.incident_for("motor-01") is None)
    assert any(r["kind"] == "incident_closed" and r.get("reason") == "false_alarm" for r in h.coord.audit.recent(400))


def test_after_confirmation_the_repair_chain_continues(two_go2):
    h = two_go2
    h.inject("motor-01", severity="high")
    assert h.until(lambda: h.coord.fleet.incident_for("motor-01").status == CONFIRMED)
    h.step(12)
    assert "waiting: no available robot model" in h.coord.fleet.incident_for("motor-01").waiting_note
    assert h.coord.mark_tools_loaded("motor-01").startswith("tools marked")
    h.coord.fleet.add(WalkingRobot("amr-1", "amr", pose=(1, 1)))
    assert h.until(lambda: "amr-1" in {r for r, _t in h.started()})
    h.faults.pop("motor-01")
    h.until(lambda: h.coord.fleet.incident_for("motor-01").status == "awaiting_repair")
    h.coord.mark_repaired("motor-01")
    assert h.coord.fleet.incident_for("motor-01").status == VERIFYING
    assert h.until(lambda: h.coord.fleet.incident_for("motor-01") is None, max_steps=200)


def test_events_from_non_robots_are_alarms_not_proof(two_go2):
    from Physical_AI_Fleet_python.schemas.fleet_event import FleetEvent
    h = two_go2
    h.coord.submit_event(FleetEvent(robot_id="sensor-gw", provenance="simulated", asset_id="motor-02", event_type="leak_suspected", observation="x", confidence=0.7))
    h.coord.tick(0.005)
    inc = h.coord.fleet.incident_for("motor-02")
    assert inc.status == OPEN and inc.needs_inspection and inc.alarm_kind == "leak" and inc.priority == "medium"
    h.coord.submit_event(FleetEvent(robot_id="operator", provenance="operator_phone", asset_id="motor-02", event_type="normal", observation="fine", confidence=0.9))
    h.coord.tick(0.005)
    assert h.coord.fleet.incident_for("motor-02") is not None


def test_repeated_alarms_raise_priority_never_lower_it_and_do_not_regress_state(two_go2):
    h = two_go2
    h.coord.report_alarm("motor-01", "overpressure", "medium")
    h.coord.report_alarm("motor-01", "overpressure", "low")
    assert h.coord.fleet.incident_for("motor-01").priority == "medium"
    h.coord.report_alarm("motor-01", "leak", "high")
    inc = h.coord.fleet.incident_for("motor-01")
    assert inc.priority == "high" and inc.alarm_kind == "leak"
    inc.status = CONFIRMED
    assert "already confirmed" in h.coord.report_alarm("motor-01", "overpressure", "high") and inc.status == CONFIRMED
    assert "Unknown asset" in h.coord.report_alarm("nope")
    assert h.coord.report_alarm("motor-02", "x", "bogus").startswith("Alarm on motor-02") and h.coord.fleet.incident_for("motor-02").priority == "medium"


def test_robot_waits_standing_still_for_a_camera_that_is_not_ready_yet(monkeypatch, tmp_path):
    state = {"n": 0}
    def cap(robot):
        state["n"] += 1
        return E.CAPTURE_RETRY if state["n"] < 6 else b"FRAME"
    h = Harness(monkeypatch, tmp_path, [WalkingRobot("go2-1", pose=(0, -3))], capture=cap)
    h.inject("motor-01")
    assert h.until(lambda: h.coord.fleet.incident_for("motor-01") and h.coord.fleet.incident_for("motor-01").status == CONFIRMED)
    assert state["n"] == 6 and h.evidence.saved and "capture_waiting" in h.kinds() and "capture_timeout" not in h.kinds()


def test_camera_that_never_delivers_does_not_stall_the_mission(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, [WalkingRobot("go2-1", pose=(0, -3))], capture=lambda r: E.CAPTURE_RETRY)
    h.inject("motor-01")
    assert h.until(lambda: h.coord.fleet.incident_for("motor-01") and h.coord.fleet.incident_for("motor-01").status == CONFIRMED, max_steps=120)
    assert "capture_timeout" in h.kinds() and h.evidence.saved == [] and "go2-1" not in h.coord.executor.runs


def test_camera_off_skips_the_wait_entirely(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, [WalkingRobot("go2-1", pose=(0, -3))], capture=lambda r: None)
    h.inject("motor-01")
    assert h.until(lambda: h.coord.fleet.incident_for("motor-01") and h.coord.fleet.incident_for("motor-01").status == CONFIRMED)
    assert h.capture_calls == ["go2-1"] and "capture_waiting" not in h.kinds()


def test_asset_added_later_is_scheduled_and_known_to_nemotron(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, [WalkingRobot("go2-1", pose=(0, -3))], n_assets=1)
    h.s.assets["valve-09"] = AssetConfig("valve-09", (-4.0, 6.0, 0.4), (-2.9, 6.0, 0.0))
    h.coord.on_assets_changed()
    assert "valve-09" in h.coord.scheduler.last
    h.inject("valve-09", severity="high")
    assert h.until(lambda: h.started()) and h.started()[0] == ("go2-1", "valve-09")
    assert {a["asset_id"] for a in json.loads(h.model.calls[0][1]["content"])["assets"]} == {"motor-01", "valve-09"}
    del h.s.assets["valve-09"]
    h.coord.on_assets_changed()
    assert "valve-09" not in h.coord.scheduler.last


def test_force_evaluate_asks_nemotron_immediately(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, [WalkingRobot("go2-1", pose=(0, -3))], n_assets=1)
    h.coord.scheduler.last["motor-01"] = h.coord._run_s - 5000
    h.coord.auto_dispatch = False
    assert "Auto Dispatch is off" in h.coord.force_evaluate()
    h.coord.auto_dispatch = True
    h.coord._last_eval_t = h.clock()
    h.coord._last_call_t = h.clock()
    assert "next step" in h.coord.force_evaluate()
    h.step()
    assert h.model.calls
