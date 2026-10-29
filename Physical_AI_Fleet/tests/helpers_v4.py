"""Fakes for the coordinator tests. They only use attributes / methods the real classes expose."""
import copy
import json
import time
from types import SimpleNamespace

from Physical_AI_Fleet_python.fleet_config.action_config import FleetActionConfig, load_action_config
from Physical_AI_Fleet_python.fleet_config.catalog import MODELS


def raw_config() -> dict:
    return json.loads(open(load_action_config().path, encoding="utf-8").read())


def make_cfg(mutator=None) -> FleetActionConfig:
    """The shipped config, with routine inspections ON and every asset due at start-up (most tests are written for that;
    the shipped defaults are routine_inspections false / initial_due false)."""
    raw = copy.deepcopy(raw_config())
    raw["schedule"]["routine_inspections"] = True
    raw["schedule"]["initial_due"] = True
    if mutator:
        mutator(raw)
    return FleetActionConfig(raw, "<test>")


class Clock:
    def __init__(self, t=10_000.0):
        self.t = t
    def __call__(self):
        return self.t
    def advance(self, s):
        self.t += s


class FakeAsset:
    def __init__(self, asset_id, xy, approach, low=2.0, high=8.0, unit="bar"):
        self.asset_id, self.position = asset_id, (xy[0], xy[1], 0.0)
        self.approach_pose, self.normal_low, self.normal_high, self.unit = approach, low, high, unit
        self.half_extents = (0.4, 0.25)


class FakeSettings:
    def __init__(self, tmp_path, assets=None):
        self.log_dir = str(tmp_path)
        self.nebius_base_url, self.nebius_api_key, self.request_timeout_s = "https://example.invalid/v1", "key", 5
        self.cosmos_base_url, self.cosmos_api_key, self.cosmos_model_id = "https://example.invalid/v1", "key", "cosmos"
        self.nemotron_model_id, self.nemotron_enabled, self.cosmos_enabled = "nvidia/nemotron-test", True, False
        self.min_confidence, self.event_ttl_s, self.telemetry_ttl_s = 0.6, 300.0, 5.0
        self.geofence = (-15.0, -15.0, 15.0, 15.0)
        self.assets = assets if assets is not None else {"pump-01": FakeAsset("pump-01", (5, 0), (4.0, 0.0, 0.0))}


class FakeRobot:
    """Idle fake; go_to / home_pose make it usable for the walk-home tests."""
    def __init__(self, robot_id, model_key, pose=(0.0, 0.0), mode="simulation"):
        self.robot_id, self.model = robot_id, MODELS[model_key]
        self.robot_type, self.mode = self.model.robot_type, mode
        self.status, self.note, self.pose = "idle", "", [pose[0], pose[1], 0.0]
        self.radius, self.neighbors_fn, self.age = 0.4, None, 0.0
        self.home_pose, self.goal, self.go_to_calls = (pose[0], pose[1], 0.0), None, []
    def telemetry_age(self): return self.age
    def update(self, dt): pass
    def stop(self): self.goal = None; self.status = "idle" if self.status == "busy" else self.status
    def reset_to_home(self): self.status, self.goal = "idle", None
    def shutdown(self): pass
    def summary(self): return f"{self.robot_id} {self.status}"
    def snapshot(self, ttl): return {"robot_id": self.robot_id}
    def go_to(self, x, y, yaw, name=""):
        self.goal, self.status = (x, y, yaw), "busy"
        self.go_to_calls.append(name)


class FakeAudit:
    def __init__(self, log_dir): self.path, self.records = log_dir, []
    def record(self, kind, data=None, **extra): self.records.append((kind, {**(data or {}), **extra}))
    def kinds(self): return [k for k, _ in self.records]
    def last(self, kind): return next(d for k, d in reversed(self.records) if k == kind)


class FakeExecutor:
    def __init__(self, settings, pool, perception, capture_fn, fault_fn, audit, on_done):
        self.runs, self.started, self.on_done = {}, [], on_done
    def start(self, assignment, robot, purpose, incident_id):
        run = SimpleNamespace(assignment=assignment, robot=robot, purpose=purpose, incident_id=incident_id, error=None, result_event=None)
        self.runs[robot.robot_id] = run
        self.started.append((assignment.skill, robot.robot_id, assignment.target_asset_id, purpose))
        robot.status = "busy"
    def tick(self, dt=0.0): pass
    def cancel(self, rid, reason):
        run = self.runs.pop(rid, None)
        if run:
            run.robot.stop()
    def finish(self, robot_id, error=None, event=None):
        run = self.runs.pop(robot_id)
        run.robot.status, run.error, run.result_event = "idle", error, event
        self.on_done(run)


class FakeVerifier:
    def __init__(self, s): self.attempts = {}
    def is_clear(self, ev): return bool(ev.gauge and ev.gauge.in_range is True and ev.confidence >= 0.6)


class FakeEvent:
    _n = 0
    def __init__(self, asset_id, robot_id, kind, value=None, in_range=None, confidence=0.9):
        FakeEvent._n += 1
        self.event_id = f"EV-{FakeEvent._n}"
        self.asset_id, self.robot_id, self.event_type, self.confidence = asset_id, robot_id, kind, confidence
        self.provenance, self.observation, self.incident_id = "simulated", "test", None
        self.gauge = SimpleNamespace(value=value, unit="bar", in_range=in_range, confidence=confidence, roi_px=100)
        self.timestamp, self.ttl_s = time.time(), 300.0
    @property
    def actionable(self): return self.event_type != "normal"
    def to_dict(self): return {"event_id": self.event_id, "asset_id": self.asset_id, "event_type": self.event_type}


def out_of_limits(asset, robot): return FakeEvent(asset, robot, "anomaly", 9.5, False)
def within_limits(asset, robot): return FakeEvent(asset, robot, "normal", 5.0, True)
def unreadable(asset, robot): return FakeEvent(asset, robot, "inconclusive", None, None, 0.3)


class FakeNemotron:
    api_key = "k"
    def __init__(self, mode="first"):
        self.mode, self.calls, self.formats = mode, [], []
    def chat(self, model, messages, *, response_format=None, **kw):
        self.formats.append(response_format)
        self.calls.append(messages)
        if callable(self.mode):
            return self.mode(messages, response_format)
        snap = json.loads(messages[1]["content"])
        if self.mode == "garbage":
            return "I think robot go2-1 should go."
        t = snap["due_tasks"][0]
        return json.dumps({"snapshot_id": snap["snapshot_id"], "decision": "assign", "task_id": t["task_id"],
                           "robot_id": t["candidates"][0]["robot_id"], "priority": t["priority"], "rationale": f"{t['action']} for {t['for_asset']}"})


def make_coordinator(monkeypatch, tmp_path, cfg=None, settings=None, nemotron=None):
    from Physical_AI_Fleet_python.coordinator import coordinator as C
    monkeypatch.setattr(C, "AuditLog", FakeAudit)
    monkeypatch.setattr(C, "SkillExecutor", FakeExecutor)
    monkeypatch.setattr(C, "Verifier", FakeVerifier)
    monkeypatch.setattr(C, "CosmosReason", lambda *a, **k: SimpleNamespace(model_id=""))
    monkeypatch.setattr(C, "PerceptionPipeline", lambda *a, **k: None)
    monkeypatch.setattr(C, "EvidenceStore", lambda *a, **k: None)
    coord = C.FleetCoordinator(settings or FakeSettings(tmp_path), action_config=cfg or make_cfg())
    coord.brain.client = nemotron or FakeNemotron()
    clock = Clock()
    coord._clock = clock
    return coord, clock


def pump(coord, clock, steps=1, step_s=0.6):
    for _ in range(steps):
        clock.advance(step_s)
        coord.tick(0.005)
        fut = coord._brain_future
        if fut is not None:
            fut.result(timeout=10)
            coord.tick(0.005)


def pump_until(coord, clock, cond, max_steps=60):
    for _ in range(max_steps):
        if cond():
            return True
        pump(coord, clock)
    return cond()
