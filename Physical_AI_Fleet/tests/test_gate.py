import time

from Physical_AI_Fleet_python.coordinator.fleet_state import FleetState
from Physical_AI_Fleet_python.fleet_config.settings import Settings
from Physical_AI_Fleet_python.robots.robot_factory import create_robot, make_planner
from Physical_AI_Fleet_python.safety.gate import APPROVED, PENDING, REJECTED, SafetyGate
from Physical_AI_Fleet_python.schemas.fleet_event import FleetEvent
from Physical_AI_Fleet_python.schemas.task_assignment import TaskAssignment


def setup():
    s = Settings()
    fleet = FleetState(s)
    fleet.add(create_robot("go2", "quadruped", "simulation", (0, -3, 0), model="go2", planner=make_planner(s)))
    fleet.add(create_robot("spot", "quadruped", "simulation", (2, -3, 0), model="spot", planner=make_planner(s)))
    fleet.add(create_robot("amr", "amr", "simulation", (1, -5, 0)))
    ev = FleetEvent(robot_id="go2", provenance="simulated", asset_id="motor-01", event_type="anomaly",
                    observation="x", confidence=0.9)
    return s, fleet, SafetyGate(s), ev


def assign(ev, robot, skill, target="motor-01", source="nemotron"):
    return TaskAssignment(ev.event_id if ev else "OPERATOR", robot, skill, target, "medium", "t", source)


def test_low_risk_approved():
    _, fleet, gate, ev = setup()
    assert gate.check(assign(ev, "go2", "inspect_asset"), ev, fleet).status == APPROVED
    assert gate.check(assign(ev, "spot", "read_gauge"), ev, fleet).status == APPROVED


def test_high_risk_needs_approval():
    _, fleet, gate, ev = setup()
    assert gate.check(assign(ev, "amr", "deliver_tools"), ev, fleet).status == PENDING


def test_capability_rejected():
    _, fleet, gate, ev = setup()
    d = gate.check(assign(ev, "amr", "read_gauge"), ev, fleet)
    assert d.status == REJECTED and any("cannot" in r for r in d.reasons)


def test_stale_event_rejected():
    _, fleet, gate, ev = setup()
    ev.timestamp = "2020-01-01T00:00:00Z"
    assert gate.check(assign(ev, "go2", "inspect_asset"), ev, fleet).status == REJECTED


def test_stale_telemetry_rejected():
    _, fleet, gate, ev = setup()
    fleet.get("go2").last_telemetry = time.time() - 60
    assert gate.check(assign(ev, "go2", "inspect_asset"), ev, fleet).status == REJECTED


def test_geofence_rejected():
    s, fleet, gate, ev = setup()
    from Physical_AI_Fleet_python.fleet_config.site_map import SiteMap
    s.site = SiteMap(walkable=[(-1, -1), (1, -1), (1, 1), (-1, 1)])  # v0.4.4+: the site map is the geofence
    assert gate.check(assign(ev, "go2", "inspect_asset"), ev, fleet).status == REJECTED


def test_low_confidence_blocks_high_risk():
    _, fleet, gate, ev = setup()
    ev.confidence = 0.3
    assert gate.check(assign(ev, "amr", "deliver_tools"), ev, fleet).status == REJECTED


def test_operator_phone_cannot_trigger_high_risk():
    _, fleet, gate, ev = setup()
    ev.provenance = "operator_phone"
    assert gate.check(assign(ev, "amr", "deliver_tools"), ev, fleet).status == REJECTED


def test_busy_and_fallen_robots_rejected():
    _, fleet, gate, ev = setup()
    fleet.get("go2").go_to(2, 2, 0)
    assert gate.check(assign(ev, "go2", "inspect_asset"), ev, fleet).status == REJECTED
    spot = fleet.get("spot")
    spot.status, spot.note = "error", "fallen"
    assert gate.check(assign(None, "spot", "return_home", None, "operator"), None, fleet).status == REJECTED
