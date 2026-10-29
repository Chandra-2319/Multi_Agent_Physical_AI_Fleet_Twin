"""Planner inside the site map, gate checks, camera warm-up, evidence names, dark-target detection."""
import json
import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.fleet_config.geometry import distance_to_polygon_edge, point_in_polygon, rotated_rect
from Physical_AI_Fleet_python.robots.locomotion.planner import GridPlanner, PolygonObstacle, RectObstacle
from test_site_and_assets import L_SHAPE, site

ROBOT_R = 0.4


def sampled(path, step=0.05):
    for a, b in zip(path, path[1:]):
        n = max(1, int(math.dist(a, b) / step))
        for k in range(n + 1):
            yield a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n


def planner_for(s, obstacles=()):
    p = GridPlanner(s.bounds, resolution=0.2)
    p.set_site(s.walkable, [z.polygon for z in s.no_go])
    p.set_obstacles(list(obstacles))
    return p


def test_path_stays_inside_an_l_shaped_walkable_area():
    s = site()
    p = planner_for(s)
    path = p.plan((18, 3), (3, 18), ROBOT_R)  # the straight line would cut the missing corner
    assert path and len(path) > 2
    for x, y in sampled(path[1:-1] if len(path) > 2 else path):
        assert point_in_polygon(x, y, s.walkable)
    assert all(point_in_polygon(x, y, s.walkable) for x, y in sampled(path))
    assert all(distance_to_polygon_edge(x, y, s.walkable) >= ROBOT_R - 0.05 for x, y in sampled(path[1:-1]))


def test_path_goes_around_a_no_go_zone_with_clearance():
    zone = [[4, 2], [8, 2], [8, 4.5], [4, 4.5]]
    s = site(zones=[zone])
    p = planner_for(s)
    path = p.plan((1, 3), (15, 3), ROBOT_R)
    assert planner_for(site(zones=[[[4, 1], [8, 1], [8, 5.5], [4, 5.5]]])).plan((1, 3), (15, 3), ROBOT_R) is None  # gaps too narrow
    assert path and len(path) > 2
    poly = s.no_go[0].polygon
    for x, y in sampled(path):
        assert not point_in_polygon(x, y, poly) and distance_to_polygon_edge(x, y, poly) >= ROBOT_R - 0.05


def test_goals_outside_the_walkable_area_or_in_a_no_go_zone_are_refused():
    s = site(zones=[[[10, 1], [14, 1], [14, 4], [10, 4]]])
    p = planner_for(s)
    assert p.plan((1, 1), (15, 15), ROBOT_R) is None  # in the missing corner of the L
    assert p.plan((1, 1), (12, 2), ROBOT_R) is None  # in the zone
    p2 = planner_for(site(zones=[[[10, 0.5], [14, 0.5], [14, 5.5], [10, 5.5]]]))  # the zone cuts the corridor completely
    assert p2.plan((2, 3), (18, 3), ROBOT_R) is None


def test_corridor_around_the_zone_is_used_when_it_exists():
    p = planner_for(site(zones=[[[10, 1], [14, 1], [14, 4], [10, 4]]]))
    path = p.plan((2, 2), (18, 2), ROBOT_R)
    assert path and max(y for _x, y in path) > 4  # it went over the zone, not through the 1 m gap below it
    for x, y in sampled(path[1:-1]):
        assert distance_to_polygon_edge(x, y, site().walkable) >= ROBOT_R - 0.05


def test_rotated_asset_footprint_blocks_exactly():
    fp = rotated_rect(10, 10, 3, 0.3, math.radians(80))  # a long, thin, almost vertical asset
    p = planner_for(site([[0, 0], [20, 0], [20, 20], [0, 20]]), [PolygonObstacle(tuple(fp), "exchanger")])
    path = p.plan((7, 10), (13, 10), ROBOT_R)
    assert path and len(path) > 2
    for x, y in sampled(path):
        assert not point_in_polygon(x, y, fp)
    assert PolygonObstacle(tuple(fp)).inflated_contains(10, 10, 0) and not PolygonObstacle(tuple(fp)).inflated_contains(12, 10, 0.2)


def test_without_a_site_the_planner_behaves_as_before():
    p = GridPlanner((-15, -15, 15, 15))
    p.set_obstacles([RectObstacle(0, 0, 1, 1)])
    path = p.plan((-5, 0), (5, 0), ROBOT_R)
    assert path and all(abs(x) > 1 or abs(y) > 1 for x, y in sampled(path))


# ------------------------------------------------------------------ gate
def test_gate_uses_the_site_map(tmp_path):
    from Physical_AI_Fleet_python.coordinator.fleet_state import FleetState
    from Physical_AI_Fleet_python.fleet_config.catalog import MODELS
    from Physical_AI_Fleet_python.safety.gate import APPROVED, REJECTED, SafetyGate
    from Physical_AI_Fleet_python.schemas.task_assignment import TaskAssignment
    s = SimpleNamespace(site=site(zones=[[[10, 1], [14, 1], [14, 4], [10, 4]]]), geofence=(0, 0, 20, 20), telemetry_ttl_s=10,
                        min_confidence=0.6, event_ttl_s=120,
                        assets={"ok": SimpleNamespace(asset_id="ok", approach_pose=(3, 3, 0)),
                                "zoned": SimpleNamespace(asset_id="zoned", approach_pose=(12, 2, 0)),
                                "corner": SimpleNamespace(asset_id="corner", approach_pose=(15, 15, 0))})
    f = FleetState(s)
    f.robots["go2-1"] = SimpleNamespace(robot_id="go2-1", status="idle", note="", model=MODELS["go2"], robot_type="quadruped",
                                        mode="simulation", telemetry_age=lambda: 0.0)
    g = SafetyGate(s)
    ta = lambda aid: TaskAssignment("x", "go2-1", "read_gauge", aid, "low", "r", source="operator")
    assert g.check(ta("ok"), None, f).status == APPROVED
    d = g.check(ta("zoned"), None, f)
    assert d.status == REJECTED and "inside no-go zone 'zone 0'" in d.reasons[0]
    assert "outside the walkable area" in g.check(ta("corner"), None, f).reasons[0]


# ------------------------------------------------------------------ camera warm-up and evidence names
class FakeCamera:
    instances = []

    def __init__(self, prim_path, resolution):
        self.prim_path, self.calls, self.frames = prim_path, 0, []
        FakeCamera.instances.append(self)

    def initialize(self):
        pass

    def get_rgba(self):
        self.calls += 1
        return self.frames.pop(0) if self.frames else np.full((4, 6, 4), 90, np.uint8)


@pytest.fixture
def fake_camera(monkeypatch):
    FakeCamera.instances = []
    mods = {n: types.ModuleType(n) for n in ("isaacsim", "isaacsim.sensors", "isaacsim.sensors.camera")}
    mods["isaacsim.sensors.camera"].Camera = FakeCamera
    for n, m in mods.items():
        monkeypatch.setitem(sys.modules, n, m)
    return FakeCamera


def test_camera_is_not_read_before_it_has_rendered_and_first_frames_are_dropped(fake_camera):
    from Physical_AI_Fleet_python.perception.frame_capture import SimCameraCapture
    cam = SimCameraCapture("/World/Fleet/go2_1/base/inspection_cam", warmup_s=0.5, discard_frames=2)
    assert cam.prepare() and cam.ready is False
    for _ in range(20):
        assert cam.grab_rgb() is None
        cam.advance(0.02)  # 20 x 0.02 = 0.4 s: still warming up
    assert fake_camera.instances[0].calls == 0  # the annotator was never asked: no "returned None" warning
    cam.advance(0.2)
    fake_camera.instances[0].frames = [None, np.zeros((0,), np.uint8)]  # empty buffers are skipped too
    got = [cam.grab_rgb() for _ in range(5)]
    assert got[:4] == [None, None, None, None] and got[4].shape == (4, 6, 3)


def test_camera_creation_failure_is_reported_not_raised(monkeypatch):
    from Physical_AI_Fleet_python.perception.frame_capture import SimCameraCapture
    monkeypatch.setitem(sys.modules, "isaacsim.sensors.camera", None)
    cam = SimCameraCapture("/x")
    assert cam.prepare() is False and cam.error and cam.grab_rgb() is None


def test_frame_stats_flags_a_black_target_in_a_lit_scene():
    from Physical_AI_Fleet_python.perception.frame_capture import frame_stats
    img = np.full((720, 1280, 3), 170, np.uint8)
    img[200:700, 320:870] = 0  # the screenshot: a black box in the middle of a bright scene
    st = frame_stats(img)
    assert st["dark_target"] is True and st["almost_black"] is False and st["centre_brightness"] < 15
    assert frame_stats(np.full((10, 10, 3), 120, np.uint8))["dark_target"] is False
    assert frame_stats(np.zeros((10, 10, 3), np.uint8))["almost_black"] is True


def test_evidence_names_carry_asset_and_robot(tmp_path):
    from Physical_AI_Fleet_python.perception.frame_capture import EvidenceStore
    ev = EvidenceStore(tmp_path)
    a = ev.save("spot-2", b"jpg", "read_gauge", asset_id="pump-01")
    b = ev.save("spot-2", b"jpg", "read_gauge", asset_id="pump-01")
    name = a.rsplit("/", 1)[-1]
    assert name.endswith("_pump-01_spot-2_read_gauge.jpg") and len(name.split("_")[0]) == 8
    assert b != a and b.endswith("_pump-01_spot-2_read_gauge_2.jpg")  # same second: never overwritten
    assert ev.save("op/x", b"j", "inspect", asset_id="a b").rsplit("/", 1)[-1].endswith("_a-b_op-x_inspect.jpg")
    assert ev.save("spot-1", b"j").endswith("_spot-1_frame.jpg")  # old call style still works


def test_perception_saves_evidence_under_asset_and_robot(tmp_path):
    from Physical_AI_Fleet_python.fleet_config.settings import Settings
    from Physical_AI_Fleet_python.perception.event_builder import PerceptionPipeline
    from Physical_AI_Fleet_python.perception.frame_capture import EvidenceStore
    s = Settings()
    pp = PerceptionPipeline(s, SimpleNamespace(available=False), EvidenceStore(tmp_path))
    ev = pp.run(robot_id="spot-2", provenance="simulated", asset_id="pump-01", purpose="read_gauge", jpeg=b"JPEG",
                sim_fault={"kind": "overpressure", "gauge_value": 18.0})
    assert ev.evidence[0].endswith("_pump-01_spot-2_read_gauge.jpg") and ev.gauge.in_range is False and ev.gauge.unit == "bar"
    leak = pp.run(robot_id="spot-1", provenance="simulated", asset_id="valve-01", purpose="read_gauge", jpeg=None,
                  sim_fault={"kind": "leak"})
    assert "control valve" in leak.observation and leak.evidence == []
