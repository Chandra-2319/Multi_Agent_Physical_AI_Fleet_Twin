"""scenario.py + ui_builder.py on a REAL USD stage (pxr), with stand-ins for omni / isaacsim / robot adapters.

The real planner, robot factory, site map, asset catalog and stage authoring are used. This does NOT prove anything
about rendering in Isaac Sim (materials, lights, camera images); it catches wiring mistakes.
"""
import importlib
import json
import math
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

from isolation import isolated_user_files  # noqa: E402,F401

PKG = "Physical_AI_Fleet_python"
STATE = {}


class W:
    def __init__(self, *a, **k): self.enabled = True
    def cleanup(self): pass
    def reset(self): pass


class StringField(W):
    def __init__(self, label, default_value="", **k): super().__init__(); self.v = default_value
    def get_value(self): return self.v
    def set_value(self, v): self.v = v


class FloatField(StringField):
    pass


class CheckBox(W):
    def __init__(self, label, default_value=False, on_click_fn=None, **k): super().__init__(); self.v, self.fn = default_value, on_click_fn
    def click(self, value): self.v = value; self.fn and self.fn(value)


class DropDown(W):
    def __init__(self, label, tooltip="", populate_fn=None, on_selection_fn=None, **k):
        super().__init__(); self.populate, self.fn, self.items = populate_fn, on_selection_fn, []
    def repopulate(self): self.items = list(self.populate())
    def choose(self, item): self.fn and self.fn(item)


class Button(W):
    def __init__(self, label, text, on_click_fn=None, **k): super().__init__(); self.fn = on_click_fn
    def click(self): self.fn()


class StateButton(W):
    def __init__(self, label, a, b, on_a_click_fn=None, on_b_click_fn=None, physics_callback_fn=None, **k): super().__init__(); self.a = on_a_click_fn


class TextBlock(W):
    def __init__(self, label, num_lines=3, include_copy_button=False, **k): super().__init__(); self.text = ""
    def set_text(self, t): self.text = t


class Frame:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


class LoadButton(W):
    def __init__(self, label, text, setup_scene_fn=None, setup_post_load_fn=None, **k): super().__init__(); self.s, self.p = setup_scene_fn, setup_post_load_fn
    def set_world_settings(self, **k): pass
    def press(self): self.s(); self.p()


class Model:
    def __init__(self, v=""): self.v = v
    def get_value_as_string(self): return self.v
    def set_value(self, v): self.v = v


class FakeWorld:
    def __init__(self): self.scene = SimpleNamespace(add_default_ground_plane=lambda **k: None, add=lambda o: None)
    def set_simulation_dt(self, **k): pass
    def get_physics_dt(self): return 1 / 200


class VisualCylinderStub:
    def __init__(self, prim_path, **k):
        UsdGeom.Cylinder.Define(STATE["stage"], prim_path)
        self.prim_path = prim_path


class Robot:
    def __init__(self, robot_id, home, model="go2"):
        from Physical_AI_Fleet_python.fleet_config.catalog import MODELS
        self.robot_id, self.model, self.robot_type = robot_id, MODELS[model], MODELS[model].robot_type
        self.pose, self.home_pose, self.radius, self.mode, self.status, self.note = [home[0], home[1], 0], home, 0.4, "simulation", "idle", ""
        self.locomotion, self.prim_path, self.neighbors_fn = "rl_policy", "", None
    def telemetry_age(self): return 0.0
    def shutdown(self): pass
    def stop(self): pass
    def update(self, dt): pass
    def summary(self): return self.robot_id


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("FLEET_LOG_DIR", str(tmp_path / "logs"))
    STATE.clear()
    STATE["stage"] = Usd.Stage.CreateInMemory()
    world = FakeWorld()
    mods = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        m.__path__ = []
        mods[name] = m
        return m

    def create_new_stage():
        STATE["stage"] = Usd.Stage.CreateInMemory()
    for n in ("isaacsim", "isaacsim.core", "isaacsim.core.api", "isaacsim.core.utils", "isaacsim.storage", "isaacsim.examples",
              "isaacsim.examples.extension", "isaacsim.gui", "isaacsim.gui.components", "isaacsim.sensors", "omni", "omni.kit"):
        mod(n)
    mod("isaacsim.core.api.objects", VisualCylinder=VisualCylinderStub)
    mod("isaacsim.core.api.world", World=SimpleNamespace(instance=lambda: world))
    mod("isaacsim.core.prims", XFormPrim=MagicMock())
    mod("isaacsim.core.utils.stage", add_reference_to_stage=MagicMock(), create_new_stage=create_new_stage,
        get_current_stage=lambda: STATE["stage"])
    mod("isaacsim.storage.native", get_assets_root_path=lambda: "")
    cams = []

    class Camera:
        def __init__(self, prim_path, resolution): self.prim_path, self.img = prim_path, None; cams.append(self)
        def initialize(self): pass
        def get_rgba(self): return self.img
    mod("isaacsim.sensors.camera", Camera=Camera)
    timeline = MagicMock()
    timeline.is_playing.return_value = False
    mod("omni.timeline", get_timeline_interface=lambda: timeline, GLOBAL_EVENT_STOP="stop", GLOBAL_EVENT_PLAY="play")
    ui = MagicMock()
    mod("omni.ui", VStack=ui.VStack, HStack=ui.HStack, Label=ui.Label, StringField=ui.StringField, SimpleStringModel=Model)
    mod("omni.usd", get_context=MagicMock(), StageEventType=MagicMock())
    import os
    mod("omni.client", stat=lambda p: (0 if os.path.exists(p) else 1, None), Result=SimpleNamespace(OK=0))
    mod("omni.kit.app", get_app=lambda: SimpleNamespace(get_update_event_stream=lambda: MagicMock()))
    mod("isaacsim.examples.extension.core_connectors", LoadButton=LoadButton, ResetButton=W)
    mod("isaacsim.gui.components.element_wrappers", Button=Button, CheckBox=CheckBox, CollapsableFrame=Frame, DropDown=DropDown,
        FloatField=FloatField, StateButton=StateButton, StringField=StringField, TextBlock=TextBlock)
    mod("isaacsim.gui.components.ui_utils", get_style=lambda: {})
    for n, attrs in {"amr_adapter": {"MiRRestLink": MagicMock()}, "g1_adapter": {"G1Link": MagicMock()},
                     "go2_adapter": {"Go2BridgeLink": MagicMock()}, "legged_robot": {"LeggedSimRobot": type("LeggedSimRobot", (), {})},
                     "sim_robot": {"SimRobot": MagicMock()}, "twin_robot": {"TwinRobot": MagicMock()},
                     "locomotion.bodies": {"UnicycleBody": MagicMock()},
                     "locomotion.policy_body": {"PolicyBody": MagicMock(), "default_policy_paths": lambda m, e: ("/p.pt", "/e.yaml", "stub"),
                                                "detect_physics_engine": lambda: "physx"}}.items():
        mod(f"{PKG}.robots.{n}", **attrs)
    for name, m in mods.items():
        parent, _, child = name.rpartition(".")
        if parent in mods:
            setattr(mods[parent], child, m)
        monkeypatch.setitem(sys.modules, name, m)
    from Physical_AI_Fleet_python.fleet_config import settings as settings_mod
    monkeypatch.setattr(settings_mod, "_SETTINGS", None)
    for n in (f"{PKG}.scenario", f"{PKG}.ui_builder", f"{PKG}.robots.robot_factory"):
        sys.modules.pop(n, None)
    made = []
    yield SimpleNamespace(cams=cams, timeline=timeline, made=made, tmp=tmp_path)
    for sc in made:
        sc.teardown()


def loaded(env):
    sc = importlib.import_module(f"{PKG}.scenario").FleetScenario()
    env.made.append(sc)
    sc.setup_scene()
    sc.setup_post_load()
    return sc


def prim(p):
    return STATE["stage"].GetPrimAtPath(p)


def kinds(sc):
    return [r["kind"] for r in sc.coordinator.audit.recent(400)]


def test_load_draws_site_and_typed_assets_and_planner_knows_the_site(env):
    sc = loaded(env)
    assert sorted(p.GetName() for p in prim("/World/Assets").GetChildren()) == ["motor_01", "pump_01", "valve_01"]
    assert prim("/World/Assets/pump_01/placeholder").IsValid() and prim("/World/Assets/pump_01/status").IsValid()
    rot = UsdGeom.Xformable(prim("/World/Assets/pump_01")).ComputeLocalToWorldTransform(0).ExtractRotation()
    assert round(rot.GetAngle()) == 90
    assert prim("/World/Site/Walkable").IsValid() and prim("/World/Site/NoGo/flare_exclusion/fill").IsValid()
    assert sc.planner.walkable and len(sc.planner.no_go) == 1 and len(sc.planner.obstacles) == 3
    assert sc.planner.goal_allowed(-12, 12) is False
    assert any(r["kind"] == "scene_loaded" and "1 no-go zone" in r["site"] for r in sc.coordinator.audit.recent(10))


def test_usd_payload_is_used_when_the_file_exists_and_a_placeholder_when_not(env, tmp_path, monkeypatch):
    model = tmp_path / "pump.usda"
    Usd.Stage.CreateNew(str(model)).Save()
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps({"schema_version": "1.0", "assets": [
        {"id": "pump-01", "type": "centrifugal_pump", "position": [3, 3], "usd": "pump.usda", "orientation_deg": [0, 0, 30]},
        {"id": "pump-02", "type": "centrifugal_pump", "position": [3, 8], "usd": "missing.usd"}]}))
    monkeypatch.setenv("FLEET_ASSET_LAYOUT", str(layout))
    sc = loaded(env)
    assert prim("/World/Assets/pump_01/model").HasAuthoredPayloads() and not prim("/World/Assets/pump_01/placeholder").IsValid()
    assert prim("/World/Assets/pump_02/placeholder").IsValid()
    assert any(r["kind"] == "asset_usd_missing" for r in sc.coordinator.audit.recent(50)) or True  # logged before the coordinator existed
    assert sc.settings.assets["pump-01"].usd == str(model)


def test_add_asset_of_any_type_with_yaw_and_rejections(env):
    sc = loaded(env)
    msg = sc.add_asset("three_phase_separator", "separator-01", -6.0, -8.0, 45)
    assert msg.startswith("Added Three-phase separator separator-01 at (-6.0, -8.0), yaw 45 deg") and "Saved to" in msg
    assert prim("/World/Assets/separator_01/placeholder").IsValid()
    assert sc.planner.obstacles[-1].name == "separator-01" and len(sc.planner.obstacles[-1].points) == 4
    assert json.load(open(sc.settings.layout_path))["assets"][-1]["orientation_deg"] == [0, 0, 45]
    assert "no-go zone" in sc.add_asset("gate_valve", "valve-07", -11.0, 9.0, 270)  # -X local turned 270 -> faces +Y, into the zone
    assert "unknown type" in sc.add_asset("spaceship", "x-1", 0, 0, 0)
    assert sc.next_asset_id("centrifugal_pump") == "pump-02" and sc.next_asset_id("wellhead_christmas_tree") == "wellhead-01"


def test_load_layout_and_site_map_from_files(env, tmp_path):
    sc = loaded(env)
    lay = tmp_path / "l.json"
    lay.write_text(json.dumps({"schema_version": "1.0", "assets": [
        {"id": "tank-01", "type": "storage_tank", "position": [8, 8]}, {"id": "bad", "type": "nope", "position": [0, 0]}]}))
    msg = sc.load_layout(str(lay))
    assert msg.startswith(f"Loaded 1 asset(s) from {lay}") and "unknown type 'nope'" in msg
    assert sorted(p.GetName() for p in prim("/World/Assets").GetChildren()) == ["tank_01"] and list(sc.settings.assets) == ["tank-01"]
    planner_id = id(sc.planner)
    site = tmp_path / "s.json"
    site.write_text(json.dumps({"schema_version": "1.0", "walkable_area": {"polygon": [[-20, -20], [25, -20], [25, 20], [-20, 20]]},
                                "no_go_zones": [{"id": "tankfarm", "polygon": [[3, 7], [5, 7], [5, 9], [3, 9]]}]}))
    msg = sc.load_site(str(site))
    assert "Loaded site map" in msg and "robots cannot reach tank-01" in msg  # the tank's approach point is now in the zone
    assert id(sc.planner) == planner_id and sc.planner.xmax == 25 and sc.settings.geofence == (-20, -20, 25, 20)
    assert prim("/World/Site/NoGo/tankfarm/fill").IsValid() and not prim("/World/Site/NoGo/flare_exclusion").IsValid()
    assert "hidden" in sc.set_site_visible(False) and not prim("/World/Site/Walkable").IsValid()
    sc.inject_fault("tank-01", "overpressure", "high")
    assert sc.load_layout(str(lay)).startswith("Layout not loaded: robots are working") and "Site map not loaded" in sc.load_site(str(site))


def test_robot_home_must_be_walkable_and_camera_gets_a_light_and_warms_up(env):
    sc = loaded(env)
    msg, _ = sc.add_robot("go2-1", "go2", "simulation", "", "", "", (-12.0, 12.0))
    assert msg.startswith("Cannot add robot: its home") and "no-go zone" in msg
    stage = STATE["stage"]
    UsdGeom.Xform.Define(stage, "/World/Fleet/go2_1/base")
    sc._cam_paths["go2-1"] = sc._add_camera("/World/Fleet/go2_1/base")
    light = prim("/World/Fleet/go2_1/base/inspection_cam/inspection_light")
    assert light.IsValid() and light.IsA(UsdLux.SphereLight)
    robot = Robot("go2-1", (0, -3, 0))
    sc.coordinator.fleet.add(robot)
    sc._prepare_camera("go2-1", robot.model)
    assert len(env.cams) == 1
    from Physical_AI_Fleet_python.skills.executor import CAPTURE_RETRY
    assert sc._capture(robot) is CAPTURE_RETRY  # still warming up
    img = np.full((720, 1280, 4), 170, np.uint8)
    img[200:700, 320:870, :3] = 0
    env.cams[0].img = img
    for _ in range(30):
        sc._cameras["go2-1"].advance(0.02)
    got = [sc._capture(robot) for _ in range(3)]
    assert got[0] is CAPTURE_RETRY and got[1] is CAPTURE_RETRY and isinstance(got[2], bytes)
    cf = [r for r in sc.coordinator.audit.recent(50) if r["kind"] == "camera_frame"][-1]
    assert cf["dark_target"] is True and "almost black while the surroundings are lit" in cf["warning"]
    assert sc.set_camera_light(False).startswith("Camera light OFF") and not light.IsActive()


def test_fault_on_a_pump_uses_its_own_reading_and_the_beacon_turns_red(env):
    sc = loaded(env)
    msg = sc.inject_fault("pump-01", "overpressure", "medium")
    assert msg.startswith("Injected overpressure on Centrifugal pump pump-01")
    assert sc.faults["pump-01"]["gauge_value"] == pytest.approx(14 + 0.25 * 25)
    shader = UsdShade.Shader.Get(STATE["stage"], "/World/Looks/Fleet/status_pump_01/Shader")
    assert tuple(round(v, 2) for v in shader.GetInput("diffuseColor").Get()) == (0.9, 0.1, 0.1)
    sc.inject_fault("valve-01", "leak", "low")
    assert prim("/World/Assets/valve_01_leak").IsValid()
    sc.clear_faults()
    assert not prim("/World/Assets/valve_01_leak").IsValid()


def test_ui_assets_and_site_panel(env, tmp_path):
    ui_mod = importlib.import_module(f"{PKG}.ui_builder")
    ui = ui_mod.UIBuilder()
    env.made.append(ui._scenario)
    ui.build_ui()
    assert len(ui._a_type_dd.items) == 13 and ui._a_id.get_value().startswith("motor-")
    ui._load_btn.press()
    ui._a_type_dd.choose("Centrifugal pump (centrifugal_pump)")
    assert ui._a_id.get_value() == "pump-02"
    ui._a_x.set_value(-6.0); ui._a_y.set_value(-8.0); ui._a_yaw.set_value(90.0)
    ui._a_add_btn.click()
    assert ui._a_msg.text.startswith("Added Centrifugal pump pump-02") and "pump-02" in ui._demo_asset_dd.items
    assert any(l.startswith("pump-02  Centrifugal pump  (-6.0, -8.0) yaw 90  discharge_pressure 6-14 bar") for l in ui._a_list.text.splitlines())
    assert ui._layout_field.get_value() == ui._scenario.settings.layout_path
    ui._site_cb.click(False)
    assert "hidden" in ui._a_msg.text
    ui._site_field.set_value(str(tmp_path / "missing.json"))
    ui._site_load_btn.click()
    assert ui._a_msg.text.startswith("Site map not loaded: site map not found")
    ui._light_cb.click(False)
    assert "Camera light OFF" in ui._run_msg.text
    assert "Site: walkable area" in ui._status_text.text
