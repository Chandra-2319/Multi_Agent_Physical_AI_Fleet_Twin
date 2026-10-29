"""v0.5.1: AI provider profiles, the VLM request, the gauge inside the asset USD, models loaded as authored, UI wiring."""
import importlib
import json
import math
import os
import stat
import sys
from types import SimpleNamespace

import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.fleet_config import ai_profiles as ap
from Physical_AI_Fleet_python.perception import cosmos_gauge as cg

NV = "nvapi-abcdefgh12345678"
NB = "nebius-key-9876543210"


# ------------------------------------------------------------------ profiles
def test_defaults_without_a_file_and_both_providers_present():
    d = ap.load_file()
    assert d["active"] == "nvidia" and set(d["profiles"]) >= {"nvidia", "nebius"}
    assert d["profiles"]["nvidia"]["vlm"]["model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert d["profiles"]["nebius"]["brain"]["url"] == "https://api.tokenfactory.nebius.com/v1" and d["profiles"]["nebius"]["brain"]["api_key"] == ""


def test_save_keeps_blank_keys_switches_active_and_is_private():
    p = ap.profiles_path()
    ap.save_profile("nvidia", {"brain": {"url": "https://integrate.api.nvidia.com/v1/", "model": "m1", "api_key": NV},
                               "vlm": {"url": "https://integrate.api.nvidia.com/v1", "model": "v1", "api_key": NV, "thinking": True}})
    ap.save_profile("nebius", {"brain": {"url": "https://api.tokenfactory.nebius.com/v1", "model": "nb", "api_key": NB},
                               "vlm": {"url": "https://api.tokenfactory.nebius.com/v1", "model": "nbv", "api_key": NB}})
    d = ap.load_file()
    assert d["active"] == "nebius" and d["profiles"]["nvidia"]["brain"]["url"] == "https://integrate.api.nvidia.com/v1"
    assert "thinking" not in d["profiles"]["nvidia"]["vlm"]  # v0.5.4: not a profile setting any more
    ap.save_profile("nvidia", {"brain": {"url": "https://x/v1", "model": "m2", "api_key": ""}, "vlm": {"model": "v2"}})
    n = ap.get_profile("nvidia")
    assert n["brain"]["api_key"] == NV and n["vlm"]["api_key"] == NV and n["brain"]["model"] == "m2" and ap.load_file()["active"] == "nvidia"
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    assert ap.forget_profile("nebius") is True and ap.forget_profile("nebius") is False
    assert ap.get_profile("nebius")["brain"]["api_key"] == ""  # back to the defaults


def test_broken_file_never_stops_anything_and_paths(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{nope")
    assert ap.load_file(p)["active"] == "nvidia"
    assert ap.normalize_path(str(tmp_path)) == tmp_path / "ai_profiles.json" and ap.normalize_path(str(p)) == p
    with pytest.raises(ValueError, match="full path"):
        ap.normalize_path("rel/dir")


def test_settings_start_with_the_saved_active_profile():
    from Physical_AI_Fleet_python.fleet_config.settings import Settings
    ap.save_profile("nebius", {"brain": {"url": "https://api.tokenfactory.nebius.com/v1", "model": "nb/brain", "api_key": NB},
                               "vlm": {"url": "https://api.tokenfactory.nebius.com/v1", "model": "nb/vl", "api_key": NB, "thinking": True}})
    s = Settings()
    assert (s.ai_provider, s.nebius_api_key, s.nemotron_model_id, s.vlm_model, s.vlm_api_key, s.vlm_thinking) == \
        ("nebius", NB, "nb/brain", "nb/vl", NB, True)
    assert NB not in s.nemotron_status() and NB not in s.vlm_status() and "thinking always on" in s.vlm_status()
    s.apply_profile("nebius", {"vlm": {"thinking": False}})
    assert s.vlm_thinking is True  # v0.5.4: cannot be switched off
    with pytest.raises(ValueError, match="https://"):
        s.apply_profile("nvidia", {"brain": {"url": "ftp://x"}})
    with pytest.raises(ValueError, match="spaces"):
        s.apply_profile("nvidia", {"vlm": {"model": "a b"}})


# ------------------------------------------------------------------ the VLM request
class Resp:
    def __init__(self, payload): self.p = payload
    def read(self): return json.dumps(self.p).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False


@pytest.mark.parametrize("url,expect", [
    ("https://integrate.api.nvidia.com/v1", "https://integrate.api.nvidia.com/v1/chat/completions"),
    ("https://integrate.api.nvidia.com/v1/", "https://integrate.api.nvidia.com/v1/chat/completions"),
    ("https://api.tokenfactory.nebius.com/v1/", "https://api.tokenfactory.nebius.com/v1/chat/completions"),
    ("https://ai.api.nvidia.com/v1/vlm/nvidia/x", "https://ai.api.nvidia.com/v1/vlm/nvidia/x"),
    ("https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos3-nano-reasoner", "https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos3-nano-reasoner"),
    ("http://127.0.0.1:8000/v1/chat/completions", "http://127.0.0.1:8000/v1/chat/completions"),
])
def test_endpoint_url(url, expect):
    assert cg.endpoint_url(url) == expect


def test_request_matches_the_working_nvidia_sample_and_parses_the_answer(monkeypatch):
    seen = {}

    def urlopen(req, timeout):
        seen.update(url=req.full_url, auth=req.headers["Authorization"], body=json.loads(req.data.decode()), timeout=timeout)
        return Resp({"choices": [{"message": {"content": '{"pressure": "high", "adjust": "none", "confidence": 0.95, '
                                                         '"needle_value": 1.5, "reason": "inside the red arc"}',
                                              "reasoning_content": "thinking..."}}]})
    monkeypatch.setattr(cg.urllib.request, "urlopen", urlopen)
    prompt = cg.build_prompt("A DIGITEN gauge.", "MPa", reasoning=False)
    c = cg.CosmosGaugeClient("https://integrate.api.nvidia.com/v1", "nvidia/nemotron-3.5-super-vl-120b-a12b", NV, prompt,
                             4096, 0.0, thinking=False)
    v, raw, lat = c.ask(b"\x89PNG\r\n\x1a\n....", 55)
    b = seen["body"]
    assert seen["url"].endswith("/v1/chat/completions") and seen["auth"] == f"Bearer {NV}" and seen["timeout"] == 55
    assert b["model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b" and b["temperature"] == 0.0 and b["stream"] is False
    assert b["chat_template_kwargs"] == {"enable_thinking": False}
    parts = b["messages"][1]["content"]
    assert parts[0]["type"] == "text" and "<think>" not in parts[0]["text"] and "What is the pressure?" in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert (v.pressure, v.adjust, v.confidence, v.needle_value) == ("high", "none", 0.95, 1.5)


def test_cosmos_models_keep_the_think_prompt_and_send_no_thinking_switch():
    assert cg.is_cosmos("nvidia/cosmos3-nano-reasoner") and not cg.is_cosmos("nvidia/nemotron-3.5-super-vl-120b-a12b")
    assert "<think>" in cg.build_prompt("g", "MPa", True)
    assert "chat_template_kwargs" not in cg.build_payload("m", b"x", "p", thinking=None)


# ------------------------------------------------------------------ the gauge inside the asset USD
pxr = pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom  # noqa: E402

NEEDLE = "/scene/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/Needle_3"


def make_motor_usd(path, mpu=0.01, up="Y"):
    """A Sketchfab-like export: default prim /scene, cm units, Y up, needle with its own translate + orient ops."""
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(st, mpu)
    UsdGeom.SetStageUpAxis(st, up)
    root = UsdGeom.Xform.Define(st, "/scene")
    st.SetDefaultPrim(root.GetPrim())
    for p in ("/scene/Meshes", "/scene/Meshes/Sketchfab_model", "/scene/Meshes/Sketchfab_model/root",
              "/scene/Meshes/Sketchfab_model/root/GLTF_SceneRootNode"):
        UsdGeom.Xform.Define(st, p)
    UsdGeom.Cube.Define(st, "/scene/Meshes/Body").CreateSizeAttr(80.0)
    n = UsdGeom.Xform.Define(st, NEEDLE)
    n.AddTranslateOp().Set(Gf.Vec3d(-45.0, 60.0, 0.0))
    n.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
    UsdGeom.Cube.Define(st, NEEDLE + "/blade").CreateSizeAttr(4.0)
    st.Save()
    return str(path)


SPEC = {"needle_name": "Needle_3", "needle_axis": "Z", "scale_min": 0.0, "scale_max": 1.6, "unit": "MPa",
        "angle_at_min_deg": 17.0, "angle_at_max_deg": -253.0, "red_from": 1.4, "normal_value": 0.4, "high_value": 1.5}


def asset(**over):
    from Physical_AI_Fleet_python.fleet_config.settings import AssetConfig
    kw = dict(asset_id="motor-01", position=(4.0, 2.0, 0.0), approach_pose=(2.5, 2.0, 0.0), half_extents=(0.4, 0.25),
              local_half=(0.4, 0.25), usd_scale=0.5, usd_rotate_deg=(90.0, 0.0, 0.0), height=0.8)
    kw.update(over)
    return AssetConfig(**kw)


@pytest.fixture
def stage_with_motor(tmp_path):
    from Physical_AI_Fleet_python import stage_assets as sa
    usd = make_motor_usd(tmp_path / "Motor.usda")
    st = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(st, "Z")
    UsdGeom.SetStageMetersPerUnit(st, 1.0)  # Isaac Sim stages are in metres
    UsdGeom.Xform.Define(st, "/World")
    sa.create_asset_prim(st, "/World/Assets/motor_01", asset(), usd)
    return st, usd


def test_model_loads_as_authored_with_only_the_json_rotation_and_scale(stage_with_motor):
    st, usd = stage_with_motor
    ops = {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(st.GetPrimAtPath("/World/Assets/motor_01/model")).GetOrderedXformOps()}
    assert tuple(round(v, 3) for v in ops["xformOp:rotateXYZ"]) == (90.0, 0.0, 0.0)
    assert tuple(round(v, 3) for v in ops["xformOp:scale"]) == (0.5, 0.5, 0.5)  # NOT 0.01 x 0.5: no automatic unit fit
    assert "as given" in st.GetPrimAtPath("/World/Assets/motor_01").GetAttribute("fleet:model_fit").Get()


def test_auto_fit_is_still_available_on_request(tmp_path):
    from Physical_AI_Fleet_python import stage_assets as sa
    usd = make_motor_usd(tmp_path / "m.usda")
    st = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(st, "Z")
    UsdGeom.SetStageMetersPerUnit(st, 1.0)  # Isaac Sim stages are in metres
    sa.create_asset_prim(st, "/World/Assets/m", asset(usd_auto_fit=True, usd_scale=1.0), usd)
    ops = {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(st.GetPrimAtPath("/World/Assets/m/model")).GetOrderedXformOps()}
    assert round(ops["xformOp:scale"][0], 4) == 0.01 and round(ops["xformOp:rotateXYZ"][0]) == 90


@pytest.mark.parametrize("needle_prim", [NEEDLE, NEEDLE.replace("/scene/", ""), "", "Meshes/wrong/Needle_3"])
def test_needle_is_found_in_the_asset_and_turned_about_its_own_pivot(stage_with_motor, needle_prim):
    from Physical_AI_Fleet_python.gauges.gauge_rig import EmbeddedGauge
    st, _ = stage_with_motor
    g = EmbeddedGauge(st, asset(), SPEC, "/World/Assets/motor_01/model", needle_prim).build()
    assert g.needle_path == "/World/Assets/motor_01/model" + NEEDLE.replace("/scene", "")
    xf = UsdGeom.Xformable(st.GetPrimAtPath(g.needle_path))
    names = [op.GetOpName() for op in xf.GetOrderedXformOps()]
    assert names == ["xformOp:translate", "xformOp:orient", "xformOp:rotateZ:fleetNeedle"]  # appended after its own ops
    assert xf.GetOrderedXformOps()[-1].Get() == pytest.approx(-50.5) and g.truth() == "normal"
    before = g.position()
    assert g.set_state("high") == pytest.approx(-236.125) and g.truth() == "high"
    assert g.position() == pytest.approx(before)  # turning the needle does not move its pivot
    g.set_state("normal")
    assert xf.GetOrderedXformOps()[-1].Get() == pytest.approx(-50.5)


def test_needle_world_position_follows_the_asset_transform_and_the_json_scale(stage_with_motor):
    from Physical_AI_Fleet_python.gauges.gauge_rig import EmbeddedGauge
    st, _ = stage_with_motor
    x, y, z = EmbeddedGauge(st, asset(), SPEC, "/World/Assets/motor_01/model", NEEDLE).build().position()
    # file point (-45, 60, 0) in its own units, x0.5 scale, rotate X 90 (Y up -> Z up), then the asset at (4, 2, 0)
    assert (round(x, 3), round(y, 3), round(z, 3)) == (4.0 - 22.5, 2.0, 30.0)


def test_missing_needle_and_missing_model_are_explained(stage_with_motor):
    from Physical_AI_Fleet_python.gauges.gauge_rig import EmbeddedGauge, GaugeError
    st, _ = stage_with_motor
    with pytest.raises(GaugeError, match="needle prim not found"):
        EmbeddedGauge(st, asset(), {**SPEC, "needle_name": "Pointer"}, "/World/Assets/motor_01/model", "").build()
    with pytest.raises(GaugeError, match="no model prim"):
        EmbeddedGauge(st, asset(), SPEC, "/World/Assets/nope/model", "").build()


def test_robot_stands_in_front_of_the_gauge_facing_it():
    from Physical_AI_Fleet_python.gauges.gauge_math import CAMERA_FORWARD_M, inspect_pose
    a = asset()  # inspected face -X at x = 3.6; approach from -X
    x, y, yaw = inspect_pose(a, (3.6, 2.3), 0.7, 0.4)
    assert (round(x, 3), round(y, 3)) == (round(3.6 - 0.7 - CAMERA_FORWARD_M, 3), 2.3) and math.cos(yaw) == pytest.approx(1.0)
    x2, _, _ = inspect_pose(a, (3.6, 2.0), 0.0, 0.4)  # too close: never inside the footprint + radius + clearance
    assert x2 == pytest.approx(4.0 - (0.4 + 0.4 + 0.15))


def test_asset_types_carry_gauge_rotation_and_scale_and_the_layout_can_override():
    from Physical_AI_Fleet_python.fleet_config import asset_catalog as ac
    from Physical_AI_Fleet_python.fleet_config.settings import AssetConfig
    cat = ac.load_catalog()
    t = cat.types["electric_motor"]
    assert t.usd_scale == 0.5 and t.usd_rotate_deg == (90.0, 0.0, 0.0) and t.gauge == {"model": "digiten_1p6mpa", "needle_prim": ""}
    a = ac.build_instance(AssetConfig, cat, {"id": "m-1", "type": "electric_motor", "position": [0, 0], "usd_scale": 0.25,
                                             "usd_rotate_deg": [90, 0, 180], "gauge": {"needle_prim": "X/Needle"}}, None, {})
    assert a.usd_scale == 0.25 and a.usd_rotate_deg == (90.0, 0.0, 180.0) and a.gauge == {"model": "digiten_1p6mpa", "needle_prim": "X/Needle"}
    off = ac.build_instance(AssetConfig, cat, {"id": "m-2", "type": "electric_motor", "position": [5, 5], "gauge": {"enabled": False}}, None, {})
    from Physical_AI_Fleet_python.gauges.config import gauge_for
    assert gauge_for({"models": {"digiten_1p6mpa": SPEC}, "camera_distance_m": 0.7}, off) is None
    assert gauge_for({"models": {"digiten_1p6mpa": SPEC}, "camera_distance_m": 0.7}, a)[2:] == ("X/Needle", 0.7)
    with pytest.raises(ac.AssetError, match="usd_scale"):
        ac.build_instance(AssetConfig, cat, {"id": "m-3", "type": "electric_motor", "position": [0, 0], "usd_scale": 0}, None, {})


# ------------------------------------------------------------------ the scenario and the UI
from test_scenario_v044 import STATE, env  # noqa: E402,F401


def setup_motor_inputs(monkeypatch, tmp_path):
    usd = make_motor_usd(tmp_path / "Motor.usda")
    from Physical_AI_Fleet_python.fleet_config import paths
    types = json.loads(open(paths.SHIPPED_DIR / "asset_types.json").read())
    types["types"]["electric_motor"]["usd"] = usd
    tp = tmp_path / "types.json"
    tp.write_text(json.dumps(types))
    lp = tmp_path / "layout.json"
    lp.write_text(json.dumps({"schema_version": "1.0", "assets": [{"id": "motor-01", "type": "electric_motor", "position": [4, 2]},
                                                                  {"id": "pump-01", "type": "centrifugal_pump", "position": [8, -4]}]}))
    monkeypatch.setenv("FLEET_ASSET_TYPES", str(tp))
    monkeypatch.setenv("FLEET_ASSET_LAYOUT", str(lp))


def test_load_finds_the_gauge_in_the_motor_and_the_flow_moves_the_needle(env, monkeypatch, tmp_path):
    setup_motor_inputs(monkeypatch, tmp_path)
    sc = importlib.import_module("Physical_AI_Fleet_python.scenario").FleetScenario()
    env.made.append(sc)
    sc.setup_scene()
    sc.setup_post_load()
    assert sorted(sc.gauges) == ["motor-01"] and sc.gauge_problems == []  # the pump has no gauge in its type
    assert not STATE["stage"].GetPrimAtPath("/World/Gauges").IsValid()  # no separate gauge model any more
    needle = UsdGeom.Xformable(STATE["stage"].GetPrimAtPath(sc.gauges["motor-01"].needle_path)).GetOrderedXformOps()[-1]
    sc.inject_fault("motor-01", "overpressure", "high")
    assert needle.Get() == pytest.approx(17 + 1.55 / 1.6 * (-253 - 17))  # v0.5.5: high_value 1.55
    sc.clear_faults()
    assert needle.Get() == pytest.approx(-50.5)
    insp = sc.coordinator.executor.gauge_inspector
    assert insp is not None and insp.handles("motor-01", "read_gauge") and not insp.handles("pump-01", "read_gauge")
    gx, gy, _ = sc.gauges["motor-01"].position()
    x, y, yaw = insp.inspect_pose(sc.settings.assets["motor-01"], 0.4)
    assert y == pytest.approx(gy) and x < gx and math.cos(yaw) == pytest.approx(1.0)


def test_vlm_client_uses_the_profile(env, monkeypatch, tmp_path):
    setup_motor_inputs(monkeypatch, tmp_path)
    sc = importlib.import_module("Physical_AI_Fleet_python.scenario").FleetScenario()
    env.made.append(sc)
    sc.setup_scene(); sc.setup_post_load()
    sc.apply_ai("nebius", {"brain": {"url": "https://api.tokenfactory.nebius.com/v1", "model": "nb/b", "api_key": NB},
                           "vlm": {"url": "https://api.tokenfactory.nebius.com/v1", "model": "nb/vl", "api_key": NB, "thinking": True}})
    c = sc._cosmos_client("motor-01")
    assert (c.base_url, c.model, c.api_key, c.thinking) == ("https://api.tokenfactory.nebius.com/v1", "nb/vl", NB, True)
    assert "<think>" not in c.prompt and "DIGITEN" in c.prompt
    assert sc.coordinator.nebius.api_key == NB and sc.coordinator.brain.model_id == "nb/b"
    assert NB not in json.dumps(sc.coordinator.audit.recent(200), default=str)


def test_ui_provider_dropdown_and_load_save_forget(env, tmp_path):
    ui_mod = importlib.import_module("Physical_AI_Fleet_python.ui_builder")
    ui = ui_mod.UIBuilder()
    env.made.append(ui._scenario)
    ui.build_ui()
    assert ui._provider_dd.items[0] == "NVIDIA" and ui._vlm_model.get_value() == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    pf = str(tmp_path / "keys")
    ui._profile_field.set_value(pf)
    ui._brain_key.set_value(NV); ui._vlm_key.set_value(NV)
    ui._ai_save_btn.click()
    target = tmp_path / "keys" / "ai_profiles.json"
    assert ui._brain_msg.text.startswith(f"Saved the nvidia profile") and str(target) in ui._brain_msg.text and target.is_file()
    assert NV not in ui._brain_msg.text and json.load(open(target))["profiles"]["nvidia"]["brain"]["api_key"] == NV
    ui._provider_dd.choose("Nebius")
    assert ui._brain_url.get_value() == "https://api.tokenfactory.nebius.com/v1" and ui._brain_key.get_value_as_string() == ""
    ui._brain_model.set_value("nb/brain"); ui._brain_key.set_value(NB); ui._vlm_model.set_value("nb/vl"); ui._vlm_key.set_value(NB)
    ui._ai_save_btn.click()
    ui._provider_dd.choose("NVIDIA")
    assert ui._brain_key.get_value_as_string() == NV  # LOAD-style fill of all six fields, keys included (masked in the UI)
    ui._ai_load_btn.click()
    assert ui._brain_msg.text.startswith("Loaded the nvidia profile") and ui._scenario.settings.ai_provider == "nvidia"
    ui._provider_dd.choose("Nebius"); ui._ai_load_btn.click()
    assert ui._scenario.settings.nebius_api_key == NB and ui._scenario.settings.vlm_model == "nb/vl"
    assert "Provider: nebius" in ui._brain_msg.text and NB not in ui._brain_msg.text
    ui._ai_forget_btn.click()
    assert ui._brain_msg.text.startswith("Deleted the nebius profile") and "nebius" not in json.load(open(target))["profiles"]
    ui._cosmos_test_btn.click()
    assert "Choose an image of the gauge first" in ui._brain_msg.text
