"""v0.5.2: the frame sent to the vision model is cropped to the gauge."""
import importlib
import io
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.perception import gauge_crop as gc

W, H = 1280, 720


def draw_dial(img, cu, cv, r, needle_deg):
    """A white disc with a black rim and a dark needle (angle in image degrees, 0 = right, 90 = up)."""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    d = np.hypot(xx - cu, yy - cv)
    img[d <= r] = (240, 240, 240)
    img[(d > r * 0.9) & (d <= r)] = (30, 30, 30)
    a = math.radians(needle_deg)
    for t in np.linspace(0, 0.85 * r, int(r) + 1):
        x, y = int(round(cu + t * math.cos(a))), int(round(cv - t * math.sin(a)))
        img[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = (20, 20, 20)
    return img


def test_projection_basics():
    pose = gc.camera_pose_from_robot(0.0, 0.0, 0.0, 0.6)
    assert np.allclose(pose[0], [0.35, 0.0, 0.75])
    fx = gc.fx_pixels(W)
    uv, depth = gc.project([[1.35, 0.0, 0.75], [1.35, 0.1, 0.75], [1.35, 0.0, 0.85], [-1, 0, 0.75]], pose, W, H, fx)
    assert np.allclose(uv[0], [640, 360]) and depth[0] == pytest.approx(1.0)
    assert uv[1][0] < 640 and uv[2][1] < 360 and np.isnan(uv[3]).all()  # +Y is left of a robot facing +X; up is up
    assert fx == pytest.approx(18 / 20.955 * 1280)
    c, r, d = gc.locate([1.35, 0.0, 0.75], 0.05, pose, W, H, fx)
    assert r == pytest.approx(fx * 0.05 / 1.0, rel=1e-6) and d == pytest.approx(1.0)
    yaw = math.radians(90)  # facing +Y: a point 1 m ahead in +Y is in the centre
    p2 = gc.camera_pose_from_robot(2.0, 3.0, yaw, 0.6)
    uv2, _ = gc.project([[2.0, 4.35, 0.75]], p2, W, H, fx)
    assert np.allclose(uv2[0], [640, 360], atol=1e-6)


def test_crop_box_is_square_shifted_into_the_frame_and_none_outside():
    assert gc.crop_box((640, 360), 50, W, H, 1.8, 96) == (550, 270, 730, 450)
    assert gc.crop_box((10, 10), 50, W, H, 1.8, 96) == (0, 0, 180, 180)  # shifted, not shrunk
    b = gc.crop_box((640, 360), 10, W, H, 1.8, 96)
    assert b[2] - b[0] == 96 == b[3] - b[1]  # never smaller than min_px
    assert gc.crop_box((640, 360), 900, W, H, 1.8, 96) == (280, 0, 1000, 720)  # never bigger than the frame
    assert gc.crop_box((-5, 300), 50, W, H, 1.8, 96) is None and gc.crop_box((float("nan"), 3), 5, W, H, 1.8, 96) is None


def test_the_real_situation_dial_is_2_percent_of_the_frame_and_fills_the_crop_afterwards():
    """Spot-like pose 1.2 m from a 6 cm-radius dial, needle in the red sector: the model gets a close-up."""
    fx = gc.fx_pixels(W)
    pose = gc.camera_pose_from_robot(0.0, 0.0, 0.0, 0.62)
    centre = np.array([1.55, 0.08, 0.95])  # a little left and above the camera axis
    uv, r_px, depth = gc.locate(centre, 0.06, pose, W, H, fx)
    img = np.full((H, W, 3), 170, np.uint8)
    draw_dial(img, uv[0], uv[1], r_px, needle_deg=-40)
    assert (math.pi * r_px ** 2) / (W * H) < 0.03  # tiny in the full frame
    crop, info = gc.crop_gauge(img, uv, r_px, gc.crop_cfg({}))
    assert info["cropped"] and crop.shape == (768, 768, 3) and info["dial_fraction"] == pytest.approx(1 / 1.8, abs=0.02)
    c = crop.astype(float).mean(axis=2)
    yy, xx = np.mgrid[0:768, 0:768]
    disc = np.hypot(xx - 384, yy - 384) < 0.8 * 384 / 1.8
    assert c[disc].mean() > 200 and (c[disc] < 60).any()  # dial centred in the crop, needle visible on it


def test_crop_reports_why_it_did_not_crop():
    img = np.zeros((H, W, 3), np.uint8)
    crop, info = gc.crop_gauge(img, None, 10, gc.crop_cfg({}))
    assert crop is None and "could not be projected" in info["reason"]
    crop, info = gc.crop_gauge(img, (2000, 300), 10, gc.crop_cfg({"crop": {"margin": 2.5}}))
    assert crop is None and "outside the camera image" in info["reason"]
    assert gc.crop_cfg({"crop": {"margin": 2.5}})["margin"] == 2.5 and gc.crop_cfg(None)["output_px"] == 768


def test_resize_works_without_pil(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_pil(name, *a, **k):
        if name.startswith("PIL"):
            raise ImportError
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_pil)
    out = gc.resize(np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3), 8)
    assert out.shape == (8, 8, 3)


# ------------------------------------------------------------------ the session sends the crop and keeps both images
def test_session_sends_the_crop_and_saves_full_frame_and_crop():
    from concurrent.futures import ThreadPoolExecutor
    from Physical_AI_Fleet_python.perception.cosmos_gauge import Verdict
    from Physical_AI_Fleet_python.skills.gauge_inspection import GaugeSession
    asked, saved, recs = [], [], []
    pool = ThreadPoolExecutor(1)

    def ask(jpeg, timeout):
        asked.append((jpeg, timeout))
        return Verdict("high", "none", 0.9, "needle in the red sector", 1.5), "raw", 9.0
    robot = SimpleNamespace(robot_id="spot-1", pose=[0, 0, 0], status="idle", goal=None, stop=lambda: None)
    s = GaugeSession(robot=robot, asset_id="motor-02", purpose="read_gauge",
                     cfg={"total_time_s": 120, "call_timeout_s": 45, "confidence_threshold": 0.75},
                     ask=ask, capture=lambda r: {"full": b"FULL", "crop": b"CROP", "info": {"cropped": True, "method": "camera"}},
                     save_evidence=lambda j, a, suffix="": saved.append((j, a, suffix)) or f"/e/x_a{a}{suffix}.jpg",
                     record=lambda k, p: recs.append((k, p)), submit=pool.submit, clock=lambda: 0.0)
    res = None
    for _ in range(200):
        res = s.step()
        if res is not None:
            break
        import time
        time.sleep(0.005)
    assert res["verdict"] == "high" and asked[0] == (b"CROP", 45.0)
    assert saved == [(b"FULL", 1, ""), (b"CROP", 1, "_crop")] and res["evidence"] == ["/e/x_a1.jpg", "/e/x_a1_crop.jpg"]
    assert ("gauge_crop", {"asset_id": "motor-02", "robot_id": "spot-1", "attempt": 1, "source": "projection", "cropped": True,
                           "method": "camera"}) in recs
    no_crop = GaugeSession(robot=robot, asset_id="motor-02", purpose="read_gauge", cfg={"total_time_s": 120}, ask=ask,
                           capture=lambda r: {"full": b"FULL", "crop": None, "info": {"cropped": False, "reason": "outside"}},
                           save_evidence=lambda j, a, suffix="": "p", record=lambda k, p: None, submit=pool.submit, clock=lambda: 0.0)
    no_crop.step()
    import time
    time.sleep(0.05)
    assert asked[-1][0] == b"FULL"  # nothing lost: the full frame goes when the gauge cannot be found


def test_prompt_judges_by_needle_direction():
    from Physical_AI_Fleet_python.perception.cosmos_gauge import build_prompt
    p = build_prompt("g", "MPa", False)
    assert "DIRECTION the needle points" in p and "even if the needle tip stops short of the arc" in p


def test_shipped_timing_and_crop_settings():
    from Physical_AI_Fleet_python.fleet_config import paths
    c = json.load(open(paths.SHIPPED_DIR / "cosmos.json"))
    assert (c["call_timeout_s"], c["total_time_s"], c["max_tokens"]) == (45, 150, 8192)
    # v0.5.4: every attempt = box call + reading call, both thinking (~8-15 s each)
    assert c["total_time_s"] >= (1 + c["max_adjustments"]) * 2 * 15 + c["max_adjustments"] * (c["settle_s"] + 5) - 30
    assert c["crop"]["enabled"] and c["crop"]["output_px"] == 768
    from Physical_AI_Fleet_python.fleet_config import ai_profiles
    assert gc.THINKING is True and "thinking" not in ai_profiles.DEFAULTS["nvidia"]["vlm"]


# ------------------------------------------------------------------ the scenario wiring (real USD stage)
pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdGeom  # noqa: E402
from test_scenario_v044 import STATE, env  # noqa: E402,F401
from test_v051 import NEEDLE, make_motor_usd, setup_motor_inputs  # noqa: E402


def test_scenario_crops_the_camera_frame_to_the_gauge(env, monkeypatch, tmp_path):
    setup_motor_inputs(monkeypatch, tmp_path)
    sc = importlib.import_module("Physical_AI_Fleet_python.scenario").FleetScenario()
    env.made.append(sc)
    sc.setup_scene()
    sc.setup_post_load()
    rig = sc.gauges["motor-01"]
    assert 0.02 <= rig.dial_radius() <= 1.0  # estimated from the needle's size
    centre = np.array(rig.position())
    insp = sc.coordinator.executor.gauge_inspector
    x, y, yaw = insp.inspect_pose(sc.settings.assets["motor-01"], 0.6)
    from Physical_AI_Fleet_python.fleet_config.catalog import MODELS
    robot = SimpleNamespace(robot_id="spot-1", mode="simulation", pose=[x, y, yaw], model=MODELS["spot"],
                            body=SimpleNamespace(_z=0.6))
    pose = gc.camera_pose_from_robot(x, y, yaw, 0.6)
    uv, r_px, _ = gc.locate(centre, rig.dial_radius(), pose, W, H, gc.fx_pixels(W))
    img = np.full((H, W, 4), 170, np.uint8)
    if 0 <= uv[0] < W and 0 <= uv[1] < H:
        draw_dial(img[..., :3], uv[0], uv[1], max(r_px, 4), -40)

    class Cam:  # no native projection: the robot-pose fallback is used
        def grab_rgb(self): return img[..., :3]
        def project(self, pts): return None
    sc._cam_paths["spot-1"] = "/World/Fleet/spot_1/body/inspection_cam"
    sc._cameras["spot-1"] = Cam()
    out = sc._capture_gauge(robot, "motor-01")
    assert set(out) == {"full", "crop", "info"} and out["full"][:2] == b"\xff\xd8"
    info = out["info"]
    if 0 <= uv[0] < W and 0 <= uv[1] < H:
        assert info["cropped"] and info["method"] == "robot_pose" and out["crop"][:2] == b"\xff\xd8"
    else:
        assert not info["cropped"] and out["crop"] is None
    # with Isaac Sim's own projection the camera result is preferred
    sc._cameras["spot-1"] = SimpleNamespace(grab_rgb=lambda: img[..., :3],
                                            project=lambda pts: np.array([[640.0, 360.0], [640.0, 300.0]]))
    out = sc._capture_gauge(robot, "motor-01")
    assert out["info"]["method"] == "camera" and out["info"]["centre_px"] == [640.0, 360.0] and out["info"]["radius_px"] == 60.0
    assert out["info"]["box"] == [532, 252, 748, 468]
    sc.cosmos_cfg = {**sc.cosmos_cfg, "crop": {"enabled": False}}
    assert sc._capture_gauge(robot, "motor-01")["crop"] is None
    assert sc.coordinator.executor.gauge_inspector.capture_gauge == sc._capture_gauge
