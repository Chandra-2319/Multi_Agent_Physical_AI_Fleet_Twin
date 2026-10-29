"""v0.5.3: the gauge's box comes from the vision model itself (works without positions, e.g. on a real quadruped)."""
import importlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.perception import cosmos_gauge as cg
from Physical_AI_Fleet_python.perception import gauge_crop as gc
from test_v052_crop import draw_dial

W, H = 1280, 720


def frame_with_dial(cu=622, cv=262, r=60):
    img = np.full((H, W, 3), 170, np.uint8)
    draw_dial(img, cu, cv, r, -40)
    return img, (cu - r, cv - r, cu + r, cv + r)


# ------------------------------------------------------------------ parsing the model's box
@pytest.mark.parametrize("fmt,box,expect", [
    ("norm1000", [440, 280, 535, 450], (563.2, 201.6, 684.8, 324.0)),
    ("norm1", [0.44, 0.28, 0.535, 0.45], (563.2, 201.6, 684.8, 324.0)),
    ("norm1000", [0.44, 0.28, 0.535, 0.45], (563.2, 201.6, 684.8, 324.0)),  # fractions are recognised whatever was asked
    ("pixels", [563, 202, 685, 324], (563.0, 202.0, 685.0, 324.0)),
])
def test_box_formats(fmt, box, expect):
    text = "<think>looking</think>" + json.dumps({"gauge_visible": True, "bbox": box, "confidence": 0.9, "adjust": "none"})
    d = cg.parse_detection(text, W, H, fmt)
    assert d.visible and d.box_px == pytest.approx(expect, abs=0.2) and d.reason == ""


@pytest.mark.parametrize("obj,reason", [
    ({"gauge_visible": False, "bbox": None, "confidence": 0.8, "adjust": "turn_left"}, "sees no gauge"),
    ({"gauge_visible": True, "bbox": None}, "no box"),
    ({"gauge_visible": True, "bbox": [1, 2, 3]}, "4 numbers"),
    ({"gauge_visible": True, "bbox": [100, 100, 900, 120]}, "not the shape"),
    ({"gauge_visible": True, "bbox": [0, 0, 1000, 1000]}, "whole image"),
    ({"gauge_visible": True, "bbox": [500, 500, 502, 503]}, "empty"),
])
def test_unusable_answers_are_rejected_with_a_reason(obj, reason):
    d = cg.parse_detection(json.dumps(obj), W, H)
    assert d.box_px is None and reason in d.reason
    if obj.get("adjust") == "turn_left":
        assert d.adjust == "turn_left" and not d.visible


def test_box_is_clipped_reordered_and_dict_boxes_accepted():
    d = cg.parse_detection(json.dumps({"gauge_visible": "yes", "bbox": [535, 450, 440, 280], "confidence": "0.8"}), W, H)
    assert d.box_px == pytest.approx((563.2, 201.6, 684.8, 324.0), abs=0.2) and d.confidence == 0.8
    d2 = cg.parse_detection(json.dumps({"gauge_visible": True, "bbox": {"xmin": 440, "ymin": 280, "xmax": 535, "ymax": 450}}), W, H)
    assert d2.box_px == pytest.approx((563.2, 201.6, 684.8, 324.0), abs=0.2)
    with pytest.raises(cg.CosmosError):
        cg.parse_detection("I see a gauge", W, H)


def test_detect_prompt_and_request(monkeypatch):
    assert "0-1000" in cg.build_detect_prompt(W, H) and "1280 x 720" in cg.build_detect_prompt(W, H, "pixels")
    seen = {}

    class Resp:
        def __init__(self, p): self.p = p
        def read(self): return json.dumps(self.p).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def urlopen(req, timeout):
        seen.update(body=json.loads(req.data.decode()), timeout=timeout)
        return Resp({"choices": [{"message": {"content": '{"gauge_visible": true, "bbox": [440, 280, 535, 450], "confidence": 0.9}'}}]})
    monkeypatch.setattr(cg.urllib.request, "urlopen", urlopen)
    c = cg.CosmosGaugeClient("https://integrate.api.nvidia.com/v1", "nvidia/nemotron-3.5-super-vl-120b-a12b", "k", "READ", 8192, 0.0, True)
    det, raw, lat = c.detect(b"\xff\xd8x", 30, W, H)
    assert det.box_px is not None and seen["timeout"] == 30
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}  # the box step is a quick call
    text = seen["body"]["messages"][1]["content"][0]["text"]
    assert "bounding box" in text and "READ" not in text


def test_crop_from_a_detected_box_centres_the_dial():
    img, box = frame_with_dial()
    crop, info = gc.crop_from_box(img, box, gc.crop_cfg({}))
    assert crop.shape == (768, 768, 3) and info["dial_box"] == [562.0, 202.0, 682.0, 322.0]
    c = crop.astype(float).mean(axis=2)
    assert c[384 - 20:384 + 20, 384 - 20:384 + 20].mean() > 100 and c[5:20, 5:20].mean() == pytest.approx(170, abs=3)
    assert gc.iou(box, box) == 1.0 and gc.iou(box, (0, 0, 10, 10)) == 0.0 and gc.iou(None, box) == 0.0


# ------------------------------------------------------------------ the session: find box -> crop -> read
def run_session(capture, detect, sources=("vlm", "projection"), detect_conf=0.5):
    from Physical_AI_Fleet_python.skills.gauge_inspection import GaugeSession
    asked, saved, recs = [], [], []
    pool = ThreadPoolExecutor(2)

    def ask(jpeg, timeout):
        asked.append(jpeg)
        return cg.Verdict("high", "none", 0.9, "red sector", 1.5), "raw", 9.0
    robot = SimpleNamespace(robot_id="spot-1", pose=[0, 0, 0], status="idle", goal=None, stop=lambda: None)
    s = GaugeSession(robot=robot, asset_id="motor-02", purpose="read_gauge",
                     cfg={"total_time_s": 120, "call_timeout_s": 45, "confidence_threshold": 0.75,
                          "crop": {"source": list(sources)}, "detect": {"timeout_s": 30, "min_confidence": detect_conf}},
                     ask=ask, capture=capture, detect=detect,
                     save_evidence=lambda j, a, suffix="": saved.append(suffix) or f"/e/a{a}{suffix}.jpg",
                     record=lambda k, p: recs.append((k, p)), submit=pool.submit, clock=time.time)
    for _ in range(400):
        res = s.step()
        if res is not None:
            return res, asked, saved, dict(recs)
        time.sleep(0.003)
    raise AssertionError("session did not finish")


def det_returning(box, visible=True, conf=0.9):
    calls = []

    def detect(jpeg, timeout, w, h):
        calls.append((len(jpeg), timeout, w, h))
        return cg.Detection(visible, box, conf, "none", "" if box else "no box"), "raw", 2.0
    detect.calls = calls
    return detect


def test_real_robot_frame_is_cropped_from_the_models_box():
    img, box = frame_with_dial()
    full = gc.encode(img)
    detect = det_returning(box)
    res, asked, saved, recs = run_session(lambda r: full, detect)  # a real robot: just JPEG bytes, no positions
    assert res["verdict"] == "high" and detect.calls[0][1:] == (30.0, W, H)
    assert asked[0] != full and gc.image_size(asked[0]) == (768, 768) and saved == ["", "_crop"]
    assert recs["gauge_crop"]["source"] == "vlm" and recs["gauge_detect"]["found"] and recs["gauge_detect"]["iou_with_projection"] is None


def test_in_the_simulation_both_boxes_are_compared():
    img, box = frame_with_dial()
    full = gc.encode(img)
    cap = {"full": full, "crop": b"PROJ", "info": {"cropped": True, "centre_px": [622, 262], "radius_px": 60}}
    res, asked, _s, recs = run_session(lambda r: cap, det_returning(box))
    assert recs["gauge_crop"]["source"] == "vlm" and recs["gauge_detect"]["iou_with_projection"] == 1.0


@pytest.mark.parametrize("detect,expect_src,expect_img", [
    (det_returning(None, visible=False), "projection", b"PROJ"),          # model: no gauge -> the projection crop
    (det_returning((562, 202, 682, 322), conf=0.2), "projection", b"PROJ"),  # model unsure -> the projection crop
])
def test_falls_back_to_the_projection_then_the_full_frame(detect, expect_src, expect_img):
    img, _ = frame_with_dial()
    full = gc.encode(img)
    cap = {"full": full, "crop": b"PROJ", "info": {"cropped": True}}
    res, asked, _s, recs = run_session(lambda r: cap, detect)
    assert recs["gauge_crop"]["source"] == expect_src and asked[0] == expect_img and res["verdict"] == "high"
    _r, asked2, saved2, recs2 = run_session(lambda r: full, det_returning(None, visible=False))  # real robot, nothing found
    assert asked2[0] == full and recs2["gauge_crop"]["source"] == "full" and saved2 == [""]
    assert "did not find the gauge" in recs2["gauge_crop"]["reason"]


def test_a_failing_box_call_never_loses_the_reading():
    img, _ = frame_with_dial()
    full = gc.encode(img)

    def boom(*a):
        raise cg.CosmosError("HTTP 500")
    res, asked, _s, recs = run_session(lambda r: full, boom)
    assert asked[0] == full and "HTTP 500" in recs["gauge_detect"]["reason"] and res["verdict"] == "high"


def test_projection_first_skips_the_box_call():
    detect = det_returning((562, 202, 682, 322))
    cap = {"full": gc.encode(frame_with_dial()[0]), "crop": b"PROJ", "info": {"cropped": True}}
    _r, asked, _s, recs = run_session(lambda r: cap, detect, sources=("projection", "vlm"))
    assert detect.calls == [] and asked[0] == b"PROJ" and recs["gauge_crop"]["source"] == "projection"


def test_shipped_config():
    from Physical_AI_Fleet_python.fleet_config import paths
    c = json.load(open(paths.SHIPPED_DIR / "cosmos.json"))
    assert c["crop"]["source"] == ["vlm", "projection"] and c["detect"] == {"format": "norm1000", "timeout_s": 45,
                                                                          "min_confidence": 0.5}
    assert gc.detect_cfg({"detect": {"thinking": False}})["thinking"] is True  # cannot be switched off




pytest.importorskip("pxr")
from test_scenario_v044 import env  # noqa: E402,F401


def test_vlm_test_writes_box_and_crop(env, tmp_path):
    sc = importlib.import_module("Physical_AI_Fleet_python.scenario")
    img, box = frame_with_dial()
    jpeg = gc.encode(img)
    asked = []
    client = SimpleNamespace(detect=lambda j, t, w, h, f, th: (cg.Detection(True, box, 0.9), "r", 2.0),
                             ask=lambda j, t: asked.append(j) or (cg.Verdict("high", "none", 0.95, "red", 1.5), "r", 9.0))
    msg = sc.vlm_test(client, jpeg, {}, tmp_path, "High_gauge", 120)
    assert msg.startswith("OK: gauge found") and "pressure high" in msg and gc.image_size(asked[0]) == (768, 768)
    assert (tmp_path / "High_gauge_vlmtest_box.jpg").is_file() and (tmp_path / "High_gauge_vlmtest_crop.jpg").is_file()
    client.detect = lambda *a: (cg.Detection(False, None, 0.9, "turn_left", "the model sees no gauge"), "r", 2.0)
    msg = sc.vlm_test(client, jpeg, {}, tmp_path, "x", 120)
    assert "gauge NOT found" in msg and asked[-1] == jpeg


def test_inspector_gives_the_session_the_box_call_with_the_configured_format():
    from Physical_AI_Fleet_python.skills.gauge_inspection import GaugeInspector
    calls = []
    client = SimpleNamespace(model="m", base_url="u", ask=lambda j, t: None,
                             detect=lambda j, t, w, h, fmt, th: calls.append((fmt, th)) or (cg.Detection(False, None), "r", 1.0))
    insp = GaugeInspector(cfg={"detect": {"format": "pixels", "thinking": True}}, specs={"m-1": {}}, mounts={},
                          client_for=lambda aid: client, truth_for=lambda aid: None, evidence_store=None,
                          audit=SimpleNamespace(record=lambda *a: None), pool=SimpleNamespace(submit=None))
    run = SimpleNamespace(assignment=SimpleNamespace(target_asset_id="m-1"), purpose="read_gauge",
                          robot=SimpleNamespace(robot_id="spot-1"))
    s = insp.start(run, lambda r: None)
    assert s.detect is not None
    s.detect(b"x", 30, W, H)
    assert calls == [("pixels", True)]
