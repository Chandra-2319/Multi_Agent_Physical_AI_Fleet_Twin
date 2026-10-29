#!/usr/bin/env python3
"""Ask the vision model where the pressure gauge is in an image, draw the box, crop, then read the gauge.

Runs outside Isaac Sim (Python 3.10+, Pillow). Key / URL / model: the active VLM in ~/.physical_ai_fleet/ai_profiles.json,
or NVIDIA_API_KEY / --model / --url.

  python3 tools/test_gauge_bbox.py evidence.jpg                    # default format norm1000
  python3 tools/test_gauge_bbox.py evidence.jpg --all-formats      # try norm1000, norm1 and pixels: pick the one whose
                                                                    #   *_box.jpg fits the dial; put it in cosmos.json detect.format
  python3 tools/test_gauge_bbox.py evidence.jpg --think            # thinking on for the box step too
Writes <image>_<format>_box.jpg (green box) and <image>_<format>_crop.jpg (what the model then reads).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Physical_AI_Fleet_python.perception import gauge_crop as gc  # noqa: E402
from Physical_AI_Fleet_python.perception.cosmos_gauge import (BBOX_FORMATS, CosmosError, CosmosGaugeClient,  # noqa: E402
                                                              build_prompt)

DESCRIPTION = ("The image shows a round pressure gauge (DIGITEN) with a white dial, a dark grey needle, an inner black scale "
               "0 to 1.6 MPa and an outer red psi scale. A solid red arc marks the high-pressure end (lower right, about "
               "1.4 to 1.6 MPa). The red psi numbers and the red DIGITEN logo are not the danger zone.")


def profile_vlm() -> dict:
    p = Path(os.environ.get("FLEET_AI_PROFILES", "") or Path.home() / ".physical_ai_fleet" / "ai_profiles.json")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return (d.get("profiles") or {}).get(d.get("active", "nvidia"), {}).get("vlm") or {}
    except (OSError, ValueError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--format", default="norm1000", choices=BBOX_FORMATS)
    ap.add_argument("--all-formats", action="store_true")
    ap.add_argument("--think", action="store_true", help="thinking on for the box step")
    ap.add_argument("--no-read", action="store_true", help="only find the box")
    ap.add_argument("--model")
    ap.add_argument("--url")
    a = ap.parse_args()
    vlm = profile_vlm()
    key = os.environ.get("NVIDIA_API_KEY", "").strip() or vlm.get("api_key", "")
    url = a.url or vlm.get("url") or "https://integrate.api.nvidia.com/v1"
    model = a.model or vlm.get("model") or "nvidia/nemotron-3.5-super-vl-120b-a12b"
    if not key:
        sys.exit("No API key: set NVIDIA_API_KEY or save the VLM key in the AI Models panel.")
    client = CosmosGaugeClient(url, model, key, build_prompt(DESCRIPTION, "MPa", "cosmos" in model.lower()), 8192, 0.0,
                               None if "cosmos" in model.lower() else bool(vlm.get("thinking", True)))
    print(f"model {model} at {url}")
    cfg = gc.crop_cfg({})
    for path in a.images:
        jpeg = Path(path).read_bytes()
        w, h = gc.image_size(jpeg)
        rgb = gc.decode(jpeg)
        print(f"\n=== {path} ({w}x{h})")
        for fmt in (BBOX_FORMATS if a.all_formats else (a.format,)):
            try:
                det, raw, lat = client.detect(jpeg, 60, w, h, fmt, a.think)
            except CosmosError as exc:
                print(f"  [{fmt}] FAILED: {exc}")
                continue
            print(f"  [{fmt}] {lat:.1f}s visible={det.visible} conf={det.confidence:.2f} model box={det.raw_box} "
                  f"-> pixels {det.box_px} {det.reason}")
            if not det.box_px:
                continue
            stem = Path(path).with_suffix("")
            boxed = rgb.copy()
            x0, y0, x1, y1 = (int(round(v)) for v in det.box_px)
            for t in range(3):
                boxed[max(y0 - t, 0), x0:x1] = boxed[min(y1 + t, h - 1), x0:x1] = (0, 255, 0)
                boxed[y0:y1, max(x0 - t, 0)] = boxed[y0:y1, min(x1 + t, w - 1)] = (0, 255, 0)
            Path(f"{stem}_{fmt}_box.jpg").write_bytes(gc.encode(boxed))
            crop, _info = gc.crop_from_box(rgb, det.box_px, cfg)
            if crop is None:
                continue
            cj = gc.encode(crop)
            Path(f"{stem}_{fmt}_crop.jpg").write_bytes(cj)
            print(f"         wrote {stem}_{fmt}_box.jpg and {stem}_{fmt}_crop.jpg")
            if not a.no_read:
                try:
                    v, _r, lat2 = client.ask(cj, 120)
                    print(f"         read {lat2:.1f}s: pressure {v.pressure} conf {v.confidence:.2f} needle {v.needle_value} - {v.reason}")
                except CosmosError as exc:
                    print(f"         read FAILED: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
