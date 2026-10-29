# SPDX-License-Identifier: Apache-2.0
"""Monitored assets that are added / removed from the UI, validated and kept between sessions (unchanged in v0.4.3).

Saved to ~/.physical_ai_fleet/assets.json (override: FLEET_ASSETS_FILE). If the file does not exist the
defaults in settings.py are used. Pure Python: no Isaac Sim imports.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path

ASSET_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,30}$")
SIDES = {"-X": (-1.0, 0.0), "+X": (1.0, 0.0), "-Y": (0.0, -1.0), "+Y": (0.0, 1.0)}
FIELDS = ("asset_id", "position", "approach_pose", "half_extents", "gauge_min", "gauge_max", "normal_low", "normal_high", "unit")
MARKER_HEIGHT = 0.8
OVERLAP_MARGIN = 0.2


class AssetError(ValueError):
    pass


def assets_path() -> Path:
    env = os.environ.get("FLEET_ASSETS_FILE", "").strip()
    return Path(env) if env else Path.home() / ".physical_ai_fleet" / "assets.json"


def prim_name(asset_id: str) -> str:
    return asset_id.replace("-", "_")


def describe_area(geofence: tuple) -> str:
    xmin, ymin, xmax, ymax = geofence
    return f"X {xmin:g} to {xmax:g} m, Y {ymin:g} to {ymax:g} m"


def to_dict(a) -> dict:
    return {"asset_id": a.asset_id, "position": list(a.position), "approach_pose": list(a.approach_pose),
            "half_extents": list(a.half_extents), "gauge_min": a.gauge_min, "gauge_max": a.gauge_max,
            "normal_low": a.normal_low, "normal_high": a.normal_high, "unit": a.unit}


def from_dict(cls, d: dict):
    missing = [k for k in FIELDS if k not in d]
    if missing:
        raise AssetError(f"asset entry missing {missing}")
    return cls(asset_id=str(d["asset_id"]), position=tuple(float(v) for v in d["position"]),
               approach_pose=tuple(float(v) for v in d["approach_pose"]), half_extents=tuple(float(v) for v in d["half_extents"]),
               gauge_min=float(d["gauge_min"]), gauge_max=float(d["gauge_max"]), normal_low=float(d["normal_low"]),
               normal_high=float(d["normal_high"]), unit=str(d["unit"]))


def load_assets(cls):
    """dict asset_id -> AssetConfig, or None when there is no (valid) saved file."""
    try:
        raw = json.loads(assets_path().read_text(encoding="utf-8"))
        items = raw["assets"]
        out = {}
        for d in items:
            a = from_dict(cls, d)
            out[a.asset_id] = a
        return out or None
    except (OSError, ValueError, KeyError, TypeError, AssetError):
        return None


def save_assets(assets: dict) -> str:
    """Atomic write. Returns the path; raises OSError."""
    path = assets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".assets_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"schema_version": "1.0", "assets": [to_dict(a) for a in assets.values()]}, f, indent=1)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(path)


def _num(v, name: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError) as exc:
        raise AssetError(f"{name} must be a number") from exc
    if not math.isfinite(f):
        raise AssetError(f"{name} must be a finite number")
    return f


def build_asset(cls, asset_id: str, x, y, size_x, size_y, side: str, gauge_min, gauge_max, normal_low, normal_high,
                unit: str, geofence: tuple, existing: dict, standoff: float = 1.1, z: float = MARKER_HEIGHT / 2):
    """Validate the UI input and return an AssetConfig. Raises AssetError with a message fit for the UI."""
    aid = (asset_id or "").strip()
    if not ASSET_ID_RE.match(aid):
        raise AssetError("Asset ID: start with a letter, then letters, digits, - or _ (max 31 characters)")
    if aid in existing or any(prim_name(k) == prim_name(aid) for k in existing):
        raise AssetError(f"Asset '{aid}' already exists (names 'a-1' and 'a_1' are the same stage prim)")
    x, y = _num(x, "X"), _num(y, "Y")
    sx, sy = _num(size_x, "Size X"), _num(size_y, "Size Y")
    if not (0.1 <= sx <= 10 and 0.1 <= sy <= 10):
        raise AssetError("Size X / Size Y must be between 0.1 and 10 m")
    if side not in SIDES:
        raise AssetError(f"Approach side must be one of {list(SIDES)}")
    gmin, gmax = _num(gauge_min, "Gauge min"), _num(gauge_max, "Gauge max")
    lo, hi = _num(normal_low, "Normal low"), _num(normal_high, "Normal high")
    if not gmin < gmax:
        raise AssetError("Gauge min must be below Gauge max")
    if not (gmin <= lo < hi <= gmax):
        raise AssetError("Normal band must satisfy gauge min <= normal low < normal high <= gauge max")
    unit = (unit or "").strip()
    if not unit or len(unit) > 16:
        raise AssetError("Unit is required (max 16 characters), e.g. bar")
    hx, hy = sx / 2, sy / 2
    dx, dy = SIDES[side]
    reach = (hx if dx else hy) + standoff
    ax, ay = x + dx * reach, y + dy * reach
    xmin, ymin, xmax, ymax = geofence
    for label, px, py in (("asset", x, y), ("approach point", ax, ay)):
        if not (xmin <= px <= xmax and ymin <= py <= ymax):
            raise AssetError(f"The {label} at ({px:.1f}, {py:.1f}) is outside the allowed work area "
                             f"({describe_area(geofence)}). Move it, pick another approach side, or widen "
                             f"Settings.geofence in fleet_config/settings.py")
    for other in existing.values():
        ox, oy = other.position[0], other.position[1]
        if abs(ox - x) < hx + other.half_extents[0] + OVERLAP_MARGIN and abs(oy - y) < hy + other.half_extents[1] + OVERLAP_MARGIN:
            raise AssetError(f"Overlaps asset '{other.asset_id}' at ({ox:.1f}, {oy:.1f}); move it at least {OVERLAP_MARGIN} m clear")
    yaw = math.atan2(y - ay, x - ax)
    return cls(asset_id=aid, position=(x, y, z), approach_pose=(ax, ay, yaw), half_extents=(hx, hy),
               gauge_min=gmin, gauge_max=gmax, normal_low=lo, normal_high=hi, unit=unit)
