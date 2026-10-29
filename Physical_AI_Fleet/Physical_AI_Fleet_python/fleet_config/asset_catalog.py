# SPDX-License-Identifier: Apache-2.0
"""Asset types (asset_types.json) and asset placement (asset_layout.json). Pure Python.

Every asset instance is validated against its type and the site map: known type, unique id, the approach point
(where the robot stands to look at the asset) is walkable and outside every no-go zone, and footprints do not overlap.
Files (v0.5.0): FLEET_ASSET_TYPES / FLEET_ASSET_LAYOUT env vars, else <extension>/inputs/asset_types.json and
asset_layout.json. 'Assets/x.usd' and '/Assets/x.usd' mean <extension>/Assets/x.usd (see paths.resolve_model_path).
SAVE LAYOUT writes inputs/asset_layout.json and keeps the previous version as asset_layout.json.bak.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import bbox, convex_overlap, rotated_rect

TYPES_PATH = Path(__file__).with_name("asset_types.json")
DEFAULT_LAYOUT_PATH = Path(__file__).with_name("asset_layout.json")
ASSET_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,30}$")
SIDES = {"-X": (-1.0, 0.0), "+X": (1.0, 0.0), "-Y": (0.0, -1.0), "+Y": (0.0, 1.0)}
OVERLAP_MARGIN = 0.2
APPROACH_CLEARANCE = 0.3  # the approach point must be this far inside the walkable area / away from no-go zones
READING_KEYS = ("name", "label", "unit", "min", "max", "normal_low", "normal_high")


class AssetError(ValueError):
    pass


@dataclass(frozen=True)
class AssetType:
    key: str
    label: str
    category: str
    usd: str
    usd_scale: float
    footprint: tuple  # (size_x, size_y) metres, local frame
    height: float
    color: tuple
    approach_side: str
    standoff: float
    primary: str
    readings: tuple  # of dicts
    usd_up_axis: str = ""  # only used with usd_auto_fit
    usd_rotate_deg: tuple = (0.0, 0.0, 0.0)  # v0.5.1: applied to the model exactly as given
    usd_auto_fit: bool = False  # v0.5.1: True = read units / up axis from the file (old v0.5.0 behaviour)
    gauge: dict = field(default_factory=dict)  # v0.5.1: {"model": <gauges.json model>, "needle_prim": <path in the USD>}


@dataclass
class AssetCatalog:
    types: dict = field(default_factory=dict)
    path: str = ""

    def labels(self) -> dict:
        return {f"{t.label} ({k})": k for k, t in self.types.items()}


def _num(v, where: str) -> float:
    if isinstance(v, bool):
        raise AssetError(f"{where} must be a number")
    try:
        f = float(v)
    except (TypeError, ValueError) as exc:
        raise AssetError(f"{where} must be a number") from exc
    if not math.isfinite(f):
        raise AssetError(f"{where} must be a finite number")
    return f


def _reading(d: dict, where: str) -> dict:
    if not isinstance(d, dict):
        raise AssetError(f"{where}: each reading is an object")
    missing = [k for k in READING_KEYS if k not in d and k != "label"]
    if missing:
        raise AssetError(f"{where}: missing {missing}")
    r = {"name": str(d["name"]), "label": str(d.get("label", d["name"])), "unit": str(d["unit"])[:16]}
    for k in ("min", "max", "normal_low", "normal_high"):
        r[k] = _num(d[k], f"{where}.{k}")
    if not r["min"] < r["max"]:
        raise AssetError(f"{where}: min must be below max")
    if not (r["min"] <= r["normal_low"] < r["normal_high"] <= r["max"]):
        raise AssetError(f"{where}: needs min <= normal_low < normal_high <= max")
    return r


def _gauge(v, where: str) -> dict:
    if v in (None, {}):
        return {}
    if not isinstance(v, dict):
        raise AssetError(f"{where} must be an object like {{\"model\": ..., \"needle_prim\": ...}}")
    out = {k: v[k] for k in ("model", "needle_prim", "enabled", "camera_distance_m") if k in v}
    if "needle_prim" in out and not isinstance(out["needle_prim"], str):
        raise AssetError(f"{where}.needle_prim must be a text path")
    return out


def parse_catalog(raw: dict, path: str = "<memory>") -> AssetCatalog:
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0":
        raise AssetError(f"{path}: schema_version must be '1.0'")
    types = {}
    for key, d in (raw.get("types") or {}).items():
        w = f"{path}: types.{key}"
        if not isinstance(d, dict):
            raise AssetError(f"{w} must be an object")
        fp = d.get("footprint_m")
        if not (isinstance(fp, list) and len(fp) == 2):
            raise AssetError(f"{w}.footprint_m must be [size_x, size_y]")
        sx, sy = _num(fp[0], w + ".footprint_m"), _num(fp[1], w + ".footprint_m")
        h = _num(d.get("height_m", 1.0), w + ".height_m")
        if min(sx, sy, h) <= 0:
            raise AssetError(f"{w}: footprint and height must be positive")
        ap = d.get("approach") or {}
        side = ap.get("side", "-X")
        if side not in SIDES:
            raise AssetError(f"{w}.approach.side must be one of {list(SIDES)}")
        readings = tuple(_reading(r, f"{w}.readings[{i}]") for i, r in enumerate(d.get("readings") or []))
        if not readings:
            raise AssetError(f"{w}: at least one reading is required")
        primary = d.get("primary", readings[0]["name"])
        if primary not in [r["name"] for r in readings]:
            raise AssetError(f"{w}.primary '{primary}' is not one of its readings")
        color = tuple(float(c) for c in (d.get("color") or (0.6, 0.6, 0.6)))[:3]
        types[key] = AssetType(key, str(d.get("label", key)), str(d.get("category", "")), str(d.get("usd", "") or ""),
                               _num(d.get("usd_scale", 1.0), w + ".usd_scale"), (sx, sy), h, color, side,
                               _num(ap.get("standoff_m", 1.1), w + ".approach.standoff_m"), primary, readings,
                               str(d.get("usd_up_axis", "") or "").upper(),
                               _triple(d.get("usd_rotate_deg"), (0.0, 0.0, 0.0), w + ".usd_rotate_deg"),
                               bool(d.get("usd_auto_fit", False)), _gauge(d.get("gauge"), w + ".gauge"))
    if not types:
        raise AssetError(f"{path}: no asset types defined")
    return AssetCatalog(types, path)


def load_catalog(path=None) -> AssetCatalog:
    from .paths import input_file, snapshot
    p = Path(path) if path else input_file("asset_types.json")
    snapshot(p)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssetError(f"asset types not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise AssetError(f"{p}: invalid JSON at line {exc.lineno}: {exc.msg}") from exc
    return parse_catalog(raw, str(p))


def user_layout_path() -> Path:
    return Path.home() / ".physical_ai_fleet" / "asset_layout.json"


def layout_path() -> Path:
    """v0.5.4: inputs/floor_layout.json (assets + robots); FLEET_ASSET_LAYOUT overrides."""
    from .paths import input_file
    return input_file("floor_layout.json")


ROBOT_KEYS = ("id", "model", "mode", "position", "yaw_deg", "walking", "connection", "map_to_world", "policy_path", "env_yaml_path",
              "usd")  # v0.5.6: usd = e.g. the twin USD (with its OmniGraph) of a mirrored real robot


def parse_robot_entry(e: dict, models: dict) -> dict:
    """One floor-layout robot -> a clean dict. Raises AssetError with a message fit for the UI."""
    if not isinstance(e, dict):
        raise AssetError("each robot is an object")
    rid = str(e.get("id", "")).strip().replace(" ", "_")
    if not ASSET_ID_RE.match(rid):
        raise AssetError(f"robot id '{rid}': start with a letter, then letters, digits, - or _")
    model = str(e.get("model", "")).strip()
    if model not in models:
        raise AssetError(f"robot '{rid}': unknown model '{model}' (known: {', '.join(sorted(models))})")
    mode = str(e.get("mode", "simulation"))
    if mode not in ("simulation", "digital_twin"):
        raise AssetError(f"robot '{rid}': mode must be simulation or digital_twin")
    pos = e.get("position")
    if not (isinstance(pos, list) and len(pos) >= 2):
        raise AssetError(f"robot '{rid}': position must be [x, y]")
    out = {"id": rid, "model": model, "mode": mode, "position": [_num(pos[0], f"{rid}.position"), _num(pos[1], f"{rid}.position")],
           "yaw_deg": _num(e.get("yaw_deg", 0.0), f"{rid}.yaw_deg"), "walking": bool(e.get("walking", True))}
    for k in ("connection", "map_to_world", "policy_path", "env_yaml_path", "usd"):
        if e.get(k):
            out[k] = str(e[k])
    if mode == "digital_twin" and not out.get("connection"):
        raise AssetError(f"robot '{rid}': a digital_twin robot needs a connection")
    return out


def load_robots(path=None, models=None) -> tuple:
    """(robot entries, problems) from the floor layout. A file without 'robots' (old asset layout) has none."""
    if models is None:
        from .catalog import MODELS as models
    p = Path(path) if path else layout_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], []
    robots, problems, seen = [], [], set()
    for i, e in enumerate(raw.get("robots") or [] if isinstance(raw, dict) else []):
        try:
            r = parse_robot_entry(e, models)
            if r["id"] in seen:
                raise AssetError(f"robot id '{r['id']}' is used twice")
            seen.add(r["id"])
            robots.append(r)
        except AssetError as exc:
            problems.append(f"robots[{i}]: {exc}")
    return robots, problems


def resolve_usd(usd: str, base: str) -> str:
    """'Assets/...' / '/Assets/...' -> <extension>/Assets/...; other relative paths are relative to the JSON file."""
    from .paths import resolve_model_path
    return resolve_model_path(usd, base)


def _triple(v, default, where) -> tuple:
    if v is None:
        return default
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return (float(v),) * 3
    if isinstance(v, list) and len(v) == 3:
        return tuple(_num(x, where) for x in v)
    raise AssetError(f"{where} must be a number or [x, y, z]")


def build_instance(cls, catalog: AssetCatalog, entry: dict, site, existing: dict, layout_file: str = ""):
    """One layout entry -> AssetConfig (cls). Raises AssetError with a message fit for the UI."""
    if not isinstance(entry, dict):
        raise AssetError("each asset is an object")
    aid = str(entry.get("id", "")).strip()
    if not ASSET_ID_RE.match(aid):
        raise AssetError(f"asset id '{aid}': start with a letter, then letters, digits, - or _ (max 31 characters)")
    if aid in existing or any(k.replace("-", "_") == aid.replace("-", "_") for k in existing):
        raise AssetError(f"asset '{aid}' already exists (names 'a-1' and 'a_1' are the same stage prim)")
    tkey = entry.get("type")
    t = catalog.types.get(tkey)
    if t is None:
        raise AssetError(f"asset '{aid}': unknown type '{tkey}' (known: {', '.join(sorted(catalog.types))})")
    pos = entry.get("position")
    if not (isinstance(pos, list) and len(pos) in (2, 3)):
        raise AssetError(f"asset '{aid}': position must be [x, y] or [x, y, z]")
    x, y = _num(pos[0], f"{aid}.position"), _num(pos[1], f"{aid}.position")
    z = _num(pos[2], f"{aid}.position") if len(pos) == 3 else 0.0
    if "orientation_deg" in entry:
        orient = _triple(entry["orientation_deg"], (0.0, 0.0, 0.0), f"{aid}.orientation_deg")
    else:
        orient = (0.0, 0.0, _num(entry.get("yaw_deg", 0.0), f"{aid}.yaw_deg"))
    scale = _triple(entry.get("scale"), (1.0, 1.0, 1.0), f"{aid}.scale")
    if min(scale) <= 0:
        raise AssetError(f"asset '{aid}': scale must be positive")
    side = entry.get("approach_side", t.approach_side)
    if side not in SIDES:
        raise AssetError(f"asset '{aid}': approach_side must be one of {list(SIDES)}")
    readings = {r["name"]: dict(r) for r in t.readings}
    for name, over in (entry.get("readings") or {}).items():
        if name not in readings or not isinstance(over, dict):
            raise AssetError(f"asset '{aid}': reading override '{name}' is not a reading of {tkey}")
        readings[name] = _reading({**readings[name], **over}, f"{aid}.readings.{name}")
    primary = entry.get("primary", t.primary)
    if primary not in readings:
        raise AssetError(f"asset '{aid}': primary '{primary}' is not a reading of {tkey}")
    p = readings[primary]
    yaw = math.radians(orient[2])
    hx, hy = t.footprint[0] * scale[0] / 2, t.footprint[1] * scale[1] / 2
    footprint = rotated_rect(x, y, hx, hy, yaw)
    lx, ly = SIDES[side]
    reach = (hx if lx else hy) + t.standoff
    c, s = math.cos(yaw), math.sin(yaw)
    ax, ay = x + (c * lx - s * ly) * reach, y + (s * lx + c * ly) * reach
    if site is not None and not site.is_walkable(ax, ay, APPROACH_CLEARANCE):
        why = site.why_not(ax, ay) or f"({ax:.1f}, {ay:.1f}) is within {APPROACH_CLEARANCE} m of a boundary or no-go zone"
        raise AssetError(f"asset '{aid}': robots cannot reach its {side} side - the approach point {why}. "
                         f"Move or turn the asset, or set approach_side.")
    for other in existing.values():
        of = getattr(other, "footprint", None) or rotated_rect(other.position[0], other.position[1], *other.half_extents, 0.0)
        if convex_overlap(footprint, of, OVERLAP_MARGIN):
            raise AssetError(f"asset '{aid}': overlaps asset '{other.asset_id}'; move it at least {OVERLAP_MARGIN} m clear")
    xmin, ymin, xmax, ymax = bbox(footprint)
    usd = resolve_usd(entry["usd"], layout_file) if entry.get("usd") else resolve_usd(t.usd, catalog.path)
    usd_scale = _num(entry.get("usd_scale", t.usd_scale), f"{aid}.usd_scale")
    if usd_scale <= 0:
        raise AssetError(f"asset '{aid}': usd_scale must be positive")
    keep = {k: v for k, v in entry.items() if not str(k).startswith("_")}
    return cls(asset_id=aid, position=(x, y, z), approach_pose=(ax, ay, math.atan2(y - ay, x - ax)),
               half_extents=((xmax - xmin) / 2, (ymax - ymin) / 2), gauge_min=p["min"], gauge_max=p["max"],
               normal_low=p["normal_low"], normal_high=p["normal_high"], unit=p["unit"], asset_type=tkey,
               label=t.label, orientation_deg=orient, scale=scale, usd=usd, usd_scale=usd_scale, height=t.height * scale[2],
               color=t.color, footprint=tuple(footprint), local_half=(hx, hy), approach_side=side, primary=primary,
               readings=tuple(readings.values()), layout_entry=keep, usd_up_axis=t.usd_up_axis,
               usd_rotate_deg=_triple(entry.get("usd_rotate_deg"), t.usd_rotate_deg, f"{aid}.usd_rotate_deg"),
               usd_auto_fit=bool(entry.get("usd_auto_fit", t.usd_auto_fit)),
               gauge={**t.gauge, **_gauge(entry.get("gauge"), f"{aid}.gauge")})


def load_layout(cls, catalog: AssetCatalog, site, path=None) -> tuple:
    """(assets dict, problems list, path). Invalid entries are skipped and reported, valid ones are kept."""
    p = Path(path) if path else layout_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssetError(f"asset layout not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise AssetError(f"{p}: invalid JSON at line {exc.lineno}: {exc.msg}") from exc
    from .paths import snapshot
    snapshot(p)
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0" or not isinstance(raw.get("assets"), list):
        raise AssetError(f"{p}: needs schema_version '1.0' and an 'assets' list")
    assets, problems = {}, []
    for i, e in enumerate(raw["assets"]):
        try:
            a = build_instance(cls, catalog, e, site, assets, str(p))
            assets[a.asset_id] = a
        except AssetError as exc:
            problems.append(f"assets[{i}]: {exc}")
    return assets, problems, str(p)


def layout_entry(a) -> dict:
    e = dict(a.layout_entry or {})
    e.update({"id": a.asset_id, "type": a.asset_type, "position": [round(v, 4) for v in a.position],
              "orientation_deg": [round(v, 3) for v in a.orientation_deg]})
    if tuple(a.scale) != (1.0, 1.0, 1.0):
        e["scale"] = list(a.scale)
    e.pop("yaw_deg", None)
    return e


def save_layout(assets: dict, path=None, robots=None) -> str:
    """Write the floor layout. robots=None keeps the robots already in the file."""
    p = Path(path) if path else layout_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if robots is None:
        try:
            robots = json.loads(p.read_text(encoding="utf-8")).get("robots") or []
        except (OSError, ValueError, AttributeError):
            robots = []
    if p.is_file():
        try:
            import shutil
            shutil.copy2(p, p.with_name(p.name + ".bak"))
        except OSError:
            pass
    data = {"schema_version": "1.0", "assets": [layout_entry(a) for a in assets.values()], "robots": list(robots)}
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".layout_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(p)


def migrate_old_assets(old_path: Path) -> list:
    """v0.4.2 ~/.physical_ai_fleet/assets.json (motors only) -> layout entries (type electric_motor)."""
    try:
        raw = json.loads(Path(old_path).read_text(encoding="utf-8"))
        out = []
        for d in raw["assets"]:
            out.append({"id": d["asset_id"], "type": "electric_motor", "position": [d["position"][0], d["position"][1]],
                        "orientation_deg": [0, 0, 0],
                        "readings": {"lube_oil_pressure": {"min": d["gauge_min"], "max": d["gauge_max"],
                                                           "normal_low": d["normal_low"], "normal_high": d["normal_high"],
                                                           "unit": d["unit"]}}})
        return out
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return []
