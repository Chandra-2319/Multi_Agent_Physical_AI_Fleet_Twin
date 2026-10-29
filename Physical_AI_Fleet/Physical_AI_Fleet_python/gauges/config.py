# SPDX-License-Identifier: Apache-2.0
"""gauges.json (needle calibration per gauge model) and cosmos.json (how the robot reads the gauge). Pure Python.

v0.5.1: the gauge is part of the asset USD. asset_types.json says, per type, {"gauge": {"model": ..., "needle_prim": ...}};
gauges.json only describes each gauge MODEL (needle angles, scale, red zone, description for the vision model).
"""
from __future__ import annotations

import json

from ..fleet_config.paths import input_file, snapshot

GAUGE_REQUIRED = ("scale_min", "scale_max", "angle_at_min_deg", "angle_at_max_deg", "red_from", "normal_value", "high_value")


class GaugeConfigError(ValueError):
    pass


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items() if not str(k).startswith("_")}
    return d


def _load(name: str) -> tuple:
    p = input_file(name)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GaugeConfigError(f"{p} not found") from exc
    except json.JSONDecodeError as exc:
        raise GaugeConfigError(f"{p}: invalid JSON at line {exc.lineno}: {exc.msg}") from exc
    snapshot(p)
    return _clean(data), str(p)


def load_gauges() -> dict:
    cfg, path = _load("gauges.json")
    for name, m in (cfg.get("models") or {}).items():
        missing = [k for k in GAUGE_REQUIRED if k not in m]
        if missing:
            raise GaugeConfigError(f"{path}: models.{name} is missing {missing}")
        if not float(m["scale_min"]) < float(m["scale_max"]):
            raise GaugeConfigError(f"{path}: models.{name}: scale_min must be below scale_max")
    cfg["_path"] = path
    return cfg


def gauge_for(cfg: dict, asset) -> tuple:
    """(model name, model spec, needle_prim, camera_distance_m) for an asset, or None when it has no gauge."""
    g = dict(getattr(asset, "gauge", None) or {})
    if not g or g.get("enabled", True) is False:
        return None
    models = cfg.get("models") or {}
    name = g.get("model") or next(iter(models), "")
    if name not in models:
        raise GaugeConfigError(f"{asset.asset_id}: gauge model '{name}' is not in gauges.json")
    dist = float(g.get("camera_distance_m", cfg.get("camera_distance_m", 0.7)))
    return name, models[name], str(g.get("needle_prim", "") or ""), dist


def load_cosmos() -> dict:
    cfg, path = _load("cosmos.json")
    cfg["_path"] = path
    return cfg
