# SPDX-License-Identifier: Apache-2.0
"""Walkable area (geofence polygon) and no-go zones from site_map.json. Pure Python.

The planner only walks inside `walkable_area` and never enters a no-go zone; the safety gate refuses targets outside.
File: FLEET_SITE_MAP env var, else <extension>/inputs/site_map.json.
v0.5.0: a no-go zone id used twice no longer rejects the whole file (that silently left the site with NO zones):
the second one is renamed <id>-2 and a warning is listed in SiteMap.warnings. The user-home copy is no longer read.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import as_polygon, bbox, distance_to_polygon_edge, point_in_polygon

DEFAULT_PATH = Path(__file__).with_name("site_map.json")
DEFAULT_RECT = [(-15.0, -15.0), (15.0, -15.0), (15.0, 15.0), (-15.0, 15.0)]


class SiteMapError(ValueError):
    pass


@dataclass(frozen=True)
class NoGoZone:
    zone_id: str
    label: str
    polygon: tuple


@dataclass
class SiteMap:
    walkable: list = field(default_factory=lambda: list(DEFAULT_RECT))
    no_go: list = field(default_factory=list)  # [NoGoZone]
    show: bool = True
    path: str = "<default>"
    warnings: list = field(default_factory=list)

    @property
    def bounds(self) -> tuple:
        return bbox(self.walkable)

    def zone_at(self, x: float, y: float):
        return next((z for z in self.no_go if point_in_polygon(x, y, z.polygon)), None)

    def is_walkable(self, x: float, y: float, clearance: float = 0.0) -> bool:
        if not point_in_polygon(x, y, self.walkable) or self.zone_at(x, y) is not None:
            return False
        if clearance > 0:
            if distance_to_polygon_edge(x, y, self.walkable) < clearance:
                return False
            if any(distance_to_polygon_edge(x, y, z.polygon) < clearance for z in self.no_go):
                return False
        return True

    def why_not(self, x: float, y: float) -> str:
        """'' if walkable, else a human-readable reason."""
        if not point_in_polygon(x, y, self.walkable):
            return f"({x:.1f}, {y:.1f}) is outside the walkable area"
        z = self.zone_at(x, y)
        if z is not None:
            return f"({x:.1f}, {y:.1f}) is inside no-go zone '{z.label or z.zone_id}'"
        return ""

    def describe(self) -> str:
        xmin, ymin, xmax, ymax = self.bounds
        return (f"walkable area: {len(self.walkable)}-point polygon within X {xmin:g}..{xmax:g} m, Y {ymin:g}..{ymax:g} m; "
                f"{len(self.no_go)} no-go zone(s)")


def site_map_path() -> Path:
    from .paths import input_file
    return input_file("site_map.json")


def parse_site_map(raw: dict, path: str = "<memory>") -> SiteMap:
    if not isinstance(raw, dict):
        raise SiteMapError(f"{path}: top level must be an object")
    raw = {k: v for k, v in raw.items() if not str(k).startswith("_")}
    if raw.get("schema_version") != "1.0":
        raise SiteMapError(f"{path}: schema_version must be '1.0'")
    wa = raw.get("walkable_area")
    if not isinstance(wa, dict) or "polygon" not in wa:
        raise SiteMapError(f"{path}: walkable_area.polygon is required")
    try:
        walkable = as_polygon(wa["polygon"], "walkable_area.polygon")
    except ValueError as exc:
        raise SiteMapError(f"{path}: {exc}") from exc
    zones, seen, warnings = [], set(), []
    for i, z in enumerate(raw.get("no_go_zones", []) or []):
        if not isinstance(z, dict):
            raise SiteMapError(f"{path}: no_go_zones[{i}] must be an object")
        zid = str(z.get("id") or f"zone-{i + 1}")
        if zid in seen:
            n = 2
            while f"{zid}-{n}" in seen:
                n += 1
            warnings.append(f"no-go zone id '{zid}' is used twice: the zone '{z.get('label', '')}' was loaded as "
                            f"'{zid}-{n}' - give every zone its own id")
            zid = f"{zid}-{n}"
        seen.add(zid)
        try:
            poly = as_polygon(z.get("polygon"), f"no_go_zones[{zid}].polygon")
        except ValueError as exc:
            raise SiteMapError(f"{path}: {exc}") from exc
        if not any(point_in_polygon(x, y, walkable) for x, y in poly):
            raise SiteMapError(f"{path}: no-go zone '{zid}' lies completely outside the walkable area")
        zones.append(NoGoZone(zid, str(z.get("label", "")), tuple(poly)))
    show = (raw.get("display") or {}).get("show", True)
    return SiteMap(walkable, zones, bool(show), path, warnings)


def load_site_map(path=None) -> SiteMap:
    p = Path(path) if path else site_map_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SiteMapError(f"site map not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise SiteMapError(f"{p}: invalid JSON at line {exc.lineno}: {exc.msg}") from exc
    from .paths import snapshot
    snapshot(p)
    return parse_site_map(raw, str(p))
