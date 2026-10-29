# SPDX-License-Identifier: Apache-2.0
"""Pure maths for the gauges: needle angle from a value, where a gauge sits on an asset's inspected face, the rotation
that turns the gauge model to face the robot, and how far the robot stands to look at it. No USD imports."""
from __future__ import annotations

import math

import numpy as np

AXES = {"+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0), "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
        "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0)}
CAMERA_FORWARD_M = 0.35  # the inspection camera sits this far in front of the robot's base (scenario._add_camera)


def value_to_angle(value: float, spec: dict) -> float:
    """Needle rotation (deg): linear from angle_at_min_deg at scale_min to angle_at_max_deg at scale_max, clamped."""
    lo, hi = float(spec["scale_min"]), float(spec["scale_max"])
    a0, a1 = float(spec["angle_at_min_deg"]), float(spec["angle_at_max_deg"])
    v = min(max(float(value), lo), hi)
    return a0 + (v - lo) / (hi - lo) * (a1 - a0)


def zone_of(value: float, spec: dict) -> str:
    return "high" if float(value) >= float(spec["red_from"]) else "normal"


def _ray_exit(cx, cy, nx, ny, polygon):
    best = None
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        ex, ey = x2 - x1, y2 - y1
        den = nx * ey - ny * ex
        if abs(den) < 1e-12:
            continue
        t = ((x1 - cx) * ey - (y1 - cy) * ex) / den
        u = ((x1 - cx) * ny - (y1 - cy) * nx) / den
        if t >= 0 and -1e-9 <= u <= 1 + 1e-9:
            best = t if best is None else max(best, t)
    return best


def inspected_face(asset) -> tuple:
    """(cx, cy, nx, ny, d): asset centre, unit normal of the inspected face (towards the approach point), distance from
    the centre to that face (from the exact footprint polygon if there is one, else half_extents)."""
    cx, cy = float(asset.position[0]), float(asset.position[1])
    nx, ny = float(asset.approach_pose[0]) - cx, float(asset.approach_pose[1]) - cy
    n = math.hypot(nx, ny)
    if n < 1e-9:
        raise ValueError(f"asset {getattr(asset, 'asset_id', '?')}: approach point equals the asset centre")
    nx, ny = nx / n, ny / n
    fp = getattr(asset, "footprint", None)
    d = _ray_exit(cx, cy, nx, ny, fp) if fp else None
    if d is None:
        hx, hy = getattr(asset, "half_extents", (0.4, 0.25))
        d = abs(nx) * float(hx) + abs(ny) * float(hy)
    return cx, cy, nx, ny, d


def mount_position(asset, mount: dict) -> tuple:
    """((x, y, z), (nx, ny)): the gauge on the inspected face. lateral_m > 0 = to the robot's right as it faces it."""
    cx, cy, nx, ny, d = inspected_face(asset)
    out, lat = float(mount.get("out_m", 0.03)), float(mount.get("lateral_m", 0.0))
    rx, ry = -ny, nx  # the robot looks along -n; its right-hand side
    return (cx + nx * (d + out) + rx * lat, cy + ny * (d + out) + ry * lat, float(mount.get("height_m", 0.5))), (nx, ny)


def inspection_distance(asset, mount: dict) -> float:
    """Distance from the asset CENTRE at which the robot's base stands so its camera is camera_distance_m from the gauge."""
    _cx, _cy, _nx, _ny, d = inspected_face(asset)
    return d + float(mount.get("out_m", 0.03)) + float(mount.get("camera_distance_m", 0.7)) + CAMERA_FORWARD_M


def facing_rotation(front_axis: str, up_axis: str, normal_xy: tuple) -> np.ndarray:
    """3x3 R (v_world = R @ v_model): the model's front_axis points along the face normal, its up_axis to world +Z."""
    f, u = np.array(AXES[front_axis]), np.array(AXES[up_axis])
    if abs(float(np.dot(f, u))) > 1e-9:
        raise ValueError(f"front_axis {front_axis} and up_axis {up_axis} must be perpendicular")
    F = np.array([normal_xy[0], normal_xy[1], 0.0])
    F /= np.linalg.norm(F)
    U = np.array([0.0, 0.0, 1.0])
    return np.column_stack([F, U, np.cross(F, U)]) @ np.column_stack([f, u, np.cross(f, u)]).T


def inspect_pose(asset, gauge_xy: tuple, camera_distance: float, robot_radius: float, clearance: float = 0.15) -> tuple:
    """(x, y, yaw) for the robot base: in front of the gauge on the inspected side, camera `camera_distance` from the dial,
    facing the gauge; never closer to the asset centre than face + robot radius + clearance (v0.5.1)."""
    cx, cy, nx, ny, d = inspected_face(asset)
    gx, gy = float(gauge_xy[0]), float(gauge_xy[1])
    rx, ry = -ny, nx
    along = (gx - cx) * nx + (gy - cy) * ny
    lateral = (gx - cx) * rx + (gy - cy) * ry
    dist = max(along + camera_distance + CAMERA_FORWARD_M, d + robot_radius + clearance)
    x, y = cx + nx * dist + rx * lateral, cy + ny * dist + ry * lateral
    return x, y, math.atan2(-ny, -nx)
