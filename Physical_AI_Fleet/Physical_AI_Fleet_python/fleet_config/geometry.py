# SPDX-License-Identifier: Apache-2.0
"""2-D polygon helpers for the site map, asset footprints and the planner. Pure Python + numpy."""
from __future__ import annotations

import math

import numpy as np


def as_polygon(points, where: str = "polygon") -> list:
    """[[x, y], ...] -> [(x, y), ...]; at least 3 finite points, closing point optional. Raises ValueError."""
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        raise ValueError(f"{where}: needs at least 3 points [[x, y], ...]")
    out = []
    for i, p in enumerate(points):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"{where}[{i}]: each point is [x, y]")
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}[{i}]: x and y must be numbers") from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"{where}[{i}]: x and y must be finite")
        out.append((x, y))
    if out[0] == out[-1]:
        out = out[:-1]
    if len(out) < 3:
        raise ValueError(f"{where}: needs at least 3 distinct points")
    if self_intersects(out):
        raise ValueError(f"{where}: the outline crosses itself; list the corners in order around the edge")
    if abs(signed_area(out)) < 1e-9:
        raise ValueError(f"{where}: the points do not enclose an area")
    return out


def signed_area(poly) -> float:
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                     for i in range(len(poly)))


def bbox(poly) -> tuple:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_cross(a, b, c, d) -> bool:
    """True if segments ab and cd properly intersect (shared end points do not count)."""
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def self_intersects(poly) -> bool:
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            if segments_cross(a, b, poly[j], poly[(j + 1) % n]):
                return True
    return False


def point_in_polygon(x: float, y: float, poly) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def points_in_polygon(xs: np.ndarray, ys: np.ndarray, poly) -> np.ndarray:
    """Vectorised even-odd test for arrays of points (same shape)."""
    inside = np.zeros(xs.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        crosses = (y1 > ys) != (y2 > ys)
        xint = (x2 - x1) * (ys - y1) / (y2 - y1) + x1
        inside ^= crosses & (xs < xint)
    return inside


def distance_to_polygon_edge(x: float, y: float, poly) -> float:
    best = math.inf
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
        best = min(best, math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
    return best


def rotated_rect(cx: float, cy: float, hx: float, hy: float, yaw: float) -> list:
    """Corners (counter-clockwise) of a hx x hy half-size rectangle rotated by yaw (radians)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + c * px - s * py, cy + s * px + c * py) for px, py in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]


def convex_overlap(a, b, margin: float = 0.0) -> bool:
    """Separating-axis test for two convex polygons; `margin` grows both slightly."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            ax, ay = -(y2 - y1), x2 - x1
            L = math.hypot(ax, ay) or 1.0
            ax, ay = ax / L, ay / L
            pa = [px * ax + py * ay for px, py in a]
            pb = [px * ax + py * ay for px, py in b]
            if max(pa) + margin < min(pb) or max(pb) + margin < min(pa):
                return False
    return True


def triangulate(poly) -> list:
    """Ear clipping for a simple polygon -> list of (i, j, k) index triples (counter-clockwise)."""
    idx = list(range(len(poly)))
    if signed_area(poly) < 0:
        idx.reverse()
    tris, guard = [], 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if _orient(a, b, c) <= 1e-12:
                continue
            if any(point_in_triangle(poly[m], a, b, c) for m in idx if m not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break  # degenerate: give up with what we have
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def point_in_triangle(p, a, b, c) -> bool:
    d1, d2, d3 = _orient(a, b, p), _orient(b, c, p), _orient(c, a, p)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))
