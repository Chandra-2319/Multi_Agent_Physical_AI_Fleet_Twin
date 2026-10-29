# SPDX-License-Identifier: Apache-2.0
"""2-D occupancy-grid motion planner (A* + line-of-sight smoothing) for walking robots.
Static obstacles (monitored assets + obstacles extracted from the environment USD) are rasterised
once into an occupancy grid, then dilated per robot footprint radius (cached), so planning stays
fast even with thousands of environment prims. Dynamic obstacles (other robots) are checked
exactly. Goals inside an inflated obstacle (an inspection standoff next to a motor) are reached by
planning to the closest free cell and walking the last short segment straight.

v0.4.4: the site map. Cells outside the walkable polygon or inside a no-go zone are blocked like obstacles (and dilated
by the robot radius), so paths stay inside the walkable area with clearance. Rotated asset footprints are rasterised
exactly (PolygonObstacle). A start cell that is blocked (robot placed on an edge) is still left via the nearest free cell,
but a GOAL outside the walkable area or inside a no-go zone is refused.
"""
from __future__ import annotations
import heapq
import math
from dataclasses import dataclass
import numpy as np
from ...fleet_config.geometry import distance_to_polygon_edge, point_in_polygon, points_in_polygon


@dataclass(frozen=True)
class RectObstacle:
    cx: float
    cy: float
    hx: float
    hy: float
    name: str = ""

    def inflated_contains(self, x: float, y: float, r: float) -> bool:
        return abs(x - self.cx) <= self.hx + r and abs(y - self.cy) <= self.hy + r


@dataclass(frozen=True)
class PolygonObstacle:
    """Any simple polygon (e.g. a rotated asset footprint): ((x, y), ...)."""
    points: tuple
    name: str = ""

    def inflated_contains(self, x: float, y: float, r: float) -> bool:
        return point_in_polygon(x, y, self.points) or distance_to_polygon_edge(x, y, self.points) <= r


def _or_shifted(out: np.ndarray, src: np.ndarray, di: int, dj: int) -> None:
    nx, ny = src.shape
    out[max(0, di):nx - max(0, -di), max(0, dj):ny - max(0, -dj)] |= \
        src[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj)]


class GridPlanner:
    def __init__(self, bounds: tuple, resolution: float = 0.2, margin: float = 0.15) -> None:
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.res = resolution
        self.margin = margin
        self.nx = int(math.ceil((self.xmax - self.xmin) / resolution)) + 1
        self.ny = int(math.ceil((self.ymax - self.ymin) / resolution)) + 1
        self.obstacles: list = []
        self.walkable = None  # polygon or None (= the whole bounds)
        self.no_go: list = []  # polygons
        self._site_occ = np.zeros((self.nx, self.ny), dtype=bool)
        self._occ = np.zeros((self.nx, self.ny), dtype=bool)
        self._inflated: dict = {}

    def _centres(self):
        xs = self.xmin + np.arange(self.nx) * self.res
        ys = self.ymin + np.arange(self.ny) * self.res
        return np.meshgrid(xs, ys, indexing="ij")

    def set_site(self, walkable=None, no_go=()) -> None:
        """walkable: polygon [(x, y), ...] or None; no_go: list of polygons. Rebuilds the grid."""
        self.walkable = list(walkable) if walkable else None
        self.no_go = [list(p) for p in no_go]
        X, Y = self._centres()
        occ = np.zeros((self.nx, self.ny), dtype=bool)
        if self.walkable:
            occ |= ~points_in_polygon(X, Y, self.walkable)
        for poly in self.no_go:
            occ |= points_in_polygon(X, Y, poly)
        self._site_occ = occ
        self.set_obstacles(self.obstacles)

    def goal_allowed(self, x: float, y: float) -> bool:
        if not (self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax):
            return False
        if self.walkable and not point_in_polygon(x, y, self.walkable):
            return False
        return not any(point_in_polygon(x, y, p) for p in self.no_go)

    def set_obstacles(self, obstacles) -> None:
        self.obstacles = list(obstacles)
        occ = self._site_occ.copy()
        X = Y = None
        for o in self.obstacles:
            if isinstance(o, PolygonObstacle):
                if X is None:
                    X, Y = self._centres()
                occ |= points_in_polygon(X, Y, o.points)
                for px, py in o.points:  # thin polygons: make sure their corners are marked
                    occ[self._cell(px, py)] = True
                continue
            i0 = int(math.ceil((o.cx - o.hx - self.xmin) / self.res))
            i1 = int(math.floor((o.cx + o.hx - self.xmin) / self.res))
            j0 = int(math.ceil((o.cy - o.hy - self.ymin) / self.res))
            j1 = int(math.floor((o.cy + o.hy - self.ymin) / self.res))
            if i1 < i0:  # thinner than a cell: mark the nearest column / row
                i0 = i1 = int(round((o.cx - self.xmin) / self.res))
            if j1 < j0:
                j0 = j1 = int(round((o.cy - self.ymin) / self.res))
            i0, i1 = max(i0, 0), min(i1, self.nx - 1)
            j0, j1 = max(j0, 0), min(j1, self.ny - 1)
            if i0 <= i1 and j0 <= j1:
                occ[i0:i1 + 1, j0:j1 + 1] = True
        self._occ = occ
        self._inflated.clear()

    def _grid_for(self, radius: float) -> np.ndarray:
        r = radius + self.margin
        key = round(r, 3)
        grid = self._inflated.get(key)
        if grid is None:
            k = int(math.ceil(r / self.res))
            # With a site map, everything beyond the grid edge is outside the walkable area: pad with blocked cells so
            # the robot keeps its radius from a walkable boundary that lies exactly on the grid edge.
            src = np.pad(self._occ, k, constant_values=True) if self.walkable else self._occ
            grid = src.copy()
            for di in range(-k, k + 1):
                for dj in range(-k, k + 1):
                    if (di or dj) and math.hypot(di, dj) * self.res <= r + 0.5 * self.res:
                        _or_shifted(grid, src, di, dj)
            if self.walkable:
                grid = grid[k:-k, k:-k] if k else grid
            self._inflated[key] = grid
        return grid

    def _cell(self, x: float, y: float) -> tuple:
        i = min(max(int(round((x - self.xmin) / self.res)), 0), self.nx - 1)
        j = min(max(int(round((y - self.ymin) / self.res)), 0), self.ny - 1)
        return i, j

    def _xy(self, c: tuple) -> tuple:
        return self.xmin + c[0] * self.res, self.ymin + c[1] * self.res

    def is_free(self, x: float, y: float, radius: float, extra=()) -> bool:
        if not (self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax):
            return False
        if self._grid_for(radius)[self._cell(x, y)]:
            return False
        r = radius + self.margin
        return not any(o.inflated_contains(x, y, r) for o in extra)

    def segment_free(self, a: tuple, b: tuple, radius: float, extra=()) -> bool:
        n = max(2, int(math.dist(a, b) / (self.res * 0.5)))
        return all(self.is_free(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n, radius, extra)
                   for k in range(n + 1))

    def _nearest_free(self, x: float, y: float, radius: float, extra=()) -> tuple | None:
        c0 = self._cell(x, y)
        for ring in range(0, 60):
            best = None
            for di in range(-ring, ring + 1):
                for dj in range(-ring, ring + 1):
                    if max(abs(di), abs(dj)) != ring:
                        continue
                    c = (c0[0] + di, c0[1] + dj)
                    if 0 <= c[0] < self.nx and 0 <= c[1] < self.ny:
                        px, py = self._xy(c)
                        if self.is_free(px, py, radius, extra):
                            d = math.hypot(px - x, py - y)
                            if best is None or d < best[0]:
                                best = (d, c)
            if best:
                return best[1]
        return None

    def _last_leg_ok(self, a: tuple, goal: tuple) -> bool:
        """The final straight walk from the nearest free cell to a goal next to an asset must stay walkable."""
        if not (self.walkable or self.no_go):
            return True
        n = max(2, int(math.dist(a, goal) / (self.res * 0.5)))
        return all(self.goal_allowed(a[0] + (goal[0] - a[0]) * k / n, a[1] + (goal[1] - a[1]) * k / n) for k in range(n + 1))

    def plan(self, start: tuple, goal: tuple, radius: float, extra=()) -> list | None:
        """Return (x, y) waypoints from start to goal (inclusive), or None if unreachable."""
        start, goal = (float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))
        if not self.goal_allowed(*goal):
            return None
        if self.segment_free(start, goal, radius, extra):
            return [start, goal]
        s = self._nearest_free(*start, radius, extra)
        g = self._nearest_free(*goal, radius, extra)
        if s is None or g is None:
            return None
        cells = self._astar(s, g, radius, extra)
        if cells is None or not self._last_leg_ok(self._xy(cells[-1]), goal):
            return None
        return self._smooth([start] + [self._xy(c) for c in cells] + [goal], radius, extra)

    def _astar(self, s: tuple, g: tuple, radius: float, extra) -> list | None:
        grid = self._grid_for(radius)
        r = radius + self.margin

        def free(c):
            if grid[c]:
                return False
            if extra:
                x, y = self._xy(c)
                return not any(o.inflated_contains(x, y, r) for o in extra)
            return True

        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        open_heap = [(0.0, 0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        while open_heap:
            _, gcost, c = heapq.heappop(open_heap)
            if c == g:
                path = []
                while c is not None:
                    path.append(c)
                    c = came[c]
                return path[::-1]
            if gcost > cost.get(c, math.inf):
                continue
            for di, dj, w in moves:
                n = (c[0] + di, c[1] + dj)
                if not (0 <= n[0] < self.nx and 0 <= n[1] < self.ny) or not free(n):
                    continue
                if di and dj and not (free((c[0] + di, c[1])) and free((c[0], c[1] + dj))):
                    continue
                ng = gcost + w
                if ng < cost.get(n, math.inf):
                    cost[n], came[n] = ng, c
                    heapq.heappush(open_heap, (ng + math.hypot(g[0] - n[0], g[1] - n[1]), ng, n))
        return None

    def _smooth(self, pts: list, radius: float, extra) -> list:
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.segment_free(pts[i], pts[j], radius, extra):
                j -= 1
            out.append(pts[j])
            i = j
        return out
