# SPDX-License-Identifier: Apache-2.0
"""Path follower: waypoints -> body velocity commands [vx, vy, wz] for an RL walking policy.

Pure pursuit with turn-in-place for large heading errors, slow-down near the goal, lateral
correction on the final approach (quadrupeds), final heading alignment, and acceleration limits so
the policy only ever sees smooth commands inside its training range.
"""
from __future__ import annotations

import math


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class PathFollower:
    LOOKAHEAD = 0.6
    YAW_GAIN = 1.6
    YAW_TOL = 0.12
    SLOW_RADIUS = 1.0
    MIN_V = 0.15

    def __init__(self, model) -> None:
        self.m = model
        self.path: list = []
        self.goal_yaw: float | None = None
        self.cmd = [0.0, 0.0, 0.0]
        self.done = True
        self._idx = 0
        self.speed_scale = 1.0

    def set_path(self, path: list, goal_yaw: float | None) -> None:
        self.path, self.goal_yaw, self._idx, self.done = list(path), goal_yaw, 0, False

    def clear(self) -> None:
        self.path, self.done = [], True

    def remaining(self, x: float, y: float) -> float:
        if not self.path:
            return 0.0
        pts = [(x, y)] + self.path[self._idx + 1:]
        return sum(math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1))

    def compute(self, pose: tuple, dt: float) -> list:
        target = self._raw_command(pose)
        a, aw = self.m.accel * dt, 2.0 * self.m.accel * dt
        self.cmd = [
            self.cmd[0] + max(-a, min(a, target[0] - self.cmd[0])),
            self.cmd[1] + max(-a, min(a, target[1] - self.cmd[1])),
            self.cmd[2] + max(-aw, min(aw, target[2] - self.cmd[2])),
        ]
        return list(self.cmd)

    def _raw_command(self, pose: tuple) -> list:
        if self.done or not self.path:
            return [0.0, 0.0, 0.0]
        x, y, yaw = pose
        gx, gy = self.path[-1]
        dist_goal = math.hypot(gx - x, gy - y)
        while self._idx < len(self.path) - 2 and math.dist((x, y), self.path[self._idx + 1]) < self.LOOKAHEAD:
            self._idx += 1
        if dist_goal <= self.m.goal_tol:
            if self.goal_yaw is None:
                self.done = True
                return [0.0, 0.0, 0.0]
            err = wrap_angle(self.goal_yaw - yaw)
            if abs(err) < self.YAW_TOL:
                self.done = True
                return [0.0, 0.0, 0.0]
            return [0.0, 0.0, max(-self.m.max_wz, min(self.m.max_wz, self.YAW_GAIN * err))]
        tx, ty = self._lookahead_point(x, y)
        err = wrap_angle(math.atan2(ty - y, tx - x) - yaw)
        wz = max(-self.m.max_wz, min(self.m.max_wz, self.YAW_GAIN * err))
        if abs(err) > self.m.turn_in_place_rad:
            return [0.0, 0.0, wz]
        slow = min(1.0, max(self.MIN_V / max(self.m.max_vx, 1e-6), dist_goal / self.SLOW_RADIUS))
        vx = self.m.max_vx * slow * max(0.0, math.cos(err)) * self.speed_scale
        vy = 0.0
        if self.m.max_vy > 0 and dist_goal < self.SLOW_RADIUS:
            lat = -math.sin(yaw) * (gx - x) + math.cos(yaw) * (gy - y)
            vy = max(-self.m.max_vy, min(self.m.max_vy, 0.8 * lat)) * self.speed_scale
        return [vx, vy, wz]

    def _lookahead_point(self, x: float, y: float) -> tuple:
        for k in range(self._idx, len(self.path) - 1):
            p, q = self.path[k], self.path[k + 1]
            if math.dist((x, y), q) >= self.LOOKAHEAD:
                dx, dy = q[0] - p[0], q[1] - p[1]
                fx, fy = p[0] - x, p[1] - y
                A = dx * dx + dy * dy
                if A < 1e-9:
                    return q
                B = 2 * (fx * dx + fy * dy)
                C = fx * fx + fy * fy - self.LOOKAHEAD ** 2
                disc = B * B - 4 * A * C
                if disc >= 0:
                    t = (-B + math.sqrt(disc)) / (2 * A)
                    if 0.0 <= t <= 1.0:
                        return (p[0] + t * dx, p[1] + t * dy)
                return q
        return self.path[-1]
