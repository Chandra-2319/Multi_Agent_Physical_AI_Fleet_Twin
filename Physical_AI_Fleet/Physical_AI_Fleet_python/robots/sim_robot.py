# SPDX-License-Identifier: Apache-2.0
"""Kinematic simulated fleet member (turn, then drive). Used for AMRs, and as an explicit fallback
for legged robots when the walking policy is switched off in the UI.

v0.5.4: with a planner (always for robots made by the scenario) the robot follows a planned path, so the AMR obeys the
same site rules as the walking robots: it stays inside the walkable area, keeps its radius clear of no-go zones and
assets, and refuses a goal it cannot reach ("no path", status error) instead of driving straight through a zone.
Without a planner (old tests / tools) it still drives straight to the goal.
"""
from __future__ import annotations

import math
import time

from .base import BUSY, ERROR, IDLE, RobotAgent

WAYPOINT_TOL = 0.12


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class SimRobot(RobotAgent):
    TURN_RATE = 1.5

    def __init__(self, *args, planner=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.speed = self.model.max_vx
        self.final_yaw: float | None = None
        self.planner = planner
        self.path: list = []
        self.neighbors_fn = None

    def go_to(self, x: float, y: float, yaw: float, name: str = "") -> None:
        if self.status == ERROR:
            return
        self.goal, self.goal_name, self.final_yaw = (x, y, yaw), name, yaw
        self.status = BUSY
        if self.planner is None:
            self.path = [(x, y)]
            return
        path = self.planner.plan(tuple(self.pose[:2]), (x, y), self.radius)
        if not path:
            self.goal, self.path = None, []
            self.status, self.note = ERROR, f"no path to {name or (round(x, 1), round(y, 1))}"
            return
        self.path = [tuple(p) for p in path[1:]] or [(x, y)]
        self.note = ""

    def stop(self) -> None:
        self.goal = None
        self.path = []
        if self.status == BUSY:
            self.status = IDLE

    def reset_to_home(self) -> None:
        super().reset_to_home()
        self.path = []

    def update(self, dt: float) -> None:
        self.last_telemetry = time.time()
        if self.goal is None:
            return
        x, y, yaw = self.pose
        if not self.path:
            self.path = [self.goal[:2]]
        gx, gy = self.path[0]
        last = len(self.path) == 1
        dist = math.hypot(gx - x, gy - y)
        if dist > (0.05 if last else WAYPOINT_TOL):
            err = wrap_angle(math.atan2(gy - y, gx - x) - yaw)
            yaw += max(-self.TURN_RATE * dt, min(self.TURN_RATE * dt, err))
            if abs(err) < 0.35:
                step = min(self.speed * dt, dist)
                x += step * math.cos(yaw)
                y += step * math.sin(yaw)
        elif not last:
            self.path.pop(0)
        else:
            err = wrap_angle(self.final_yaw - yaw)
            yaw += max(-self.TURN_RATE * dt, min(self.TURN_RATE * dt, err))
            if abs(err) < 0.05:
                self.pose = [gx, gy, self.final_yaw]
                self.goal, self.path = None, []
                self.status = IDLE
                return
        self.pose = [x, y, wrap_angle(yaw)]
