# SPDX-License-Identifier: Apache-2.0
"""Body interface between a legged fleet member and whatever actually moves it.

`PolicyBody` (policy_body.py, Isaac Sim only) runs an RL walking policy on a physics articulation.
`UnicycleBody` integrates the same velocity commands with a first-order lag for headless runs/tests.
"""
from __future__ import annotations

import math


class LocomotionBody:
    ready: bool = True
    status_note: str = ""
    error: str = ""  # non-empty = the body cannot be driven (e.g. policy keeps failing); the robot goes to ERROR

    def step(self, dt: float, command: list) -> None:
        raise NotImplementedError

    def pose(self) -> tuple:
        raise NotImplementedError

    def is_fallen(self) -> bool:
        return False

    holding: bool = False

    def hold(self, dt: float) -> None:
        """Idle: keep standing as cheaply as possible (v0.5.4). Default: a zero-velocity step."""
        self.step(dt, [0.0, 0.0, 0.0])

    def reset_to(self, x: float, y: float, yaw: float) -> None:
        raise NotImplementedError

    def request_reinit(self) -> None:
        """World reset / timeline stop: physics handles are invalid until the next step."""

    def destroy(self) -> None:
        pass


class UnicycleBody(LocomotionBody):
    TAU = 0.25

    def __init__(self, x: float, y: float, yaw: float) -> None:
        self.x, self.y, self.yaw = x, y, yaw
        self.v = [0.0, 0.0, 0.0]
        self.fallen = False

    def step(self, dt: float, command: list) -> None:
        k = min(1.0, dt / self.TAU)
        self.v = [v + k * (c - v) for v, c in zip(self.v, command)]
        vx, vy, wz = self.v
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * vx - s * vy) * dt
        self.y += (s * vx + c * vy) * dt
        self.yaw = (self.yaw + wz * dt + math.pi) % (2 * math.pi) - math.pi

    def pose(self) -> tuple:
        return (self.x, self.y, self.yaw)

    def is_fallen(self) -> bool:
        return self.fallen

    def reset_to(self, x: float, y: float, yaw: float) -> None:
        self.x, self.y, self.yaw, self.v, self.fallen = x, y, yaw, [0.0, 0.0, 0.0], False
