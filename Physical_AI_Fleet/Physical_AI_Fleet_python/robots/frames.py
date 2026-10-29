# SPDX-License-Identifier: Apache-2.0
"""2-D transforms between a robot's own map frame (SLAM / MiR map) and the shared world frame."""
from __future__ import annotations

import math


class Frame2D:
    """world = R(theta) * map + t"""

    def __init__(self, tx: float = 0.0, ty: float = 0.0, theta: float = 0.0) -> None:
        self.tx, self.ty, self.theta = tx, ty, theta

    @staticmethod
    def parse(text: str) -> "Frame2D":
        if not text.strip():
            return Frame2D()
        tx, ty, th = (float(v) for v in text.split(","))
        return Frame2D(tx, ty, math.radians(th))

    def to_world(self, x: float, y: float, yaw: float) -> tuple:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return (c * x - s * y + self.tx, s * x + c * y + self.ty, yaw + self.theta)

    def to_text(self) -> str:
        """'tx,ty,theta_deg' - the format of the Map -> World field / floor-layout map_to_world."""
        return f"{self.tx:.4f},{self.ty:.4f},{math.degrees(self.theta):.3f}"

    @staticmethod
    def from_correspondence(map_pose: tuple, world_pose: tuple) -> "Frame2D":
        """The transform that puts a robot reporting `map_pose` (x, y, yaw) exactly at `world_pose` (v0.5.6 ALIGN TWIN)."""
        mx, my, myaw = (float(v) for v in map_pose[:3])
        wx, wy, wyaw = (float(v) for v in world_pose[:3])
        th = (wyaw - myaw + math.pi) % (2 * math.pi) - math.pi
        c, s = math.cos(th), math.sin(th)
        return Frame2D(wx - (c * mx - s * my), wy - (s * mx + c * my), th)

    def to_map(self, x: float, y: float, yaw: float) -> tuple:
        c, s = math.cos(self.theta), math.sin(self.theta)
        dx, dy = x - self.tx, y - self.ty
        return (c * dx + s * dy, -s * dx + c * dy, yaw - self.theta)
