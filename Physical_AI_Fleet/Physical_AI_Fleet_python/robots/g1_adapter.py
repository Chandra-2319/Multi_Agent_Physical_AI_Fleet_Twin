# SPDX-License-Identifier: Apache-2.0
"""Real humanoid (G1 / H1) link - STRETCH GOAL. Disabled until repeatable safe runs are validated."""
from __future__ import annotations

from .twin_robot import RealLink


class G1Link(RealLink):
    def __init__(self, conn: str, frame=None) -> None:
        self.conn, self.frame = conn, frame

    def read_state(self) -> dict:
        return {"x": 0.0, "y": 0.0, "yaw": 0.0, "busy": False,
                "error": "real humanoid link not enabled (stretch goal; see robots/g1_adapter.py)"}

    def send_goal(self, x, y, yaw, name: str = "") -> None:
        raise RuntimeError("real humanoid control is disabled until safe repeatable runs are validated")

    def stop(self) -> None:
        pass
