# SPDX-License-Identifier: Apache-2.0
"""Robot interface shared by simulated agents and digital-twin (real) robots."""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod

from ..fleet_config.catalog import capabilities_for, model_for

IDLE, BUSY, ERROR, OFFLINE = "idle", "busy", "error", "offline"


class RobotAgent(ABC):
    def __init__(self, robot_id: str, robot_type: str, mode: str, home_pose: tuple, prim_path: str = "",
                 model: str = "") -> None:
        self.model = model_for(model or robot_type)
        if self.model.robot_type != robot_type:
            raise ValueError(f"model {self.model.key} is a {self.model.robot_type}, not a {robot_type}")
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.mode = mode
        self.provenance = "simulated" if mode == "simulation" else "real"
        self.home_pose = tuple(home_pose)
        self.pose = list(home_pose)
        self.prim_path = prim_path
        self.status = IDLE
        self.current_skill: str | None = None
        self.goal: tuple | None = None
        self.goal_name: str | None = None
        self.last_telemetry = time.time()
        self.note = ""
        self.locomotion = "kinematic"

    @property
    def radius(self) -> float:
        return self.model.radius

    @abstractmethod
    def go_to(self, x: float, y: float, yaw: float, name: str = "") -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def update(self, dt: float) -> None:
        """Called every physics step on the main thread. Must never block."""

    def reset_to_home(self) -> None:
        self.goal = None
        self.pose = list(self.home_pose)
        self.status = IDLE
        self.note = ""

    def shutdown(self) -> None:
        """Release sockets / threads."""

    @property
    def capabilities(self) -> tuple:
        return capabilities_for(self.robot_type)

    def arrived(self, tol: float | None = None) -> bool:
        if self.goal is None:
            return True
        return math.dist(self.pose[:2], self.goal[:2]) <= (tol if tol is not None else self.model.goal_tol)

    def telemetry_age(self) -> float:
        return time.time() - self.last_telemetry

    def snapshot(self, telemetry_ttl_s: float) -> dict:
        return {
            "robot_id": self.robot_id, "type": self.robot_type, "model": self.model.key, "mode": self.mode,
            "provenance": self.provenance, "status": self.status, "pose": [round(v, 2) for v in self.pose],
            "capabilities": list(self.capabilities), "current_skill": self.current_skill,
            "fresh": self.telemetry_age() <= telemetry_ttl_s,
        }

    def summary(self) -> str:
        x, y, yaw = self.pose
        skill = f" [{self.current_skill}]" if self.current_skill else ""
        fresh = "" if self.telemetry_age() < 10 else " (stale)"
        return (f"{self.robot_id:<10} {self.model.key:<5} {self.mode:<12} {self.locomotion:<9} {self.status:<7}{skill} "
                f"({x:5.2f},{y:5.2f},{math.degrees(yaw):4.0f}deg){fresh} {self.note}")
