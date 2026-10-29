# SPDX-License-Identifier: Apache-2.0
"""Fleet state: robots, assets and incidents, plus the compact snapshot sent to Nemotron (legacy format).

v0.4.3: `homing` = robots that finished a task and are walking back to their home pose. They are reported as idle
(effective_status) so Nemotron and the safety gate can give them a new task at once; the new goal replaces the walk home.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field

OPEN, CONFIRMED, TOOLS_DISPATCHED, AWAITING_REPAIR, VERIFYING, CLOSED, NEEDS_OPERATOR = ("open", "confirmed", "tools_dispatched", "awaiting_repair", "verifying", "closed", "needs_operator")
TOOLS_PICKUP, TOOLS_LOADED = "tools_pickup", "tools_loaded"
BUSY = "busy"


@dataclass
class Incident:
    asset_id: str
    incident_id: str = field(default_factory=lambda: f"INC-{uuid.uuid4().hex[:6]}")
    status: str = OPEN
    opened_at: float = field(default_factory=time.time)
    problem_reports: int = 0
    reporters: set = field(default_factory=set)
    assignments: int = 0
    last_event_id: str = ""
    history: list = field(default_factory=list)
    unreadable: int = 0
    needs_reinspect: bool = False
    last_inspector: str = ""
    waiting_note: str = ""
    priority: str | None = None
    alarm_kind: str = ""
    needs_inspection: bool = False

    def note(self, text: str) -> None:
        self.history.append(f"{time.strftime('%H:%M:%S')} {text}")


class FleetState:
    def __init__(self, settings) -> None:
        self.s = settings
        self.robots: dict = {}
        self.incidents: dict = {}
        self.closed: list = []
        self.homing: set = set()

    def add(self, robot) -> None:
        if robot.robot_id in self.robots:
            raise ValueError(f"robot id '{robot.robot_id}' already exists")
        self.robots[robot.robot_id] = robot
        if hasattr(robot, "neighbors_fn"):
            robot.neighbors_fn = lambda rid=robot.robot_id: self.neighbors(rid)

    def neighbors(self, robot_id: str) -> list:
        return [(r.robot_id, r.pose[0], r.pose[1], r.radius) for r in self.robots.values() if r.robot_id != robot_id]

    def remove(self, robot_id: str):
        self.homing.discard(robot_id)
        robot = self.robots.pop(robot_id, None)
        if robot:
            robot.shutdown()
        return robot

    def get(self, robot_id: str):
        return self.robots.get(robot_id)

    def effective_status(self, robot) -> str:
        """'idle' for a robot that is only walking home; otherwise the robot's own status."""
        if robot.robot_id in self.homing and robot.status == BUSY:
            return "idle"
        return robot.status

    def is_returning_home(self, robot) -> bool:
        return robot.robot_id in self.homing and robot.status == BUSY

    def incident_for(self, asset_id: str):
        return self.incidents.get(asset_id)

    def open_incident(self, asset_id: str) -> Incident:
        if asset_id not in self.incidents:
            self.incidents[asset_id] = Incident(asset_id)
        return self.incidents[asset_id]

    def close_incident(self, asset_id: str):
        inc = self.incidents.pop(asset_id, None)
        if inc:
            inc.status = CLOSED
            self.closed.append(inc)
        return inc

    def snapshot(self) -> dict:
        assets = {}
        for aid, a in self.s.assets.items():
            inc = self.incidents.get(aid)
            assets[aid] = {"position": list(a.position[:2]), "normal_band": [a.normal_low, a.normal_high, a.unit], "incident": inc.status if inc else None,
                           "confirmed": bool(inc and inc.status in (CONFIRMED, TOOLS_PICKUP, TOOLS_LOADED, TOOLS_DISPATCHED, AWAITING_REPAIR))}
        return {"robots": [r.snapshot(self.s.telemetry_ttl_s) for r in self.robots.values()], "assets": assets, "min_confidence": self.s.min_confidence}
