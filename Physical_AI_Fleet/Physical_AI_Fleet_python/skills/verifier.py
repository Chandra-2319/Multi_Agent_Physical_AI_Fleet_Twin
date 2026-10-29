# SPDX-License-Identifier: Apache-2.0
"""Verification pass: after the repair an inspection-capable robot re-checks the asset."""
from __future__ import annotations

import math

from ..fleet_config.catalog import robot_can
from ..schemas.task_assignment import TaskAssignment

MAX_VERIFY_ATTEMPTS = 2


class Verifier:
    def __init__(self, settings) -> None:
        self.s = settings
        self.attempts: dict = {}

    def pick_robot(self, fleet, asset_id: str, prefer_real: bool = True):
        ax, ay = self.s.assets[asset_id].position[:2]
        best = None
        for r in fleet.robots.values():
            if r.status != "idle" or r.telemetry_age() > self.s.telemetry_ttl_s or not robot_can(r.robot_type, "reinspect_asset"):
                continue
            score = math.dist(r.pose[:2], (ax, ay)) + (0 if (r.provenance == "real") == prefer_real else 3)
            if best is None or score < best[0]:
                best = (score, r)
        return best[1] if best else None

    def make_assignment(self, incident, robot):
        n = self.attempts.get(incident.incident_id, 0)
        if n >= MAX_VERIFY_ATTEMPTS:
            return None
        self.attempts[incident.incident_id] = n + 1
        return TaskAssignment(event_id=incident.last_event_id or "VERIFY", robot_id=robot.robot_id,
                              skill="reinspect_asset", target_asset_id=incident.asset_id, priority="medium",
                              rationale=f"Verification pass {n + 1} after repair of {incident.asset_id}.",
                              source="rule_based")

    @staticmethod
    def is_clear(event) -> bool:
        return event.event_type == "verification"
