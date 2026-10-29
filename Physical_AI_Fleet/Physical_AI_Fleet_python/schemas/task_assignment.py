# SPDX-License-Identifier: Apache-2.0
"""Strict task-assignment schema: what Nemotron (or the rule-based fallback) returns."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..fleet_config.catalog import SKILLS
from .validate import SchemaError, bounded_text, no_extra_keys, reject_forbidden_keys, require, utc_now_iso

PRIORITIES = ("low", "medium", "high")
SOURCES = ("nemotron", "rule_based", "operator")


@dataclass
class TaskAssignment:
    event_id: str
    robot_id: str
    skill: str
    target_asset_id: Optional[str]
    priority: str
    rationale: str
    source: str = "nemotron"
    assignment_id: str = field(default_factory=lambda: f"TA-{uuid.uuid4().hex[:10]}")
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def risk(self) -> str:
        return SKILLS[self.skill].risk

    @staticmethod
    def from_model_dict(d: dict, *, source: str) -> "TaskAssignment":
        if not isinstance(d, dict):
            raise SchemaError("assignment must be an object")
        reject_forbidden_keys(d)
        no_extra_keys(d, {"event_id", "robot_id", "skill", "target_asset_id", "priority", "rationale"}, "task_assignment")
        if source not in SOURCES:
            raise SchemaError("bad source")
        skill = require(d, "skill", str, allowed=set(SKILLS))
        target = require(d, "target_asset_id", str, nullable=True)
        if SKILLS[skill].needs_target and not target:
            raise SchemaError(f"skill '{skill}' needs target_asset_id")
        return TaskAssignment(
            event_id=bounded_text(require(d, "event_id", str), "event_id", 64),
            robot_id=bounded_text(require(d, "robot_id", str), "robot_id", 64),
            skill=skill, target_asset_id=target,
            priority=require(d, "priority", str, allowed=set(PRIORITIES)),
            rationale=bounded_text(require(d, "rationale", str), "rationale", 400),
            source=source,
        )


ASSIGNMENT_OUTPUT_SPEC = {
    "event_id": "string, copy from the triggering event",
    "robot_id": "string, one robot_id from the fleet snapshot",
    "skill": "one of the catalog skill names",
    "target_asset_id": "asset id string, or null for hold_position/return_home",
    "priority": "low | medium | high",
    "rationale": "one short sentence (max 300 chars)",
}
