# SPDX-License-Identifier: Apache-2.0
"""What Nemotron returns: ONE decision that references a due task and one of that task's candidate robots."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

from .validate import SchemaError, bounded_text, no_extra_keys, reject_forbidden_keys, require, utc_now_iso

DECISIONS = ("assign", "no_action")
PRIORITIES = ("low", "medium", "high")
KEYS = {"snapshot_id", "decision", "task_id", "robot_id", "priority", "rationale"}


@dataclass
class FleetDecision:
    snapshot_id: str
    decision: str
    task_id: Optional[str]
    robot_id: Optional[str]
    priority: str
    rationale: str
    source: str  # "nemotron" | "rule_based"
    action: str = ""
    target_id: str = ""
    for_asset: str = ""
    decision_id: str = field(default_factory=lambda: f"FD-{uuid.uuid4().hex[:10]}")
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


def output_json_schema(snapshot: dict) -> dict:
    """JSON Schema for constrained decoding: only ids from this snapshot are even expressible."""
    task_ids = [t["task_id"] for t in snapshot["due_tasks"]]
    robot_ids = sorted({c["robot_id"] for t in snapshot["due_tasks"] for c in t["candidates"]})

    def id_field(ids):
        return {"anyOf": [{"type": "string", "enum": ids}, {"type": "null"}]} if ids else {"type": "null"}

    return {
        "type": "object", "additionalProperties": False, "required": sorted(KEYS),
        "properties": {
            "snapshot_id": {"type": "string", "enum": [snapshot["snapshot_id"]]},
            "decision": {"type": "string", "enum": list(DECISIONS)},
            "task_id": id_field(task_ids),
            "robot_id": id_field(robot_ids),
            "priority": {"type": "string", "enum": list(PRIORITIES)},
            "rationale": {"type": "string", "maxLength": 300},
        },
    }


def validate_decision(d, snapshot: dict, source: str) -> FleetDecision:
    if not isinstance(d, dict):
        raise SchemaError("decision must be a JSON object")
    reject_forbidden_keys(d)
    no_extra_keys(d, KEYS, "fleet_decision")
    sid = require(d, "snapshot_id", str)
    if sid != snapshot["snapshot_id"]:
        raise SchemaError(f"snapshot_id '{sid}' does not match '{snapshot['snapshot_id']}'")
    decision = require(d, "decision", str, allowed=set(DECISIONS))
    task_id = require(d, "task_id", str, nullable=True)
    robot_id = require(d, "robot_id", str, nullable=True)
    priority = require(d, "priority", str, allowed=set(PRIORITIES))
    rationale = bounded_text(require(d, "rationale", str), "rationale", 300)
    if decision == "no_action":
        if task_id or robot_id:
            raise SchemaError("no_action must have task_id and robot_id set to null")
        return FleetDecision(sid, decision, None, None, priority, rationale, source)
    tasks = {t["task_id"]: t for t in snapshot["due_tasks"]}
    if task_id not in tasks:
        raise SchemaError(f"task_id '{task_id}' is not one of the due tasks {sorted(tasks)}")
    task = tasks[task_id]
    cands = [c["robot_id"] for c in task["candidates"]]
    if robot_id not in cands:
        raise SchemaError(f"robot_id '{robot_id}' is not a candidate for {task_id} (candidates: {cands})")
    return FleetDecision(sid, decision, task_id, robot_id, priority, rationale, source,
                         action=task["action"], target_id=task["target_id"], for_asset=task["for_asset"])
