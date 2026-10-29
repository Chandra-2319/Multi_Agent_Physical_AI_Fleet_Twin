# SPDX-License-Identifier: Apache-2.0
"""Nemotron task assigner: compact fleet snapshot + event -> one validated TaskAssignment.

Nemotron only picks *which robot* runs *which catalogue skill* on *which asset*. If Nemotron is
not configured or its output fails validation, a deterministic rule-based assigner is used
(labelled source="rule_based" in the audit log).
"""
from __future__ import annotations

import json
import math

from ..fleet_config.catalog import SKILLS, catalog_for_prompt, robot_can
from ..schemas.fleet_event import FleetEvent
from ..schemas.task_assignment import ASSIGNMENT_OUTPUT_SPEC, TaskAssignment
from ..schemas.validate import SchemaError, extract_json_object
from .token_factory_client import LLMError, TokenFactoryClient

SYSTEM_PROMPT = """You are the dispatcher for an industrial inspection robot fleet (quadrupeds, humanoids, AMRs).
You receive one fleet event and a compact fleet snapshot. Choose exactly ONE action.
Rules:
- Use only skills from the catalog, and only for a robot whose capabilities include the skill's capability.
- Only choose robots with status "idle" and fresh telemetry. Prefer the closest suitable robot.
- Prefer real robots for real-provenance events; simulated robots may handle simulated events.
- If confidence is low or the gauge reading is missing/under-resolved, choose "reinspect_asset" or "read_gauge".
- If the fault is confirmed and an AMR is available, you may choose "deliver_tools".
- Never output velocities, joint values, trajectories or any field not in the output schema.
Return ONLY a JSON object with exactly these fields:
""" + json.dumps(ASSIGNMENT_OUTPUT_SPEC, indent=1)


class NemotronAssigner:
    def __init__(self, client: TokenFactoryClient | None, model_id: str, audit=None) -> None:
        self.client = client
        self.model_id = model_id
        self.audit = audit
        self.use_nemotron = True

    @property
    def nemotron_available(self) -> bool:
        return bool(self.client and self.client.api_key and self.model_id)

    def assign(self, event: FleetEvent, snapshot: dict) -> TaskAssignment | None:
        if self.use_nemotron and self.nemotron_available:
            try:
                return self._assign_with_nemotron(event, snapshot)
            except (LLMError, SchemaError) as exc:
                self._log("assigner_fallback", reason=str(exc))
        elif self.use_nemotron:
            self._log("assigner_fallback", reason="Nemotron not configured (NEBIUS_API_KEY / NEMOTRON_MODEL_ID)")
        return rule_based_assign(event, snapshot)

    def _assign_with_nemotron(self, event: FleetEvent, snapshot: dict) -> TaskAssignment:
        user = json.dumps({"event": _compact_event(event), "fleet": snapshot, "catalog": catalog_for_prompt()}, default=str)
        reply = self.client.chat(self.model_id, [{"role": "system", "content": SYSTEM_PROMPT},
                                                 {"role": "user", "content": user}],
                                 temperature=0.0, max_tokens=800, tag="nemotron_assign")
        assignment = TaskAssignment.from_model_dict(extract_json_object(reply), source="nemotron")
        if assignment.event_id != event.event_id:
            raise SchemaError("assignment event_id does not match the triggering event")
        return assignment

    def _log(self, kind: str, **kw) -> None:
        if self.audit:
            self.audit.record(kind, kw)


def _compact_event(ev: FleetEvent) -> dict:
    d = {"event_id": ev.event_id, "asset_id": ev.asset_id, "event_type": ev.event_type, "provenance": ev.provenance,
         "confidence": ev.confidence, "observation": ev.observation, "reporting_robot": ev.robot_id}
    if ev.gauge:
        d["gauge"] = {"value": ev.gauge.value, "unit": ev.gauge.unit, "in_range": ev.gauge.in_range,
                      "confidence": ev.gauge.confidence, "roi_px": ev.gauge.roi_px}
    return d


def rule_based_assign(event: FleetEvent, snapshot: dict) -> TaskAssignment | None:
    """Deterministic fallback with the same contract as Nemotron."""
    low_conf = event.confidence < snapshot.get("min_confidence", 0.6) or event.event_type == "inconclusive"
    gauge_unread = event.gauge is not None and event.gauge.value is None
    if low_conf or gauge_unread:
        skill = "read_gauge" if gauge_unread else "reinspect_asset"
    elif event.event_type in ("anomaly", "leak_suspected") or (event.gauge and event.gauge.in_range is False):
        confirmed = snapshot.get("assets", {}).get(event.asset_id, {}).get("confirmed")
        skill = "deliver_tools" if confirmed else "reinspect_asset"
    else:
        return None
    asset_pos = snapshot.get("assets", {}).get(event.asset_id, {}).get("position", (0, 0))
    candidates = []
    for r in snapshot.get("robots", []):
        if r["status"] != "idle" or not r["fresh"] or not robot_can(r["type"], skill):
            continue
        penalty = 5.0 if (skill == "reinspect_asset" and r["robot_id"] == event.robot_id
                          and len(snapshot["robots"]) > 1) else 0.0
        prov_penalty = 0.0 if (r["provenance"] == "real") == (event.provenance == "real") else 2.0
        candidates.append((math.dist(r["pose"][:2], asset_pos[:2]) + penalty + prov_penalty, r["robot_id"]))
    if not candidates:
        return None
    _, robot_id = min(candidates)
    return TaskAssignment(
        event_id=event.event_id, robot_id=robot_id, skill=skill,
        target_asset_id=event.asset_id if SKILLS[skill].needs_target else None,
        priority="high" if skill == "deliver_tools" else "medium",
        rationale=f"Rule-based: {event.event_type} on {event.asset_id}, closest capable idle robot.",
        source="rule_based",
    )
