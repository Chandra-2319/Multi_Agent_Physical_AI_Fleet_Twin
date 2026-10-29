# SPDX-License-Identifier: Apache-2.0
"""Deterministic safety gate. No model is involved: same inputs -> same decision.

Checks: event freshness (TTL), robot availability + telemetry freshness, capability (from robots_actions.json
for configured actions and follow-ups), target existence + site map (walkable area, no-go zones), confidence and
provenance for high-risk skills, real-robot policy, and operator approval (config approval, high-risk legacy skills,
real robots).
v0.4.3: a robot that is only walking home after a task counts as idle (FleetState.effective_status).
v0.4.4: targets are checked against the site map polygon and no-go zones (bounding box only if there is no site map).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from ..fleet_config.catalog import SKILLS, robot_can
from ..schemas.validate import age_seconds
APPROVED, REJECTED, PENDING = "approved", "rejected", "pending_approval"
SAFE_SKILLS = ("hold_position", "return_home")
@dataclass
class GateDecision:
    status: str
    reasons: list = field(default_factory=list)
    def to_dict(self): return {"status": self.status, "reasons": self.reasons}
class SafetyGate:
    def __init__(self, settings, action_cfg=None): self.s = settings; self.cfg = action_cfg
    def check(self, assignment, event, fleet_state, action=None, follow_up=None):
        reasons = []; spec = SKILLS.get(assignment.skill)
        if spec is None: return GateDecision(REJECTED, [f"unknown skill {assignment.skill}"])
        aspec = None; fspec = None
        if (action is not None or follow_up is not None) and self.cfg is None: return GateDecision(REJECTED, ["no action configuration loaded"])
        if action is not None:
            if action not in self.cfg.actions: return GateDecision(REJECTED, [f"unknown action {action}"])
            aspec = self.cfg.actions[action]
            if aspec.executor_skill != assignment.skill: return GateDecision(REJECTED, [f"action {action} runs {aspec.executor_skill}, not {assignment.skill}"])
        if follow_up is not None:
            fspec = follow_up if not isinstance(follow_up, str) else self.cfg.follow_ups.get(follow_up)
            if fspec is None: return GateDecision(REJECTED, [f"unknown follow-up {follow_up}"])
            if fspec.executor_skill != assignment.skill: return GateDecision(REJECTED, [f"follow-up {fspec.name} runs {fspec.executor_skill}, not {assignment.skill}"])
        if event is not None:
            if event.event_id != assignment.event_id: reasons.append("assignment does not reference the triggering event")
            elif age_seconds(event.timestamp) > min(event.ttl_s, self.s.event_ttl_s): reasons.append(f"event {event.event_id} is stale")
        robot = fleet_state.get(assignment.robot_id)
        if robot is None: return GateDecision(REJECTED, reasons + [f"unknown robot {assignment.robot_id}"])
        if getattr(robot, "control", "") == "mirror":  # v0.5.6: a mirrored real robot is never driven by the fleet
            return GateDecision(REJECTED, reasons + [f"{robot.robot_id} mirrors a real robot: it is driven by its remote or JOG, not by tasks"])
        status = fleet_state.effective_status(robot) if hasattr(fleet_state, "effective_status") else robot.status
        if status != "idle" and assignment.skill not in SAFE_SKILLS: reasons.append(f"robot {robot.robot_id} is {status}")
        if robot.status == "error" and assignment.skill == "return_home": reasons.append(f"robot {robot.robot_id} is in error ({robot.note}); use Reset Robot")
        if robot.telemetry_age() > self.s.telemetry_ttl_s: reasons.append(f"robot {robot.robot_id} telemetry is stale ({robot.telemetry_age():.0f}s)")
        key = getattr(getattr(robot, "model", None), "key", "")
        if aspec is not None:
            if not self.cfg.can(key, action): reasons.append(f"model {key or '?'} is not configured for {action}")
        elif fspec is not None:
            if key not in self.cfg.robots_with_role(fspec.robot_role): reasons.append(f"model {key or '?'} does not have the role '{fspec.robot_role}' for {fspec.name}")
        elif not robot_can(robot.robot_type, assignment.skill): reasons.append(f"{robot.robot_type} cannot run {assignment.skill}")
        configured = aspec is not None or fspec is not None; real = getattr(robot, "mode", "") == "digital_twin"
        autonomy = self.cfg.policy.get("real_robot_autonomy", "approve_each") if self.cfg is not None else "approve_each"
        if configured and real:
            if aspec is not None and not aspec.real_robot_allowed: reasons.append(f"{action} is not allowed on real robots")
            if fspec is not None: reasons.append(f"follow-up {fspec.name} is not allowed on real robots")
            if autonomy == "off": reasons.append("real-robot autonomy is off")
        if aspec is not None and aspec.target_kind == "location":
            loc = self.cfg.locations.get(aspec.target_id)
            if loc is None: reasons.append(f"unknown location {aspec.target_id}")
            else: self._geofence(loc["pose"], f"location {aspec.target_id}", reasons)
        elif spec.needs_target or (aspec is not None and aspec.target_kind == "asset"):
            asset = self.s.assets.get(assignment.target_asset_id or "")
            if asset is None: reasons.append(f"unknown asset {assignment.target_asset_id}")
            else: self._geofence(asset.approach_pose, f"asset {asset.asset_id} approach pose", reasons)
        risk = aspec.risk if aspec is not None else spec.risk
        if event is not None and event.confidence < self.s.min_confidence and risk == "high": reasons.append(f"confidence {event.confidence:.2f} below {self.s.min_confidence} for high-risk skill")
        if event is not None and event.provenance == "operator_phone" and risk == "high": reasons.append("operator-phone evidence cannot trigger high-risk skills on its own")
        if reasons: return GateDecision(REJECTED, reasons)
        why = self._approval_reason(assignment, spec, aspec, fspec, real, autonomy)
        return GateDecision(PENDING, [why]) if why else GateDecision(APPROVED, ["all checks passed"])
    def _geofence(self, pose, label, reasons):
        x, y = pose[0], pose[1]
        site = getattr(self.s, "site", None)
        if site is not None:
            why = site.why_not(x, y)
            if why: reasons.append(f"{label} outside geofence: {why}")
            return
        xmin, ymin, xmax, ymax = self.s.geofence
        if not (xmin <= x <= xmax and ymin <= y <= ymax): reasons.append(f"{label} outside geofence")
    @staticmethod
    def _approval_reason(assignment, spec, aspec, fspec, real, autonomy):
        if aspec is not None or fspec is not None:
            approval = aspec.approval if aspec is not None else fspec.approval
            if approval == "operator": return f"{aspec.name if aspec else fspec.name} requires operator approval"
            if real:
                if autonomy == "approve_each": return "real robot: operator approval is required for every action"
                if autonomy == "auto_low_risk" and (aspec is None or aspec.risk != "low"): return "real robot: only low-risk actions run automatically"
            return None
        if spec.risk == "high" and assignment.source != "operator": return f"{assignment.skill} is high-risk: operator approval required"
        return None
