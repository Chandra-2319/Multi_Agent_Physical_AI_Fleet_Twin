# SPDX-License-Identifier: Apache-2.0
"""Fleet snapshot for Nemotron: due tasks, eligible candidates and fleet facts. Pure Python.

Nemotron picks ONE (task_id, robot_id) pair from this snapshot. Capability, availability, freshness and
real-robot rules are applied here, so a robot only appears as a candidate if it may actually do the task.
Incident statuses are plain strings here (same values as coordinator/fleet_state.py) to avoid an import cycle.

v0.4.3
  * Routine inspections are opt-in (schedule.routine_inspections / Run Fleet panel) and are counted on the
    scheduler's own clock, which the coordinator drives with RUNNING time only (paused / stopped time does not count).
  * A robot that is walking home after a task is shown as idle ("returning_home": true) and is a valid candidate.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field

from ..schemas.validate import utc_now_iso

PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
_OPEN, _CONFIRMED, _VERIFYING = "open", "confirmed", "verifying"


@dataclass
class DueTask:
    task_id: str
    action: str
    target_id: str
    for_asset: str
    reason: str
    priority: str
    due_since: float
    incident_id: str | None = None
    prefer_not: str = ""
    candidates: list = field(default_factory=list)  # [(distance_m, robot_id)]


class Scheduler:
    """When each asset was last inspected and when its next ROUTINE inspection is due (if routine inspections are on).

    `clock` is the scheduler's own time source (the coordinator passes its running-time counter); explicit `now`
    arguments override it (used by tests).
    """

    def __init__(self, cfg, asset_ids, now: float | None = None, clock=None) -> None:
        self.cfg = cfg
        self.clock = clock or time.time
        sched = cfg.schedule
        self.enabled = bool(sched.get("routine_inspections", False))
        self.interval_s = float(sched.get("default_interval_s", 900))
        self.initial_due = bool(sched.get("initial_due", False))
        self.last: dict = {}
        self.ensure(asset_ids, now)

    def _t(self, now) -> float:
        return self.clock() if now is None else now

    def interval_for(self, asset_id: str) -> float:
        per = self.cfg.schedule.get("per_asset_interval_s", {}) or {}
        return float(per.get(asset_id, self.interval_s))

    def ensure(self, asset_ids, now: float | None = None) -> None:
        t = self._t(now)
        for a in asset_ids:
            if a not in self.last:
                self.last[a] = None if self.initial_due else t

    def restart(self, asset_ids, now: float | None = None) -> None:
        """A new run: forget every inspection time."""
        self.last.clear()
        self.ensure(asset_ids, now)

    def set_routine(self, enabled: bool, interval_s: float | None = None, now: float | None = None) -> None:
        if interval_s is not None:
            self.interval_s = float(interval_s)
        if enabled and not self.enabled:  # switching ON starts a fresh interval instead of making everything due at once
            t = self._t(now)
            for a in list(self.last):
                self.last[a] = None if self.initial_due else t
        self.enabled = bool(enabled)

    def mark_inspected(self, asset_id: str, t: float | None = None) -> None:
        self.last[asset_id] = self._t(t)

    def due_in(self, asset_id: str, now: float | None = None) -> float:
        last = self.last.get(asset_id)
        return 0.0 if last is None else last + self.interval_for(asset_id) - self._t(now)


def compute_due_tasks(cfg, fleet, scheduler, busy_assets: set, now: float, sched_now: float | None = None) -> list:
    tasks = []
    assets = fleet.s.assets
    reinspect = cfg.actions.get("reinspect")
    for aid, inc in fleet.incidents.items():
        if aid not in assets or aid in busy_assets:
            continue
        avoid = inc.last_inspector if (reinspect and reinspect.prefer_different_robot) else ""
        if inc.status == _OPEN and inc.needs_inspection and "inspect" in cfg.actions:
            kind = inc.alarm_kind or "sensor"
            tasks.append(DueTask(f"inspect:{aid}", "inspect", aid, aid, f"{kind} alarm on {aid}: go and read the gauge",
                                 inc.priority or "medium", inc.opened_at, inc.incident_id))
        elif inc.status == _CONFIRMED and "pickup_tool" in cfg.actions:
            tasks.append(DueTask(f"pickup_tool:{aid}", "pickup_tool", cfg.actions["pickup_tool"].target_id, aid,
                                 f"pressure out of limits on {aid}: load repair tools onto the AMR", inc.priority or "high",
                                 inc.opened_at, inc.incident_id))
        elif inc.status == _VERIFYING and reinspect:
            tasks.append(DueTask(f"reinspect:{aid}", "reinspect", aid, aid,
                                 f"repair completed on {aid}: confirm pressure is back within limits", inc.priority or "high",
                                 inc.opened_at, inc.incident_id, prefer_not=avoid))
        elif inc.status == _OPEN and inc.needs_reinspect and reinspect:
            tasks.append(DueTask(f"reinspect:{aid}", "reinspect", aid, aid,
                                 f"last reading on {aid} was unreadable, low confidence or unconfirmed", inc.priority or "medium",
                                 inc.opened_at, inc.incident_id, prefer_not=avoid))
    if "inspect" in cfg.actions and scheduler.enabled:
        sn = now if sched_now is None else sched_now
        for aid in assets:
            if aid in fleet.incidents or aid in busy_assets:
                continue
            due_in = scheduler.due_in(aid, sn)
            if due_in <= 0:
                tasks.append(DueTask(f"inspect:{aid}", "inspect", aid, aid, "routine inspection due", "low", now + due_in))
    return tasks


def target_pose(cfg, fleet, task: DueTask):
    if cfg.actions[task.action].target_kind == "location":
        return cfg.location_pose(task.target_id)
    asset = fleet.s.assets.get(task.target_id)
    return tuple(asset.approach_pose) if asset is not None else None


def robot_eligible(cfg, robot, action: str, busy_robots: set, ttl_s: float, homing=()) -> tuple:
    spec = cfg.actions[action]
    if getattr(robot, "control", "") == "mirror":  # v0.5.6
        return False, "real robot in mirror mode (driven by its remote / JOG, not by tasks)"
    key = getattr(getattr(robot, "model", None), "key", "")
    if not cfg.can(key, action):
        return False, f"model {key or '?'} not configured for {action}"
    if robot.robot_id in busy_robots:
        return False, "already running or reserved for a task"
    status = "idle" if (robot.robot_id in homing and robot.status == "busy") else robot.status  # walking home = available
    if status != "idle":
        return False, f"status {status}"
    if robot.telemetry_age() > ttl_s:
        return False, "telemetry stale"
    if getattr(robot, "mode", "") == "digital_twin":
        if not spec.real_robot_allowed:
            return False, f"{action} not allowed on real robots"
        if cfg.policy.get("real_robot_autonomy", "approve_each") == "off":
            return False, "real-robot autonomy is off"
    return True, ""


def attach_candidates(cfg, fleet, tasks: list, busy_robots: set) -> list:
    ttl = fleet.s.telemetry_ttl_s
    homing = getattr(fleet, "homing", ())
    for t in tasks:
        pose = target_pose(cfg, fleet, t)
        cands = []
        for r in fleet.robots.values():
            ok, _ = robot_eligible(cfg, r, t.action, busy_robots, ttl, homing)
            if ok and pose is not None:
                cands.append((round(math.dist(r.pose[:2], pose[:2]), 2), r.robot_id))
        cands.sort()
        if t.prefer_not and len(cands) > 1:
            cands.sort(key=lambda c: (c[1] == t.prefer_not, c[0]))
        t.candidates = cands
    return tasks


def explain_no_candidates(cfg, fleet, task: DueTask, busy_robots: set) -> str:
    """Why nobody can take a task (shown in the audit log and the incident note)."""
    ttl = fleet.s.telemetry_ttl_s
    homing = getattr(fleet, "homing", ())
    if not any(cfg.can(k, task.action) for k in cfg.models):
        return f"no available robot model is configured for {task.action} (see robots_actions.json)"
    reasons = []
    for r in fleet.robots.values():
        ok, why = robot_eligible(cfg, r, task.action, busy_robots, ttl, homing)
        if not ok:
            reasons.append(f"{r.robot_id}: {why}")
    return "; ".join(reasons[:4]) or f"no {cfg.actions[task.action].robot_types[0]} robot is in the fleet"


def build_snapshot(cfg, fleet, tasks: list, scheduler, active: dict, trigger: dict, recent, now: float | None = None,
                   sched_now: float | None = None) -> dict:
    now = time.time() if now is None else now
    sn = now if sched_now is None else sched_now
    ttl = fleet.s.telemetry_ttl_s
    homing = getattr(fleet, "homing", ())
    robots, modes = [], set()
    for r in fleet.robots.values():
        modes.add(r.mode)
        key = getattr(getattr(r, "model", None), "key", "")
        returning = r.robot_id in homing and r.status == "busy"
        robots.append({
            "robot_id": r.robot_id, "model": key, "robot_type": r.robot_type,
            "provenance": "real" if r.mode == "digital_twin" else "simulated",
            "status": "idle" if returning else r.status, "returning_home": returning, "busy_with": active.get(r.robot_id),
            "pose_xy": [round(float(r.pose[0]), 2), round(float(r.pose[1]), 2)],
            "telemetry_age_s": round(float(r.telemetry_age()), 1), "fresh": r.telemetry_age() <= ttl,
            "configured_actions": list(cfg.actions_for_model(key)), "note": (r.note or "")[:80],
        })
    assets = []
    for aid, a in fleet.s.assets.items():
        inc = fleet.incidents.get(aid)
        last = scheduler.last.get(aid)
        assets.append({
            "asset_id": aid, "position_xy": [round(float(a.position[0]), 2), round(float(a.position[1]), 2)],
            "gauge_limits": [getattr(a, "normal_low", None), getattr(a, "normal_high", None), getattr(a, "unit", "")],
            "incident": inc.status if inc else None,
            "incident_priority": (inc.priority if inc else None),
            "alarm": (inc.alarm_kind or None) if inc else None,
            "last_inspected_s_ago": None if last is None else round(sn - last),
            "inspection_due_in_s": round(max(scheduler.due_in(aid, sn), 0.0)) if scheduler.enabled else None,
        })
    keep = int(cfg.policy.get("recent_decisions_in_snapshot", 5))
    return {
        "schema_version": "1.0",
        "snapshot_id": f"SNAP-{uuid.uuid4().hex[:10]}",
        "created_at": utc_now_iso(),
        "created_ts": now,
        "valid_for_s": float(cfg.nemotron.get("snapshot_valid_for_s", 90)),
        "mode": "mixed" if len(modes) > 1 else (next(iter(modes)) if modes else "simulation"),
        "trigger": trigger,
        "robots": robots,
        "assets": assets,
        "locations": {k: {"label": v["label"], "pose_xy": list(v["pose"][:2])} for k, v in cfg.locations.items()},
        "active_tasks": [{"robot_id": rid, "task": t} for rid, t in active.items()],
        "due_tasks": [{
            "task_id": t.task_id, "action": t.action, "target_id": t.target_id, "for_asset": t.for_asset,
            "reason": t.reason, "priority": t.priority, "waiting_s": round(max(now - t.due_since, 0.0)),
            "candidates": [{"robot_id": rid, "distance_m": d} for d, rid in t.candidates],
        } for t in tasks],
        "actions": cfg.catalogue_for_prompt(),
        "policy": {"assignments_per_call": 1, "min_confidence": fleet.s.min_confidence,
                   "real_robot_autonomy": cfg.policy.get("real_robot_autonomy", "approve_each"),
                   "routine_inspections": scheduler.enabled},
        "recent_decisions": list(recent)[-keep:] if keep > 0 else [],
    }
