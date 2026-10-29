# SPDX-License-Identifier: Apache-2.0
"""Fleet coordinator with Nemotron as the central brain (v0.4.3).

  alarms / events / schedule / operator buttons -> incident state -> due tasks + eligible robots -> fleet snapshot
  -> FleetBrain (Nemotron, worker thread, ONE decision) -> dispatch-time re-check -> safety gate
  -> (operator approval) -> skill executor (deterministic, per robot) -> result event -> incident state ...

v0.4.3
  * A quadruped that finishes an inspection / re-inspection walks back to its home pose (config: return_home_after).
    While it walks it is reported as idle and can be given a new task at once; the new goal replaces the walk home.
  * reset_tasks(): everything left over from an earlier run is cleared (incidents, queued events, approvals, running
    skills, blocked pairs, routine timers) and an answer Nemotron is still computing for the old run is discarded.
    Called when the world is reset / the timeline stops, and when RUN starts a fresh run.
  * Routine inspections are opt-in and counted in RUNNING time (paused / stopped time does not count).
  * The "alarm_raised" audit record no longer loses its name (its payload field "kind" is now "alarm_kind").
v0.5.5
  * A robot whose task failed (timeout, navigation timeout, stuck) walks home instead of standing where it stopped.
  * A robot that got STUCK (not fallen) is released once and walks home; only if that fails too it waits for the operator.
  * Before a robot is sent to a spot, idle robots standing within CLEAR_SPOT_M of it are sent home.
  * Robot events (robot_woke from a parked robot) are written to the audit log.
Pure Python (no omni imports) so it runs headless in tests and tools/run_headless.py.
`tick(dt)` must be called from one thread (the Isaac Sim physics step).
"""
from __future__ import annotations

import math
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..common.audit_log import AuditLog
from ..fleet_config.action_config import get_action_config
from ..fleet_config.settings import get_settings
from ..perception.cosmos_reason import CosmosReason
from ..perception.event_builder import PerceptionPipeline
from ..perception.frame_capture import EvidenceStore
from ..reasoning.fleet_brain import FleetBrain
from ..reasoning.nemotron_assigner import NemotronAssigner
from ..reasoning.snapshot_builder import (Scheduler, attach_candidates, build_snapshot, compute_due_tasks,
                                          explain_no_candidates)
from ..reasoning.token_factory_client import TokenFactoryClient
from ..robots.base import BUSY
from ..safety.approval import ApprovalQueue
from ..safety.gate import APPROVED, PENDING, SafetyGate
from ..schemas.fleet_event import FleetEvent
from ..schemas.task_assignment import TaskAssignment
from ..schemas.validate import SchemaError
from ..skills.executor import SkillExecutor
from ..skills.verifier import Verifier
from .fleet_state import (AWAITING_REPAIR, CONFIRMED, NEEDS_OPERATOR, OPEN, TOOLS_DISPATCHED, TOOLS_LOADED,
                          TOOLS_PICKUP, VERIFYING, FleetState)

MAX_ASSIGNMENTS_PER_INCIDENT = 6
CHECK_PERIOD_S = 0.5
SEVERITIES = ("low", "medium", "high")
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}
PAUSE_GAP_S = 5.0  # a gap between two ticks longer than this is a pause / stop: it does not count as running time
HOME_TOL_M = 0.35  # already this close to home: no walk home needed
CLEAR_SPOT_M = 1.2  # v0.5.5: idle robots this close to a robot's target are sent home first


class FleetCoordinator:
    def __init__(self, settings=None, capture_fn=None, fault_fn=None, action_config=None) -> None:
        self.s = settings or get_settings()
        self.acfg = action_config or get_action_config()
        self._clock = time.time  # replaceable in tests
        self._run_s = 0.0  # RUNNING time of the current run (advances only while tick() is being called)
        self._last_tick_t = None
        self._epoch = 0  # bumped by reset_tasks(): answers computed for an older epoch are never executed
        self.audit = AuditLog(self.s.log_dir)
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fleet")
        self.fleet = FleetState(self.s)
        self.gate = SafetyGate(self.s, self.acfg)
        self.approvals = ApprovalQueue(self.audit)
        self.verifier = Verifier(self.s)
        self.nebius = TokenFactoryClient(self.s.nebius_base_url, self.s.nebius_api_key, self.s.request_timeout_s, audit=self.audit)
        self.cosmos_client = TokenFactoryClient(self.s.cosmos_base_url, self.s.cosmos_api_key, self.s.request_timeout_s, audit=self.audit)
        self.assigner = NemotronAssigner(self.nebius, self.s.nemotron_model_id, self.audit)
        self.brain = FleetBrain(self.nebius, self.s.nemotron_model_id, self.acfg, self.audit,
                                out_dir=Path(self.s.log_dir) / "nemotron", enabled_fn=lambda: self.assigner.use_nemotron)
        self.cosmos = CosmosReason(self.cosmos_client, self.s.cosmos_model_id)
        self.perception = PerceptionPipeline(self.s, self.cosmos, EvidenceStore(self.s.log_dir), self.audit)
        self.capture_fn = capture_fn or (lambda robot: None)
        self.fault_fn = fault_fn or (lambda asset_id: None)
        self.executor = SkillExecutor(self.s, self.pool, self.perception, lambda r: self.capture_fn(r),
                                      lambda a: self.fault_fn(a), self.audit, self._on_skill_done)
        self.scheduler = Scheduler(self.acfg, list(self.s.assets), clock=lambda: self._run_s)
        self._home_after = self.acfg.home_after_skills()
        self.auto_dispatch = True
        self._inbox: queue.Queue = queue.Queue()
        self._events: dict = {}
        self._error_reported: set = set()
        trig = self.acfg.nemotron.get("triggers", {})
        self._on_events = bool(trig.get("on_events", True))
        self._periodic_s = float(trig.get("periodic_s", 60))
        self._min_interval_s = float(trig.get("min_interval_s", 5))
        self._brain_future = None
        self._brain_snapshot = None
        self._last_decision = None
        self._last_call_t = -1e18
        self._last_eval_t = -1e18
        self._next_check = 0.0
        self._wake_flag = False
        self._wake_reasons: set = set()
        self._robots_seen: set = set()
        self._recent: list = []
        self._meta: dict = {}
        self._blocked: dict = {}
        self._waiting_noted: set = set()
        self._delivered: dict = {}  # v0.5.4: asset_id -> AMR that delivered the tools there (sent home after the repair)
        self._home_retry: set = set()  # v0.5.5: robots released once after getting stuck (a second stuck needs the operator)
        self.executor.clear_spot_fn = self._clear_spot
        self.audit.record("coordinator_started", {"nemotron": self.s.nemotron_enabled, "cosmos": self.s.cosmos_enabled,
                                                  "nemotron_model": self.s.nemotron_model_id,
                                                  "nebius_base_url": self.s.nebius_base_url, "log": str(self.audit.path),
                                                  "routine_inspections": self.scheduler.enabled,
                                                  "actions_config": self.acfg.path, "actions": sorted(self.acfg.actions)})

    # ------------------------------------------------------------------ configuration at run time
    def refresh_credentials(self) -> None:
        s = self.s
        self.nebius.api_key, self.nebius.base_url = s.nebius_api_key, s.nebius_base_url.rstrip("/")
        self.cosmos_client.api_key, self.cosmos_client.base_url = s.cosmos_api_key, s.cosmos_base_url.rstrip("/")
        self.assigner.model_id = s.nemotron_model_id
        self.brain.set_model(s.nemotron_model_id)
        self.cosmos.model_id = s.cosmos_model_id
        self.audit.record("credentials_updated", {"nemotron": s.nemotron_enabled, "model": self.brain.model_id,
                                                  "cosmos": s.cosmos_enabled})
        self._wake_up("credentials_updated")

    def set_routine(self, enabled: bool, interval_s: float | None = None) -> None:
        """Routine inspections on / off, every `interval_s` seconds of RUNNING time."""
        self.scheduler.set_routine(enabled, interval_s)
        self.audit.record("routine_changed", {"enabled": self.scheduler.enabled, "interval_s": self.scheduler.interval_s})
        self._wake_up("routine_changed")

    def on_assets_changed(self) -> None:
        self.scheduler.ensure(self.s.assets.keys())
        for aid in [a for a in self.scheduler.last if a not in self.s.assets]:
            del self.scheduler.last[aid]
        self._wake_up("assets_changed")

    def asset_in_use(self, asset_id: str) -> bool:
        return self.fleet.incident_for(asset_id) is not None or asset_id in self._busy_assets()

    def force_evaluate(self) -> str:
        if not self.auto_dispatch:
            return "Auto Dispatch is off - turn it on first."
        if self._brain_future is not None:
            return "Nemotron is already working on a decision."
        self._wake_up("operator_request")
        self._last_call_t = -1e18
        self._next_check = 0.0
        return "Nemotron will be asked on the next step."

    # ------------------------------------------------------------------ a clean start
    def reset_tasks(self, reason: str = "reset") -> dict:
        """Forget everything left over from an earlier run. Robots are NOT moved (see on_world_reset)."""
        had = {"incidents": len(self.fleet.incidents), "running": len(self.executor.runs), "queued_events": self._inbox.qsize(),
               "approvals": len(self.approvals.pending()), "reserved": len(self._meta),
               "decision_in_flight": self._brain_future is not None, "homing": len(self.fleet.homing)}
        for rid in list(self.executor.runs):
            self.executor.cancel(rid, reason)
        self._meta.clear()
        self._blocked.clear()
        self._recent.clear()
        self._events.clear()
        self._waiting_noted.clear()
        self._error_reported.clear()
        self._delivered.clear()
        self._home_retry.clear()
        while True:
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                break
        for a in list(self.approvals.pending()):
            self.approvals.resolve(a.assignment_id, False, operator=reason)
        self.fleet.incidents.clear()
        self.fleet.closed.clear()
        self.fleet.homing.clear()
        self.verifier.attempts.clear()
        self._epoch += 1
        self._brain_future = None  # a model answer still being computed belongs to the old run: it is never executed
        self._brain_snapshot = None
        self._last_decision = None
        self._wake_flag = False
        self._wake_reasons.clear()
        self._last_call_t = self._last_eval_t = -1e18
        self._next_check = 0.0
        self._run_s = 0.0
        self._last_tick_t = None
        self.scheduler.restart(self.s.assets.keys())
        self.audit.record("tasks_reset", {"reason": reason, "cleared": had})
        return had

    # ------------------------------------------------------------------ inputs
    def submit_event(self, event) -> bool:
        try:
            ev = event if (isinstance(event, FleetEvent) or hasattr(event, "event_id")) else FleetEvent.from_dict(event)
        except SchemaError as exc:
            self.audit.record("event_rejected", {"error": str(exc)})
            return False
        self._inbox.put(ev)
        return True

    def report_alarm(self, asset_id: str, kind: str = "overpressure", severity: str = "medium", source: str = "sensor-gw") -> str:
        if asset_id not in self.s.assets:
            return f"Unknown asset '{asset_id}'."
        severity = severity if severity in SEVERITIES else "medium"
        inc = self.fleet.incident_for(asset_id)
        new = inc is None
        inc = inc or self.fleet.open_incident(asset_id)
        if new or inc.priority is None or SEVERITY_ORDER[severity] > SEVERITY_ORDER[inc.priority]:
            inc.priority = severity
        inc.alarm_kind = kind
        if inc.status == OPEN:
            inc.needs_inspection = True
        inc.note(f"{source}: {kind} alarm, priority {inc.priority}")
        self.audit.record("alarm_raised", {"asset_id": asset_id, "alarm_kind": kind, "priority": inc.priority, "source": source,
                                           "incident_id": inc.incident_id, "incident_status": inc.status})
        self._wake_up(f"alarm:{asset_id}")
        if inc.status == OPEN:
            return f"Alarm on {asset_id} ({kind}, priority {inc.priority}): an inspection will be assigned."
        return f"Alarm on {asset_id} noted (priority {inc.priority}); incident is already {inc.status}."

    def operator_dispatch(self, robot_id: str, skill: str, asset_id: str | None) -> str:
        try:
            a = TaskAssignment.from_model_dict(
                {"event_id": "OPERATOR", "robot_id": robot_id, "skill": skill, "target_asset_id": asset_id or None,
                 "priority": "medium", "rationale": "Manual operator command from the extension UI."}, source="operator")
        except SchemaError as exc:
            self.audit.record("operator_command_rejected", {"error": str(exc)})
            return f"rejected: {exc}"
        return self._gate_and_dispatch(a, None, None)

    def approve(self, assignment_id: str, approved: bool) -> str:
        a, ev = self.approvals.resolve(assignment_id, approved)
        if a is None:
            return "nothing to approve"
        meta = self._meta.get(a.assignment_id, {})
        if not approved:
            self._meta.pop(a.assignment_id, None)
            self._block(meta)
            inc = self.fleet.incident_for(a.target_asset_id or "")
            if inc:
                inc.status = NEEDS_OPERATOR
                inc.note(f"operator rejected {a.skill}")
            self._wake_up("approval_rejected")
            return "rejected by operator"
        decision = self._check(a, ev, meta)
        if decision.status == PENDING:
            return self._dispatch(a, ev)
        self._meta.pop(a.assignment_id, None)
        self.audit.record("gate_decision", {"assignment_id": a.assignment_id, **decision.to_dict(), "after_approval": True})
        self._wake_up("approval_gate_rejected")
        return decision.status

    def mark_repaired(self, asset_id: str) -> str:
        inc = self.fleet.incident_for(asset_id)
        if inc is None:
            return "no open incident"
        inc.status = VERIFYING
        inc.needs_reinspect = False
        inc.needs_inspection = False
        inc.unreadable = 0
        inc.waiting_note = ""
        inc.note("operator marked repaired")
        self.audit.record("repair_reported", {"incident_id": inc.incident_id, "asset_id": asset_id})
        amr = self.fleet.get(self._delivered.pop(asset_id, "") or "")
        if amr is not None:  # v0.5.4: the repair is done, the tool carrier is no longer needed at the asset
            self._send_home(amr, "repair_done")
        self._wake_up(f"repair_completed:{asset_id}")
        return "repair recorded - a re-inspection will be assigned"

    def mark_tools_loaded(self, asset_id: str) -> str:
        inc = self.fleet.incident_for(asset_id)
        if inc is None:
            return "no open incident"
        if inc.status not in (CONFIRMED, TOOLS_PICKUP, NEEDS_OPERATOR):
            return f"incident is {inc.status}, tools cannot be marked as loaded"
        inc.status = TOOLS_LOADED
        inc.waiting_note = ""
        inc.note("operator marked tools loaded on the AMR")
        self.audit.record("tools_loaded", {"incident_id": inc.incident_id, "asset_id": asset_id, "by": "operator"})
        self._wake_up(f"tools_loaded:{asset_id}")
        return "tools marked as loaded - the AMR will deliver them"

    def reset_robot(self, robot_id: str) -> str:
        robot = self.fleet.get(robot_id)
        if robot is None:
            return "unknown robot"
        self.executor.cancel(robot_id, "operator reset")
        self._drop_meta(robot_id)
        self.fleet.homing.discard(robot_id)
        self._home_retry.discard(robot_id)
        robot.reset_to_home()
        self._error_reported.discard(robot_id)
        self.audit.record("robot_reset", {"robot_id": robot_id})
        self._wake_up(f"robot_reset:{robot_id}")
        return f"{robot_id} reset to home"

    def on_world_reset(self) -> None:
        """World reset / timeline stop: robots back home AND every task of the old run forgotten."""
        self.reset_tasks("world reset")
        for r in self.fleet.robots.values():
            if hasattr(r, "on_world_reset"):
                r.on_world_reset()
            elif r.mode == "simulation":
                r.reset_to_home()
        self._wake_up("world_reset")

    def stop_all(self) -> None:
        for rid in list(self.executor.runs):
            self.executor.cancel(rid, "operator stop")
        self._meta.clear()
        self.fleet.homing.clear()
        for r in self.fleet.robots.values():
            r.stop()
        self.audit.record("stop_all", {})

    # ------------------------------------------------------------------ main loop
    def tick(self, dt: float) -> None:
        now = self._clock()
        if self._last_tick_t is not None and 0.0 <= now - self._last_tick_t <= PAUSE_GAP_S:
            self._run_s += now - self._last_tick_t  # only time that really ran counts
        self._last_tick_t = now
        for robot in list(self.fleet.robots.values()):
            robot.update(dt)
            pop = getattr(robot, "pop_events", None)
            if callable(pop):
                for kind, payload in pop():
                    self.audit.record(kind, payload)
            if robot.status == "error" and robot.robot_id not in self._error_reported:
                self._error_reported.add(robot.robot_id)
                self.audit.record("robot_error", {"robot_id": robot.robot_id, "note": robot.note})
                if robot.robot_id not in self.executor.runs:  # stuck on its own walk (e.g. home): try once to recover
                    self._recover_and_home(robot, "stuck")
        self._end_homing()
        self.executor.tick(dt)
        while True:
            try:
                ev = self._inbox.get_nowait()
            except queue.Empty:
                break
            self._handle_event(ev)
        self._brain_tick()

    @property
    def busy(self) -> bool:
        return bool(self.executor.runs or self._brain_future is not None or not self._inbox.empty())

    def shutdown(self) -> None:
        for rid in list(self.fleet.robots):
            self.fleet.remove(rid)
        self.pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ walking home after a task
    def _send_home(self, robot, after_skill: str) -> None:
        """The robot walks home but stays available: any new task simply replaces this goal."""
        try:
            if robot.status == "error" or robot.robot_id in self.executor.runs:
                return
            hx, hy = robot.home_pose[0], robot.home_pose[1]
            if math.dist(tuple(robot.pose[:2]), (hx, hy)) < HOME_TOL_M:
                return
            robot.go_to(*robot.home_pose, name="home")
            if robot.status == BUSY:
                self.fleet.homing.add(robot.robot_id)
                self.audit.record("returning_home", {"robot_id": robot.robot_id, "after": after_skill})
            else:
                self.audit.record("return_home_failed", {"robot_id": robot.robot_id, "note": robot.note})
        except Exception as exc:  # noqa: BLE001 - a failed walk home must never break the mission flow
            self.audit.record("return_home_failed", {"robot_id": getattr(robot, "robot_id", "?"), "error": str(exc)[:160]})

    def _end_homing(self) -> None:
        for rid in list(self.fleet.homing):
            r = self.fleet.robots.get(rid)
            if r is None or r.status != BUSY or rid in self.executor.runs:
                self.fleet.homing.discard(rid)
                if r is not None and r.status != "error":
                    self._home_retry.discard(rid)  # it made it (or got a new task): a later stuck may be released again

    def _recover_and_home(self, robot, after: str) -> None:
        """v0.5.5: after a failed task the robot walks home. A robot in error only because it is STUCK is released once
        first; fallen robots and a second stuck in a row stay in error for the operator (RESET ROBOT)."""
        rid = robot.robot_id
        if robot.status == "error":
            clear = getattr(robot, "clear_error", None)
            if not str(robot.note).startswith("stuck") or rid in self._home_retry or not callable(clear):
                return
            note = robot.note
            if not clear():
                return
            self._home_retry.add(rid)
            self._error_reported.discard(rid)
            self.audit.record("robot_recovering", {"robot_id": rid, "was": note, "action": "walking home"})
        self._send_home(robot, after)

    def _clear_spot(self, xy, robot_id: str) -> None:
        """Send idle robots that stand on (or next to) another robot's target home first."""
        for r in list(self.fleet.robots.values()):
            if r.robot_id == robot_id or r.robot_id in self.executor.runs or r.robot_id in self._busy_robots():
                continue
            if getattr(r, "mode", "") == "digital_twin":
                continue  # v0.5.6: a real robot is never moved to make room
            if math.dist(tuple(r.pose[:2]), tuple(xy)) >= CLEAR_SPOT_M:
                continue
            if r.status == "idle" or (r.status == "error" and str(r.note).startswith("stuck")):
                self.audit.record("spot_cleared", {"robot_id": r.robot_id, "for": robot_id,
                                                   "spot": [round(float(v), 2) for v in xy]})
                self._recover_and_home(r, f"clear spot for {robot_id}")

    # ------------------------------------------------------------------ brain triggers
    def _wake_up(self, reason: str) -> None:
        self._wake_flag = True
        if len(self._wake_reasons) < 12:
            self._wake_reasons.add(reason)

    def _brain_tick(self) -> None:
        now = self._clock()
        if self._brain_future is not None:
            if not self._brain_future.done():
                return
            fut, snap = self._brain_future, self._brain_snapshot
            self._brain_future = self._brain_snapshot = None
            try:
                decision = fut.result()
            except Exception as exc:  # noqa: BLE001
                self.audit.record("assigner_error", {"error": str(exc)})
                return
            self._on_decision(decision, snap, now)
            return
        if not self.auto_dispatch or now < self._next_check:
            return
        self._next_check = now + CHECK_PERIOD_S
        seen = set(self.fleet.robots)
        if seen != self._robots_seen:
            self._robots_seen = seen
            if seen:
                self._wake_up("robots_changed")
        self._expire_pending(now)
        self._run_follow_ups(now)
        event_due = self._on_events and self._wake_flag and now - self._last_call_t >= self._min_interval_s
        periodic_due = now - self._last_eval_t >= self._periodic_s
        if not (event_due or periodic_due):
            return
        self._last_eval_t = now
        tasks = self._due_tasks(now)
        ready = [t for t in tasks if t.candidates]
        self._note_waiting(tasks, now)
        reasons = sorted(self._wake_reasons) if event_due else ["periodic"]
        self._wake_flag = False
        self._wake_reasons.clear()
        if not ready:
            return
        trigger = {"kind": "event" if event_due else "periodic", "reasons": reasons[:6], "due_tasks": len(ready)}
        active = {m["robot"]: f"{m.get('action') or m.get('follow_up') or 'task'} on {m.get('asset')}"
                  for m in self._meta.values() if m.get("state") == "running"}
        snapshot = build_snapshot(self.acfg, self.fleet, ready, self.scheduler, active, trigger, self._recent, now,
                                  sched_now=self._run_s)
        self._last_call_t = now
        self._brain_snapshot = snapshot
        self.audit.record("snapshot_sent", {"snapshot_id": snapshot["snapshot_id"], "robots": len(snapshot["robots"]),
                                            "due_tasks": [f"{t.task_id}[{t.priority}]" for t in ready], "trigger": trigger})
        self._brain_future = self.pool.submit(self.brain.decide, snapshot)

    def _due_tasks(self, now: float) -> list:
        self.scheduler.ensure(self.s.assets.keys())
        tasks = compute_due_tasks(self.acfg, self.fleet, self.scheduler, self._busy_assets(), now, sched_now=self._run_s)
        attach_candidates(self.acfg, self.fleet, tasks, self._busy_robots())
        for t in tasks:
            t.candidates = [c for c in t.candidates if self._blocked.get((t.task_id, c[1]), 0.0) <= now]
        return tasks

    def _busy_robots(self) -> set:
        return set(self.executor.runs) | {m["robot"] for m in self._meta.values()}

    def _busy_assets(self) -> set:
        return {m["asset"] for m in self._meta.values() if m.get("asset")}

    def _note_waiting(self, tasks: list, now: float) -> None:
        keys = set()
        for t in tasks:
            if t.candidates:
                continue
            keys.add(t.task_id)
            if t.task_id in self._waiting_noted:
                continue
            self._waiting_noted.add(t.task_id)
            why = explain_no_candidates(self.acfg, self.fleet, t, self._busy_robots())
            inc = self.fleet.incident_for(t.for_asset)
            if inc is not None:
                inc.waiting_note = f"waiting: {why}"[:120]
                inc.note(inc.waiting_note)
            self.audit.record("task_waiting", {"task_id": t.task_id, "action": t.action, "reason": why})
        self._waiting_noted &= keys

    # ------------------------------------------------------------------ decisions
    def _remember(self, decision, outcome: str) -> None:
        self._recent.append({"task_id": decision.task_id, "robot_id": decision.robot_id, "outcome": outcome,
                             "source": decision.source})
        del self._recent[:-20]

    def _block(self, meta: dict) -> None:
        if meta.get("task_id") and meta.get("robot"):
            self._blocked[(meta["task_id"], meta["robot"])] = self._clock() + float(self.acfg.policy.get("retry_block_s", 120))

    def _on_decision(self, decision, snapshot: dict, now: float) -> None:
        self._last_decision = decision
        self.audit.record("fleet_decision", {"decision": decision.to_dict(), "snapshot_id": snapshot["snapshot_id"]})
        if decision.decision == "no_action":
            self._remember(decision, "no_action")
            return
        age = now - snapshot["created_ts"]
        if age > snapshot["valid_for_s"]:
            self._remember(decision, "stale")
            self.audit.record("decision_dropped", {"decision_id": decision.decision_id, "reason": f"snapshot is {age:.0f}s old"})
            self._wake_up("stale_decision")
            return
        task = next((t for t in self._due_tasks(now) if t.task_id == decision.task_id), None)
        if task is None or decision.robot_id not in [c[1] for c in task.candidates]:
            self._remember(decision, "state_changed")
            self.audit.record("decision_dropped", {"decision_id": decision.decision_id, "reason": "task or robot no longer eligible"})
            self._wake_up("state_changed")
            return
        aspec = self.acfg.actions[task.action]
        inc = self.fleet.incident_for(task.for_asset)
        if inc is not None and inc.assignments >= MAX_ASSIGNMENTS_PER_INCIDENT:
            inc.status = NEEDS_OPERATOR
            inc.note("too many automatic assignments - operator needed")
            self.audit.record("incident_escalated", {"incident_id": inc.incident_id})
            self._remember(decision, "escalated")
            return
        a = TaskAssignment(event_id=decision.decision_id, robot_id=decision.robot_id, skill=aspec.executor_skill,
                           target_asset_id=task.for_asset, priority=decision.priority, rationale=decision.rationale[:400],
                           source=decision.source)
        self.audit.record("task_assignment", {"assignment": a.to_dict(), "action": task.action, "task_id": task.task_id})
        meta = {"action": task.action, "task_id": task.task_id, "asset": task.for_asset, "robot": a.robot_id,
                "kind": "decision", "state": "pending", "since": now}
        result = self._gate_and_dispatch(a, None, inc, action=task.action, meta=meta)
        outcome = "dispatched" if result.startswith("dispatched") else ("pending_approval" if result.startswith("pending") else "rejected")
        self._remember(decision, outcome)
        if outcome == "rejected":
            self._block(meta)
            self._wake_up("gate_rejected")
        else:
            self._wake_up("task_dispatched")

    def _check(self, a, ev, meta: dict):
        kind = meta.get("kind")
        return self.gate.check(a, ev, self.fleet, action=meta.get("action") if kind == "decision" else None,
                               follow_up=meta.get("follow_up") if kind == "follow_up" else None)

    def _gate_and_dispatch(self, a: TaskAssignment, ev, inc, action=None, follow_up=None, meta=None) -> str:
        decision = self.gate.check(a, ev, self.fleet, action=action, follow_up=follow_up)
        self.audit.record("gate_decision", {"assignment_id": a.assignment_id, "skill": a.skill, "robot_id": a.robot_id,
                                            **decision.to_dict()})
        if decision.status == PENDING:
            if meta is not None:
                meta["state"] = "pending"
                self._meta[a.assignment_id] = meta
            self.approvals.submit(a, ev)
            return "pending operator approval"
        if decision.status != APPROVED:
            if inc:
                inc.note(f"gate rejected {a.skill}: {'; '.join(decision.reasons)}")
            return "rejected: " + "; ".join(decision.reasons)
        if meta is not None:
            self._meta[a.assignment_id] = meta
        return self._dispatch(a, ev)

    def _dispatch(self, a: TaskAssignment, ev) -> str:
        robot = self.fleet.get(a.robot_id)
        inc = self.fleet.incident_for(a.target_asset_id or "")
        purpose = "verify" if inc and inc.status == VERIFYING and a.skill == "reinspect_asset" else ""
        meta = self._meta.setdefault(a.assignment_id, {"action": "", "asset": a.target_asset_id, "robot": a.robot_id,
                                                       "kind": "operator", "since": self._clock()})
        meta["state"] = "running"
        self.fleet.homing.discard(a.robot_id)  # the new task replaces the walk home
        try:
            self.executor.start(a, robot, purpose, inc.incident_id if inc else None)
        except Exception as exc:  # noqa: BLE001
            self._meta.pop(a.assignment_id, None)
            self.audit.record("dispatch_failed", {"assignment_id": a.assignment_id, "error": str(exc)[:200]})
            return f"rejected: executor error: {exc}"
        if inc:
            if meta.get("kind") != "operator":
                inc.assignments += 1
            if a.skill == "pick_tool":
                inc.status = TOOLS_PICKUP
            elif a.skill == "deliver_tools":
                inc.status = TOOLS_DISPATCHED
            inc.waiting_note = ""
            inc.note(f"{a.robot_id} -> {a.skill} ({a.source})")
        return f"dispatched {a.skill} to {a.robot_id}"

    def _expire_pending(self, now: float) -> None:
        limit = float(self.acfg.policy.get("approval_timeout_s", 300))
        for aid, m in list(self._meta.items()):
            if m.get("state") == "pending" and now - m.get("since", now) > limit:
                try:
                    self.approvals.resolve(aid, False)
                except Exception:  # noqa: BLE001
                    pass
                self._meta.pop(aid, None)
                self._block(m)
                self.audit.record("approval_expired", {"assignment_id": aid, "task_id": m.get("task_id")})
                self._wake_up("approval_expired")

    def _drop_meta(self, robot_id: str) -> None:
        for aid in [k for k, m in self._meta.items() if m.get("robot") == robot_id]:
            self._meta.pop(aid, None)

    # ------------------------------------------------------------------ automatic AMR follow-up
    def _run_follow_ups(self, now: float) -> None:
        for fu in self.acfg.follow_ups.values():
            if fu.after != "tools_loaded":
                continue
            for aid, inc in list(self.fleet.incidents.items()):
                if inc.status != TOOLS_LOADED or aid in self._busy_assets() or aid not in self.s.assets:
                    continue
                task_id = f"{fu.name}:{aid}"
                robot = self._pick_carrier(fu, aid, task_id, now)
                if robot is None:
                    if inc.waiting_note != "waiting for an idle AMR":
                        inc.waiting_note = "waiting for an idle AMR"
                        inc.note(inc.waiting_note)
                        self.audit.record("task_waiting", {"task_id": task_id, "reason": "no idle AMR"})
                    continue
                a = TaskAssignment(event_id=f"FOLLOWUP:{inc.incident_id}", robot_id=robot.robot_id,
                                   skill=fu.executor_skill, target_asset_id=aid, priority="high",
                                   rationale=f"Automatic: tools loaded, {robot.robot_id} delivers them to {aid}.",
                                   source="rule_based")
                self.audit.record("task_assignment", {"assignment": a.to_dict(), "follow_up": fu.name})
                meta = {"follow_up": fu.name, "task_id": task_id, "asset": aid, "robot": robot.robot_id,
                        "kind": "follow_up", "state": "pending", "since": now, "action": ""}
                result = self._gate_and_dispatch(a, None, inc, follow_up=fu, meta=meta)
                if result.startswith("rejected"):
                    self._block(meta)

    def _pick_carrier(self, fu, asset_id: str, task_id: str, now: float):
        keys = self.acfg.robots_with_role(fu.robot_role)
        ttl = self.s.telemetry_ttl_s
        busy = self._busy_robots()
        station = next((self.acfg.location_pose(x.target_id) for x in self.acfg.actions.values()
                        if x.target_kind == "location"), None)
        ref = station[:2] if station else tuple(self.s.assets[asset_id].approach_pose[:2])
        best = None
        for r in self.fleet.robots.values():
            key = getattr(getattr(r, "model", None), "key", "")
            if key not in keys or r.robot_id in busy or r.status != "idle" or getattr(r, "mode", "") == "digital_twin":
                continue
            if r.telemetry_age() > ttl or self._blocked.get((task_id, r.robot_id), 0.0) > now:
                continue
            d = math.dist(r.pose[:2], ref)
            if best is None or d < best[0]:
                best = (d, r)
        return best[1] if best else None

    # ------------------------------------------------------------------ events and results
    def _handle_event(self, ev) -> None:
        self._events[ev.event_id] = ev
        if len(self._events) > 500:
            self._events.pop(next(iter(self._events)))
        self.audit.record("fleet_event", {"event": ev.to_dict()})
        min_conf = self.s.min_confidence
        inc = self.fleet.incident_for(ev.asset_id)
        external = ev.robot_id not in self.fleet.robots
        problem = ev.event_type in ("anomaly", "leak_suspected") or (ev.gauge is not None and ev.gauge.in_range is False)
        unreadable = (ev.event_type == "inconclusive" or (ev.gauge is not None and ev.gauge.value is None)
                      or ev.confidence < min_conf)
        if inc is not None and ev.robot_id and not external:
            inc.last_inspector = ev.robot_id
        max_unreadable = int(self.acfg.policy.get("max_unreadable_before_operator", 2))
        self._wake_up(f"event:{ev.event_type}")
        if inc is not None and inc.status == VERIFYING:
            if self.verifier.is_clear(ev):
                self.fleet.close_incident(ev.asset_id)
                inc.note("verification passed - closed")
                self.audit.record("incident_closed", {"incident_id": inc.incident_id, "asset_id": ev.asset_id})
                self._send_everyone_home()
            elif problem and not unreadable:
                inc.status = CONFIRMED
                inc.unreadable = 0
                inc.note("verification failed - still out of limits, repair triggered again")
                self.audit.record("repair_failed_verification", {"incident_id": inc.incident_id})
            else:
                inc.unreadable += 1
                inc.note("verification reading unreadable")
                if inc.unreadable >= max_unreadable:
                    inc.status = NEEDS_OPERATOR
                    inc.note("gauge unreadable twice - operator needed")
            return
        if external and (problem or unreadable):
            self.report_alarm(ev.asset_id, "leak" if ev.event_type == "leak_suspected" else "overpressure",
                              "medium", source=ev.robot_id or "external")
            return
        if not (ev.actionable or problem or unreadable):
            if inc is not None and inc.status == OPEN and not external:
                self.fleet.close_incident(ev.asset_id)
                inc.note("inspection found no problem - closed as false alarm")
                self.audit.record("incident_closed", {"incident_id": inc.incident_id, "reason": "false_alarm"})
            return
        inc = inc or self.fleet.open_incident(ev.asset_id)
        inc.last_event_id = ev.event_id
        inc.last_inspector = ev.robot_id or inc.last_inspector
        ev.incident_id = inc.incident_id
        if not external:
            inc.needs_inspection = False
        if problem and not unreadable:
            inc.problem_reports += 1
            inc.reporters.add(ev.robot_id)
            inc.unreadable = 0
            inc.needs_reinspect = False
            if inc.status == OPEN:
                if inc.problem_reports >= int(self.acfg.policy.get("confirm_reports_required", 1)):
                    inc.status = CONFIRMED
                    inc.note(f"pressure out of limits, confirmed by {sorted(inc.reporters)}")
                    self.audit.record("incident_confirmed", {"incident_id": inc.incident_id, "reporters": sorted(inc.reporters)})
                else:
                    inc.needs_reinspect = True
        elif unreadable and inc.status == OPEN:
            inc.unreadable += 1
            inc.needs_reinspect = True
            if inc.unreadable >= max_unreadable:
                inc.status = NEEDS_OPERATOR
                inc.note("gauge unreadable repeatedly - operator needed")
                self.audit.record("incident_escalated", {"incident_id": inc.incident_id, "reason": "unreadable"})

    def _on_skill_done(self, run) -> None:
        a = run.assignment
        meta = self._meta.pop(a.assignment_id, {})
        asset = a.target_asset_id or ""
        inc = self.fleet.incident_for(asset)
        if a.skill in ("read_gauge", "reinspect_asset") and not run.error and asset:
            self.scheduler.mark_inspected(asset)
        if run.error:
            self._block(meta)
            if inc and inc.status == TOOLS_PICKUP:
                inc.status = CONFIRMED
                inc.note(f"tool pick-up failed: {run.error}")
            elif inc and inc.status == TOOLS_DISPATCHED:
                inc.status = NEEDS_OPERATOR
                inc.note(f"tool delivery failed: {run.error}")
        elif a.skill == "pick_tool" and inc:
            inc.status = TOOLS_LOADED
            inc.note("tools loaded on the AMR")
            self.audit.record("tools_loaded", {"incident_id": inc.incident_id, "robot_id": a.robot_id})
        elif a.skill == "deliver_tools" and inc:
            inc.status = AWAITING_REPAIR
            self._delivered[asset] = a.robot_id
            inc.note("tools delivered - waiting for the technician")
            self.audit.record("tools_delivered", {"incident_id": inc.incident_id, "robot_id": a.robot_id})
        robot = getattr(run, "robot", None)
        if robot is not None and run.error and a.skill != "return_home":
            self._recover_and_home(robot, f"{a.skill} failed")  # v0.5.5: never left standing where the task failed
        elif robot is not None and not run.error and a.skill in self._home_after:
            self._send_home(robot, a.skill)  # reading done: walk home, but stay available for the next task
        self._wake_up(f"task_finished:{a.skill}")
        if getattr(run, "result_event", None) is not None:
            self.submit_event(run.result_event)

    def _send_everyone_home(self) -> None:
        busy = self._busy_robots()
        for r in self.fleet.robots.values():
            if r.robot_id not in busy and r.status == "idle":
                self._gate_and_dispatch(TaskAssignment(event_id="RESET", robot_id=r.robot_id, skill="return_home",
                                                       target_asset_id=None, priority="low",
                                                       rationale="Incident closed - return home.", source="rule_based"),
                                        None, None)

    # ------------------------------------------------------------------ status
    def status_lines(self) -> list:
        lines = [r.summary() for r in self.fleet.robots.values()] or ["(no robots - add one in Fleet Builder)"]
        walking = [rid for rid in self.fleet.homing if self.fleet.is_returning_home(self.fleet.robots[rid])] if self.fleet.homing else []
        if walking:
            lines.append("RETURNING HOME (available for a new task): " + ", ".join(sorted(walking)))
        for inc in self.fleet.incidents.values():
            prio = f" [{inc.priority}]" if inc.priority else ""
            lines.append(f"INCIDENT {inc.incident_id} {inc.asset_id}{prio}: {inc.status} | {inc.history[-1] if inc.history else ''}")
        last = self._last_decision
        if last is not None:
            what = f"{last.action} -> {last.robot_id} ({last.for_asset})" if last.decision == "assign" else "no action"
            lines.append(f"BRAIN [{last.source}] {what} | {last.rationale[:70]}")
        lines.append(self.brain.health_line())
        return lines
