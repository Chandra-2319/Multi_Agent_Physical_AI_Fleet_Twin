# SPDX-License-Identifier: Apache-2.0
"""Runs approved high-level skills: navigate -> (capture -> perceive) -> done.
Standoff distances account for the robot's footprint (Spot stands further from the motor than Go2).
v0.5.5: time limits count SIMULATION time (tick(dt)): a whole skill SKILL_TIMEOUT_S, the walk to the target
NAV_BASE_S + NAV_FACTOR x (distance / top speed). Before a robot is sent to a spot, clear_spot_fn (set by the
coordinator) sends any idle robot standing there home.
v0.4.1: the capture phase waits (up to CAPTURE_WAIT_S) when the camera says it is not ready yet.
v0.5.1: the robot stands in front of the gauge built into the asset USD (GaugeInspector.inspect_pose).
v0.5.0: when a GaugeInspector is attached (scenario) and the asset has a gauge, read_gauge / reinspect / verify stand
at the gauge's camera distance and the capture phase becomes a Cosmos read-and-adjust session (phase "gauge").
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..robots.base import ERROR, IDLE

SKILL_TIMEOUT_S = 900.0  # simulation seconds (v0.5.5; was 300 s of wall time)
NAV_BASE_S = 60.0
NAV_FACTOR = 3.0
CLEAR_SPOT_M = 1.2
CAPTURE_WAIT_S = 3.0  # how long a robot waits (standing still) for its camera's first frame
CLEARANCE = 0.15
CAPTURE_RETRY = object()  # capture_fn may return this: "camera not ready, ask again next tick"


@dataclass
class SkillRun:
    assignment: object
    robot: object
    purpose: str
    incident_id: str | None
    phase: str = "start"
    started: float = field(default_factory=time.time)
    future: object = None
    result_event: object = None
    error: str = ""
    capture_started: float = 0.0
    session: object = None
    sim_s: float = 0.0
    nav_s: float = 0.0
    nav_cap_s: float = 0.0


def standoff_pose(asset, robot_radius: float, preferred: float) -> tuple:
    ax, ay = asset.position[:2]
    px, py, _ = asset.approach_pose
    d = math.hypot(px - ax, py - ay) or 1.0
    ux, uy = (px - ax) / d, (py - ay) / d
    face = abs(ux) * asset.half_extents[0] + abs(uy) * asset.half_extents[1]
    dist = max(preferred, face + robot_radius + CLEARANCE)
    x, y = ax + ux * dist, ay + uy * dist
    return x, y, math.atan2(ay - y, ax - x)


class SkillExecutor:
    def __init__(self, settings, pool, perception, capture_fn, fault_fn, audit, on_done) -> None:
        self.s = settings
        self.pool = pool
        self.perception = perception
        self.capture_fn = capture_fn
        self.fault_fn = fault_fn
        self.audit = audit
        self.on_done = on_done
        self.runs: dict = {}
        self.gauge_inspector = None  # set by the scenario (v0.5.0)
        self.clear_spot_fn = None  # (xy, robot_id) -> None; set by the coordinator (v0.5.5)

    def start(self, assignment, robot, purpose: str = "", incident_id: str | None = None) -> None:
        if robot.robot_id in self.runs:
            self.cancel(robot.robot_id, "preempted")
        purpose = purpose or {"inspect_asset": "inspect", "reinspect_asset": "reinspect",
                              "read_gauge": "read_gauge"}.get(assignment.skill, "")
        self.runs[robot.robot_id] = SkillRun(assignment, robot, purpose, incident_id)
        robot.current_skill = assignment.skill
        self.audit.record("skill_started", {"robot_id": robot.robot_id, "skill": assignment.skill,
                                            "target": assignment.target_asset_id, "purpose": purpose,
                                            "locomotion": robot.locomotion})

    def cancel(self, robot_id: str, reason: str) -> None:
        run = self.runs.pop(robot_id, None)
        if run:
            run.robot.stop()
            run.robot.current_skill = None
            self.audit.record("skill_cancelled", {"robot_id": robot_id, "reason": reason})

    def tick(self, dt: float = 0.0) -> None:
        for robot_id, run in list(self.runs.items()):
            run.sim_s += float(dt)
            if run.phase == "navigate":
                run.nav_s += float(dt)
            try:
                self._step(run)
            except Exception as exc:  # noqa: BLE001
                run.error, run.phase = f"{type(exc).__name__}: {exc}", "done"
            if run.phase == "navigate" and run.nav_cap_s and run.nav_s > run.nav_cap_s:
                run.error, run.phase = f"navigation timeout ({run.nav_s:.0f} s simulated)", "done"
                run.robot.stop()
            if run.phase != "done" and run.sim_s > SKILL_TIMEOUT_S:
                run.error, run.phase = "timeout", "done"
                run.robot.stop()
            if run.phase == "done":
                self.runs.pop(robot_id, None)
                run.robot.current_skill = None
                self.audit.record("skill_finished", {"robot_id": robot_id, "skill": run.assignment.skill,
                                                     "error": run.error or None})
                self.on_done(run)

    def _gauge(self, run: SkillRun) -> bool:
        gi = self.gauge_inspector
        return gi is not None and gi.handles(run.assignment.target_asset_id or "", run.purpose)

    def _step(self, run: SkillRun) -> None:
        a, robot = run.assignment, run.robot
        asset = self.s.assets.get(a.target_asset_id or "")
        if run.phase == "start":
            if a.skill == "hold_position":
                robot.stop()
                run.phase = "done"
                return
            if a.skill == "return_home":
                goal, name = tuple(robot.home_pose), "home"
            elif self._gauge(run):
                goal, name = self.gauge_inspector.inspect_pose(asset, robot.radius), asset.asset_id
            elif a.skill == "read_gauge":
                goal, name = standoff_pose(asset, robot.radius, 0.6), asset.asset_id
            elif a.skill == "deliver_tools":
                goal, name = standoff_pose(asset, robot.radius, 1.4), asset.asset_id
            else:
                ax, ay = asset.position[:2]
                preferred = math.hypot(asset.approach_pose[0] - ax, asset.approach_pose[1] - ay)
                goal, name = standoff_pose(asset, robot.radius, preferred), asset.asset_id
            if a.skill != "return_home" and self.clear_spot_fn is not None:
                try:
                    self.clear_spot_fn(tuple(goal[:2]), robot.robot_id)
                except Exception as exc:  # noqa: BLE001 - never block the task
                    self.audit.record("clear_spot_failed", {"robot_id": robot.robot_id, "error": str(exc)[:160]})
            dist = math.dist(tuple(robot.pose[:2]), tuple(goal[:2]))
            run.nav_cap_s = NAV_BASE_S + NAV_FACTOR * dist / max(float(getattr(robot.model, "max_vx", 0.5) or 0.5), 0.1)
            robot.go_to(*goal, name=name)
            run.phase = "navigate"
            return
        if run.phase == "navigate":
            if robot.status == ERROR:
                run.error, run.phase = f"robot error: {robot.note}", "done"
            elif robot.goal is None and robot.status == IDLE:
                run.phase = "capture" if run.purpose else "done"
            return
        if run.phase == "capture" and self._gauge(run):
            run.session = self.gauge_inspector.start(run, self.capture_fn)
            run.phase = "gauge"
        if run.phase == "gauge":
            result = run.session.step()
            if result is not None:
                run.result_event = self.gauge_inspector.event_for(result, run, self.s)
                run.phase = "done"
            return
        if run.phase == "capture":
            jpeg = self.capture_fn(robot)
            if jpeg is CAPTURE_RETRY:  # camera has no frame yet: the robot stands still and asks again next tick
                if not run.capture_started:
                    run.capture_started = time.time()
                    self.audit.record("capture_waiting", {"robot_id": robot.robot_id, "asset": a.target_asset_id})
                if time.time() - run.capture_started < CAPTURE_WAIT_S:
                    return
                self.audit.record("capture_timeout", {"robot_id": robot.robot_id, "asset": a.target_asset_id,
                                                      "waited_s": CAPTURE_WAIT_S})
                jpeg = None
            run.future = self.pool.submit(
                self.perception.run, robot_id=robot.robot_id, provenance=robot.provenance,
                asset_id=a.target_asset_id, purpose=run.purpose, jpeg=jpeg,
                sim_fault=self.fault_fn(a.target_asset_id), incident_id=run.incident_id,
            )
            run.phase = "perceive"
            return
        if run.phase == "perceive" and run.future.done():
            try:
                run.result_event = run.future.result()
            except Exception as exc:  # noqa: BLE001
                run.error = f"perception failed: {exc}"
            run.phase = "done"
