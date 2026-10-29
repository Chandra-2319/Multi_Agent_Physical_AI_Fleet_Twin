# SPDX-License-Identifier: Apache-2.0
"""Simulated legged fleet member (Go2, Spot, H1, G1) that really walks.

    skill goal -> GridPlanner (A*, obstacle-aware) -> PathFollower (pure pursuit, rate-limited)
               -> [vx, vy, wz] -> LocomotionBody (RL walking policy in Isaac Sim) -> joint targets

The pose is always read back from the simulated body, never written. Falls are reported as an
error so the safety gate stops dispatching to the robot until the operator resets it.
v0.5.5: an idle robot PARKS (stops running its walking policy) only at home, upright, with no task, after standing still
for PARK_AFTER_S; while parked its height / tilt are watched and the policy takes over again before a fall (logged as
robot_woke). Every wake is a clean re-stand. Progress / stuck checks count SIMULATION time, and turning towards the goal
heading counts as progress. A robot that is stuck can be released with clear_error() (the coordinator walks it home).
v0.3.1: a body that cannot be driven (body.error, e.g. the walking policy keeps failing) also puts the
robot into ERROR with the reason as the note, instead of raising on every physics step.
"""
from __future__ import annotations

import math
import time

from .base import BUSY, ERROR, IDLE, RobotAgent
from .locomotion.path_follower import wrap_angle
from .locomotion.bodies import LocomotionBody
from .locomotion.path_follower import PathFollower
from .locomotion.planner import GridPlanner, RectObstacle

PARK_AFTER_S = 3.0  # v0.5.5: idle at home and still for this long -> parked (walking policy not run)
IDLE_HOLD_AFTER_S = PARK_AFTER_S  # old name
HOME_PARK_TOL_M = 0.35
MAX_DRIFT_WAKES = 2  # a robot that keeps sagging while parked is not parked again in this run
BLOCKED_REPLAN_S = 2.5
NO_PROGRESS_S = 10.0
MAX_REPLANS = 4


class LeggedSimRobot(RobotAgent):
    def __init__(self, robot_id: str, robot_type: str, home_pose: tuple, body: LocomotionBody, planner: GridPlanner,
                 prim_path: str = "", model: str = "", locomotion: str = "rl_policy") -> None:
        super().__init__(robot_id, robot_type, "simulation", home_pose, prim_path, model)
        self.body = body
        self.planner = planner
        self.follower = PathFollower(self.model)
        self.locomotion = locomotion
        self.neighbors_fn = None
        self.path: list = []
        self._blocked_t = 0.0
        self._progress_ref = (0.0, math.inf)
        self._replans = 0
        self.last_command = [0.0, 0.0, 0.0]
        self.idle_hold = True  # v0.5.4 (Run Fleet > Skip Policy When Idle)
        self._idle_s = 0.0
        self._sim_t = 0.0  # v0.5.5: simulation time seen by this robot (progress checks)
        self._drift_wakes = 0
        self._events: list = []

    def go_to(self, x: float, y: float, yaw: float, name: str = "") -> None:
        if self.status == ERROR:
            return
        self.goal, self.goal_name = (x, y, yaw), name
        self.follower.YAW_TOL = PathFollower.YAW_TOL
        self._replans = 0
        self.status = BUSY
        if not self._plan(avoid_robots=False):
            self.goal = None
            self.status, self.note = ERROR, f"no path to {name or (round(x, 1), round(y, 1))}"

    def turn_to(self, yaw: float) -> None:
        """Rotate in place to `yaw` (no planning: the robot does not move away from where it stands). v0.5.0"""
        if self.status == ERROR:
            return
        x, y = float(self.pose[0]), float(self.pose[1])
        self.goal, self.goal_name, self.status = (x, y, yaw), "turn", BUSY
        self.follower.YAW_TOL = 0.05  # ~3 deg: a 10 deg step must really turn (the walking tolerance is ~7 deg)
        self.follower.set_path([(x, y), (x, y)], yaw)
        self._progress_ref = (self._sim_t, self._progress_metric())

    @property
    def holding(self) -> bool:
        return bool(getattr(self.body, "holding", False))

    def pop_events(self) -> list:
        ev, self._events = self._events, []
        return ev

    def clear_error(self) -> bool:
        """v0.5.5: release a robot that is in error only because it got stuck (the body is fine), so it can walk home."""
        if self.status != ERROR or self.body.error or self.body.is_fallen():
            return False
        self.goal = None
        self.follower.clear()
        self.status, self.note = IDLE, ""
        return True

    def _parkable(self) -> bool:
        if not self.idle_hold or self._drift_wakes >= MAX_DRIFT_WAKES:
            return False
        if self.goal is not None or self.status != IDLE or self.current_skill is not None:
            return False
        if max(abs(c) for c in self.follower.cmd) >= 1e-3:
            return False
        return math.dist(tuple(self.pose[:2]), tuple(self.home_pose[:2])) <= HOME_PARK_TOL_M

    def stop(self) -> None:
        self.goal = None
        self.follower.clear()
        if self.status == BUSY:
            self.status = IDLE

    def reset_to_home(self) -> None:
        self.stop()
        self.body.reset_to(*self.home_pose)  # also clears a latched body.error
        self.pose = list(self.home_pose)
        self.follower.cmd = [0.0, 0.0, 0.0]
        self.status, self.note = IDLE, ""
        self._drift_wakes, self._idle_s = 0, 0.0

    def on_world_reset(self) -> None:
        self.stop()
        self.body.request_reinit()
        self.body.reset_to(*self.home_pose)
        self.pose = list(self.home_pose)
        self.follower.cmd = [0.0, 0.0, 0.0]
        self.status, self.note = IDLE, ""
        self._drift_wakes, self._idle_s = 0, 0.0

    def update(self, dt: float) -> None:
        self.last_telemetry = time.time()
        self._sim_t += dt
        self.pose = list(self.body.pose())
        if self.body.error:  # the body cannot be driven: report why, and let the gate stop dispatching to it
            if self.status != ERROR or self.note != self.body.error[:80]:
                self.goal = None
                self.follower.clear()
                self.status, self.note = ERROR, self.body.error[:80]
            return
        if not self.body.ready:
            self.note = self.body.status_note
            self.body.step(dt, [0.0, 0.0, 0.0])
            return
        if self.body.is_fallen():
            if self.status != ERROR:
                self.status, self.note = ERROR, "fallen - press RESET ROBOT"
                self.goal = None
                self.follower.clear()
            self.body.step(dt, [0.0, 0.0, 0.0])
            return
        if self.status != ERROR and self.note.startswith(("initialising", "re-initialising")):
            self.note = ""
        if self._parkable():
            self._idle_s += dt
            if self._idle_s >= PARK_AFTER_S:
                self.last_command = [0.0, 0.0, 0.0]
                self.body.hold(dt)
                woke = getattr(self.body, "last_wake", "")
                if woke:
                    self.body.last_wake = ""
                    self._idle_s = 0.0
                    self._drift_wakes += 1
                    self._events.append(("robot_woke", {"robot_id": self.robot_id, "reason": woke,
                                                        "parking_disabled": self._drift_wakes >= MAX_DRIFT_WAKES}))
                return
        else:
            self._idle_s = 0.0
        self.follower.speed_scale = self._yield_scale(dt) if self.goal is not None else 1.0
        cmd = self.follower.compute(tuple(self.pose), dt)
        self.last_command = cmd
        self.body.step(dt, cmd)
        if self.goal is None:
            return
        if self.follower.done and max(abs(c) for c in cmd) < 0.05:
            self.goal = None
            self.status = IDLE
            return
        self._watch_progress()

    def _plan(self, avoid_robots: bool) -> bool:
        extra = []
        if avoid_robots and self.neighbors_fn:
            extra = [RectObstacle(nx, ny, nr, nr, nid) for nid, nx, ny, nr in self.neighbors_fn()]
        path = self.planner.plan(self.pose[:2], self.goal[:2], self.radius, extra) if self.goal else None
        if not path:
            return False
        self.path = path
        self.follower.set_path(path, self.goal[2])
        self._blocked_t = 0.0
        self._progress_ref = (self._sim_t, self._progress_metric())
        return True

    def _yield_scale(self, dt: float) -> float:
        if not self.neighbors_fn:
            return 1.0
        x, y, yaw = self.pose
        c, s = math.cos(yaw), math.sin(yaw)
        scale = 1.0
        for _nid, nx, ny, nr in self.neighbors_fn():
            dx, dy = nx - x, ny - y
            fwd, lat = c * dx + s * dy, -s * dx + c * dy
            clear = self.radius + nr
            if fwd > 0 and abs(lat) < clear + 0.1:
                gap = fwd - clear
                scale = min(scale, 0.0 if gap < 0.3 else min(1.0, gap / 1.2))
        if scale == 0.0:
            self._blocked_t += dt
            if self._blocked_t > BLOCKED_REPLAN_S:
                self._replan("blocked by another robot")
        else:
            self._blocked_t = 0.0
        return scale

    def _progress_metric(self) -> float:
        """Distance still to walk plus the heading still to turn (1 rad counts as 0.5 m): turning is progress too."""
        rem = self.follower.remaining(*self.pose[:2])
        if self.goal is not None and self.goal[2] is not None:
            rem += 0.5 * abs(wrap_angle(self.goal[2] - self.pose[2]))
        return rem

    def _watch_progress(self) -> None:
        t0, ref = self._progress_ref
        m = self._progress_metric()
        if m < ref - 0.15:
            self._progress_ref = (self._sim_t, m)
        elif self._sim_t - t0 > NO_PROGRESS_S:
            self._replan("no progress")

    def _replan(self, reason: str) -> None:
        self._replans += 1
        if self._replans > MAX_REPLANS or not self._plan(avoid_robots=True):
            self.stop()
            self.status, self.note = ERROR, f"stuck ({reason})"
        else:
            self.note = f"replanned: {reason}"
