# SPDX-License-Identifier: Apache-2.0
"""A real robot MIRRORED in the simulation (v0.5.6, Digital Twin phase 1).

The real Go2 is driven by its own remote (or the JOG buttons); the simulation only SHOWS where it is. The fleet never
gives it a task (it is not a candidate and the safety gate refuses it), but simulated robots see it and plan around it.
Its pose comes from Go2UdpLink (world = Map -> World of the robot's odometry). Real robots are never teleported.
"""
from __future__ import annotations

import math
import time

from .base import BUSY, ERROR, IDLE, OFFLINE, RobotAgent
from .unitree_twin_link import JOG_MOVE_S, JOG_TURN_DEG, JOG_V, JOG_W, STALE_S

JOG_KINDS = {"forward": (JOG_V, 0.0, 0.0, JOG_MOVE_S), "back": (-JOG_V, 0.0, 0.0, JOG_MOVE_S),
             "turn_left": (0.0, 0.0, JOG_W, math.radians(JOG_TURN_DEG) / JOG_W),
             "turn_right": (0.0, 0.0, -JOG_W, math.radians(JOG_TURN_DEG) / JOG_W)}


class MirrorRobot(RobotAgent):
    control = "mirror"

    def __init__(self, robot_id, robot_type, home_pose, link, prim_path: str = "", model: str = "", jog=None) -> None:
        super().__init__(robot_id, robot_type, "digital_twin", home_pose, prim_path, model)
        self.link, self.jog_client = link, jog
        self.locomotion = "mirror"
        self.status, self.last_telemetry = OFFLINE, 0.0
        self.world_pose = None  # (x, y, z, (qw, qx, qy, qz)) for the twin model
        self.src, self.age = "", None
        self.joints = None  # v0.5.7: 12 joint angles, Isaac order, smoothed (None until data arrives)
        self.joint_age, self.rate_hz = None, 0.0
        self.neighbors_fn = None

    def update(self, dt: float) -> None:
        st = self.link.state()
        self.age, self.src = st["age"], st["src"]
        self.joint_age, self.rate_hz = st.get("joint_age"), st.get("rate_hz", 0.0)
        fresh_joints = st.get("joint_age") is not None and st["joint_age"] <= STALE_S
        self.joints = st.get("joints_isaac") if fresh_joints else None
        if st.get("error"):
            self.status, self.note = ERROR, st["error"][:90]
            return
        if st["age"] is None or st["age"] > STALE_S:
            self.status = OFFLINE
            self.note = ("waiting for robot data on UDP port %d (go2_udp_sender.py on the Go2 sends it)" % self.link.port
                         if st["age"] is None else f"no robot data for {st['age']:.0f} s")
            return
        if not st["have_pose"]:
            self.status = OFFLINE
            self.note = "robot data has no position: run the v0.5.6 go2_udp_sender.py on the Go2"
            return
        self.last_telemetry = time.time() - st["age"]
        self.pose = [st["x"], st["y"], st["yaw"]]
        self.world_pose = (st["x"], st["y"], st["z"], st["quat_world"])
        jog = self.jog_client is not None and self.jog_client.active
        self.status = BUSY if (st["moving"] or jog) else IDLE
        self.note = ""

    # the fleet never drives a mirror; these exist so generic code (stop all, reset) is safe
    def go_to(self, x: float, y: float, yaw: float, name: str = "") -> None:
        self.note = "mirror only: drive the real robot with its remote or JOG"

    def stop(self) -> None:
        if self.jog_client is not None:
            self.jog_client.stop()

    def reset_to_home(self) -> None:
        self.stop()  # never teleport a real robot

    def jog(self, kind: str, site=None) -> str:
        if self.jog_client is None:
            return (f"{self.robot_id}: no jog address - add ';jog=<robot ip>:9999' to its Connection "
                    "(go2_vel_server.py must run on the Go2)")
        if kind not in JOG_KINDS:
            return f"unknown jog '{kind}'"
        if self.status in (OFFLINE, ERROR):
            return f"{self.robot_id}: not jogging - the twin is {self.status} ({self.note}); its position is not known"
        vx, vy, wz, dur = JOG_KINDS[kind]
        if vx and site is not None:
            d = vx * dur
            x, y = self.pose[0] + math.cos(self.pose[2]) * d, self.pose[1] + math.sin(self.pose[2]) * d
            why = site.why_not(x, y)
            if why:
                return f"{self.robot_id}: not jogging {kind} - {why}"
        self.jog_client.pulse(vx, vy, wz, dur)
        what = f"{abs(vx * dur) * 100:.0f} cm" if vx else f"{JOG_TURN_DEG:g} deg"
        return f"{self.robot_id}: jog {kind} ({what}, then stop)"

    def shutdown(self) -> None:
        try:
            self.stop()
        finally:
            self.link.close()

    def summary(self) -> str:
        x, y, yaw = self.pose
        data = "no data" if self.age is None else f"data {self.age:.1f}s {self.rate_hz:.0f}/s"
        if self.joint_age is not None:
            data += f", joints {self.joint_age:.1f}s"
        src = f" from {self.src}" if self.src else ""
        return (f"{self.robot_id:<10} {self.model.key:<5} real-mirror  {self.status:<7} "
                f"({x:5.2f},{y:5.2f},{math.degrees(yaw):4.0f}deg) {data}{src} {self.note}")
