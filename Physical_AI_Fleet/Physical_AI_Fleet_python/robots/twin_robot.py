# SPDX-License-Identifier: Apache-2.0
"""Digital-twin fleet member: a real robot drives the sim pose (network I/O in a background thread)."""
from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod

from .base import BUSY, ERROR, IDLE, OFFLINE, RobotAgent


class RealLink(ABC):
    """Transport to a physical robot. All methods may block; they run off the main thread."""

    @abstractmethod
    def read_state(self) -> dict:
        """Return {"x": m, "y": m, "yaw": rad, "busy": bool, "error": str|None} in the WORLD frame."""

    @abstractmethod
    def send_goal(self, x: float, y: float, yaw: float, name: str = "") -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def close(self) -> None:
        pass


class TwinRobot(RobotAgent):
    def __init__(self, robot_id, robot_type, home_pose, link: RealLink, prim_path: str = "", model: str = "",
                 poll_hz: float = 5.0) -> None:
        super().__init__(robot_id, robot_type, "digital_twin", home_pose, prim_path, model)
        self.locomotion = "real"
        self.link = link
        self._period = 1.0 / poll_hz
        self._lock = threading.Lock()
        self._commands: list = []
        self._latest: dict | None = None
        self._goal_sent_at = 0.0
        self._running = True
        self.status = OFFLINE
        self.last_telemetry = 0.0
        self._thread = threading.Thread(target=self._loop, name=f"twin-{robot_id}", daemon=True)
        self._thread.start()

    def go_to(self, x, y, yaw, name: str = "") -> None:
        self.goal, self.goal_name = (x, y, yaw), name
        self.status = BUSY
        self._goal_sent_at = time.time()
        with self._lock:
            self._commands.append(("goal", (x, y, yaw, name)))

    def stop(self) -> None:
        self.goal = None
        with self._lock:
            self._commands.append(("stop", None))

    def reset_to_home(self) -> None:
        self.stop()  # never teleport a real robot

    def update(self, dt: float) -> None:
        with self._lock:
            state, self._latest = self._latest, None
        if state is None:
            return
        self.last_telemetry = state["_t"]
        if state.get("error"):
            self.status, self.note = ERROR, str(state["error"])[:60]
            return
        self.note = ""
        self.pose = [state["x"], state["y"], state["yaw"]]
        if self.goal is not None:
            settled = time.time() - self._goal_sent_at > 2.0 and not state.get("busy", False)
            if self.arrived(0.35) or settled:
                self.goal = None
                self.status = IDLE
        elif self.status in (OFFLINE, ERROR, BUSY):
            self.status = BUSY if state.get("busy") else IDLE

    def shutdown(self) -> None:
        self._running = False
        try:
            self.link.close()
        except Exception:  # noqa: BLE001
            pass

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                cmds, self._commands = self._commands, []
            for kind, arg in cmds:
                try:
                    if kind == "goal":
                        self.link.send_goal(*arg)
                    else:
                        self.link.stop()
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        self._latest = {"_t": time.time(), "error": f"{kind} failed: {exc}"}
            try:
                st = self.link.read_state()
                st["_t"] = time.time()
                if not all(isinstance(st.get(k), (int, float)) and math.isfinite(st[k]) for k in ("x", "y", "yaw")):
                    raise ValueError("bad pose in robot state")
                with self._lock:
                    self._latest = st
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self._period)
