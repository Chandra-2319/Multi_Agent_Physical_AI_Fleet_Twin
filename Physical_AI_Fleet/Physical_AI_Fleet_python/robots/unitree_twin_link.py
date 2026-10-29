# SPDX-License-Identifier: Apache-2.0
"""Real Unitree Go2 mirrored in the simulation (v0.5.6, v0.5.7: no ROS on the PC). Stdlib only.

v0.5.7:   Go2: go2_udp_sender.py  --UDP 9870-->  PC: Go2UdpLink (this extension, inside Isaac Sim)
The extension reads the robot's packets itself: body pose (odometry, Map -> World) AND the 12 joint angles, reordered
from Unitree to Isaac order and smoothed exactly like go2_udp_receiver.py did. go2_udp_receiver.py, ROS 2 on the PC and
the OmniGraph joint pipeline are no longer needed. (v0.5.6 setups that forward to udp:9871 still work.)

Each packet is the sender's JSON: q / dq / tau (12 joints), quat (IMU, w x y z), gyro, acc and - from v0.5.6 - "odom":
{"px", "py", "pz", "qw", "qx", "qy", "qz", "src"} = the robot's own position estimate (lidar odometry, SLAM localisation
or the sport-mode estimate; "src" says which). The pose in the WORLD is Map -> World (Frame2D) applied to that odometry.

JogClient sends short, time-limited velocity pulses to go2_vel_server.py on the Go2 (same text protocol as your
go2_isaac_control.py: "vx,vy,vyaw\\n", one TCP connection per command, 10 Hz while a pulse lasts, then "0,0,0").
"""
from __future__ import annotations

import json
import math
import socket
import threading
import time

from .frames import Frame2D

STALE_S = 1.0  # no packet for this long -> the twin is offline
DEFAULT_PORT = 9870  # v0.5.7: the port go2_udp_sender.py sends to (PC_IP:9870)
ISAAC_JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
# Unitree motor order: 0-2 FR, 3-5 FL, 6-8 RR, 9-11 RL (hip, thigh, calf) -> Isaac order above (from go2_udp_receiver.py)
REMAP = (3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8)
JOINT_ALPHA = 0.4  # same smoothing as go2_udp_receiver.py
RATE_WINDOW_S = 2.0


def unitree_to_isaac(q) -> list:
    return [float(q[i]) for i in REMAP]


def smooth(prev, new, alpha: float = JOINT_ALPHA) -> list:
    return list(new) if prev is None else [alpha * n + (1.0 - alpha) * p for n, p in zip(new, prev)]
MOVING_V, MOVING_W = 0.05, 0.10  # m/s, rad/s: above this the real robot counts as moving (busy)
JOG_HZ = 10.0
JOG_V, JOG_W = 0.25, 0.5  # m/s, rad/s - deliberately slower than the remote
JOG_MOVE_S, JOG_TURN_DEG = 0.6, 15.0


def yaw_from_quat(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_mul(a, b) -> tuple:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def _finite(vals) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals)


def parse_packet(data) -> dict | None:
    """Sender JSON -> {"odom": (px, py, pz, qw, qx, qy, qz) | None, "src", "imu_quat", "wz", "joints"}; None if unusable."""
    try:
        d = json.loads(data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(d, dict):
        return None
    out = {"odom": None, "src": "", "imu_quat": None, "wz": 0.0, "joints": None}
    o = d.get("odom")
    if isinstance(o, dict):
        vals = [o.get(k) for k in ("px", "py", "pz", "qw", "qx", "qy", "qz")]
        if vals[2] is None:
            vals[2] = 0.0
        if _finite(vals) and abs(math.hypot(*vals[3:])) > 1e-6:
            n = math.sqrt(sum(v * v for v in vals[3:]))
            out["odom"] = tuple(float(v) for v in vals[:3]) + tuple(float(v) / n for v in vals[3:])
            out["src"] = str(o.get("src", d.get("src", "")))[:60]
    q = d.get("quat")
    if isinstance(q, (list, tuple)) and len(q) == 4 and _finite(q):
        out["imu_quat"] = tuple(float(v) for v in q)
    g = d.get("gyro")
    if isinstance(g, (list, tuple)) and len(g) == 3 and _finite(g):
        out["wz"] = float(g[2])
    j = d.get("q")
    if isinstance(j, (list, tuple)) and len(j) == 12 and _finite(j):
        out["joints"] = [float(v) for v in j]
    if out["odom"] is None and out["imu_quat"] is None:
        return None
    return out


class Go2UdpLink:
    """Listens for the forwarded robot state. Thread-safe; never blocks the physics step."""

    def __init__(self, port: int = DEFAULT_PORT, host: str = "0.0.0.0", map_to_world: Frame2D | None = None, clock=time.time):
        self.port, self.host = int(port), host
        self.frame = map_to_world or Frame2D()
        self.clock = clock
        self._lock = threading.Lock()
        self._last = None  # (t, parsed)
        self._prev_odom = None  # (t, x, y, yaw) in the map frame, for the speed estimate
        self._speed = (0.0, 0.0)
        self.packets = self.bad_packets = 0
        self._joints = None  # smoothed, Isaac order
        self._joint_t = None
        self._times: list = []  # packet arrival times (rate)
        self.error = ""
        self._sock = None
        self._thread = None
        self._running = False

    @staticmethod
    def parse_connection(conn: str) -> dict:
        """'udp:9870' or 'udp:9870;jog=10.85.82.106:9999' -> {"port": 9870, "jog": "10.85.82.106:9999" | ""}."""
        text = (conn or "").strip()
        if not text.lower().startswith("udp:"):
            raise ValueError("a mirrored Go2 needs Connection 'udp:<port>' (e.g. udp:9870), optionally ';jog=<robot ip>:9999'")
        parts = [p.strip() for p in text[4:].split(";") if p.strip()]
        try:
            port = int(parts[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"bad UDP port in '{conn}'") from exc
        if not 1 <= port <= 65535:
            raise ValueError(f"UDP port {port} out of range")
        jog = ""
        for p in parts[1:]:
            k, _, v = p.partition("=")
            if k.strip().lower() == "jog":
                jog = v.strip()
                JogClient.parse_addr(jog)
        return {"port": port, "jog": jog}

    def start(self) -> bool:
        if self._running:
            return True
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # no SO_REUSEADDR: a second reader must fail loudly
            s.bind((self.host, self.port))
            s.settimeout(0.2)
        except OSError as exc:
            hint = " - is go2_udp_receiver.py still running? It is not needed any more: stop it" if self.port == DEFAULT_PORT else ""
            self.error = f"cannot listen on UDP port {self.port}: {exc.strerror or exc}{hint}"
            return False
        self._sock, self._running, self.error = s, True, ""
        self._thread = threading.Thread(target=self._loop, name=f"go2-twin-udp-{self.port}", daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        while self._running:
            try:
                data, _addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            self.feed(data)

    def feed(self, data, t: float | None = None) -> bool:
        parsed = parse_packet(data)
        t = self.clock() if t is None else t
        with self._lock:
            if parsed is None:
                self.bad_packets += 1
                return False
            self.packets += 1
            self._times.append(t)
            if len(self._times) > 400 or (self._times and t - self._times[0] > RATE_WINDOW_S):
                self._times = [x for x in self._times[-400:] if t - x <= RATE_WINDOW_S]
            if parsed["joints"] is not None:
                self._joints = smooth(self._joints, unitree_to_isaac(parsed["joints"]))
                self._joint_t = t
            if parsed["odom"] is not None:
                px, py, _pz, qw, qx, qy, qz = parsed["odom"]
                yaw = yaw_from_quat(qw, qx, qy, qz)
                if self._prev_odom is not None and t - self._prev_odom[0] > 1e-3:
                    dt = t - self._prev_odom[0]
                    v = math.hypot(px - self._prev_odom[1], py - self._prev_odom[2]) / dt
                    w = abs((yaw - self._prev_odom[3] + math.pi) % (2 * math.pi) - math.pi) / dt
                    a = 0.3  # smoothing
                    self._speed = (a * v + (1 - a) * self._speed[0], a * w + (1 - a) * self._speed[1])
                self._prev_odom = (t, px, py, yaw)
            self._last = (t, parsed)
        return True

    def state(self) -> dict:
        """{"age", "have_pose", "map_pose", "x", "y", "z", "yaw", "quat_world", "moving", "src", "joints"} (age None = never)."""
        with self._lock:
            last, speed = self._last, self._speed
            joints, jt, times = (list(self._joints) if self._joints else None), self._joint_t, list(self._times)
        now = self.clock()
        st = {"age": None, "have_pose": False, "moving": False, "src": "", "joints": None, "error": self.error,
              "joints_isaac": joints, "joint_age": None if jt is None else max(0.0, now - jt),
              "rate_hz": (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0}
        if last is None:
            return st
        t, p = last
        st["age"] = max(0.0, self.clock() - t)
        st["src"], st["joints"] = p["src"], p["joints"]
        if p["odom"] is None:
            return st
        px, py, pz, qw, qx, qy, qz = p["odom"]
        myaw = yaw_from_quat(qw, qx, qy, qz)
        x, y, yaw = self.frame.to_world(px, py, myaw)
        h = self.frame.theta / 2.0
        st.update(have_pose=True, map_pose=(px, py, myaw), x=x, y=y, z=pz, yaw=(yaw + math.pi) % (2 * math.pi) - math.pi,
                  quat_world=quat_mul((math.cos(h), 0.0, 0.0, math.sin(h)), (qw, qx, qy, qz)),
                  moving=speed[0] > MOVING_V or speed[1] > MOVING_W or abs(p["wz"]) > 2 * MOVING_W)
        return st

    def close(self) -> None:
        """Stop listening and FREE the port before returning (v0.5.7: a socket closed while its reader thread still waits
        in recvfrom stays bound on Linux, so a LOAD right after would fail with 'Address already in use')."""
        self._running = False
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)  # recvfrom times out every 0.2 s
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._thread = None


class JogClient:
    """Short velocity pulses to go2_vel_server.py. Every pulse ends with "0,0,0"; stop() interrupts at once."""

    def __init__(self, addr: str, timeout_s: float = 0.4, send=None, sleep=time.sleep, clock=time.time):
        self.host, self.port = self.parse_addr(addr)
        self.timeout_s = timeout_s
        self._send = send or self._tcp_send
        self._sleep, self._clock = sleep, clock
        self._cancel = threading.Event()
        self._thread = None
        self.last_error = ""
        self.sent = 0

    @staticmethod
    def parse_addr(addr: str) -> tuple:
        host, _, port = (addr or "").strip().rpartition(":")
        try:
            p = int(port)
        except ValueError as exc:
            raise ValueError(f"jog address must be <robot ip>:<port>, got '{addr}'") from exc
        if not host or not 1 <= p <= 65535:
            raise ValueError(f"jog address must be <robot ip>:<port>, got '{addr}'")
        return host, p

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _tcp_send(self, line: str) -> None:
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as s:
            s.sendall(line.encode())
            s.recv(64)

    def _send_cmd(self, vx: float, vy: float, wz: float) -> bool:
        try:
            self._send(f"{vx:.3f},{vy:.3f},{wz:.3f}\n")
            self.sent += 1
            self.last_error = ""
            return True
        except OSError as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def pulse(self, vx: float, vy: float, wz: float, duration_s: float) -> None:
        """Send (vx, vy, wz) at 10 Hz for duration_s (clamped to the jog limits), then zero. Replaces a running pulse."""
        self.stop(send_zero=False, wait=True)
        vx = max(-JOG_V, min(JOG_V, float(vx)))
        vy = max(-JOG_V, min(JOG_V, float(vy)))
        wz = max(-JOG_W, min(JOG_W, float(wz)))
        duration_s = max(0.1, min(2.0, float(duration_s)))
        self._cancel = threading.Event()
        cancel = self._cancel

        def run():
            t_end = self._clock() + duration_s
            while not cancel.is_set() and self._clock() < t_end:
                if not self._send_cmd(vx, vy, wz):
                    break
                self._sleep(1.0 / JOG_HZ)
            self._send_cmd(0.0, 0.0, 0.0)
        self._thread = threading.Thread(target=run, name="go2-jog", daemon=True)
        self._thread.start()

    def stop(self, send_zero: bool = True, wait: bool = False) -> None:
        """Interrupt a pulse and send zero velocity. Non-blocking by default (the robot may be unreachable)."""
        self._cancel.set()

        def run():
            if self._thread is not None and self._thread is not threading.current_thread():
                self._thread.join(timeout=1.0)
            if send_zero:
                for _ in range(2):
                    if self._send_cmd(0.0, 0.0, 0.0):
                        break
        if wait or not send_zero:
            run()
        else:
            threading.Thread(target=run, name="go2-jog-stop", daemon=True).start()
