# SPDX-License-Identifier: Apache-2.0
"""Real Unitree Go2 (or Spot) link via the team's existing UDP/TCP bridge.

ASSUMED WIRE FORMAT (edit `_request` to match your bridge): newline-delimited JSON over TCP.
  -> {"cmd": "get_state"}                       <- {"x":..,"y":..,"yaw":..,"busy":bool}
  -> {"cmd": "goto", "x":..,"y":..,"yaw":..}    <- {"ok": true}
  -> {"cmd": "stop"}                            <- {"ok": true}
  -> {"cmd": "get_frame"}                       <- {"jpeg_b64": "..."}   (optional)
"""
from __future__ import annotations

import base64
import json
import socket
import threading

from .frames import Frame2D
from .twin_robot import RealLink


class Go2BridgeLink(RealLink):
    def __init__(self, host: str, port: int = 9000, map_to_world: Frame2D | None = None, timeout_s: float = 2.0) -> None:
        self.host, self.port = host, port
        self.frame = map_to_world or Frame2D()
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self._buf = b""
        self._lock = threading.Lock()

    @staticmethod
    def from_connection(conn: str, frame: Frame2D) -> "Go2BridgeLink":
        host, _, port = conn.partition(":")
        return Go2BridgeLink(host or "127.0.0.1", int(port or 9000), frame)

    def _connect(self) -> socket.socket:
        if self._sock is None:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            self._buf = b""
        return self._sock

    def _request(self, msg: dict) -> dict:
        with self._lock:
            try:
                sock = self._connect()
                sock.sendall((json.dumps(msg) + "\n").encode())
                while b"\n" not in self._buf:
                    chunk = sock.recv(65536)
                    if not chunk:
                        raise ConnectionError("bridge closed the connection")
                    self._buf += chunk
                line, self._buf = self._buf.split(b"\n", 1)
                return json.loads(line)
            except Exception:
                self._close_locked()
                raise

    def read_state(self) -> dict:
        r = self._request({"cmd": "get_state"})
        x, y, yaw = self.frame.to_world(float(r["x"]), float(r["y"]), float(r["yaw"]))
        return {"x": x, "y": y, "yaw": yaw, "busy": bool(r.get("busy", False)), "error": r.get("error")}

    def send_goal(self, x, y, yaw, name: str = "") -> None:
        mx, my, myaw = self.frame.to_map(x, y, yaw)
        r = self._request({"cmd": "goto", "x": mx, "y": my, "yaw": myaw})
        if not r.get("ok", False):
            raise RuntimeError(r.get("error", "goal rejected by bridge"))

    def stop(self) -> None:
        self._request({"cmd": "stop"})

    def get_frame_jpeg(self) -> bytes | None:
        data = self._request({"cmd": "get_frame"}).get("jpeg_b64")
        return base64.b64decode(data) if data else None

    def _close_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()
