# SPDX-License-Identifier: Apache-2.0
"""Real MiR AMR link through the MiR REST API (v2.0.0).

  MIR_MISSIONS='{"motor-01": "<mission-guid>", "home": "<mission-guid>"}'
  MIR_AUTH='Basic <base64 user:sha256(password)>'
"""
from __future__ import annotations

import json
import math
import os
import urllib.request

from .frames import Frame2D
from .twin_robot import RealLink

BUSY_STATES = {"Executing", "Starting"}


class MiRRestLink(RealLink):
    def __init__(self, base_url: str, auth: str, missions: dict, map_to_world: Frame2D | None = None,
                 timeout_s: float = 3.0) -> None:
        self.base = base_url.rstrip("/") + "/api/v2.0.0"
        self.auth = auth
        self.missions = missions
        self.frame = map_to_world or Frame2D()
        self.timeout_s = timeout_s

    @staticmethod
    def from_connection(conn: str, frame: Frame2D) -> "MiRRestLink":
        url = conn if conn.startswith("http") else f"http://{conn}"
        return MiRRestLink(url, os.environ.get("MIR_AUTH", ""), json.loads(os.environ.get("MIR_MISSIONS", "{}")), frame)

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode() if body is not None else None, method=method,
            headers={"Authorization": self.auth, "Content-Type": "application/json", "Accept-Language": "en_US"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else {}

    def read_state(self) -> dict:
        st = self._call("GET", "/status")
        p = st.get("position", {})
        x, y, yaw = self.frame.to_world(float(p["x"]), float(p["y"]), math.radians(float(p["orientation"])))
        errors = st.get("errors") or []
        return {"x": x, "y": y, "yaw": yaw, "busy": st.get("state_text") in BUSY_STATES,
                "error": errors[0].get("description") if errors else None}

    def send_goal(self, x, y, yaw, name: str = "") -> None:
        guid = self.missions.get(name)
        if not guid:
            raise RuntimeError(f"no MiR mission configured for target '{name}' (set MIR_MISSIONS)")
        self._call("PUT", "/status", {"state_id": 3})
        self._call("POST", "/mission_queue", {"mission_id": guid})

    def stop(self) -> None:
        self._call("PUT", "/status", {"state_id": 4})
