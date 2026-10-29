# SPDX-License-Identifier: Apache-2.0
"""Fake Go2 bridge speaking the wire format assumed by robots/go2_adapter.py.

  python tools/mock_go2_bridge.py --port 9000 [--image some_gauge.jpg]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import socketserver
import threading
import time


class FakeGo2:
    def __init__(self, image: bytes | None) -> None:
        self.x, self.y, self.yaw = 0.0, -3.0, 0.0
        self.goal = None
        self.image = image
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            with self.lock:
                if self.goal:
                    gx, gy, gyaw = self.goal
                    d = math.hypot(gx - self.x, gy - self.y)
                    if d < 0.05:
                        self.x, self.y, self.yaw, self.goal = gx, gy, gyaw, None
                    else:
                        step = min(0.08, d)
                        self.yaw = math.atan2(gy - self.y, gx - self.x)
                        self.x += step * math.cos(self.yaw)
                        self.y += step * math.sin(self.yaw)
            time.sleep(0.1)

    def handle(self, msg: dict) -> dict:
        with self.lock:
            cmd = msg.get("cmd")
            if cmd == "get_state":
                return {"x": self.x, "y": self.y, "yaw": self.yaw, "busy": self.goal is not None}
            if cmd == "goto":
                self.goal = (float(msg["x"]), float(msg["y"]), float(msg["yaw"]))
                return {"ok": True}
            if cmd == "stop":
                self.goal = None
                return {"ok": True}
            if cmd == "get_frame":
                return {"jpeg_b64": base64.b64encode(self.image).decode() if self.image else None}
            return {"ok": False, "error": f"unknown cmd {cmd}"}


def serve(port: int, image: bytes | None = None, block: bool = True):
    robot = FakeGo2(image)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                self.wfile.write((json.dumps(robot.handle(json.loads(line))) + "\n").encode())

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    if block:
        print(f"Fake Go2 bridge on 127.0.0.1:{port}")
        server.serve_forever()
    else:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, robot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--image", default="")
    args = ap.parse_args()
    serve(args.port, open(args.image, "rb").read() if args.image else None)
