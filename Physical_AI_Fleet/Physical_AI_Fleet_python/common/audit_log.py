# SPDX-License-Identifier: Apache-2.0
"""JSONL audit log: model calls, events, assignments, gate decisions, approvals, evidence.

v0.4.3: payload fields called "ts" or "kind" can no longer overwrite the entry's own timestamp / kind. Before, an
alarm record that carried a field named "kind" showed up in the log as kind="overpressure" instead of "alarm_raised".
Such a field is now stored as "payload_kind" / "payload_ts".
"""
from __future__ import annotations
import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
RESERVED = ("ts", "kind")
class AuditLog:
    def __init__(self, log_dir: Path, keep_in_memory: int = 400) -> None:
        self._lock = threading.Lock()
        self._recent: deque = deque(maxlen=keep_in_memory)
        self.path: Path | None = None
        try:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
        except OSError:
            self.path = None
        self._listeners = []
    def add_listener(self, fn) -> None:
        self._listeners.append(fn)
    def record(self, kind: str, payload: dict | None = None, **extra) -> dict:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind}
        for k, v in {**(payload or {}), **extra}.items():
            entry[f"payload_{k}" if k in RESERVED else k] = v
        with self._lock:
            self._recent.append(entry)
            if self.path is not None:
                try:
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, default=str) + "\n")
                except OSError:
                    pass
        for fn in list(self._listeners):
            try:
                fn(entry)
            except Exception:  # noqa: BLE001
                pass
        return entry
    def recent(self, n: int = 20) -> list:
        with self._lock:
            return list(self._recent)[-n:]
    @staticmethod
    def one_line(entry: dict) -> str:
        body = {k: v for k, v in entry.items() if k not in ("ts", "kind")}
        return f"{entry.get('ts', '')[11:19]} {entry.get('kind', ''):<18} {json.dumps(body, default=str)[:160]}"
