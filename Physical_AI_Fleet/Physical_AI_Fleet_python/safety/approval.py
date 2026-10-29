# SPDX-License-Identifier: Apache-2.0
"""Operator approval queue for high-risk assignments (approved/rejected from the extension UI)."""
from __future__ import annotations

import time
from collections import OrderedDict


class ApprovalQueue:
    def __init__(self, audit=None, timeout_s: float = 300.0) -> None:
        self._pending: "OrderedDict[str, tuple]" = OrderedDict()
        self.audit = audit
        self.timeout_s = timeout_s

    def submit(self, assignment, event) -> None:
        self._pending[assignment.assignment_id] = (assignment, event, time.time())
        if self.audit:
            self.audit.record("approval_requested", {"assignment": assignment.to_dict()})

    def pending(self) -> list:
        self._expire()
        return [a for a, _e, _t in self._pending.values()]

    def oldest(self):
        self._expire()
        return next(iter(self._pending.values()), (None, None, None))[:2]

    def resolve(self, assignment_id: str, approved: bool, operator: str = "operator"):
        item = self._pending.pop(assignment_id, None)
        if item and self.audit:
            self.audit.record("approval_resolved", {"assignment_id": assignment_id, "approved": approved, "by": operator})
        return item[:2] if item else (None, None)

    def _expire(self) -> None:
        now = time.time()
        for aid in [k for k, (_a, _e, t) in self._pending.items() if now - t > self.timeout_s]:
            self._pending.pop(aid)
            if self.audit:
                self.audit.record("approval_expired", {"assignment_id": aid})
