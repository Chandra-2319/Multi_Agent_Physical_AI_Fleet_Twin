# SPDX-License-Identifier: Apache-2.0
"""Strict fleet-event schema: what a robot (real or simulated) reports to the coordinator."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..fleet_config.catalog import PROVENANCE
from .validate import (
    SchemaError, bounded_text, no_extra_keys, parse_iso, reject_forbidden_keys, require, unit_interval, utc_now_iso,
)

SCHEMA_VERSION = "1.0"
EVENT_TYPES = ("normal", "anomaly", "gauge_reading", "leak_suspected", "verification", "robot_status", "inconclusive")


@dataclass
class GaugeReading:
    value: Optional[float]
    unit: str
    in_range: Optional[bool]
    confidence: float
    roi_px: int = 0

    @staticmethod
    def from_dict(d: dict) -> "GaugeReading":
        no_extra_keys(d, {"value", "unit", "in_range", "confidence", "roi_px"}, "gauge")
        value = require(d, "value", (int, float), nullable=True)
        unit = bounded_text(require(d, "unit", str), "unit", 16)
        in_range = require(d, "in_range", bool, nullable=True)
        conf = unit_interval(require(d, "confidence", (int, float)), "gauge.confidence")
        roi = d.get("roi_px", 0)
        if not isinstance(roi, int) or isinstance(roi, bool) or roi < 0:
            raise SchemaError("gauge.roi_px must be a non-negative int")
        return GaugeReading(None if value is None else float(value), unit, in_range, conf, roi)


@dataclass
class FleetEvent:
    robot_id: str
    provenance: str
    asset_id: str
    event_type: str
    observation: str
    confidence: float
    gauge: Optional[GaugeReading] = None
    evidence: list = field(default_factory=list)
    ttl_s: int = 120
    incident_id: Optional[str] = None
    event_id: str = field(default_factory=lambda: f"EV-{uuid.uuid4().hex[:10]}")
    timestamp: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def actionable(self) -> bool:
        if self.event_type in ("anomaly", "leak_suspected", "inconclusive"):
            return True
        if self.event_type == "gauge_reading" and self.gauge is not None:
            return self.gauge.in_range is False or self.gauge.value is None
        return False

    @staticmethod
    def from_dict(d: dict) -> "FleetEvent":
        if not isinstance(d, dict):
            raise SchemaError("fleet event must be an object")
        reject_forbidden_keys(d)
        allowed = {"robot_id", "provenance", "asset_id", "event_type", "observation", "confidence", "gauge",
                   "evidence", "ttl_s", "incident_id", "event_id", "timestamp", "schema_version"}
        no_extra_keys(d, allowed, "fleet_event")
        if d.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise SchemaError("unsupported schema_version")
        gauge_raw = d.get("gauge")
        if gauge_raw is not None and not isinstance(gauge_raw, dict):
            raise SchemaError("gauge must be an object or null")
        gauge = GaugeReading.from_dict(gauge_raw) if gauge_raw is not None else None
        evidence = d.get("evidence", [])
        if not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence):
            raise SchemaError("evidence must be a list of strings")
        ttl = d.get("ttl_s", 120)
        if not isinstance(ttl, int) or isinstance(ttl, bool) or not 1 <= ttl <= 3600:
            raise SchemaError("ttl_s must be an int in [1, 3600]")
        ev = FleetEvent(
            robot_id=bounded_text(require(d, "robot_id", str), "robot_id", 64),
            provenance=require(d, "provenance", str, allowed=set(PROVENANCE)),
            asset_id=bounded_text(require(d, "asset_id", str), "asset_id", 64),
            event_type=require(d, "event_type", str, allowed=set(EVENT_TYPES)),
            observation=bounded_text(require(d, "observation", str), "observation", 600),
            confidence=unit_interval(require(d, "confidence", (int, float)), "confidence"),
            gauge=gauge, evidence=evidence, ttl_s=ttl, incident_id=d.get("incident_id"),
        )
        if "event_id" in d:
            ev.event_id = bounded_text(require(d, "event_id", str), "event_id", 64)
        if "timestamp" in d:
            parse_iso(require(d, "timestamp", str))
            ev.timestamp = d["timestamp"]
        return ev


PERCEPTION_OUTPUT_SPEC = {
    "event_type": "one of normal | anomaly | leak_suspected | inconclusive",
    "observation": "one or two plain sentences about what is visible (max 300 chars)",
    "leak_visible": "true | false",
    "confidence": "number 0..1",
}
