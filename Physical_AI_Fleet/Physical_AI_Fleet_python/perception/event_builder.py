# SPDX-License-Identifier: Apache-2.0
"""Turns a captured frame (+ sim fault state when mocking) into one validated FleetEvent.

v0.4.4: evidence files carry the asset id as well as the robot id; mock observations name the equipment type.
"""
from __future__ import annotations
from ..schemas.fleet_event import FleetEvent, GaugeReading
from .frame_capture import decode_jpeg, encode_jpeg
from .gauge_reader import DialSpec, read_gauge


class PerceptionPipeline:
    def __init__(self, settings, cosmos, evidence_store, audit=None) -> None:
        self.s = settings
        self.cosmos = cosmos
        self.evidence = evidence_store
        self.audit = audit

    def run(self, *, robot_id: str, provenance: str, asset_id: str, purpose: str, jpeg, sim_fault: dict | None,
            incident_id: str | None = None) -> FleetEvent:
        """purpose: inspect | reinspect | read_gauge | verify. Runs in a worker thread."""
        asset = self.s.assets[asset_id]
        if callable(jpeg):  # deferred network capture (digital twin) runs here, off the main thread
            try:
                jpeg = jpeg()
            except Exception as exc:  # noqa: BLE001
                if self.audit:
                    self.audit.record("frame_capture_failed", {"robot_id": robot_id, "error": str(exc)})
                jpeg = None
        evidence = [self.evidence.save(robot_id, jpeg, purpose, asset_id=asset_id)] if jpeg else []
        if jpeg is not None and self.cosmos.available and provenance != "simulated":
            scene, gauge = self._real(jpeg, asset, robot_id, evidence)
        else:
            scene, gauge = self._mock(asset, sim_fault, purpose)
        problem = scene["event_type"] in ("anomaly", "leak_suspected") or (gauge is not None and gauge.in_range is False)
        if purpose == "verify":
            event_type = "anomaly" if problem else ("verification" if scene["event_type"] != "inconclusive" else "inconclusive")
        elif scene["event_type"] in ("anomaly", "leak_suspected", "inconclusive"):
            event_type = scene["event_type"]
        elif gauge is not None:
            event_type = "gauge_reading"
        else:
            event_type = "normal"
        confidence = scene["confidence"] if gauge is None else min(scene["confidence"], gauge.confidence)
        event = FleetEvent(robot_id=robot_id, provenance=provenance, asset_id=asset_id, event_type=event_type,
                           observation=scene["observation"], confidence=round(confidence, 2), gauge=gauge,
                           evidence=[e for e in evidence if e], ttl_s=int(self.s.event_ttl_s), incident_id=incident_id)
        return FleetEvent.from_dict(event.to_dict())

    def _real(self, jpeg: bytes, asset, robot_id: str, evidence: list):
        scene = self.cosmos.describe_scene(jpeg, asset.asset_id)
        g = read_gauge(decode_jpeg(jpeg), DialSpec(asset.gauge_min, asset.gauge_max),
                       min_roi_px=self.s.gauge_min_roi_px, upscale=self.s.gauge_upscale)
        value, conf = g.value, g.confidence
        if g.crop is not None and g.crop.size:
            crop_jpeg = encode_jpeg(g.crop)
            evidence.append(self.evidence.save(robot_id, crop_jpeg, "gauge_crop", asset_id=asset.asset_id))
            c_val, c_conf = self.cosmos.read_gauge(crop_jpeg, asset.gauge_min, asset.gauge_max, asset.unit)
            if c_val is not None and c_conf >= conf:
                value, conf = c_val, c_conf
            if g.roi and g.roi.side_px < self.s.gauge_min_roi_px:
                conf = min(conf, 0.3)
        if self.audit:
            self.audit.record("gauge_pipeline", {"robot_id": robot_id, "asset_id": asset.asset_id, "value": value,
                                                 "confidence": conf, "note": g.note})
        gauge = GaugeReading(value, asset.unit, None if value is None else asset.normal_low <= value <= asset.normal_high,
                             round(conf, 2), g.roi.side_px if g.roi else 0)
        return scene, gauge

    def _mock(self, asset, fault: dict | None, purpose: str):
        tag = "[MOCK perception] "
        what = (getattr(asset, "label", "") or "equipment").lower()
        healthy = (asset.normal_low + asset.normal_high) / 2
        if not fault:
            scene = {"event_type": "normal", "observation": tag + f"No leak or damage visible on the {what}; gauge in normal band.",
                     "leak_visible": False, "confidence": 0.9}
            value = healthy
        elif fault.get("kind") == "leak":
            scene = {"event_type": "leak_suspected", "observation": tag + f"Coloured fluid pooled below the {what}.",
                     "leak_visible": True, "confidence": 0.82 if purpose == "inspect" else 0.9}
            value = healthy
        else:
            value = float(fault.get("gauge_value", asset.normal_high + 2.5))
            scene = {"event_type": "anomaly", "observation": tag + f"Gauge needle above normal band (~{value:.1f} {asset.unit}).",
                     "leak_visible": False, "confidence": 0.8 if purpose == "inspect" else 0.9}
        gauge = GaugeReading(round(value, 2), asset.unit, asset.normal_low <= value <= asset.normal_high, 0.85,
                             160 if purpose in ("read_gauge", "reinspect", "verify") else 96)
        return scene, gauge
