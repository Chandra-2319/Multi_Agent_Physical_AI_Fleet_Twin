# SPDX-License-Identifier: Apache-2.0
"""Gauge reading with Cosmos at an asset (v0.5.0). Pure Python: robot, camera, Cosmos and clock are injected.

  capture -> ask Cosmos -> high / normal / low with confidence >= threshold                      -> done
                        -> unclear + turn_left / turn_right -> rotate in place one step -> settle -> capture again
                        -> unclear + move_back              -> step back a little       -> settle -> capture again
The whole stay at the asset is limited (cosmos.json total_time_s, 60 s): then the reading is UNREADABLE and the robot
is released. Cosmos errors, invalid answers and low confidence without a usable suggestion are UNREADABLE (no mock).
low pressure counts as normal.
v0.5.3: before reading, the gauge can be FOUND IN THE IMAGE by the vision model (bounding box, cosmos.json "detect") and
the frame is cropped to it - no robot / asset / gauge positions needed, so it works the same on a real quadruped.
crop.source sets the order: "vlm" (box from the model), "projection" (box from the known gauge position, simulation only).
The result becomes the FleetEvent the coordinator already understands:
  high -> anomaly (out of limits) | normal -> gauge_reading / verification | unreadable -> inconclusive.
"""
from __future__ import annotations

import json
import math
import time

UNREADABLE = "unreadable"


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class GaugeSession:
    def __init__(self, *, robot, asset_id, purpose, cfg, ask, capture, save_evidence, record, submit, clock,
                 walkable=None, truth=None, retry_marker=None, detect=None):
        self.robot, self.asset_id, self.purpose, self.cfg = robot, asset_id, purpose, cfg
        self.ask, self.capture, self.save_evidence, self.record, self.submit = ask, capture, save_evidence, record, submit
        self.clock, self.walkable, self.truth, self.retry_marker = clock, walkable, truth, retry_marker
        self.t0 = clock()
        self.state, self.attempt, self.adjustments = "capture", 0, 0
        self.turned, self.backed = 0.0, 0.0
        self.evidence, self.answers = [], []
        self._future, self._move_t0, self._settle_until, self._goal = None, 0.0, 0.0, None
        self.result = None
        self.detect = detect  # v0.5.3: (jpeg, timeout, width, height) -> (Detection, raw, latency)
        self._frame = None

    def _left(self) -> float:
        return float(self.cfg.get("total_time_s", 60)) - (self.clock() - self.t0)

    def _finish(self, verdict: str, reason: str, last=None) -> dict:
        if self.state in ("moving", "settling") and verdict == UNREADABLE:
            self.robot.stop()
        truth = self.truth() if self.truth else None
        self.result = {"verdict": verdict, "reason": reason, "confidence": getattr(last, "confidence", 0.0) if last else 0.0,
                       "needle_value": getattr(last, "needle_value", None) if last else None, "attempts": self.attempt,
                       "adjustments": self.adjustments, "turned_deg": round(math.degrees(self.turned), 1),
                       "backed_m": round(self.backed, 2), "evidence": list(self.evidence),
                       "elapsed_s": round(self.clock() - self.t0, 1), "truth": truth}
        self.state = "done"
        self.record("gauge_inspection_done", {"asset_id": self.asset_id, "robot_id": self.robot.robot_id, "purpose": self.purpose,
                                              "verdict": verdict, "reason": reason, "attempts": self.attempt,
                                              "adjustments": self.adjustments, "elapsed_s": self.result["elapsed_s"], "truth": truth,
                                              "cosmos_agrees_with_truth": None if truth is None or verdict == UNREADABLE
                                              else truth == verdict})
        return self.result

    def _plan(self, adjust: str):
        cfg = self.cfg
        if self.adjustments >= int(cfg.get("max_adjustments", 3)):
            return "adjustment limit reached"
        x, y, yaw = (float(v) for v in self.robot.pose[:3])
        if adjust in ("turn_left", "turn_right"):
            step = math.radians(float(cfg.get("turn_step_deg", 10))) * (1 if adjust == "turn_left" else -1)
            if abs(self.turned + step) > math.radians(float(cfg.get("max_turn_deg", 30))) + 1e-9:
                return "turn limit reached"
            return ("turn", x, y, _wrap(yaw + step), step, 0.0)
        if adjust == "move_back":
            d = float(cfg.get("back_step_m", 0.15))
            if self.backed + d > float(cfg.get("max_back_m", 0.45)) + 1e-9:
                return "move-back limit reached"
            nx, ny = x - math.cos(yaw) * d, y - math.sin(yaw) * d
            why = self.walkable(nx, ny) if self.walkable else ""
            if why:
                return f"cannot step back: {why}"
            return ("move", nx, ny, yaw, 0.0, d)
        return "no adjustment suggested"

    # ------------------------------------------------------------------ v0.5.3: which image goes to the model
    def _sources(self) -> list:
        src = (self.cfg.get("crop") or {}).get("source", ["vlm", "projection"])
        return [src] if isinstance(src, str) else list(src)

    def _use_detection(self, full, proj_crop) -> bool:
        crop_on = (self.cfg.get("crop") or {}).get("enabled", True)
        src = self._sources()
        if not (crop_on and self.detect and full and "vlm" in src):
            return False
        return src[0] == "vlm" or proj_crop is None

    def _fallback(self, why):
        """(image, note): the next source after the vision model's box, else the uncropped frame."""
        f = self._frame
        if "projection" in self._sources() and f["proj_crop"]:
            return f["proj_crop"], "projection", f["proj_info"]
        info = dict(f["proj_info"] or {})
        info.update({"cropped": False, "reason": why or info.get("reason") or "no crop source available"})
        return f["full"] or f["proj_crop"], "full", info

    def _after_detection(self):
        f = self._frame
        w, h = f["size"]
        rec = {"asset_id": self.asset_id, "robot_id": self.robot.robot_id, "attempt": self.attempt}
        try:
            det, raw, latency = self._future.result()
        except Exception as exc:  # noqa: BLE001 - detection failed: still read the gauge from what we have
            self.record("gauge_detect", {**rec, "found": False, "reason": f"error: {str(exc)[:200]}"})
            return self._send(*self._fallback(f"box detection failed: {str(exc)[:80]}"))
        dcfg = self.cfg.get("detect") or {}
        ok = det.visible and det.box_px is not None and det.confidence >= float(dcfg.get("min_confidence", 0.5))
        pinfo = f["proj_info"] or {}
        from ..perception import gauge_crop as gc
        proj_box = gc.dial_box(pinfo.get("centre_px"), pinfo.get("radius_px")) if pinfo.get("cropped") else None
        self.record("gauge_detect", {**rec, "found": ok, "box_px": det.box_px, "raw_box": det.raw_box,
                                     "confidence": det.confidence, "adjust": det.adjust, "reason": det.reason,
                                     "latency_s": round(latency, 1), "frame": [w, h],
                                     "iou_with_projection": round(gc.iou(det.box_px, proj_box), 2) if proj_box and det.box_px else None,
                                     "raw": raw[-400:]})
        if not ok:
            why = det.reason or ("confidence %.2f" % det.confidence)
            return self._send(*self._fallback(f"vision model did not find the gauge ({why})"))
        try:
            crop, info = gc.crop_from_box(gc.decode(f["full"]), det.box_px, gc.crop_cfg(self.cfg))
        except Exception as exc:  # noqa: BLE001
            return self._send(*self._fallback(f"crop failed: {exc}"))
        if crop is None:
            return self._send(*self._fallback(info.get("reason")))
        return self._send(gc.encode(crop), "vlm", {**info, "detect_confidence": det.confidence})

    def _send(self, image, source: str, info: dict):
        if source != "full":
            cpath = self.save_evidence(image, self.attempt, "_crop")
            if cpath:
                self.evidence.append(cpath)
        self.record("gauge_crop", {"asset_id": self.asset_id, "robot_id": self.robot.robot_id, "attempt": self.attempt,
                                   "source": source, **(info or {})})
        timeout = max(1.0, min(float(self.cfg.get("call_timeout_s", 55)), self._left() - 0.5))
        self._future = self.submit(self.ask, image, timeout)
        self.state = "asking"
        return None

    def step(self):
        """Call every tick. None while busy; the result dict when finished."""
        if self.state == "done":
            return self.result
        if self._left() <= 0:
            return self._finish(UNREADABLE, f"time limit {self.cfg.get('total_time_s', 60)} s reached ({self.state})",
                                self.answers[-1] if self.answers else None)
        if self.state == "capture":
            jpeg = self.capture(self.robot)
            if self.retry_marker is not None and jpeg is self.retry_marker:
                return None  # camera warming up: stand still and ask again (bounded by the time limit)
            if callable(jpeg):
                try:
                    jpeg = jpeg()
                except Exception:  # noqa: BLE001
                    jpeg = None
            self.attempt += 1
            proj_crop, proj_info = None, None
            if isinstance(jpeg, dict):  # v0.5.2: {"full": frame, "crop": close-up from the projection or None, "info": {...}}
                full, proj_crop, proj_info = jpeg.get("full"), jpeg.get("crop"), jpeg.get("info") or {}
            else:
                full = jpeg
            if not full and not proj_crop:
                return self._finish(UNREADABLE, "no camera frame (is Capture Quadruped Camera on?)")
            path = self.save_evidence(full or proj_crop, self.attempt)
            if path:
                self.evidence.append(path)
            self._frame = {"full": full, "proj_crop": proj_crop, "proj_info": proj_info}
            if self._use_detection(full, proj_crop):
                try:
                    from ..perception.gauge_crop import image_size
                    w, h = image_size(full)
                except Exception as exc:  # noqa: BLE001 - not a decodable image: read it uncropped
                    self.record("gauge_detect", {"asset_id": self.asset_id, "robot_id": self.robot.robot_id,
                                                 "attempt": self.attempt, "found": False, "reason": f"image: {exc}"})
                    return self._send(*self._fallback("image could not be decoded"))
                self._frame["size"] = (w, h)
                dto = float((self.cfg.get("detect") or {}).get("timeout_s", 30))
                self._future = self.submit(self.detect, full, max(1.0, min(dto, self._left() - 0.5)), w, h)
                self.state = "detecting"
                return None
            return self._send(*self._fallback(None))
        if self.state == "detecting":
            if not self._future.done():
                return None
            return self._after_detection()
        if self.state == "asking":
            if not self._future.done():
                return None
            try:
                verdict, raw, latency = self._future.result()
            except Exception as exc:  # noqa: BLE001 - CosmosError or anything else: unreadable, never a guess
                self.record("cosmos_failed", {"asset_id": self.asset_id, "robot_id": self.robot.robot_id,
                                              "attempt": self.attempt, "error": str(exc)[:300]})
                return self._finish(UNREADABLE, f"Cosmos unavailable or invalid reply: {str(exc)[:160]}")
            self.answers.append(verdict)
            self.record("cosmos_answer", {"asset_id": self.asset_id, "robot_id": self.robot.robot_id, "attempt": self.attempt,
                                          **verdict.to_dict(), "latency_s": round(latency, 1),
                                          "evidence": self.evidence[-1] if self.evidence else None, "raw": raw[-600:]})
            thr = float(self.cfg.get("confidence_threshold", 0.75))
            if verdict.pressure in ("high", "normal", "low") and verdict.confidence >= thr:
                return self._finish("high" if verdict.pressure == "high" else "normal",
                                    f"Cosmos: {verdict.pressure} ({verdict.confidence:.2f}) {verdict.reason}", verdict)
            plan = self._plan(verdict.adjust)
            if isinstance(plan, str):
                why = "gauge unclear" if verdict.pressure == "unclear" else f"confidence {verdict.confidence:.2f} below {thr}"
                return self._finish(UNREADABLE, f"{why}; {plan}", verdict)
            kind, x, y, yaw, dturn, dback = plan
            turn = getattr(self.robot, "turn_to", None)
            if kind == "turn" and callable(turn):
                turn(yaw)
            else:
                self.robot.go_to(x, y, yaw, name=f"{self.asset_id}:{verdict.adjust}")
            self.adjustments += 1
            self.turned += dturn
            self.backed += dback
            self._goal, self._move_t0, self.state = (x, y, yaw), self.clock(), "moving"
            self.record("gauge_adjust", {"asset_id": self.asset_id, "robot_id": self.robot.robot_id, "adjust": verdict.adjust,
                                         "turned_deg": round(math.degrees(self.turned), 1), "backed_m": round(self.backed, 2)})
            return None
        if self.state == "moving":
            if self.robot.status == "error":
                return self._finish(UNREADABLE, f"robot error while adjusting: {getattr(self.robot, 'note', '')}")
            arrived = getattr(self.robot, "goal", None) is None and self.robot.status == "idle"
            if arrived or self.clock() - self._move_t0 > float(self.cfg.get("move_timeout_s", 12)):
                if not arrived:
                    self.robot.stop()
                    self.record("gauge_adjust_timeout", {"asset_id": self.asset_id, "robot_id": self.robot.robot_id})
                gx, gy, gyaw = self._goal
                self.record("gauge_adjust_result", {"robot_id": self.robot.robot_id,
                                                    "yaw_error_deg": round(math.degrees(_wrap(self.robot.pose[2] - gyaw)), 1),
                                                    "pos_error_m": round(math.hypot(self.robot.pose[0] - gx, self.robot.pose[1] - gy), 2)})
                self._settle_until, self.state = self.clock() + float(self.cfg.get("settle_s", 0.6)), "settling"
            return None
        if self.state == "settling" and self.clock() >= self._settle_until:
            self.state = "capture"
        return None


class GaugeInspector:
    """Plugged into SkillExecutor by the scenario. Knows which assets have a gauge, where to stand, how to ask Cosmos."""

    PURPOSES = ("read_gauge", "reinspect", "verify", "inspect")

    def __init__(self, *, cfg, specs: dict, mounts: dict, client_for, truth_for, evidence_store, audit, pool, site=None,
                 is_free=None, clock=time.time, retry_marker=None, gauge_pos=None, capture_gauge=None):
        self.cfg, self.specs, self.mounts = cfg, specs, mounts  # asset_id -> gauge model spec / {"camera_distance_m": ..}
        self.gauge_pos = gauge_pos  # v0.5.1: asset_id -> (x, y, z) of the needle (gauge inside the asset USD)
        self.capture_gauge = capture_gauge  # v0.5.2: (robot, asset_id) -> {"full", "crop", "info"} | CAPTURE_RETRY | None
        self.client_for, self.truth_for = client_for, truth_for
        self.evidence, self.audit, self.pool, self.site, self.is_free = evidence_store, audit, pool, site, is_free
        self.clock, self.retry_marker = clock, retry_marker

    def handles(self, asset_id: str, purpose: str) -> bool:
        return asset_id in self.specs and purpose in self.PURPOSES

    def standoff(self, asset) -> float:
        from ..gauges.gauge_math import inspection_distance
        return inspection_distance(asset, self.mounts.get(asset.asset_id, {}))

    def inspect_pose(self, asset, robot_radius: float) -> tuple:
        """Where the robot stands to read this asset's gauge (in front of the dial, facing it)."""
        from ..gauges.gauge_math import inspect_pose
        pos = self.gauge_pos(asset.asset_id) if self.gauge_pos else None
        dist = float(self.mounts.get(asset.asset_id, {}).get("camera_distance_m", 0.7))
        if pos is None:
            pos = (asset.position[0], asset.position[1], 0.0)
        return inspect_pose(asset, pos[:2], dist, robot_radius)

    def _walkable(self, x, y):
        why = self.site.why_not(x, y) if self.site is not None else ""
        if not why and self.is_free is not None and not self.is_free(x, y):
            why = f"({x:.1f}, {y:.1f}) is too close to an obstacle"
        return why

    def start(self, run, capture_fn) -> GaugeSession:
        aid = run.assignment.target_asset_id
        client = self.client_for(aid)

        def save(jpeg, attempt, suffix=""):
            try:
                return self.evidence.save(run.robot.robot_id, jpeg, f"{run.purpose}_a{attempt}{suffix}", asset_id=aid)
            except Exception:  # noqa: BLE001
                return ""

        def record(kind, payload):
            self.audit.record(kind, payload)
            if kind == "cosmos_answer" and payload.get("evidence"):
                try:
                    with open(str(payload["evidence"]).rsplit(".", 1)[0] + ".cosmos.json", "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=1, default=str)
                except OSError:
                    pass

        self.audit.record("gauge_inspection_started", {"asset_id": aid, "robot_id": run.robot.robot_id, "purpose": run.purpose,
                                                       "cosmos_model": getattr(client, "model", ""),
                                                       "cosmos_url": getattr(client, "base_url", "")})
        return GaugeSession(robot=run.robot, asset_id=aid, purpose=run.purpose, cfg=self.cfg, ask=client.ask,
                            capture=(lambda robot: self.capture_gauge(robot, aid)) if self.capture_gauge else capture_fn,
                            save_evidence=save, record=record, submit=self.pool.submit,
                            clock=self.clock, walkable=self._walkable, truth=lambda: self.truth_for(aid),
                            retry_marker=self.retry_marker, detect=self._detector(client))

    def _detector(self, client):
        """v0.5.3: the bounding-box call, configured by cosmos.json "detect"."""
        if not hasattr(client, "detect"):
            return None
        from ..perception.gauge_crop import detect_cfg
        d = detect_cfg(self.cfg)
        return lambda jpeg, timeout, w, h: client.detect(jpeg, timeout, w, h, d["format"], d["thinking"])

    def event_for(self, result: dict, run, settings):
        from ..schemas.fleet_event import FleetEvent, GaugeReading
        aid = run.assignment.target_asset_id
        spec = self.specs[aid]
        verdict = result["verdict"]
        if verdict == "high":
            etype, in_range = "anomaly", False
        elif verdict == "normal":
            etype, in_range = ("verification" if run.purpose == "verify" else "gauge_reading"), True
        else:
            etype, in_range = "inconclusive", None
        conf = round(max(float(result["confidence"]), float(self.cfg.get("confidence_threshold", 0.75))), 2) \
            if verdict != UNREADABLE else 0.3
        value = result.get("needle_value")
        note = ""
        if verdict == UNREADABLE:
            value = None
        elif not isinstance(value, (int, float)):
            value, note = float(spec["high_value"] if verdict == "high" else spec["normal_value"]), " (value from the needle zone)"
        obs = f"[Cosmos] {verdict}: {result['reason']}{note}"
        ev = FleetEvent(robot_id=run.robot.robot_id, provenance=getattr(run.robot, "provenance", "simulated"), asset_id=aid,
                        event_type=etype, observation=obs[:600], confidence=conf,
                        gauge=GaugeReading(value, str(spec.get("unit", ""))[:16], in_range, conf, 0),
                        evidence=[e for e in result["evidence"] if e], ttl_s=int(getattr(settings, "event_ttl_s", 120)),
                        incident_id=run.incident_id)
        return FleetEvent.from_dict(ev.to_dict())
