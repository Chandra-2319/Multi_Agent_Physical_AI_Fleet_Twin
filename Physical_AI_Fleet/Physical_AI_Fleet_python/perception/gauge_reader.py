# SPDX-License-Identifier: Apache-2.0
"""Gauge pipeline: localize -> crop ROI -> upscale -> read.

If the ROI is smaller than `min_roi_px` the reading is marked low-confidence and the coordinator
asks for a closer standoff (read_gauge / reinspect_asset).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class GaugeROI:
    cx: int
    cy: int
    r: int
    method: str

    @property
    def side_px(self) -> int:
        return 2 * self.r


@dataclass
class GaugeResult:
    value: float | None
    confidence: float
    roi: GaugeROI | None
    crop: np.ndarray | None
    note: str = ""


@dataclass
class DialSpec:
    min_value: float = 0.0
    max_value: float = 10.0
    min_angle_deg: float = 225.0
    max_angle_deg: float = -45.0


def localize(rgb: np.ndarray, fallback_roi: tuple | None = None) -> GaugeROI | None:
    h, w = rgb.shape[:2]
    if cv2 is not None:
        gray = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(h, w) // 8, param1=120, param2=40,
                                   minRadius=max(8, min(h, w) // 60), maxRadius=min(h, w) // 2)
        if circles is not None:
            cx, cy, r = max(circles[0], key=lambda c: c[2])
            return GaugeROI(int(cx), int(cy), int(r), "hough")
    if fallback_roi:
        fx, fy, fr = fallback_roi
        return GaugeROI(int(fx * w), int(fy * h), int(fr * min(h, w)), "fallback")
    return None


def crop_and_upscale(rgb: np.ndarray, roi: GaugeROI, scale: int = 4, margin: float = 1.15) -> np.ndarray:
    h, w = rgb.shape[:2]
    half = int(roi.r * margin)
    crop = rgb[max(0, roi.cy - half):min(h, roi.cy + half), max(0, roi.cx - half):min(w, roi.cx + half)]
    if crop.size == 0:
        return crop
    if cv2 is not None:
        return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return np.kron(crop, np.ones((scale, scale, 1), dtype=crop.dtype))


def read_needle(crop: np.ndarray, dial: DialSpec, radius_px: float | None = None) -> tuple:
    """Radial ray casting: the needle is the darkest ray inside the dial sweep. Returns (value, confidence)."""
    if crop.size == 0:
        return None, 0.0
    gray = crop.astype(np.float32).mean(axis=2) if crop.ndim == 3 else crop.astype(np.float32)
    h, w = gray.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = radius_px or 0.85 * min(h, w) / 2.0
    sweep = dial.min_angle_deg - dial.max_angle_deg
    angles = np.radians(dial.min_angle_deg - np.linspace(0.0, sweep, int(sweep * 2) + 1))
    radii = np.linspace(0.2 * r, 0.8 * r, 40)
    xs = (cx + np.outer(np.cos(angles), radii)).round().astype(int).clip(0, w - 1)
    ys = (cy - np.outer(np.sin(angles), radii)).round().astype(int).clip(0, h - 1)
    darkness = 255.0 - gray[ys, xs].mean(axis=1)
    best = int(np.argmax(darkness))
    contrast = float(darkness[best] - np.median(darkness))
    if contrast < 25.0:
        return None, 0.1
    value = dial.min_value + best / (len(angles) - 1) * (dial.max_value - dial.min_value)
    return round(value, 2), round(float(min(0.95, 0.4 + contrast / 200.0)), 2)


def read_gauge(rgb: np.ndarray, dial: DialSpec, *, min_roi_px: int = 48, upscale: int = 4,
               fallback_roi: tuple | None = (0.5, 0.5, 0.15)) -> GaugeResult:
    roi = localize(rgb, fallback_roi)
    if roi is None:
        return GaugeResult(None, 0.0, None, None, "gauge not found in frame")
    crop = crop_and_upscale(rgb, roi, upscale)
    value, conf = read_needle(crop, dial, radius_px=roi.r * upscale)
    note = f"roi {roi.side_px}px via {roi.method}"
    if roi.side_px < min_roi_px:
        conf = min(conf, 0.3)
        note += f"; under-resolved (<{min_roi_px}px): move closer or zoom"
    if roi.method == "fallback":
        conf = min(conf, 0.5)
    return GaugeResult(value, conf, roi, crop, note)
