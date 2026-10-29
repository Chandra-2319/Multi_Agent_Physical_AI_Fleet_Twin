import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from Physical_AI_Fleet_python.perception.gauge_reader import DialSpec, read_gauge  # noqa: E402


def synth_gauge(value, size=900, r=120, center=(600, 380)):
    img = np.full((size * 9 // 16, size, 3), 200, np.uint8)
    cv2.circle(img, center, r, (255, 255, 255), -1)
    cv2.circle(img, center, r, (20, 20, 20), 4)
    dial = DialSpec()
    frac = (value - dial.min_value) / (dial.max_value - dial.min_value)
    ang = math.radians(dial.min_angle_deg - frac * (dial.min_angle_deg - dial.max_angle_deg))
    tip = (int(center[0] + 0.85 * r * math.cos(ang)), int(center[1] - 0.85 * r * math.sin(ang)))
    cv2.line(img, center, tip, (200, 0, 0), 5)
    return img[..., ::-1].copy()


@pytest.mark.parametrize("value", [2.0, 5.0, 8.5])
def test_reads_synthetic_gauge(value):
    res = read_gauge(synth_gauge(value), DialSpec())
    assert res.roi and res.roi.method == "hough"
    assert res.value is not None and abs(res.value - value) < 0.6


def test_small_gauge_flagged_under_resolved():
    assert read_gauge(synth_gauge(5.0, r=18), DialSpec(), min_roi_px=48).confidence <= 0.3
