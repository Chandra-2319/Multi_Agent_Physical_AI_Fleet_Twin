# SPDX-License-Identifier: Apache-2.0
"""Crop the robot's camera frame to the pressure gauge before the vision model sees it (v0.5.2).

Why: from where the robot stands the dial is only ~2 % of a 1280 x 720 frame; the model downsizes the whole image and
cannot see the needle or the red arc, so it guesses ("0.4 MPa, normal"). The same model reads a close-up correctly.

How: the gauge's world position is known (its needle pivot in the asset USD), so it is projected into the image -
with the Isaac Sim camera's own projection when available, otherwise from the robot pose and the camera mount below -
and a square around the dial (crop.margin x the dial diameter) is cut out and enlarged to crop.output_px.
Pure numpy (PIL / cv2 only used for a nicer resize when they are installed).
"""
from __future__ import annotations

import math

import numpy as np

CAM_OFFSET = (0.35, 0.0, 0.15)  # inspection camera position in the robot's base frame (scenario._add_camera)
FOCAL_MM = 18.0
H_APERTURE_MM = 20.955  # USD default horizontal aperture
DEFAULTS = {"enabled": True, "margin": 1.8, "output_px": 768, "min_px": 96, "dial_radius_m": 0.0,
            "source": ["vlm", "projection"]}  # v0.5.3: where the gauge's box comes from, in order of preference
THINKING = True  # v0.5.4: every vision query reasons (box and reading); deliberately not configurable
DETECT_DEFAULTS = {"format": "norm1000", "timeout_s": 45, "min_confidence": 0.5}


def detect_cfg(cosmos_cfg: dict) -> dict:
    d = dict(DETECT_DEFAULTS)
    d.update((cosmos_cfg or {}).get("detect") or {})
    d["thinking"] = THINKING  # a "thinking" key in cosmos.json is ignored
    return d


def crop_cfg(cosmos_cfg: dict) -> dict:
    c = dict(DEFAULTS)
    c.update((cosmos_cfg or {}).get("crop") or {})
    return c


def fx_pixels(width: int, focal_mm: float = FOCAL_MM, h_aperture_mm: float = H_APERTURE_MM) -> float:
    """RTX fits the horizontal aperture to the image width; pixels are square."""
    return float(focal_mm) / float(h_aperture_mm) * float(width)


def camera_pose_from_robot(x: float, y: float, yaw: float, base_z: float, offset=CAM_OFFSET) -> tuple:
    """(position, forward, right, up) of the inspection camera for a robot base at (x, y, base_z) facing yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    pos = np.array([x + c * offset[0] - s * offset[1], y + s * offset[0] + c * offset[1], base_z + offset[2]])
    fwd = np.array([c, s, 0.0])
    right = np.array([s, -c, 0.0])
    return pos, fwd, right, np.array([0.0, 0.0, 1.0])


def project(points, cam_pose: tuple, width: int, height: int, fx: float):
    """World points (N, 3) -> pixel coords (N, 2) (u right, v down) and depth (N,). Points behind the camera: nan."""
    pos, fwd, right, up = cam_pose
    p = np.asarray(points, float).reshape(-1, 3) - pos
    depth = p @ fwd
    xr, yu = p @ right, p @ up
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2.0 + fx * xr / depth
        v = height / 2.0 - fx * yu / depth
    uv = np.stack([u, v], axis=1)
    uv[depth <= 1e-6] = np.nan
    return uv, depth


def crop_box(centre_uv, radius_px: float, width: int, height: int, margin: float, min_px: int):
    """Square (x0, y0, x1, y1) of side max(min_px, 2 * margin * radius_px), shifted (not shrunk) to stay in the frame.
    None when the centre is not in the frame."""
    u, v = float(centre_uv[0]), float(centre_uv[1])
    if not (math.isfinite(u) and math.isfinite(v)) or not (0 <= u < width and 0 <= v < height):
        return None
    side = int(round(max(float(min_px), 2.0 * float(margin) * max(float(radius_px), 1.0))))
    side = min(side, width, height)
    x0 = int(round(u - side / 2.0))
    y0 = int(round(v - side / 2.0))
    x0 = min(max(x0, 0), width - side)
    y0 = min(max(y0, 0), height - side)
    return x0, y0, x0 + side, y0 + side


def resize(rgb: np.ndarray, size: int) -> np.ndarray:
    try:
        from PIL import Image
        return np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)).resize((size, size), Image.LANCZOS))
    except Exception:  # noqa: BLE001 - no PIL: nearest neighbour
        h, w = rgb.shape[:2]
        ys = (np.arange(size) * h / size).astype(int).clip(0, h - 1)
        xs = (np.arange(size) * w / size).astype(int).clip(0, w - 1)
        return rgb[ys][:, xs, :3]


def locate(centre_xyz, radius_m: float, cam_pose: tuple, width: int, height: int, fx: float) -> tuple:
    """(centre_uv, radius_px, depth) of the dial for a camera pose."""
    c = np.asarray(centre_xyz, float)
    up = np.asarray(cam_pose[3], float)
    uv, depth = project(np.stack([c, c + up * float(radius_m)]), cam_pose, width, height, fx)
    r_px = float(np.hypot(*(uv[1] - uv[0]))) if np.all(np.isfinite(uv)) else float("nan")
    return uv[0], r_px, float(depth[0])


def crop_gauge(rgb: np.ndarray, centre_uv, radius_px: float, cfg: dict) -> tuple:
    """(crop RGB or None, info dict)."""
    h, w = rgb.shape[:2]
    info = {"centre_px": [round(float(centre_uv[0]), 1), round(float(centre_uv[1]), 1)] if centre_uv is not None else None,
            "radius_px": round(float(radius_px), 1) if radius_px == radius_px else None}
    if centre_uv is None or not np.all(np.isfinite(np.asarray(centre_uv, float))):
        return None, {**info, "cropped": False, "reason": "gauge position could not be projected into the image"}
    box = crop_box(centre_uv, radius_px if radius_px == radius_px else 0.0, w, h, float(cfg["margin"]), int(cfg["min_px"]))
    if box is None:
        return None, {**info, "cropped": False, "reason": "the gauge is outside the camera image"}
    x0, y0, x1, y1 = box
    out = resize(np.asarray(rgb)[y0:y1, x0:x1], int(cfg["output_px"]))
    return out, {**info, "cropped": True, "box": [x0, y0, x1, y1], "dial_fraction": round(2 * float(radius_px) / (x1 - x0), 2)
                 if radius_px == radius_px else None}


# ------------------------------------------------------------------ v0.5.3: crop from a box found in the image itself
def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dial_box(centre_uv, radius_px) -> tuple | None:
    if centre_uv is None or radius_px is None:
        return None
    u, v, r = float(centre_uv[0]), float(centre_uv[1]), float(radius_px)
    return (u - r, v - r, u + r, v + r)


def crop_from_box(rgb: np.ndarray, box_px: tuple, cfg: dict) -> tuple:
    """Crop around a detected dial box (square, margin x the box's larger side). (crop RGB or None, info)."""
    x0, y0, x1, y1 = box_px
    centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    crop, info = crop_gauge(rgb, centre, max(x1 - x0, y1 - y0) / 2.0, cfg)
    info["dial_box"] = [round(float(v), 1) for v in box_px]
    return crop, info


def image_size(jpeg: bytes) -> tuple:
    from PIL import Image
    import io
    with Image.open(io.BytesIO(jpeg)) as im:
        return im.size


def decode(jpeg: bytes) -> np.ndarray:
    from PIL import Image
    import io
    with Image.open(io.BytesIO(jpeg)) as im:
        return np.asarray(im.convert("RGB"))


def encode(rgb: np.ndarray, quality: int = 92) -> bytes:
    from PIL import Image
    import io
    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)).save(out, format="JPEG", quality=quality)
    return out.getvalue()
