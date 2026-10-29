# SPDX-License-Identifier: Apache-2.0
"""Frame capture for real, simulated and operator-phone sources + evidence storage.

v0.4.4
  * Evidence files are named <date>_<time>_<asset>_<robot>_<purpose>.jpg (e.g. 20261022_121258_pump-01_spot-2_read_gauge.jpg).
  * SimCameraCapture is created when the robot is added (prepare) and only reads the annotator after the camera has
    rendered for `warmup_s` of simulated time; the first `discard_frames` valid frames are dropped (they can still
    hold an empty / not fully shaded buffer). This avoids Isaac Sim's "Annotator 'rgb' returned None" warning.
  * frame_stats(): brightness of the whole frame and of its centre (where the asset is); "dark_target" flags a
    centre that is almost black while the rest of the frame is lit.
"""
from __future__ import annotations
import io
import re
import time
from pathlib import Path
import numpy as np
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    rgb = np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)
    if cv2 is not None:
        ok, buf = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            return buf.tobytes()
    from PIL import Image
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="JPEG", quality=quality)
    return out.getvalue()


def decode_jpeg(data: bytes) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img[..., ::-1].copy()
    from PIL import Image
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def frame_stats(rgb: np.ndarray, centre: float = 0.4) -> dict:
    """mean brightness of the frame and of its centre box (`centre` = fraction of width / height)."""
    a = np.asarray(rgb)[..., :3].astype(np.float32)
    h, w = a.shape[:2]
    ch, cw = max(1, int(h * centre)), max(1, int(w * centre))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    whole, mid = float(a.mean()), float(a[y0:y0 + ch, x0:x0 + cw].mean())
    return {"mean_brightness": round(whole, 1), "centre_brightness": round(mid, 1),
            "dark_target": bool(mid < 15.0 and whole > 40.0), "almost_black": bool(whole < 3.0)}


def _safe(part: str) -> str:
    return _SAFE.sub("-", str(part or "")).strip("-") or "x"


class EvidenceStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root) / "evidence"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def path_for(self, robot_id: str, tag: str = "frame", asset_id: str = "") -> Path:
        parts = [time.strftime("%Y%m%d_%H%M%S")] + ([_safe(asset_id)] if asset_id else []) + [_safe(robot_id), _safe(tag)]
        path = self.root / ("_".join(parts) + ".jpg")
        n = 2
        while path.exists():  # two captures in the same second: never overwrite
            path = self.root / ("_".join(parts) + f"_{n}.jpg")
            n += 1
        return path

    def save(self, robot_id: str, jpeg: bytes, tag: str = "frame", asset_id: str = "") -> str:
        path = self.path_for(robot_id, tag, asset_id)
        try:
            path.write_bytes(jpeg)
            return str(path)
        except OSError:
            return ""


class SimCameraCapture:
    """Wraps an Isaac Sim camera prim. The Isaac Sim import is lazy so this module imports outside Isaac Sim."""

    def __init__(self, prim_path: str, resolution=(1280, 720), warmup_s: float = 0.5, discard_frames: int = 2) -> None:
        self.prim_path = prim_path
        self.resolution = resolution
        self.warmup_s = warmup_s
        self.discard_frames = discard_frames
        self._cam = None
        self._rendered_s = 0.0
        self._discarded = 0
        self.error = ""

    def prepare(self) -> bool:
        """Create and initialise the camera now (robot added), so it has rendered for a while before it is needed."""
        if self._cam is not None:
            return True
        try:
            from isaacsim.sensors.camera import Camera
            self._cam = Camera(prim_path=self.prim_path, resolution=self.resolution)
            self._cam.initialize()
            self._rendered_s = 0.0
            self._discarded = 0
            self.error = ""
            return True
        except Exception as exc:  # noqa: BLE001
            self._cam, self.error = None, f"{type(exc).__name__}: {exc}"
            return False

    active: bool = True

    def set_active(self, on: bool) -> bool:
        """v0.5.4: render only while the robot is about to capture. Off = the camera's render product is paused (no extra
        rendered view per robot per frame); on = resumed and warmed up again before a frame is used. True if it changed."""
        on = bool(on)
        if self._cam is None or on == self.active:
            return False
        fn = getattr(self._cam, "resume" if on else "pause", None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                self.error = f"{'resume' if on else 'pause'}: {type(exc).__name__}: {exc}"
                return False
        self.active = on
        if on:
            self._rendered_s, self._discarded = 0.0, 0
        return True

    def reset(self) -> None:
        """v0.5.5: after STOP / RESET the render product may not resume reliably: forget it, it is created again."""
        self._cam = None
        self.active = True
        self._rendered_s, self._discarded = 0.0, 0

    def advance(self, dt: float) -> None:
        if self._cam is not None and self.active:
            self._rendered_s += float(dt)

    @property
    def ready(self) -> bool:
        return self._cam is not None and self._rendered_s >= self.warmup_s

    def grab_rgb(self) -> np.ndarray | None:
        """None until the camera is warm (the caller waits and asks again); then an RGB uint8 frame."""
        if self._cam is None and not self.prepare():
            return None
        if not self.active:
            self.set_active(True)
            return None
        if self._rendered_s < self.warmup_s:
            return None
        try:
            rgba = self._cam.get_rgba()
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            return None
        if rgba is None or getattr(rgba, "size", 0) == 0 or np.ndim(rgba) != 3:
            return None
        if self._discarded < self.discard_frames:
            self._discarded += 1
            return None
        return np.asarray(rgba)[..., :3].astype(np.uint8)


    def project(self, points):
        """World points -> pixel coords (N, 2) with Isaac Sim's own camera model, or None when it is not available.
        v0.5.2: used to crop the frame to the gauge."""
        fn = getattr(self._cam, "get_image_coords_from_world_points", None) if self._cam is not None else None
        if not callable(fn):
            return None
        try:
            uv = np.asarray(fn(np.asarray(points, dtype=float).reshape(-1, 3)), dtype=float).reshape(-1, 2)
        except Exception as exc:  # noqa: BLE001
            self.error = f"projection: {type(exc).__name__}: {exc}"
            return None
        return uv if np.all(np.isfinite(uv)) else None


def load_operator_photo(path: str) -> bytes | None:
    p = Path(path.strip().strip('"'))
    return p.read_bytes() if p.is_file() else None
