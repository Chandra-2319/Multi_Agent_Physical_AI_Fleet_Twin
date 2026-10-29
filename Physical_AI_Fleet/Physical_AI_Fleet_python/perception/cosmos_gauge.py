# SPDX-License-Identifier: Apache-2.0
"""Vision-model gauge reader (v0.5.0 Cosmos, v0.5.1 any OpenAI-compatible VLM): one image -> {pressure, adjust, ...}.

v0.5.1: the URL is used as-is when it already names an endpoint (".../chat/completions", NVIDIA per-model ".../vlm/...",
".../cosmos/..."); thinking is switched with chat_template_kwargs.enable_thinking (Nemotron VL, Nebius models); the
Cosmos-style "<think>" instruction is only added for Cosmos models. temperature 0, one call per image.

OpenAI-compatible POST {base_url}/chat/completions. Works with NVIDIA's hosted free endpoint
(https://integrate.api.nvidia.com/v1, model nvidia/cosmos3-nano-reasoner) and with a local server. The image is sent
inline as a base64 JPEG data URL. Stdlib only. No mock: every failure is a CosmosError (the caller treats it as
unreadable).
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

PRESSURES = ("high", "normal", "low", "unclear")
ADJUSTS = ("none", "turn_left", "turn_right", "move_back")
_ALIASES = {"left": "turn_left", "turn left": "turn_left", "rotate_left": "turn_left", "right": "turn_right",
            "turn right": "turn_right", "rotate_right": "turn_right", "back": "move_back", "move back": "move_back",
            "step_back": "move_back", "backward": "move_back", "": "none", "null": "none"}
SYSTEM_PROMPT = "You inspect industrial pressure gauges for an inspection robot. Look carefully at the needle."


class CosmosError(RuntimeError):
    pass


@dataclass
class Verdict:
    pressure: str
    adjust: str
    confidence: float
    reason: str = ""
    needle_value: float | None = None
    needle_zone: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_prompt(description: str, unit: str, reasoning: bool = True) -> str:
    fmt = ('{"pressure": "high|normal|low|unclear", "adjust": "none|turn_left|turn_right|move_back", '
           '"confidence": 0.0-1.0, "needle_value": number or null, "needle_zone": "red|white|unknown", "reason": "short"}')
    text = (f"{description}\n"
            "Question: What is the pressure? If unclear, suggest which way the quadruped should move to get a clear image.\n"
            "Rules: judge by the DIRECTION the needle points from its centre pivot. Pressure is HIGH if the needle points into "
            "the angular sector covered by the solid red arc - even if the needle tip stops short of the arc (the needle is "
            "shorter than the dial); otherwise NORMAL (LOW if it points near the start of the scale). If the dial is cut off, blurred, too small, seen at a steep angle or "
            "hidden, answer \"unclear\" and set adjust: turn_left or turn_right if the gauge is toward that side of the "
            "image or partly out of view on that side (left/right as seen in the image), move_back if the gauge is too "
            f"close or cut off at the top and bottom. needle_value in {unit}.\n"
            f"Final answer: ONLY this JSON object: {fmt}")
    if reasoning:
        text += ("\nAnswer the question using the following format: <think> Your reasoning. </think> "
                 "Write your final answer immediately after the </think> tag.")
    return text


def image_data_url(jpeg: bytes) -> str:
    kind = "png" if jpeg[:8] == b"\x89PNG\r\n\x1a\n" else "jpeg"
    return f"data:image/{kind};base64," + base64.b64encode(jpeg).decode("ascii")


def build_payload(model: str, jpeg: bytes, prompt: str, max_tokens: int = 4096, temperature: float = 0.0,
                  thinking: bool | None = None) -> dict:
    body = {"model": model, "max_tokens": int(max_tokens), "temperature": float(temperature), "stream": False,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "image_url", "image_url": {"url": image_data_url(jpeg)}}]}]}
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    return body


def endpoint_url(base_url: str) -> str:
    u = (base_url or "").rstrip("/")
    if u.endswith("/chat/completions") or "/vlm/" in u or "/cosmos/" in u:
        return u
    return f"{u}/chat/completions"


def is_cosmos(model: str) -> bool:
    return "cosmos" in (model or "").lower()


def parse_reply(text: str) -> Verdict:
    """Strict: the final answer (after </think>) must contain a JSON object with a valid "pressure"."""
    final = text or ""
    if "</think>" in final:
        final = final.rsplit("</think>", 1)[1]
    final = re.sub(r"<think>.*", "", final, flags=re.DOTALL).strip()
    final = re.sub(r"^```(?:json)?\s*|\s*```$", "", final, flags=re.IGNORECASE)
    start, end = final.find("{"), final.rfind("}")
    if start < 0 or end <= start:
        raise CosmosError(f"no JSON object in the reply: {final[:120]!r}")
    try:
        obj = json.loads(final[start:end + 1])
    except json.JSONDecodeError as exc:
        raise CosmosError(f"malformed JSON in the reply: {exc}") from exc
    if not isinstance(obj, dict):
        raise CosmosError("reply JSON is not an object")
    pressure = str(obj.get("pressure", "")).strip().lower()
    if pressure not in PRESSURES:
        raise CosmosError(f"pressure {pressure!r} not in {PRESSURES}")
    adjust = str(obj.get("adjust", "none") or "none").strip().lower()
    adjust = _ALIASES.get(adjust, adjust.replace(" ", "_"))
    if adjust not in ADJUSTS:
        adjust = "none"
    try:
        conf = min(max(float(obj.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.0
    nv = obj.get("needle_value")
    try:
        nv = float(nv) if nv is not None and str(nv).strip() != "" else None
    except (TypeError, ValueError):
        nv = None
    return Verdict(pressure, adjust, conf, str(obj.get("reason", ""))[:300], nv, str(obj.get("needle_zone", ""))[:16])


# ------------------------------------------------------------------ v0.5.3: find the gauge in the image (bounding box)
BBOX_FORMATS = ("norm1000", "norm1", "pixels")


@dataclass
class Detection:
    visible: bool
    box_px: tuple | None  # (x0, y0, x1, y1) in pixels of the image that was sent, validated; None when not usable
    confidence: float = 0.0
    adjust: str = "none"
    reason: str = ""
    raw_box: list | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_detect_prompt(width: int, height: int, fmt: str = "norm1000", reasoning: bool = False) -> str:
    if fmt == "pixels":
        coords = f"pixel coordinates in this {width} x {height} image (x to the right, y down)"
    elif fmt == "norm1":
        coords = "fractions 0.0-1.0 of the image width and height (0,0 = top-left corner, 1,1 = bottom-right)"
    else:
        coords = "integers 0-1000 relative to the image width and height (0,0 = top-left corner, 1000,1000 = bottom-right)"
    text = ("Is there a round pressure gauge (a dial with a needle) in this image? If yes, give the bounding box of the "
            "whole round dial including its rim, without the stem or fitting below it. "
            f"Box = [x0, y0, x1, y1] (top-left, bottom-right) in {coords}. If no gauge is visible, or only part of it, "
            "say which way the robot should turn or move so the whole gauge is in view (left/right as seen in the image).\n"
            'Final answer: ONLY this JSON object: {"gauge_visible": true|false, "bbox": [x0, y0, x1, y1] or null, '
            '"confidence": 0.0-1.0, "adjust": "none|turn_left|turn_right|move_back"}')
    if reasoning:
        text += ("\nAnswer the question using the following format: <think> Your reasoning. </think> "
                 "Write your final answer immediately after the </think> tag.")
    return text


def _json_obj(text: str) -> dict:
    final = text or ""
    if "</think>" in final:
        final = final.rsplit("</think>", 1)[1]
    final = re.sub(r"<think>.*", "", final, flags=re.DOTALL).strip()
    final = re.sub(r"^```(?:json)?\s*|\s*```$", "", final, flags=re.IGNORECASE)
    start, end = final.find("{"), final.rfind("}")
    if start < 0 or end <= start:
        raise CosmosError(f"no JSON object in the reply: {final[:120]!r}")
    try:
        obj = json.loads(final[start:end + 1])
    except json.JSONDecodeError as exc:
        raise CosmosError(f"malformed JSON in the reply: {exc}") from exc
    if not isinstance(obj, dict):
        raise CosmosError("reply JSON is not an object")
    return obj


def box_to_pixels(raw, width: int, height: int, fmt: str = "norm1000") -> tuple:
    """Model box -> (x0, y0, x1, y1) pixels, clipped. Values that are all <= 1 are read as fractions whatever the format."""
    if isinstance(raw, dict):
        raw = [raw.get(k) for k in ("x0", "y0", "x1", "y1")] if "x0" in raw else \
              [raw.get(k) for k in ("xmin", "ymin", "xmax", "ymax")]
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("bbox is not a list of 4 numbers")
    v = [float(x) for x in raw]
    if all(0.0 <= x <= 1.0 for x in v):
        sx, sy = width, height
    elif fmt == "pixels":
        sx = sy = 1.0
    else:
        sx, sy = width / 1000.0, height / 1000.0
    x0, y0, x1, y1 = v[0] * sx, v[1] * sy, v[2] * sx, v[3] * sy
    x0, x1 = sorted((min(max(x0, 0.0), width), min(max(x1, 0.0), width)))
    y0, y1 = sorted((min(max(y0, 0.0), height), min(max(y1, 0.0), height)))
    return x0, y0, x1, y1


def check_box(box: tuple, width: int, height: int) -> str:
    """'' when the box can be a round dial, else why not."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4:
        return "box is (almost) empty"
    if not 0.33 <= bw / bh <= 3.0:
        return f"box is {bw:.0f} x {bh:.0f} px - not the shape of a round dial"
    frac = bw * bh / float(width * height)
    if frac < 0.0002:
        return "box is too small to be read"
    if frac > 0.95:
        return "box covers the whole image"
    return ""


def parse_detection(text: str, width: int, height: int, fmt: str = "norm1000") -> Detection:
    obj = _json_obj(text)
    visible = obj.get("gauge_visible", obj.get("visible", False))
    visible = visible if isinstance(visible, bool) else str(visible).strip().lower() in ("true", "yes", "1")
    try:
        conf = min(max(float(obj.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.0
    adjust = str(obj.get("adjust", "none") or "none").strip().lower()
    adjust = _ALIASES.get(adjust, adjust.replace(" ", "_"))
    adjust = adjust if adjust in ADJUSTS else "none"
    raw = obj.get("bbox", obj.get("box"))
    if not visible or raw in (None, [], ""):
        return Detection(False, None, conf, adjust, "the model sees no gauge" if not visible else "no box given", None)
    try:
        box = box_to_pixels(raw, width, height, fmt)
    except (TypeError, ValueError) as exc:
        return Detection(True, None, conf, adjust, f"unusable box {raw!r}: {exc}", list(raw) if isinstance(raw, list) else None)
    why = check_box(box, width, height)
    return Detection(True, None if why else tuple(round(v, 1) for v in box), conf, adjust, why,
                     list(raw) if isinstance(raw, (list, tuple)) else None)


class CosmosGaugeClient:
    def __init__(self, base_url: str, model: str, api_key: str, prompt: str, max_tokens: int = 4096,
                 temperature: float = 0.0, thinking: bool | None = None):
        self.base_url, self.model, self.api_key, self.prompt = base_url.rstrip("/"), model, api_key, prompt
        self.max_tokens, self.temperature, self.thinking = max_tokens, temperature, thinking

    def ask(self, jpeg: bytes, timeout_s: float) -> tuple:
        """(Verdict, raw_text, latency_s). Raises CosmosError for network / HTTP / timeout / bad reply."""
        text, latency = self._post(jpeg, self.prompt, timeout_s, self.thinking)
        return parse_reply(text), text, latency

    def detect(self, jpeg: bytes, timeout_s: float, width: int, height: int, fmt: str = "norm1000",
               thinking: bool | None = False) -> tuple:
        """v0.5.3: (Detection, raw_text, latency_s) - where is the gauge in this image? Raises CosmosError."""
        cosmos = "cosmos" in (self.model or "").lower()
        prompt = build_detect_prompt(width, height, fmt, reasoning=cosmos and bool(thinking))
        text, latency = self._post(jpeg, prompt, timeout_s, None if cosmos else thinking)
        return parse_detection(text, width, height, fmt), text, latency

    def _post(self, jpeg: bytes, prompt: str, timeout_s: float, thinking) -> tuple:
        if timeout_s <= 1:
            raise CosmosError("no time left for a Cosmos call")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if (self.api_key or "").strip():
            headers["Authorization"] = f"Bearer {self.api_key.strip()}"
        body = json.dumps(build_payload(self.model, jpeg, prompt, self.max_tokens, self.temperature,
                                        thinking)).encode("utf-8")
        req = urllib.request.Request(endpoint_url(self.base_url), data=body, method="POST", headers=headers)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            hint = {401: " (API key rejected)", 403: " (API key not allowed for this model)", 404: " (wrong model id or URL)",
                    429: " (rate limited)"}.get(exc.code, "")
            raise CosmosError(f"HTTP {exc.code}{hint}: {exc.read()[:200]!r}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise CosmosError(f"{type(exc).__name__}: {exc}") from exc
        latency = time.time() - t0
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise CosmosError(f"unexpected response: {str(data)[:200]}") from exc
        text = msg.get("content") or ""
        if "</think>" not in text and msg.get("reasoning_content"):
            text = f"<think>{msg['reasoning_content']}</think>{text}"
        return text, latency
