# SPDX-License-Identifier: Apache-2.0
"""Cosmos Reason perception client (NVIDIA open VLM for physical AI), via an OpenAI-compatible endpoint."""
from __future__ import annotations

import json

from ..reasoning.token_factory_client import LLMError, TokenFactoryClient, image_part
from ..schemas.fleet_event import PERCEPTION_OUTPUT_SPEC
from ..schemas.validate import SchemaError, extract_json_object, reject_forbidden_keys, unit_interval

SCENE_PROMPT = """You are inspecting industrial equipment ({asset_id}) from a robot camera.
Look for: fluid leaks or puddles under/around the equipment, smoke, visible damage, abnormal gauge position.
Answer with reasoning in <think></think> then ONLY this JSON object inside <answer></answer>:
""" + json.dumps(PERCEPTION_OUTPUT_SPEC, indent=1)

GAUGE_PROMPT = """This is an upscaled crop of an analog pressure gauge (dial range {lo} to {hi} {unit}).
Read the needle. Answer with reasoning in <think></think> then ONLY this JSON inside <answer></answer>:
{{"value": number or null if unreadable, "confidence": number 0..1}}"""


class CosmosReason:
    def __init__(self, client: TokenFactoryClient | None, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    @property
    def available(self) -> bool:
        return bool(self.client and self.client.api_key and self.model_id)

    def _ask(self, prompt: str, jpeg: bytes, tag: str) -> dict:
        reply = self.client.chat(self.model_id, [{"role": "user", "content": [image_part(jpeg), {"type": "text", "text": prompt}]}],
                                 temperature=0.2, max_tokens=900, tag=tag)
        if "<answer>" in reply:
            reply = reply.split("<answer>", 1)[1].split("</answer>", 1)[0]
        obj = extract_json_object(reply)
        reject_forbidden_keys(obj)
        return obj

    def describe_scene(self, jpeg: bytes, asset_id: str) -> dict:
        try:
            obj = self._ask(SCENE_PROMPT.format(asset_id=asset_id), jpeg, "cosmos_scene")
            et = obj.get("event_type")
            if et not in ("normal", "anomaly", "leak_suspected", "inconclusive"):
                raise SchemaError(f"bad event_type {et!r}")
            return {"event_type": et, "observation": str(obj.get("observation", ""))[:300],
                    "leak_visible": bool(obj.get("leak_visible", False)),
                    "confidence": unit_interval(obj.get("confidence", 0.0), "confidence")}
        except (LLMError, SchemaError) as exc:
            return {"event_type": "inconclusive", "observation": f"perception failed: {exc}"[:300],
                    "leak_visible": False, "confidence": 0.0}

    def read_gauge(self, crop_jpeg: bytes, lo: float, hi: float, unit: str) -> tuple:
        try:
            obj = self._ask(GAUGE_PROMPT.format(lo=lo, hi=hi, unit=unit), crop_jpeg, "cosmos_gauge")
            val = obj.get("value")
            conf = unit_interval(obj.get("confidence", 0.0), "confidence")
            if val is None:
                return None, conf
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not lo - 0.5 <= val <= hi + 0.5:
                raise SchemaError("gauge value out of dial range")
            return float(val), conf
        except (LLMError, SchemaError):
            return None, 0.0
