# SPDX-License-Identifier: Apache-2.0
"""Shared JSON validation helpers. Every inbound and outbound JSON passes through here."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from ..fleet_config.catalog import FORBIDDEN_KEYS


class SchemaError(ValueError):
    """Raised when a payload does not match its schema. The payload is rejected, never repaired."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception as exc:  # noqa: BLE001
        raise SchemaError(f"bad timestamp: {ts!r}") from exc


def age_seconds(ts: str) -> float:
    return (datetime.now(timezone.utc) - parse_iso(ts)).total_seconds()


def extract_json_object(text: str) -> dict:
    """Extract exactly one JSON object from a model reply (tolerates ```json fences and <think> blocks)."""
    if not isinstance(text, str):
        raise SchemaError("model reply is not text")
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise SchemaError("no JSON object in model reply")
    try:
        obj = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SchemaError(f"malformed JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise SchemaError("JSON root is not an object")
    return obj


def reject_forbidden_keys(obj, path: str = "$") -> None:
    """Guardrail: no velocity / joint / torque style fields anywhere in the payload."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                raise SchemaError(f"forbidden raw-motion field '{k}' at {path}")
            reject_forbidden_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            reject_forbidden_keys(v, f"{path}[{i}]")


def require(obj: dict, key: str, types, *, allowed=None, nullable: bool = False):
    if key not in obj:
        raise SchemaError(f"missing field '{key}'")
    val = obj[key]
    if val is None and nullable:
        return None
    type_tuple = types if isinstance(types, tuple) else (types,)
    if not isinstance(val, type_tuple) or (isinstance(val, bool) and bool not in type_tuple):
        raise SchemaError(f"field '{key}' has wrong type {type(val).__name__}")
    if allowed is not None and val not in allowed:
        raise SchemaError(f"field '{key}'={val!r} not in {sorted(allowed)}")
    return val


def no_extra_keys(obj: dict, allowed: set, name: str) -> None:
    extra = set(obj) - allowed
    if extra:
        raise SchemaError(f"{name}: unexpected fields {sorted(extra)}")


def bounded_text(val: str, key: str, max_len: int) -> str:
    if len(val) > max_len:
        raise SchemaError(f"field '{key}' longer than {max_len} chars")
    return val


def unit_interval(val, key: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not 0.0 <= float(val) <= 1.0:
        raise SchemaError(f"field '{key}' must be a number in [0, 1]")
    return float(val)
