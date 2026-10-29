# SPDX-License-Identifier: Apache-2.0
"""Minimal OpenAI-compatible chat client for Nebius Token Factory (stdlib only)."""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status  # HTTP status when the server answered, else None


class TokenFactoryClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 45.0, max_retries: int = 2, audit=None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.audit = audit

    # ------------------------------------------------------------------ transport
    def _request(self, payload: dict, *, tag: str, model: str, retries: int, timeout_s: float):
        """POST /chat/completions. Returns (parsed JSON, latency_s). Raises LLMError (with .status when known)."""
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        last_err, last_status = None, None
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                data["choices"][0]  # shape check (KeyError / IndexError handled below)
                return data, time.time() - t0
            except urllib.error.HTTPError as exc:
                last_status = exc.code
                last_err = f"HTTP {exc.code}: {exc.read()[:200]!r}"
                retriable = exc.code in (408, 429, 500, 502, 503, 504)
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_status = None
                last_err = f"{type(exc).__name__}: {exc}"
                retriable = True
            self._log(tag, model, time.time() - t0, ok=False, error=last_err, attempt=attempt)
            if not retriable or attempt == retries:
                break
            time.sleep(1.5 * (attempt + 1))
        raise LLMError(last_err or "unknown error", last_status)

    def chat(self, model: str, messages: list, *, temperature: float = 0.1, max_tokens: int = 600,
             tag: str = "llm", response_format: dict | None = None) -> str:
        if not (self.api_key and model):
            raise LLMError("API key or model id not configured")
        payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if response_format is not None:
            payload["response_format"] = response_format
        data, latency = self._request(payload, tag=tag, model=model, retries=self.max_retries, timeout_s=self.timeout_s)
        choice = data["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        if choice.get("finish_reason") == "length":
            self._log(tag, model, latency, ok=False, error="reply truncated", usage=data.get("usage"))
            raise LLMError("reply truncated (finish_reason=length): raise max_tokens")
        self._log(tag, model, latency, ok=True, usage=data.get("usage"), reply=text[:400], structured=response_format is not None)
        return text

    # ------------------------------------------------------------------ connection test (UI button)
    def ping(self, model: str, timeout_s: float = 15.0) -> dict:
        """One tiny request, no retries. {ok, status, message, latency_s}. Never raises; never contains the key."""
        if not self.api_key:
            return {"ok": False, "status": None, "message": "No API key set.", "latency_s": 0.0}
        if not model:
            return {"ok": False, "status": None, "message": "No model id set.", "latency_s": 0.0}
        payload = {"model": model, "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                   "max_tokens": 32, "temperature": 0}
        t0 = time.time()
        try:
            data, latency = self._request(payload, tag="connection_test", model=model, retries=0, timeout_s=timeout_s)
        except LLMError as exc:
            return {"ok": False, "status": exc.status, "message": self._explain(exc), "latency_s": round(time.time() - t0, 2)}
        reply = ((data["choices"][0].get("message") or {}).get("content") or "").strip()[:40]
        self._log("connection_test", model, latency, ok=True, usage=data.get("usage"))
        return {"ok": True, "status": 200, "latency_s": round(latency, 2),
                "message": f"Connected: {model} answered in {latency:.1f}s" + (f" ({reply!r})" if reply else "")}

    def _explain(self, exc: LLMError) -> str:
        host = urlparse(self.base_url).netloc or self.base_url
        if exc.status in (401, 403):
            return f"The server rejected the API key (HTTP {exc.status}). Check the key is a Nebius Token Factory key."
        if exc.status == 404:
            return "HTTP 404: the model id or the base URL is wrong. Check both."
        if exc.status == 429:
            return "Rate limited (HTTP 429). Wait a moment and try again."
        if exc.status is None:
            return f"Cannot reach {host}: {exc}"[:200]
        return str(exc)[:200]

    def _log(self, tag, model, latency, **kw) -> None:
        if self.audit:
            self.audit.record("model_call", {"tag": tag, "model": model, "endpoint": self.base_url,
                                             "latency_s": round(latency, 2), **kw})


def image_part(jpeg_bytes: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")}}
