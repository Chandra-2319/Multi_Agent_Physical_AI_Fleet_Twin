# SPDX-License-Identifier: Apache-2.0
"""Fleet brain: one Nemotron call per decision, strict validation, one corrective retry, rule-based fallback.

Saves <out_dir>/<snapshot_id>.json (exact input) and <snapshot_id>_decision.json (raw replies, validation
errors, final decision). Nemotron never produces motion: it can only pick ids the snapshot offered.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from ..schemas.fleet_decision import FleetDecision, output_json_schema, validate_decision
from ..schemas.validate import SchemaError, extract_json_object
from .snapshot_builder import PRIORITY_RANK
from .token_factory_client import LLMError
SYSTEM_PROMPT = """You are the central dispatcher ("fleet brain") of an industrial inspection robot fleet.
Each robot executes its task deterministically; you only decide WHICH due task goes to WHICH robot.

You receive one JSON fleet snapshot. Return exactly ONE decision:
- "assign": one task_id from due_tasks and one robot_id from THAT task's candidates.
- "no_action": only if no due task should start now; explain why.

Rules:
1. Copy ids exactly from the snapshot. Never invent ids.
2. Handle higher priority first (high > medium > low), even if a lower-priority task has a nearer robot. For equal priority prefer the task waiting longest.
3. Prefer the nearest candidate (smallest distance_m) unless the snapshot gives a reason not to.
4. Candidates are already filtered for capability, availability and safety. Do not add robots. A robot with returning_home=true is available: it is walking back to its home position and will turn around for the new task.
5. For reinspect tasks the first candidate is the independent robot; prefer it.
6. Do not repeat a task/robot pair that recently failed (see recent_decisions).
7. Never output velocities, joint values, waypoints or any field not listed below.
Return ONLY a JSON object with keys: snapshot_id, decision, task_id, robot_id, priority, rationale.
rationale: one short sentence (max 300 characters). task_id and robot_id are null for no_action."""
def prompt_view(snapshot): return {k: v for k, v in snapshot.items() if k != "created_ts"}
class FleetBrain:
    def __init__(self, client, model_id, cfg, audit=None, out_dir=None, enabled_fn=None):
        self.client = client; self.cfg = cfg; self._settings_model = model_id; self.model_id = cfg.nemotron.get("model_id") or model_id
        self.audit = audit; self.out_dir = Path(out_dir) if out_dir else None; self._enabled_fn = enabled_fn; self._enabled = True
        self.structured = cfg.nemotron.get("structured_output", "json_schema") == "json_schema"; self.last = None
        self.calls = self.ok = self.failed = 0; self.last_latency_s = 0.0; self.last_error = ""; self.last_source = ""
    @property
    def use_nemotron(self): return bool(self._enabled_fn()) if self._enabled_fn else self._enabled
    @use_nemotron.setter
    def use_nemotron(self, value): self._enabled = bool(value)
    @property
    def nemotron_available(self): return bool(self.client and self.client.api_key and self.model_id)
    def set_model(self, model_id): self._settings_model = model_id; self.model_id = self.cfg.nemotron.get("model_id") or model_id
    def health_line(self):
        if not self.calls: return "Brain: no decision yet"
        line = f"Brain: Nemotron answered {self.ok}/{self.calls} | last {self.last_source} in {self.last_latency_s:.1f}s"
        return line + (f" | last problem: {self.last_error[:90]}" if self.last_error else "")
    def decide(self, snapshot):
        t0 = time.time(); record = {"snapshot_id": snapshot["snapshot_id"], "model": self.model_id, "attempts": []}; decision = None
        if self.use_nemotron and self.nemotron_available: decision = self._ask_nemotron(snapshot, record)
        else: record["fallback_reason"] = ("Nemotron disabled in the UI" if not self.use_nemotron else "Nemotron not configured (no API key or model id)")
        if decision is None:
            decision = rule_based_decide(snapshot); record.setdefault("fallback_reason", "Nemotron output rejected")
            self._log("assigner_fallback", snapshot_id=snapshot["snapshot_id"], reason=record["fallback_reason"])
        self._note_health(decision, record, time.time() - t0); record["decision"] = decision.to_dict(); self._save(snapshot, record); self.last = decision
        return decision
    def _note_health(self, decision, record, elapsed):
        self.calls += 1; self.last_source = decision.source; self.last_latency_s = elapsed
        if decision.source == "nemotron": self.ok += 1; self.last_error = ""
        else: self.failed += 1; self.last_error = record.get("fallback_reason", "")
    def _ask_nemotron(self, snapshot, record):
        retries = int(self.cfg.nemotron.get("corrective_retries", 1))
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(prompt_view(snapshot), separators=(",", ":"), default=str)}]
        for attempt in range(retries + 1):
            t0 = time.time()
            try: reply = self._chat(messages, snapshot)
            except LLMError as exc:
                record["attempts"].append({"attempt": attempt, "error": str(exc)[:300]}); record["fallback_reason"] = f"Nemotron call failed: {exc}"[:200]; return None
            entry = {"attempt": attempt, "latency_s": round(time.time() - t0, 2), "structured": self.structured, "raw_reply": reply[:2000]}; record["attempts"].append(entry)
            try: return validate_decision(extract_json_object(reply), snapshot, source="nemotron")
            except SchemaError as exc:
                entry["validation_error"] = str(exc); self._log("nemotron_output_rejected", snapshot_id=snapshot["snapshot_id"], attempt=attempt, error=str(exc))
                messages = messages + [{"role": "assistant", "content": reply[:2000]}, {"role": "user", "content": f"Your answer was rejected: {exc}. Return a corrected JSON object only."}]
        record["fallback_reason"] = "Nemotron output failed validation after the corrective retry"; return None
    def _chat(self, messages, snapshot):
        kw = {"temperature": float(self.cfg.nemotron.get("temperature", 0.0)), "max_tokens": int(self.cfg.nemotron.get("max_tokens", 1200)), "tag": "nemotron_decide"}
        if self.structured:
            rf = {"type": "json_schema", "json_schema": {"name": "fleet_decision", "strict": True, "schema": output_json_schema(snapshot)}}
            try: return self.client.chat(self.model_id, messages, response_format=rf, **kw)
            except LLMError as exc:
                if "HTTP 400" not in str(exc) and "HTTP 422" not in str(exc): raise
                self.structured = False; self._log("structured_output_disabled", error=str(exc)[:200])
        return self.client.chat(self.model_id, messages, **kw)
    def _save(self, snapshot, record):
        if self.out_dir is None: return
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True); sid = snapshot["snapshot_id"]
            (self.out_dir / f"{sid}.json").write_text(json.dumps(snapshot, indent=1, default=str), encoding="utf-8")
            (self.out_dir / f"{sid}_decision.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        except OSError as exc: self._log("snapshot_save_failed", error=str(exc))
    def _log(self, kind, **kw):
        if self.audit: self.audit.record(kind, kw)
def rule_based_decide(snapshot):
    tasks = [t for t in snapshot["due_tasks"] if t["candidates"]]
    if not tasks: return FleetDecision(snapshot["snapshot_id"], "no_action", None, None, "low", "Rule-based: no due task has an eligible robot.", "rule_based")
    tasks.sort(key=lambda t: (PRIORITY_RANK.get(t["priority"], 3), -t["waiting_s"])); t = tasks[0]
    return FleetDecision(snapshot["snapshot_id"], "assign", t["task_id"], t["candidates"][0]["robot_id"], t["priority"], f"Rule-based: {t['action']} for {t['for_asset']}, first eligible candidate.", "rule_based", action=t["action"], target_id=t["target_id"], for_asset=t["for_asset"])
