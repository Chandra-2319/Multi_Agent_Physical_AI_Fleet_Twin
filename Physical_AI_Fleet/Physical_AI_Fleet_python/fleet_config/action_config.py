# SPDX-License-Identifier: Apache-2.0
"""Robots, policies and actions from robots_actions.json: the single source of truth for what Nemotron may assign.

Adding a robot model, a policy file, or an action that reuses an existing executor skill is a config edit only.
Keys starting with "_" are comments. Unknown keys are rejected so typos do not pass silently.
v0.4.3: `return_home_after` per action (the robot walks home after the task) and `schedule.routine_inspections`.
"""
from __future__ import annotations
import json, math, os, re
from dataclasses import dataclass
from pathlib import Path
from .catalog import MODELS, SKILLS
DEFAULT_PATH = Path(__file__).with_name("robots_actions.json")
RISKS = ("low", "medium", "high"); APPROVALS = ("none", "operator"); TARGET_KINDS = ("asset", "location")
AUTONOMY = ("off", "approve_each", "auto_low_risk"); FOLLOW_UP_TRIGGERS = ("tools_loaded",)
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
ACTION_KEYS = {"label", "description", "robot_types", "target", "executor_skill", "risk", "approval", "real_robot_allowed", "prefer_different_robot", "return_home_after"}
MODEL_KEYS = {"label", "robot_type", "available", "unavailable_reason", "role", "locomotion", "real_robot", "actions"}
ROOT_KEYS = {"schema_version", "nemotron", "policy", "schedule", "locations", "actions", "follow_ups", "robot_models"}
class ConfigError(ValueError): pass
@dataclass(frozen=True)
class ActionSpec:
    name: str; label: str; description: str; robot_types: tuple; target_kind: str; target_id: str; executor_skill: str; risk: str; approval: str; real_robot_allowed: bool; prefer_different_robot: bool
    return_home_after: bool = False
@dataclass(frozen=True)
class ModelSpec:
    key: str; label: str; robot_type: str; actions: tuple; available: bool; unavailable_reason: str; role: str; policy_path: str; env_yaml_path: str; real_supported: bool
@dataclass(frozen=True)
class FollowUpSpec:
    name: str; after: str; robot_role: str; executor_skill: str; approval: str
def _clean(d):
    if isinstance(d, dict): return {k: _clean(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list): return [_clean(v) for v in d]
    return d
def _need(d, key, typ, where, default=...):
    if key not in d:
        if default is ...: raise ConfigError(f"{where}: missing '{key}'")
        return default
    v = d[key]
    if typ is float and isinstance(v, int) and not isinstance(v, bool): v = float(v)
    if not isinstance(v, typ) or (typ is not bool and isinstance(v, bool)): raise ConfigError(f"{where}.{key}: expected {typ.__name__}, got {type(v).__name__}")
    return v
def _one_of(v, allowed, where):
    if v not in allowed: raise ConfigError(f"{where}: '{v}' not in {list(allowed)}")
    return v
def _no_unknown(d, allowed, where):
    extra = set(d) - allowed
    if extra: raise ConfigError(f"{where}: unknown keys {sorted(extra)}")
def _pose(v, where):
    ok = isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
    if not ok or not all(math.isfinite(float(x)) for x in v): raise ConfigError(f"{where}: pose must be [x, y, yaw_rad] with finite numbers")
    return tuple(float(x) for x in v)
def _positive(d, key, where, default):
    v = d.get(key, default)
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0: raise ConfigError(f"{where}.{key}: must be a positive number")
    return float(v)
class FleetActionConfig:
    def __init__(self, raw, path="<memory>"):
        self.path = path
        if not isinstance(raw, dict): raise ConfigError(f"{path}: top level must be an object")
        raw = _clean(raw); _no_unknown(raw, ROOT_KEYS, "root")
        if raw.get("schema_version") != "1.0": raise ConfigError(f"{path}: schema_version must be '1.0'")
        self.nemotron = _need(raw, "nemotron", dict, "root"); self.policy = _need(raw, "policy", dict, "root"); self.schedule = _need(raw, "schedule", dict, "root")
        self.locations = {}
        for k, v in _need(raw, "locations", dict, "root").items():
            if not isinstance(v, dict): raise ConfigError(f"locations.{k}: must be an object")
            self.locations[k] = {"label": _need(v, "label", str, f"locations.{k}", k), "pose": _pose(_need(v, "pose", list, f"locations.{k}"), f"locations.{k}.pose")}
        self.actions = {n: self._action(n, d) for n, d in _need(raw, "actions", dict, "root").items()}
        self.follow_ups = {n: self._follow_up(n, d) for n, d in _need(raw, "follow_ups", dict, "root", {}).items()}
        self.models = {k: self._model(k, d) for k, d in _need(raw, "robot_models", dict, "root").items()}
        self._check_sections()
    def _action(self, name, d):
        w = f"actions.{name}"
        if not NAME_RE.match(name): raise ConfigError(f"{w}: action names are lower_snake_case")
        if not isinstance(d, dict): raise ConfigError(f"{w}: must be an object")
        _no_unknown(d, ACTION_KEYS, w)
        target = _need(d, "target", dict, w); kind = _one_of(_need(target, "kind", str, w + ".target"), TARGET_KINDS, w + ".target.kind"); target_id = ""
        if kind == "location":
            target_id = _need(target, "id", str, w + ".target")
            if target_id not in self.locations: raise ConfigError(f"{w}.target.id: unknown location '{target_id}'")
        skill = _need(d, "executor_skill", str, w)
        if skill not in SKILLS: raise ConfigError(f"{w}.executor_skill: '{skill}' is not an executor skill {sorted(SKILLS)}")
        types = _need(d, "robot_types", list, w)
        if not types or not all(isinstance(t, str) for t in types): raise ConfigError(f"{w}.robot_types: non-empty list of strings")
        return ActionSpec(name, _need(d, "label", str, w, name), _need(d, "description", str, w), tuple(types), kind, target_id, skill, _one_of(_need(d, "risk", str, w), RISKS, w + ".risk"),
                          _one_of(_need(d, "approval", str, w), APPROVALS, w + ".approval"), _need(d, "real_robot_allowed", bool, w, False), _need(d, "prefer_different_robot", bool, w, False),
                          _need(d, "return_home_after", bool, w, False))
    def _model(self, key, d):
        w = f"robot_models.{key}"
        if key not in MODELS: raise ConfigError(f"{w}: unknown model key (known: {sorted(MODELS)})")
        if not isinstance(d, dict): raise ConfigError(f"{w}: must be an object")
        _no_unknown(d, MODEL_KEYS, w); rtype = _need(d, "robot_type", str, w)
        if rtype != MODELS[key].robot_type: raise ConfigError(f"{w}.robot_type: '{rtype}' but the catalogue says '{MODELS[key].robot_type}'")
        actions = tuple(_need(d, "actions", list, w, []))
        for a in actions:
            if a not in self.actions: raise ConfigError(f"{w}.actions: unknown action '{a}'")
            if rtype not in self.actions[a].robot_types: raise ConfigError(f"{w}.actions: '{a}' is not allowed for robot type {rtype}")
        loco = _need(d, "locomotion", dict, w, {}); pt = _need(loco, "policy_path", str, w + ".locomotion", ""); env = _need(loco, "env_yaml_path", str, w + ".locomotion", "")
        if bool(pt) != bool(env): raise ConfigError(f"{w}.locomotion: policy_path and env_yaml_path must be set together (a policy without its env.yaml is not loadable)")
        real = _need(d, "real_robot", dict, w, {})
        return ModelSpec(key, _need(d, "label", str, w, MODELS[key].label), rtype, actions, _need(d, "available", bool, w, True), _need(d, "unavailable_reason", str, w, ""), _need(d, "role", str, w, ""), pt, env, _need(real, "supported", bool, w + ".real_robot", False))
    def _follow_up(self, name, d):
        w = f"follow_ups.{name}"
        if not isinstance(d, dict): raise ConfigError(f"{w}: must be an object")
        _no_unknown(d, {"after", "robot_role", "executor_skill", "approval"}, w); skill = _need(d, "executor_skill", str, w)
        if skill not in SKILLS: raise ConfigError(f"{w}.executor_skill: '{skill}' is not an executor skill")
        return FollowUpSpec(name, _one_of(_need(d, "after", str, w), FOLLOW_UP_TRIGGERS, w + ".after"), _need(d, "robot_role", str, w), skill, _one_of(_need(d, "approval", str, w), APPROVALS, w + ".approval"))
    def _check_sections(self):
        _one_of(self.policy.get("real_robot_autonomy", "approve_each"), AUTONOMY, "policy.real_robot_autonomy")
        _one_of(self.nemotron.get("structured_output", "json_schema"), ("json_schema", "none"), "nemotron.structured_output")
        trig = self.nemotron.get("triggers", {}); _positive(trig, "periodic_s", "nemotron.triggers", 60); _positive(trig, "min_interval_s", "nemotron.triggers", 5)
        _positive(self.schedule, "default_interval_s", "schedule", 900); _positive(self.nemotron, "snapshot_valid_for_s", "nemotron", 30)
        _positive(self.policy, "approval_timeout_s", "policy", 300); _positive(self.policy, "retry_block_s", "policy", 120)
        if not isinstance(self.schedule.get("routine_inspections", False), bool): raise ConfigError("schedule.routine_inspections: must be true or false")
        for k in ("confirm_reports_required", "max_unreadable_before_operator"):
            v = self.policy.get(k, 1)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1: raise ConfigError(f"policy.{k}: must be an integer >= 1")
    def actions_for_model(self, k):
        m = self.models.get(k); return m.actions if (m and m.available) else ()
    def can(self, k, action): return action in self.actions_for_model(k)
    def robots_with_role(self, role): return {k for k, m in self.models.items() if m.role == role and m.available}
    def home_after_skills(self): return {a.executor_skill for a in self.actions.values() if a.return_home_after}
    def location_pose(self, loc_id): return self.locations[loc_id]["pose"]
    def interval_for(self, aid):
        per = self.schedule.get("per_asset_interval_s", {}) or {}; return float(per.get(aid, self.schedule.get("default_interval_s", 900)))
    def policy_override(self, k):
        m = self.models.get(k); return (m.policy_path, m.env_yaml_path) if (m and m.policy_path) else ("", "")
    def check_model_available(self, k):
        m = self.models.get(k)
        if m is not None and not m.available: raise ConfigError(f"{m.label} is not available: {m.unavailable_reason or 'disabled in robots_actions.json'}")
    def catalogue_for_prompt(self):
        return [{"action": a.name, "label": a.label, "description": a.description, "robot_types": list(a.robot_types), "target": a.target_kind, "risk": a.risk, "needs_operator_approval": a.approval == "operator"} for a in self.actions.values()]
_CACHE = None
def load_action_config(path=None):
    from .paths import input_file, snapshot
    p = Path(path) if path else input_file("robots_actions.json")
    try: raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc: raise ConfigError(f"config not found: {p}") from exc
    except json.JSONDecodeError as exc: raise ConfigError(f"{p}: invalid JSON at line {exc.lineno}: {exc.msg}") from exc
    return FleetActionConfig(raw, str(p))
def get_action_config(reload=False):
    global _CACHE
    if _CACHE is None or reload: _CACHE = load_action_config()
    return _CACHE
