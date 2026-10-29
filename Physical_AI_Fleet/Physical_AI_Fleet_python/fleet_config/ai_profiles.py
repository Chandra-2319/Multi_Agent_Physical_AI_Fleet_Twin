# SPDX-License-Identifier: Apache-2.0
"""AI provider profiles (v0.5.1): one JSON file with the fleet-brain model and the vision (gauge-reading) model for each
provider. The UI's Provider drop-down fills the six fields from here; LOAD / SAVE / FORGET read and write this file.

  ~/.physical_ai_fleet/ai_profiles.json   (outside the extension folder: it holds API keys; owner-only on Linux)
  {
    "active": "nvidia",
    "profiles": {
      "nvidia": {"brain": {"url": ..., "model": ..., "api_key": "nvapi-..."},
                 "vlm":   {"url": ..., "model": ..., "api_key": "nvapi-...", "thinking": false}},
      "nebius": {"brain": {...}, "vlm": {...}}
    }
  }
Keys are typed straight into this file (or into the UI and saved). They are never written to the audit log.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

PROVIDERS = ("nvidia", "nebius")
DEFAULTS = {
    "nvidia": {"brain": {"url": "https://integrate.api.nvidia.com/v1", "model": "nvidia/nemotron-3-super-120b-a12b", "api_key": ""},
               "vlm": {"url": "https://integrate.api.nvidia.com/v1", "model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
                       "api_key": ""}},
    "nebius": {"brain": {"url": "https://api.tokenfactory.nebius.com/v1", "model": "", "api_key": ""},
               "vlm": {"url": "https://api.tokenfactory.nebius.com/v1", "model": "", "api_key": ""}},
}
ROLES = ("brain", "vlm")


def profiles_path() -> Path:
    env = os.environ.get("FLEET_AI_PROFILES", "").strip()
    return Path(env) if env else Path.home() / ".physical_ai_fleet" / "ai_profiles.json"


def normalize_path(text: str) -> Path:
    t = (text or "").strip().strip('"').strip("'")
    if not t:
        return profiles_path()
    p = Path(t).expanduser()
    if not p.is_absolute():
        raise ValueError("Profile File needs a full path (starting with / or a drive letter)")
    return p if p.suffix.lower() == ".json" else p / "ai_profiles.json"


def _clean(profile: dict | None) -> dict:
    out = {}
    for role in ROLES:
        d = dict((profile or {}).get(role) or {})
        out[role] = {"url": str(d.get("url", "") or "").strip().rstrip("/"), "model": str(d.get("model", "") or "").strip(),
                     "api_key": str(d.get("api_key", "") or "").strip()}
    return out


def load_file(path: Path | None = None) -> dict:
    """{'active': str, 'profiles': {name: profile}}; missing / broken file -> defaults (never raises)."""
    p = path or profiles_path()
    try:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    profs = raw.get("profiles") if isinstance(raw, dict) else None
    profiles = {k: _clean(DEFAULTS[k]) for k in PROVIDERS}
    if isinstance(profs, dict):
        for k, v in profs.items():
            if isinstance(v, dict):
                profiles[str(k)] = _clean(v)
    active = raw.get("active") if isinstance(raw, dict) else None
    return {"active": active if active in profiles else "nvidia", "profiles": profiles}


def get_profile(provider: str, path: Path | None = None) -> dict:
    return load_file(path)["profiles"].get(provider) or _clean(DEFAULTS.get(provider))


def _write(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".ai_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(path)


def save_profile(provider: str, profile: dict, path: Path | None = None, keep_blank_keys: bool = True) -> str:
    """Write `profile` for `provider` and make it the active one. A blank api_key keeps the stored key."""
    p = path or profiles_path()
    data = load_file(p)
    new = _clean(profile)
    old = data["profiles"].get(provider) or _clean(None)
    if keep_blank_keys:
        for role in ROLES:
            if not new[role]["api_key"]:
                new[role]["api_key"] = old[role]["api_key"]
    data["profiles"][provider] = new
    data["active"] = provider
    return _write(p, data)


def forget_profile(provider: str, path: Path | None = None) -> bool:
    """Remove the stored profile (back to the built-in defaults, no keys). True if something was removed."""
    p = path or profiles_path()
    if not Path(p).is_file():
        return False
    try:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    profs = raw.get("profiles") if isinstance(raw, dict) else None
    if not isinstance(profs, dict) or provider not in profs:
        return False
    del profs[provider]
    _write(p, raw)
    return True


def set_active(provider: str, path: Path | None = None) -> None:
    p = path or profiles_path()
    data = load_file(p)
    data["active"] = provider
    _write(p, data)
