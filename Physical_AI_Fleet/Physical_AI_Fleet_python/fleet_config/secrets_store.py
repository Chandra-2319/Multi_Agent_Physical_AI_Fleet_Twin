# SPDX-License-Identifier: Apache-2.0
"""Where the Nebius API key is saved, and which location is remembered between sessions (unchanged in v0.4.3).

Files (all outside the extension folder, owner-only where the OS allows it):
  * the key file      any folder / .json file the user chooses (default ~/.physical_ai_fleet/secrets.json)
  * the location file ~/.physical_ai_fleet/location.json : one line saying which key file was used last

At start-up the key is read from the first of these that exists:
  1. the file named by the environment variable FLEET_SECRETS_FILE
  2. the last used location (written by SAVE KEY / LOAD KEY in the Nemotron Brain panel)
  3. the default location
The key is never written to the audit log, the status panel or any snapshot / decision file.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

KEYS = ("nebius_api_key", "nebius_base_url", "nemotron_model_id", "cosmos_model_id", "cosmos_api_key", "cosmos_base_url")
FILE_NAME = "secrets.json"


def app_dir() -> Path:
    return Path.home() / ".physical_ai_fleet"


def default_secrets_path() -> Path:
    return app_dir() / FILE_NAME


def pointer_path() -> Path:
    env = os.environ.get("FLEET_LOCATION_FILE", "").strip()
    return Path(env) if env else app_dir() / "location.json"


def _env_path() -> Path | None:
    env = os.environ.get("FLEET_SECRETS_FILE", "").strip()
    return Path(env) if env else None


def remembered_path() -> Path | None:
    """The key file used last time (SAVE KEY / LOAD KEY), or None."""
    try:
        data = json.loads(pointer_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("secrets_file") if isinstance(data, dict) else None
    return Path(value) if isinstance(value, str) and value.strip() else None


def candidate_paths() -> list:
    """[(path, label)] in priority order, without duplicates."""
    out: list = []
    for path, label in ((_env_path(), "environment variable FLEET_SECRETS_FILE"),
                        (remembered_path(), "last used location"),
                        (default_secrets_path(), "default location")):
        if path is not None and all(path != q for q, _ in out):
            out.append((path, label))
    return out


def secrets_path() -> Path:
    """The key file to read: the first candidate that exists; if none does, where a save would go."""
    cands = candidate_paths()
    for path, _label in cands:
        if path.is_file():
            return path
    return cands[0][0]


def secrets_path_source(path: Path | None = None) -> str:
    path = path or secrets_path()
    for p, label in candidate_paths():
        if p == path:
            return label
    return "chosen in the UI"


def normalize_secrets_path(text: str) -> Path:
    """What the user typed or picked (a folder or a .json file) -> the key file path. Raises ValueError with a UI message."""
    t = (text or "").strip().strip('"').strip("'")
    if not t:
        raise ValueError("Key File is empty: browse to a folder or type a full path")
    if "://" in t:
        raise ValueError("Key File must be a path on this PC (not omniverse:// or http://)")
    path = Path(t).expanduser()
    if not path.is_absolute():
        raise ValueError("Key File needs a full path (starting with / or a drive letter)")
    if path.suffix.lower() == ".json":
        return path
    if path.suffix and path.is_file():
        raise ValueError("Key File must be a folder or a .json file")
    return path / FILE_NAME  # a folder: the file inside it


def load_secrets(path: Path | None = None) -> dict:
    """{} when the file is missing or unreadable (a broken file must never stop the extension)."""
    try:
        data = json.loads((path or secrets_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if k in KEYS and isinstance(v, str)} if isinstance(data, dict) else {}


def _atomic_write_json(path: Path, data: dict, private: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".fleet_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        if private:
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass  # e.g. Windows
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_secrets(updates: dict, path: Path | None = None) -> str:
    """Merge `updates` into the key file at `path` (default: the current one). Returns the path written. Raises OSError."""
    bad = set(updates) - set(KEYS)
    if bad:
        raise ValueError(f"unknown secret keys {sorted(bad)}")
    path = path or secrets_path()
    data = load_secrets(path)
    data.update({k: str(v).strip() for k, v in updates.items() if v is not None})
    _atomic_write_json(path, data, private=True)
    return str(path)


def remember_location(path: Path) -> None:
    """Remember `path` as the place to read the key from next time. Raises OSError."""
    _atomic_write_json(pointer_path(), {"secrets_file": str(path)}, private=False)


def forget_secrets(path: Path | None = None) -> bool:
    """Delete the key file. True if it existed. Raises OSError if it exists but cannot be deleted."""
    path = path or secrets_path()
    if not path.is_file():
        return False
    path.unlink()
    return True


def mask_key(key: str) -> str:
    key = (key or "").strip()
    if not key:
        return "(none)"
    return "****" if len(key) <= 8 else "****" + key[-4:]


def describe_secrets(path: Path | None = None) -> dict:
    """What is in the key file, without revealing the key."""
    path = path or secrets_path()
    data = load_secrets(path)
    remembered = remembered_path()
    return {"path": str(path), "exists": path.is_file(), "has_key": bool(data.get("nebius_api_key")),
            "key": mask_key(data.get("nebius_api_key", "")), "model": data.get("nemotron_model_id", ""),
            "base_url": data.get("nebius_base_url", ""), "remembered": remembered is not None and remembered == path}
