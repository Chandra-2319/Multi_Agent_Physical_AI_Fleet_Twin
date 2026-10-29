# SPDX-License-Identifier: Apache-2.0
"""Where the extension's input files and models live (v0.5.0).

  <extension>/inputs/      the ONLY folder you edit: robots_actions.json, asset_types.json, floor_layout.json (v0.5.4),
                           site_map.json, gauges.json, cosmos.json (created from fleet_config/ the first time)
  <extension>/Assets/      USD models (Motor.usd, Pressure_gauge.usd, ...)
  ~/.physical_ai_fleet/last_loaded/   copies of the input files as they were last loaded (never read back)

Overrides: FLEET_INPUTS_DIR (whole folder), and per file FLEET_ACTIONS_CONFIG, FLEET_ASSET_TYPES, FLEET_ASSET_LAYOUT,
FLEET_SITE_MAP (tests use these). Pure Python.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

SHIPPED_DIR = Path(__file__).resolve().parent  # .../Physical_AI_Fleet_python/fleet_config
PKG_DIR = SHIPPED_DIR.parent  # .../Physical_AI_Fleet_python
EXT_ROOT = PKG_DIR.parent  # .../Physical_AI_Fleet
INPUT_FILES = ("robots_actions.json", "asset_types.json", "floor_layout.json", "site_map.json", "gauges.json", "cosmos.json")
LEGACY_LAYOUT = "asset_layout.json"  # v0.5.4: replaced by floor_layout.json (assets AND robots)
ENV_FOR = {"robots_actions.json": "FLEET_ACTIONS_CONFIG", "asset_types.json": "FLEET_ASSET_TYPES",
           "floor_layout.json": "FLEET_ASSET_LAYOUT", "asset_layout.json": "FLEET_ASSET_LAYOUT", "site_map.json": "FLEET_SITE_MAP"}


def inputs_dir() -> Path:
    env = os.environ.get("FLEET_INPUTS_DIR", "").strip()
    return Path(env) if env else EXT_ROOT / "inputs"


def assets_dirs() -> list:
    """Folders searched for 'Assets/...' paths: the extension root first, then the python package."""
    return [EXT_ROOT / "Assets", PKG_DIR / "Assets"]


def last_loaded_dir() -> Path:
    return Path.home() / ".physical_ai_fleet" / "last_loaded"


def ensure_inputs() -> list:
    """Create inputs/ and copy any missing input file from fleet_config/. Returns the names that were created."""
    created = []
    d = inputs_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "floor_layout.json").exists() and (d / LEGACY_LAYOUT).exists():
            import json as _json
            try:  # v0.5.4: your asset layout becomes the floor layout (assets kept, no robots yet)
                old = _json.loads((d / LEGACY_LAYOUT).read_text(encoding="utf-8"))
                old.setdefault("robots", [])
                (d / "floor_layout.json").write_text(_json.dumps(old, indent=1), encoding="utf-8")
                created.append(f"floor_layout.json (from your {LEGACY_LAYOUT})")
            except (OSError, ValueError):
                pass
        for name in INPUT_FILES:
            if not (d / name).exists() and (SHIPPED_DIR / name).exists():
                shutil.copy2(SHIPPED_DIR / name, d / name)
                created.append(name)
    except OSError:
        pass
    return created


def input_file(name: str) -> Path:
    """The file to read for `name`: its FLEET_* variable, else inputs/<name> (created from the shipped default)."""
    env = os.environ.get(ENV_FOR.get(name, ""), "").strip() if name in ENV_FOR else ""
    if env:
        return Path(env)
    p = inputs_dir() / name
    if not p.exists():
        ensure_inputs()
    return p if p.exists() else SHIPPED_DIR / name


def snapshot(path) -> str:
    """Copy a file that was just loaded to ~/.physical_ai_fleet/last_loaded/ (best effort). Returns the copy's path."""
    try:
        src = Path(path)
        if not src.is_file():
            return ""
        d = last_loaded_dir()
        d.mkdir(parents=True, exist_ok=True)
        dst = d / src.name
        shutil.copy2(src, dst)
        with (d / "loaded_at.txt").open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {src.name}  <- {src}\n")
        return str(dst)
    except OSError:
        return ""


def resolve_model_path(usd: str, base: str = "") -> str:
    """'Assets/Motor.usd', './Assets/Motor.usd' and '/Assets/Motor.usd' -> <extension>/Assets/Motor.usd (or the
    package's Assets/ folder if the file is there). /Isaac/..., URLs and existing absolute paths pass through; other
    relative paths are relative to the JSON file that named them (`base`)."""
    usd = (usd or "").strip().strip('"')
    if not usd or usd.startswith("/Isaac/") or "://" in usd:
        return usd
    if os.path.isabs(usd) and Path(usd).exists():
        return usd
    rel = usd.replace("\\", "/").lstrip("/")
    if rel.startswith("./"):
        rel = rel[2:]
    if rel.lower().startswith("assets/"):
        tail = rel.split("/", 1)[1]
        cands = [d / tail for d in assets_dirs()]
        return str(next((c for c in cands if c.exists()), cands[0]))
    if os.path.isabs(usd):
        return usd  # an absolute path that does not exist: kept, reported as missing by the caller
    return str((Path(base).parent / rel).resolve()) if base and not base.startswith("<") else usd
