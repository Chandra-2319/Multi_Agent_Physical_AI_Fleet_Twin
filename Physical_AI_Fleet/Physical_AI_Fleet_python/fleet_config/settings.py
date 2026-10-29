# SPDX-License-Identifier: Apache-2.0
"""Runtime settings.

Nemotron / Cosmos configuration, highest priority first:
  1. values applied in the "Nemotron Brain" panel (APPLY = this session, SAVE KEY = also written to the key file)
  2. environment variables NEBIUS_API_KEY, NEBIUS_BASE_URL, NEMOTRON_MODEL_ID, COSMOS_MODEL_ID ...
  3. the saved key file: the last used location (SAVE KEY / LOAD KEY), else ~/.physical_ai_fleet/secrets.json
  4. defaults: Token Factory URL and nvidia/nemotron-3-super-120b-a12b
The Cosmos model id stays empty by default: until you set one, perception is labelled [MOCK].

v0.4.4: assets come from asset_types.json + asset_layout.json (any equipment type, any position / orientation) and the
work area from site_map.json (walkable polygon + no-go zones). `geofence` is the bounding box of the walkable area.
Broken files never stop the extension: defaults are used and the problem is listed in `load_problems`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import asset_catalog
from .secrets_store import load_secrets, mask_key, secrets_path
from .site_map import SiteMap, SiteMapError, load_site_map

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
DEFAULT_NEMOTRON_MODEL = "nvidia/nemotron-3-super-120b-a12b"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _pick(env_name: str, secret_key: str, default: str = "") -> str:
    return _env(env_name) or str(load_secrets().get(secret_key, "") or "").strip() or default


def _key_source() -> str:
    if _env("NEBIUS_API_KEY"):
        return "environment variable NEBIUS_API_KEY"
    if load_secrets().get("nebius_api_key"):
        return f"saved file {secrets_path()}"
    return "not set"


@dataclass
class AssetConfig:
    """A monitored asset (motor, pump, valve, separator...) in the world frame.

    gauge_* / normal_* / unit describe the PRIMARY reading (what inspection checks and INJECT FAULT drives).
    half_extents is the world-axis bounding box of the (possibly rotated) footprint; footprint is the exact polygon.
    """
    asset_id: str
    position: tuple  # (x, y, z) metres
    approach_pose: tuple  # (x, y, yaw_rad) where an inspecting robot should stand
    half_extents: tuple = (0.4, 0.25)
    gauge_min: float = 0.0
    gauge_max: float = 10.0
    normal_low: float = 2.0
    normal_high: float = 6.0
    unit: str = "bar"
    asset_type: str = "electric_motor"
    label: str = "Electric motor"
    orientation_deg: tuple = (0.0, 0.0, 0.0)  # roll, pitch, yaw
    scale: tuple = (1.0, 1.0, 1.0)
    usd: str = ""  # payload; empty = placeholder box
    usd_scale: float = 1.0
    height: float = 0.8
    color: tuple = (0.2, 0.6, 0.85)
    footprint: tuple = ()  # ((x, y), ...) world polygon
    local_half: tuple = ()  # half sizes in the asset's own frame
    approach_side: str = "-X"
    primary: str = ""
    readings: tuple = ()
    layout_entry: dict = field(default_factory=dict)
    usd_up_axis: str = ""
    usd_rotate_deg: tuple = (0.0, 0.0, 0.0)
    usd_auto_fit: bool = False
    gauge: dict = field(default_factory=dict)


def _load_site_and_assets() -> tuple:
    """(site, catalog, assets, layout_path, problems). Never raises: broken files fall back and are reported."""
    problems = []
    try:
        site = load_site_map()
        problems += [f"site map {site.path}: {w}" for w in site.warnings]
    except SiteMapError as exc:
        site = SiteMap()
        problems.append(f"site map: {exc} - using the default 30 x 30 m area with NO no-go zones")
    try:
        catalog = asset_catalog.load_catalog()
    except asset_catalog.AssetError as exc:
        return site, None, {}, "", problems + [f"asset types: {exc}"]
    old = Path.home() / ".physical_ai_fleet" / "assets.json"
    if not _env("FLEET_ASSET_LAYOUT") and not asset_catalog.layout_path().is_file() and old.is_file():
        assets = {}
        for e in asset_catalog.migrate_old_assets(old):
            try:
                a = asset_catalog.build_instance(AssetConfig, catalog, e, site, assets)
                assets[a.asset_id] = a
            except asset_catalog.AssetError as exc:
                problems.append(f"{old}: {exc}")
        if assets:
            problems.append(f"imported {len(assets)} asset(s) from the old {old}; press SAVE LAYOUT to keep them in the new format")
            return site, catalog, assets, str(asset_catalog.layout_path()), problems
    try:
        assets, bad, path = asset_catalog.load_layout(AssetConfig, catalog, site)
    except asset_catalog.AssetError as exc:
        return site, catalog, {}, "", problems + [f"asset layout: {exc}"]
    return site, catalog, assets, path, problems + bad


@dataclass
class Settings:
    # --- Nebius Token Factory (OpenAI-compatible) ---
    nebius_api_key: str = field(default_factory=lambda: _pick("NEBIUS_API_KEY", "nebius_api_key"))
    nebius_key_source: str = field(default_factory=_key_source)
    nebius_base_url: str = field(default_factory=lambda: _pick("NEBIUS_BASE_URL", "nebius_base_url", DEFAULT_BASE_URL))
    nemotron_model_id: str = field(default_factory=lambda: _pick("NEMOTRON_MODEL_ID", "nemotron_model_id", DEFAULT_NEMOTRON_MODEL))
    # --- Cosmos Reason (follows the Nebius key / URL unless COSMOS_* is set separately) ---
    cosmos_api_key: str = field(default_factory=lambda: _env("COSMOS_API_KEY") or _pick("NEBIUS_API_KEY", "nebius_api_key"))
    cosmos_base_url: str = field(default_factory=lambda: _env("COSMOS_BASE_URL") or _pick("NEBIUS_BASE_URL", "nebius_base_url", DEFAULT_BASE_URL))
    cosmos_model_id: str = field(default_factory=lambda: _pick("COSMOS_MODEL_ID", "cosmos_model_id"))
    # --- v0.5.0 gauge reading: key for the endpoint in inputs/cosmos.json (env COSMOS_API_KEY or the key file's
    #     cosmos_api_key; never the Nebius key, which a different endpoint would reject) ---
    cosmos_gauge_key: str = field(default_factory=lambda: _pick("COSMOS_API_KEY", "cosmos_api_key"))
    # --- v0.5.1 vision model that reads the gauges (filled from the active AI profile, see ai_profiles.py) ---
    ai_provider: str = ""
    vlm_base_url: str = "https://integrate.api.nvidia.com/v1"
    vlm_model: str = "nvidia/nemotron-3.5-super-vl-120b-a12b"
    vlm_api_key: str = ""
    vlm_thinking: bool = True  # v0.5.4: ALWAYS on, set by the code (THINKING_ALWAYS_ON); not a user option
    request_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 45.0))
    # --- Safety gate ---
    event_ttl_s: float = 120.0
    telemetry_ttl_s: float = 10.0
    min_confidence: float = 0.6
    geofence: tuple = (-15.0, -15.0, 15.0, 15.0)  # bounding box of the walkable area (set from the site map)
    # --- Gauge pipeline ---
    gauge_min_roi_px: int = 48
    gauge_upscale: int = 4
    # --- Simulation physics (RL walking policies are trained at a fixed rate; the policy's env.yaml wins) ---
    physics_dt: float = 1.0 / 200.0
    rendering_dt: float = 1.0 / 50.0
    # --- Custom humanoid policy (e.g. Unitree G1 trained in Isaac Lab): exported policy.pt + env.yaml ---
    g1_policy_path: str = field(default_factory=lambda: _env("G1_POLICY_PATH"))
    g1_env_path: str = field(default_factory=lambda: _env("G1_ENV_PATH"))
    # --- Environment USD referenced under /World/Environment on LOAD (browse in the UI) ---
    environment_usd: str = field(default_factory=lambda: _env("FLEET_ENV_USD"))
    add_ground_plane: bool = True
    environment_obstacles: bool = True
    # --- Inspection camera ---
    camera_light: bool = True  # a small spot light on each robot camera, like a real inspection robot carries
    camera_light_intensity: float = 20000.0  # UsdLux sphere-light intensity; lower it if frames are washed out
    camera_warmup_s: float = 0.5  # simulated time a new camera renders before its first frame is used
    # --- Logging ---
    log_dir: Path = field(default_factory=lambda: Path(_env("FLEET_LOG_DIR", str(Path.home() / "physical_ai_fleet_logs"))))
    # --- Assets and site map: asset_types.json + asset_layout.json + site_map.json ---
    assets: dict = field(default_factory=dict)
    site: SiteMap = field(default_factory=SiteMap)
    catalog: object = None
    layout_path: str = ""
    load_problems: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.assets and self.catalog is None:
            self.site, self.catalog, self.assets, self.layout_path, self.load_problems = _load_site_and_assets()
        self.geofence = self.site.bounds
        from . import ai_profiles
        if ai_profiles.profiles_path().is_file():  # v0.5.1: the saved active profile wins over env / old key file
            data = ai_profiles.load_file()
            self.apply_profile(data["active"], data["profiles"][data["active"]], source=f"profile file {ai_profiles.profiles_path()}")

    def apply_profile(self, provider: str, profile: dict, source: str = "entered in the UI") -> list:
        """Use a provider profile (brain + vlm). Blank fields keep the current value. Returns what changed."""
        b, v = profile.get("brain") or {}, profile.get("vlm") or {}
        for url in (b.get("url"), v.get("url")):
            if url and not str(url).lower().startswith(("http://", "https://")):
                raise ValueError("URLs must start with https:// (or http:// for a local server)")
        for m in (b.get("model"), v.get("model")):
            if m and " " in str(m).strip():
                raise ValueError("Model ids must not contain spaces")
        changed = []
        if b.get("url") and b["url"].rstrip("/") != self.nebius_base_url:
            self.nebius_base_url = b["url"].rstrip("/"); changed.append("Nemotron URL")
        if b.get("model") and b["model"] != self.nemotron_model_id:
            self.nemotron_model_id = b["model"]; changed.append("Nemotron model")
        if b.get("api_key") and b["api_key"] != self.nebius_api_key:
            self.nebius_api_key = b["api_key"]; self.nebius_key_source = source; changed.append("Nemotron key")
        if v.get("url") and v["url"].rstrip("/") != self.vlm_base_url:
            self.vlm_base_url = v["url"].rstrip("/"); changed.append("VLM URL")
        if v.get("model") and v["model"] != self.vlm_model:
            self.vlm_model = v["model"]; changed.append("VLM model")
        if v.get("api_key") and v["api_key"] != self.vlm_api_key:
            self.vlm_api_key = v["api_key"]; changed.append("VLM key")
        self.vlm_thinking = True  # v0.5.4: a "thinking" value in the profile is ignored
        if provider != self.ai_provider:
            self.ai_provider = provider; changed.append(f"provider {provider}")
        return changed

    def vlm_status(self) -> str:
        if not self.vlm_model:
            return "VLM: OFF - no model id"
        key = mask_key(self.vlm_api_key) if self.vlm_api_key else "MISSING"
        return f"VLM: {self.vlm_model} at {self.vlm_base_url} (key {key}, thinking always on)"

    @property
    def nemotron_enabled(self) -> bool:
        return bool(self.nebius_api_key and self.nemotron_model_id)

    @property
    def cosmos_enabled(self) -> bool:
        return bool(self.cosmos_api_key and self.cosmos_model_id)

    def nemotron_status(self) -> str:
        if not self.nebius_api_key:
            return "OFF - no API key (enter it in the Nemotron Brain panel or set NEBIUS_API_KEY) -> rule-based"
        if not self.nemotron_model_id:
            return "OFF - no model id -> rule-based"
        return f"ON {self.nemotron_model_id} (key {mask_key(self.nebius_api_key)}, from {self.nebius_key_source})"

    def apply_nebius(self, api_key: str | None = None, base_url: str | None = None, model_id: str | None = None,
                     cosmos_model_id: str | None = None) -> list:
        """Change the Nebius settings at run time. None / "" = keep the current value (cosmos_model_id may be "")."""
        changed = []
        if base_url:
            base_url = base_url.strip().rstrip("/")
            if not base_url.lower().startswith(("http://", "https://")):
                raise ValueError("Base URL must start with https://")
        if model_id and (" " in model_id.strip()):
            raise ValueError("Model id must not contain spaces")
        if cosmos_model_id and (" " in cosmos_model_id.strip()):
            raise ValueError("Cosmos model id must not contain spaces")
        if api_key and api_key.strip():
            follows = self.cosmos_api_key in ("", self.nebius_api_key)
            self.nebius_api_key = api_key.strip()
            self.nebius_key_source = "entered in the UI"
            if follows:
                self.cosmos_api_key = self.nebius_api_key
            changed.append("API key")
        if base_url and base_url != self.nebius_base_url:
            follows = self.cosmos_base_url in ("", self.nebius_base_url)
            self.nebius_base_url = base_url
            if follows:
                self.cosmos_base_url = base_url
            changed.append("base URL")
        if model_id and model_id.strip():
            if model_id.strip() != self.nemotron_model_id:
                changed.append("Nemotron model")
            self.nemotron_model_id = model_id.strip()
        if cosmos_model_id is not None and cosmos_model_id.strip() != self.cosmos_model_id:
            self.cosmos_model_id = cosmos_model_id.strip()
            changed.append("Cosmos model")
        return changed

    def apply_saved_file(self, path: Path) -> list:
        """Use the key (and URL / model ids) stored in the key file at `path`. Raises ValueError if it has no key."""
        data = load_secrets(path)
        if not data.get("nebius_api_key"):
            raise ValueError(f"there is no API key in {path}")
        changed = self.apply_nebius(api_key=data["nebius_api_key"], base_url=data.get("nebius_base_url") or None,
                                    model_id=data.get("nemotron_model_id") or None,
                                    cosmos_model_id=data["cosmos_model_id"] if "cosmos_model_id" in data else None)
        self.nebius_key_source = f"saved file {path}"
        return changed or ["API key"]


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
    return _SETTINGS
