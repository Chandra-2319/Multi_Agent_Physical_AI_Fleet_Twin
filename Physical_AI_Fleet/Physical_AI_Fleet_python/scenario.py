# SPDX-License-Identifier: Apache-2.0
"""FleetScenario: the Isaac Sim backend behind the UI (same role as ExampleScenario in the templates).

Owns the stage (environment USD, ground, light, site map, monitored assets, robots) and the pure-Python
FleetCoordinator.

Simulation mode, legged robots (Go2, Spot, H1, G1): spawned and driven by their RL walking policy
  (isaacsim.robot.policy.examples): planner -> path follower -> [vx, vy, wz] -> policy.forward().
Simulation mode AMR / Digital Twin: kinematic puppet posed from the coordinator.
Physics rate: walking policies only work at the rate they were trained at (sim.dt in env.yaml).

v0.4.3: a fresh run starts clean (start_run), routine inspections are opt-in, physics-rate conflicts are reported.
v0.4.4
  * Assets of any equipment type from asset_types.json, placed by asset_layout.json (position, roll/pitch/yaw, scale);
    the USD is loaded as a payload, a coloured placeholder box is used when there is none. A status beacon above
    each asset shows ok / alarm / work in progress. LOAD LAYOUT / SAVE LAYOUT, ADD ASSET with a type and yaw.
  * Site map (site_map.json): walkable polygon + no-go zones, drawn in the stage; robots plan only inside the
    walkable area and around no-go zones; LOAD SITE MAP reloads it.
  * Inspection camera: created when the robot is added, warmed up before its first frame is used, a small spot light
    on the camera, and every frame's centre brightness is logged (dark target warning). Evidence names carry the asset.
v0.5.0
  * All input JSON files are read from <extension>/inputs/; models from <extension>/Assets/ ('Assets/Motor.usd').
  * A pressure gauge (inputs/gauges.json) on every asset's inspected face; its needle prim is rotated: INJECT FAULT
    (overpressure) -> high value, repair / reset / stop / new run -> normal value. Leak: unchanged.
  * Robots read the gauge with Cosmos (inputs/cosmos.json) and turn / step back when Cosmos asks (GaugeInspector).
v0.5.6
  * v0.5.7: no ROS on the PC: the twin's joints are set from the robot's UDP packets (port 9870), ROS graphs in the
    twin USD are switched off.
  * Digital Twin (mirror): mode digital_twin + Connection 'udp:<port>[;jog=<robot ip>:<port>]' adds a REAL Go2 whose twin
    USD (with your OmniGraph for the joints) follows the robot's own odometry, Map -> World. The fleet never gives it a
    task; simulated robots see it. ALIGN TWIN computes Map -> World; JOG sends short, slow pulses to go2_vel_server.py.
v0.5.1
  * The gauge is part of the asset USD (asset_types.json "gauge": {"model", "needle_prim"}); only its needle is turned.
  * Asset models load exactly as authored, then usd_rotate_deg / usd_scale from the JSON are applied.
  * AI models come from provider profiles (NVIDIA / Nebius): fleet brain + vision model (VLM) that reads the gauge.
"""
from __future__ import annotations

import math
import os
import threading
import traceback
from pathlib import Path

import numpy as np
from isaacsim.core.api.objects import VisualCylinder
from isaacsim.core.api.world import World
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

from . import stage_assets
from .coordinator.coordinator import FleetCoordinator
from .coordinator.fleet_state import AWAITING_REPAIR, TOOLS_DISPATCHED, VERIFYING
from .fleet_config import asset_catalog, secrets_store
from .fleet_config.action_config import get_action_config
from .fleet_config.catalog import MODELS
from .fleet_config.settings import AssetConfig, get_settings
from .fleet_config.site_map import SiteMapError, load_site_map
from .perception.frame_capture import SimCameraCapture, encode_jpeg, frame_stats
from .reasoning.token_factory_client import TokenFactoryClient
from .robots.legged_robot import LeggedSimRobot
from .robots.locomotion.planner import RectObstacle
from .robots.locomotion.policy_body import PolicyBody, default_policy_paths, detect_physics_engine
from .robots.robot_factory import create_robot, make_planner
from .skills.executor import CAPTURE_RETRY
from .fleet_config import secrets_store as _secrets
from .fleet_config.paths import ensure_inputs, inputs_dir, resolve_model_path
from .fleet_config import ai_profiles
from .gauges.config import GaugeConfigError, gauge_for, load_cosmos, load_gauges
from .gauges.gauge_rig import EmbeddedGauge, GaugeError
from .perception.cosmos_gauge import CosmosError, CosmosGaugeClient, build_prompt, is_cosmos
from .skills.gauge_inspection import GaugeInspector
from .perception import gauge_crop
from . import twin_stage
from .robots.frames import Frame2D
from .robots.mirror_robot import MirrorRobot
from .robots.twin_body import TwinPose
from .robots.unitree_twin_link import ISAAC_JOINT_NAMES, STALE_S, Go2UdpLink, JogClient

FLEET_ROOT = "/World/Fleet"
ASSET_ROOT = "/World/Assets"
ENV_ROOT = "/World/Environment"
SITE_ROOT = "/World/Site"
COLORS = {"ok": np.array([0.2, 0.8, 0.2]), "alarm": np.array([0.9, 0.1, 0.1]), "working": np.array([1.0, 0.6, 0.0])}
USD_EXTS = (".usd", ".usda", ".usdc", ".usdz")
MAX_ENV_OBSTACLES = 5000
FAULT_GAUGE_OFFSET = {"low": 0.1, "medium": 0.25, "high": 0.4}  # x (gauge span) above the normal band
ROBOT_CLEARANCE = 0.3  # a new asset must be at least this far from every robot (plus the robot's radius)
ROUTINE_MIN_RANGE = (1.0, 1440.0)  # minutes
_ASSETS_ROOT: str | None = None


def assets_root() -> str:
    global _ASSETS_ROOT
    if _ASSETS_ROOT is None:
        try:
            _ASSETS_ROOT = get_assets_root_path() or ""
        except Exception:  # noqa: BLE001
            _ASSETS_ROOT = ""
    return _ASSETS_ROOT


def resolve_asset_path(path: str) -> str:
    path = (path or "").strip().strip('"')
    if path.startswith("/Isaac/"):
        root = assets_root()
        return (root + path) if root else ""
    return path


def path_exists(path: str) -> bool | None:
    """True / False, or None if the check itself is not possible."""
    if not path:
        return False
    try:
        import omni.client
        result, _entry = omni.client.stat(path)
        return result == omni.client.Result.OK
    except Exception:  # noqa: BLE001
        return os.path.exists(path) if "://" not in path else None


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])


def prim_name(asset_id: str) -> str:
    return asset_id.replace("-", "_")


def id_prefix(type_key: str) -> str:
    return "wellhead" if type_key.startswith("wellhead") else type_key.split("_")[-1]


def _routine_defaults() -> tuple:
    try:
        sched = get_action_config().schedule
        return bool(sched.get("routine_inspections", False)), float(sched.get("default_interval_s", 900)) / 60.0
    except Exception:  # noqa: BLE001 - a broken config is reported by the coordinator at LOAD
        return False, 15.0


def vlm_test(client, jpeg: bytes, cfg: dict, out_dir: Path, stem: str, timeout: float) -> str:
    """TEST VLM (v0.5.3): the same two steps a robot uses - find the gauge's box in the image, crop, read.
    Writes <stem>_vlmtest_box.jpg (the box drawn on the image) and <stem>_vlmtest_crop.jpg (what was read)."""
    from .perception.cosmos_gauge import CosmosError as _CE
    parts = []
    image = jpeg
    dcfg, ccfg = gauge_crop.detect_cfg(cfg), gauge_crop.crop_cfg(cfg)
    if ccfg.get("enabled", True) and "vlm" in (ccfg.get("source") if isinstance(ccfg.get("source"), list) else [ccfg.get("source")]):
        try:
            w, h = gauge_crop.image_size(jpeg)
            det, _raw, lat = client.detect(jpeg, min(float(dcfg["timeout_s"]), timeout), w, h, dcfg["format"], dcfg["thinking"])
            if det.visible and det.box_px and det.confidence >= float(dcfg["min_confidence"]):
                rgb = gauge_crop.decode(jpeg)
                crop, _info = gauge_crop.crop_from_box(rgb, det.box_px, ccfg)
                if crop is not None:
                    image = gauge_crop.encode(crop)
                    try:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        boxed = rgb.copy()
                        x0, y0, x1, y1 = (int(round(v)) for v in det.box_px)
                        for t in range(3):
                            boxed[max(y0 - t, 0), x0:x1] = boxed[min(y1 + t, h - 1), x0:x1] = (0, 255, 0)
                            boxed[y0:y1, max(x0 - t, 0)] = boxed[y0:y1, min(x1 + t, w - 1)] = (0, 255, 0)
                        (out_dir / f"{stem}_vlmtest_box.jpg").write_bytes(gauge_crop.encode(boxed))
                        (out_dir / f"{stem}_vlmtest_crop.jpg").write_bytes(image)
                    except OSError:
                        pass
                parts.append(f"gauge found in {lat:.1f}s at {[int(v) for v in det.box_px]} px of {w}x{h} "
                             f"(conf {det.confidence:.2f}), cropped")
            else:
                parts.append(f"gauge NOT found in {lat:.1f}s ({det.reason or 'confidence %.2f' % det.confidence}; "
                             f"model box {det.raw_box}) - reading the whole image")
        except (_CE, OSError, ValueError) as exc:
            parts.append(f"box step failed ({str(exc)[:120]}) - reading the whole image")
    try:
        v, _raw, lat = client.ask(image, timeout)
        parts.append(f"read in {lat:.1f}s: pressure {v.pressure}, confidence {v.confidence:.2f}, adjust {v.adjust}, "
                     f"needle {v.needle_value} - {v.reason}")
        return "OK: " + " | ".join(parts)
    except _CE as exc:
        return "FAILED: " + " | ".join(parts + [str(exc)])


class FleetScenario:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.planner = make_planner(self.settings)
        self.coordinator: FleetCoordinator | None = None
        self.faults: dict = {}
        self._xforms: dict = {}
        self._z: dict = {}
        self._cameras: dict = {}
        self._cam_paths: dict = {}
        self._asset_markers: dict = {}
        self._puddles: dict = {}
        self._robot_dt: dict = {}  # robot_id -> physics step its walking policy was trained with
        self._env_extra: list = []  # planner obstacles from the environment USD (kept when assets change)
        self.save_sim_frames = True  # "Capture Quadruped Camera": read the robot camera when it inspects
        self.routine_enabled, self.routine_minutes = _routine_defaults()
        self.run_fresh = True  # True until RUN starts the next run; a pause / resume does not make a run fresh
        self.on_refresh = None
        self._since_refresh = 0.0
        self._stage_id = None
        self._loading = False
        self.required_dt = self.settings.physics_dt
        self.configured_dt: float | None = None
        self.measured_dt: float | None = None
        self.env_info = "none"
        self._load_log: list = []
        self.key_file: Path = secrets_store.secrets_path()  # the key file in use (last used location at start-up)
        self.connection_result: dict | None = None  # set by the connection-test thread, read by the UI
        self._ping_thread: threading.Thread | None = None
        self.inputs_created = ensure_inputs()  # v0.5.0: <extension>/inputs/ is created from the shipped defaults once
        self.gauges: dict = {}  # asset_id -> GaugeRig
        self.gauge_cfg: dict = {}
        self.cosmos_cfg: dict = {}
        self.gauge_problems: list = []
        self.cosmos_test_result: str | None = None
        self.robot_entries: dict = {}  # v0.5.4: robot_id -> floor-layout entry (written to floor_layout.json)
        self.idle_hold = True  # v0.5.4: idle walking robots stop running their policy (Run Fleet > Skip Policy When Idle)
        self.camera_on_demand = True  # v0.5.4: robot cameras render only while a capture is near
        self._twins: dict = {}  # v0.5.6: robot_id -> TwinPose (mirrored real robots)
        self._cam_task: dict = {}  # v0.5.5: robot_id -> the run its camera was switched on for (stays on until it ends)
        self._perf = {"steps": 0, "wall_s": 0.0, "ext_s": 0.0, "last": None, "fps": 0.0, "ext_ms": 0.0}
        self._cosmos_thread: threading.Thread | None = None

    def owns_current_stage(self) -> bool:
        stage = get_current_stage()
        return self._loading or (stage is not None and stage.GetRootLayer().identifier == self._stage_id)

    # ------------------------------------------------------------------ diagnostics
    def _audit(self, kind: str, **payload) -> None:
        """Audit log when the scene is loaded, always the console (Isaac Sim's Console panel / terminal)."""
        print(f"[Physical_AI_Fleet] {kind}: {payload}")
        if self.coordinator:
            self.coordinator.audit.record(kind, payload)

    # ------------------------------------------------------------------ policy info for the UI
    def default_policy_info(self, model_key: str) -> dict:
        m = MODELS[model_key]
        if not m.legged:
            return {"policy": "", "env": "", "note": "kinematic robot - no walking policy"}
        if m.custom_policy:
            return {"policy": self.settings.g1_policy_path, "env": self.settings.g1_env_path,
                    "note": "custom Isaac Lab policy - browse to your policy.pt and env.yaml"}
        engine = detect_physics_engine()
        pt, env, source = default_policy_paths(m, engine)
        pt_full, env_full = resolve_asset_path(pt), resolve_asset_path(env)
        found = path_exists(pt_full), path_exists(env_full)
        flag = {True: "found", False: "NOT FOUND", None: "not checked"}
        return {"policy": pt_full or pt, "env": env_full or env,
                "note": f"default for {engine} (from {source}): policy {flag[found[0]]}, env {flag[found[1]]}"}

    # ------------------------------------------------------------------ scene
    def setup_scene(self, env_usd: str = "", add_ground: bool = True, env_obstacles: bool = True) -> None:
        """LoadButton setup_scene_fn: fresh stage + optional environment USD + site map + monitored assets."""
        self._loading = True
        self._load_log = []
        self.teardown()
        self._asset_markers.clear()
        self._puddles.clear()
        create_new_stage()
        stage = get_current_stage()
        self._stage_id = stage.GetRootLayer().identifier
        UsdGeom.Xform.Define(stage, FLEET_ROOT)
        UsdGeom.Xform.Define(stage, ASSET_ROOT)
        UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight")).CreateIntensityAttr(1200)
        world = World.instance()
        extra = []
        env_usd = (env_usd or "").strip().strip('"')
        self.env_info = "none"
        if env_usd:
            extra = self._load_environment(stage, env_usd, env_obstacles)
        if add_ground or not env_usd:
            world.scene.add_default_ground_plane(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        stage_assets.draw_site(stage, self.settings.site, SITE_ROOT)
        for aid, a in self.settings.assets.items():
            try:
                self._asset_markers[aid], _note = self._author_asset(stage, a)
            except Exception as exc:  # noqa: BLE001 - one bad asset must not stop LOAD
                self._load_log.append({"asset": aid, "error": str(exc)})
        self._env_extra = list(extra)
        self._rebuild_planner()
        self._load_gauge_configs()
        self._build_gauges(stage)

    # ------------------------------------------------------------------ gauges (v0.5.0)
    def _load_gauge_configs(self) -> None:
        self.gauge_problems = []
        try:
            self.gauge_cfg = load_gauges()
        except (GaugeConfigError, OSError) as exc:
            self.gauge_cfg = {}
            self.gauge_problems.append(f"gauges.json: {exc} - no gauges")
        try:
            self.cosmos_cfg = load_cosmos()
        except (GaugeConfigError, OSError) as exc:
            self.cosmos_cfg = {}
            self.gauge_problems.append(f"cosmos.json: {exc} - gauge reading falls back to the v0.4.4 perception")

    def _build_gauges(self, stage, only=None) -> None:
        for aid in list(self.gauges):
            if only is None or aid in only:
                self.gauges.pop(aid).remove()
        if not self.gauge_cfg:
            return
        for aid, a in self.settings.assets.items():
            if only is not None and aid not in only:
                continue
            try:
                g = gauge_for(self.gauge_cfg, a)
                if g is None:
                    continue
                name, spec, needle, dist = g
                rig = EmbeddedGauge(stage, a, spec, f"{ASSET_ROOT}/{prim_name(aid)}/model", needle).build()
                rig.camera_distance = dist
                if aid in self.faults and self.faults[aid].get("kind") != "leak":
                    rig.set_state("high")
                self.gauges[aid] = rig
                self._load_log.append({"gauge": aid, "model": name, "notes": rig.notes})
            except (GaugeError, GaugeConfigError, ValueError, KeyError) as exc:
                self.gauge_problems.append(f"gauge on {aid}: {exc}")
        if self.coordinator:
            self._attach_inspector()

    def _sync_gauges(self) -> None:
        if not self.coordinator:
            return
        stage = get_current_stage()
        gone = [aid for aid in self.gauges if aid not in self.settings.assets]
        for aid in gone:
            self.gauges.pop(aid).remove()
        new = [aid for aid in self.settings.assets if aid not in self.gauges]
        if new:
            self._build_gauges(stage, only=set(new))
        else:
            self._attach_inspector()

    def _set_gauge(self, asset_id: str, state: str) -> None:
        rig = self.gauges.get(asset_id)
        if rig is None:
            return
        try:
            angle = rig.set_state(state)
            if self.coordinator:
                self.coordinator.audit.record("gauge_state", {"asset_id": asset_id, "state": state, "value": rig.value,
                                                              "needle_deg": round(angle, 1)})
        except Exception as exc:  # noqa: BLE001
            self._audit("gauge_state_failed", asset=asset_id, error=str(exc))

    def cosmos_key(self) -> str:
        return (self.settings.vlm_api_key or os.environ.get("COSMOS_API_KEY", "").strip() or self.settings.cosmos_gauge_key or "").strip()

    def _cosmos_client(self, asset_id: str = "", spec: dict | None = None) -> CosmosGaugeClient:
        c, s = self.cosmos_cfg or {}, self.settings
        if spec is None:
            spec = self.gauges[asset_id].model if asset_id in self.gauges else next(iter((self.gauge_cfg.get("models") or {}).values()), {})
        cosmos = is_cosmos(s.vlm_model)
        prompt = build_prompt(spec.get("description", "A round pressure gauge."), spec.get("unit", ""), cosmos and gauge_crop.THINKING)
        return CosmosGaugeClient(s.vlm_base_url, s.vlm_model, self.cosmos_key(), prompt, int(c.get("max_tokens", 4096)),
                                 float(c.get("temperature", 0.0)), None if cosmos else gauge_crop.THINKING)

    def _attach_inspector(self) -> None:
        c = self.coordinator
        if c is None:
            return
        if not self.cosmos_cfg or not self.gauges:
            c.executor.gauge_inspector = None
            return
        c.executor.gauge_inspector = GaugeInspector(
            cfg=self.cosmos_cfg, specs={aid: r.model for aid, r in self.gauges.items()},
            mounts={aid: {"camera_distance_m": getattr(r, "camera_distance", 0.7)} for aid, r in self.gauges.items()},
            client_for=self._cosmos_client,
            truth_for=lambda aid: self.gauges[aid].truth() if aid in self.gauges else None,
            evidence_store=c.perception.evidence, audit=c.audit, pool=c.pool, site=self.settings.site,
            is_free=lambda x, y: self.planner.is_free(x, y, 0.4), retry_marker=CAPTURE_RETRY,
            gauge_pos=lambda aid: self.gauges[aid].position() if aid in self.gauges else None,
            capture_gauge=self._capture_gauge)

    def cosmos_status(self) -> str:
        n = len(self.gauges)
        extra = f" | gauges on {n} asset(s)" if self.coordinator else ""
        prob = f" | {self.gauge_problems[-1]}" if self.gauge_problems else ""
        return self.settings.vlm_status() + extra + prob

    # ------------------------------------------------------------------ AI provider profiles (v0.5.1)
    def profile_fields(self, provider: str, path_text: str = "") -> dict:
        """The stored profile for `provider` (keys included, for filling the masked fields)."""
        try:
            path = ai_profiles.normalize_path(path_text)
        except ValueError:
            path = None
        return ai_profiles.get_profile(provider, path)

    def apply_ai(self, provider: str, profile: dict) -> str:
        try:
            changed = self.settings.apply_profile(provider, profile)
        except ValueError as exc:
            return f"Not applied: {exc}"
        if self.coordinator:
            self.coordinator.refresh_credentials()
            self._attach_inspector()
        return ("Applied for this session: " + ", ".join(changed) + ".") if changed else "Nothing changed."

    def save_ai(self, provider: str, profile: dict, path_text: str) -> str:
        try:
            path = ai_profiles.normalize_path(path_text)
        except ValueError as exc:
            return f"Not saved: {exc}"
        applied = self.apply_ai(provider, profile)
        if applied.startswith("Not applied"):
            return applied.replace("Not applied", "Not saved", 1)
        try:
            written = ai_profiles.save_profile(provider, profile, path)
        except OSError as exc:
            return f"Not saved: could not write {path}: {exc.strerror or exc}"
        stored = ai_profiles.get_profile(provider, path)
        self.settings.apply_profile(provider, stored, source=f"profile file {written}")
        keys = ", ".join(f"{r} key {_secrets.mask_key(stored[r]['api_key'])}" for r in ai_profiles.ROLES)
        self._audit("ai_profile_saved", provider=provider, path=written)
        return f"Saved the {provider} profile ({keys}) to {written}; it is the active profile at the next start."

    def load_ai(self, provider: str, path_text: str) -> tuple:
        """(message, profile or None)."""
        try:
            path = ai_profiles.normalize_path(path_text)
        except ValueError as exc:
            return f"Not loaded: {exc}", None
        if not path.is_file():
            return f"Not loaded: there is no profile file at {path} (SAVE creates it).", None
        prof = ai_profiles.get_profile(provider, path)
        msg = self.apply_ai(provider, prof)
        if msg.startswith("Not applied"):
            return msg.replace("Not applied", "Not loaded", 1), None
        self.settings.nebius_key_source = f"profile file {path}"
        try:
            ai_profiles.set_active(provider, path)
        except OSError:
            pass
        missing = [r for r in ai_profiles.ROLES if not prof[r]["api_key"]]
        return (f"Loaded the {provider} profile from {path}." + (f" No API key stored for: {', '.join(missing)}." if missing else "")), prof

    def forget_ai(self, provider: str, path_text: str) -> str:
        try:
            path = ai_profiles.normalize_path(path_text)
            done = ai_profiles.forget_profile(provider, path)
        except (ValueError, OSError) as exc:
            return f"Nothing deleted: {exc}"
        if not done:
            return f"Nothing to delete: no {provider} profile in {path}."
        self._audit("ai_profile_deleted", provider=provider, path=str(path))
        return f"Deleted the {provider} profile from {path}. The values stay active until you close Isaac Sim."

    def start_cosmos_test(self, image_path: str) -> str:
        if self._cosmos_thread is not None and self._cosmos_thread.is_alive():
            return "A VLM test is already running."
        p = Path((image_path or "").strip().strip('"'))
        if not p.is_file():
            return f"Choose an image of the gauge first (not found: {p})."
        if not self.cosmos_cfg or not self.gauge_cfg:
            self._load_gauge_configs()
        try:
            client = self._cosmos_client()
        except Exception as exc:  # noqa: BLE001
            return f"Cannot test the VLM: {exc}"
        jpeg = p.read_bytes()
        self.cosmos_test_result = None
        timeout = float((self.cosmos_cfg or {}).get("call_timeout_s", 120))

        cfg = dict(self.cosmos_cfg or {})
        out_dir = Path(self.settings.log_dir) / "evidence"

        def job() -> None:
            self.cosmos_test_result = vlm_test(client, jpeg, cfg, out_dir, p.stem, max(timeout, 30.0))
        self._cosmos_thread = threading.Thread(target=job, name="fleet-vlm-test", daemon=True)
        self._cosmos_thread.start()
        return f"Asking {client.model} about {p.name} ..."

    def _author_asset(self, stage, a: AssetConfig) -> tuple:
        """Create the asset prim (payload or placeholder). Returns (marker, note)."""
        usd, note = "", ""
        if a.usd:
            full = resolve_asset_path(a.usd)
            found = path_exists(full)
            if found is False or not full:
                note = f"USD not found ({a.usd}); a placeholder box is shown instead"
                self._audit("asset_usd_missing", asset=a.asset_id, usd=a.usd)
            else:
                usd = full
        path = f"{ASSET_ROOT}/{prim_name(a.asset_id)}"
        marker = stage_assets.create_asset_prim(stage, path, a, usd, COLORS["ok"])
        if not stage_assets.marker_exists(stage, path):
            raise RuntimeError("the prim is not on the stage after creation")
        return marker, note

    def _rebuild_planner(self) -> None:
        """Same planner object (robots keep a reference to it), rebuilt for the current site map and assets."""
        fresh = make_planner(self.settings, self._env_extra)
        self.planner.__dict__.update(fresh.__dict__)

    def _load_environment(self, stage, env_usd: str, env_obstacles: bool) -> list:
        path = resolve_asset_path(env_usd)
        if not path.lower().endswith(USD_EXTS):
            self.env_info = f"ignored (not a USD file): {env_usd}"
            return []
        try:
            add_reference_to_stage(path, ENV_ROOT)
        except Exception as exc:  # noqa: BLE001
            self.env_info = f"FAILED to load {path}: {exc}"
            return []
        root = stage.GetPrimAtPath(ENV_ROOT)
        if not root.IsValid() or not root.GetChildren():
            self.env_info = f"FAILED to load {path} (empty or not found)"
            return []
        disabled = 0
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.Scene):
                prim.SetActive(False)
                disabled += 1
        obstacles = self._environment_obstacles(stage, root) if env_obstacles else []
        self.env_info = (f"{os.path.basename(path)} | planner obstacles: {len(obstacles)}"
                         + (f" | disabled {disabled} physics scene(s)" if disabled else ""))
        self._load_log.append({"environment": path, "obstacles": len(obstacles), "physics_scenes_disabled": disabled})
        return obstacles

    def _environment_obstacles(self, stage, root) -> list:
        """Footprints of the environment's collision geometry between 8 cm and 1.8 m height."""
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                                  useExtentsHint=True)
        xmin, ymin, xmax, ymax = self.settings.geofence
        area_limit = 0.25 * (xmax - xmin) * (ymax - ymin)
        prims = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Imageable)]
        if not prims:
            prims = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Gprim)]
        obstacles = []
        for prim in prims[:MAX_ENV_OBSTACLES]:
            try:
                rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            except Exception:  # noqa: BLE001
                continue
            if rng.IsEmpty():
                continue
            lo, hi = rng.GetMin(), rng.GetMax()
            if hi[2] < 0.08 or lo[2] > 1.8:
                continue
            w, d = hi[0] - lo[0], hi[1] - lo[1]
            if w * d > area_limit:
                continue
            obstacles.append(RectObstacle((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, w / 2, d / 2, prim.GetName()))
        return obstacles

    def setup_post_load(self) -> None:
        """LoadButton setup_post_load_fn: world is loaded, timeline paused at t=0."""
        self.coordinator = FleetCoordinator(self.settings, capture_fn=self._capture, fault_fn=self.faults.get)
        self.coordinator.set_routine(self.routine_enabled, self.routine_minutes * 60.0)
        self._attach_inspector()
        for p in self.gauge_problems:
            self.coordinator.audit.record("gauge_problem", {"problem": p})
        self.run_fresh = True
        self.required_dt = self.settings.physics_dt
        self.enforce_physics_dt(self.required_dt)
        self.coordinator.audit.record("scene_loaded", {"environment": self.env_info, "physics_dt": self.configured_dt,
                                                       "assets": sorted(self.settings.assets), "site": self.settings.site.describe(),
                                                       "load_problems": list(self.settings.load_problems),
                                                       "inputs": str(inputs_dir()), "gauges": sorted(self.gauges),
                                                       "cosmos": self.cosmos_status(),
                                                       "details": self._load_log})
        self._loading = False
        self.robot_entries.clear()
        self.layout_robots_note = self.spawn_layout_robots()

    def enforce_physics_dt(self, dt: float) -> None:
        world = World.instance()
        if world is None:
            return
        try:
            world.set_simulation_dt(physics_dt=dt, rendering_dt=max(dt, self.settings.rendering_dt))
        except Exception as exc:  # noqa: BLE001
            if self.coordinator:
                self.coordinator.audit.record("physics_dt_not_set", {"wanted": dt, "error": str(exc)})
        try:
            self.configured_dt = float(world.get_physics_dt())
        except Exception:  # noqa: BLE001
            self.configured_dt = None

    def physics_status(self) -> str:
        want = self.required_dt
        have = self.measured_dt if self.measured_dt is not None else self.configured_dt
        if have is None:
            return f"Physics: wanted {1 / want:.0f} Hz (unknown actual)"
        ok = abs(have - want) <= 0.1 * want
        src = "measured" if self.measured_dt is not None else "configured"
        text = f"Physics: {1 / have:.0f} Hz {src} (policies need {1 / want:.0f} Hz)"
        return text + ("" if ok else "  <-- MISMATCH: walking robots will fall. Press RESET / LOAD.")

    def teardown(self) -> None:
        if self.coordinator:
            self.coordinator.shutdown()
        self.coordinator = None
        self._xforms.clear()
        self._cameras.clear()
        self._cam_paths.clear()
        self._robot_dt.clear()
        self._cam_task.clear()
        self._twins.clear()
        self.robot_entries.clear()
        self.faults.clear()
        self.gauges.clear()
        self.measured_dt = None
        self.run_fresh = True

    # ------------------------------------------------------------------ runs
    def start_run(self) -> str:
        if not self.coordinator or not self.run_fresh:
            return ""
        had = self.coordinator.reset_tasks("new run")
        self.clear_faults()
        self.run_fresh = False
        names = {"incidents": "incident", "running": "running task", "queued_events": "queued event", "approvals": "pending approval",
                 "reserved": "reserved task"}
        gone = [f"{had[k]} {label}{'s' if had[k] != 1 else ''}" for k, label in names.items() if had.get(k)]
        if had.get("decision_in_flight"):
            gone.append("a Nemotron answer still in flight")
        return "New run: nothing is carried over from the previous run." + (f" Cleared: {', '.join(gone)}." if gone else "")

    def _reprepare_cameras(self) -> None:
        """v0.5.5: STOP / RESET can invalidate a paused render product: create every robot camera again."""
        self._cam_task.clear()
        for rid, cam in self._cameras.items():
            cam.reset()
            if cam.prepare() and self.camera_on_demand:
                cam.set_active(False)
            elif cam.error:
                self._audit("camera_not_ready", robot_id=rid, error=cam.error)

    def reset(self) -> None:
        if not self.coordinator:
            return
        self._reprepare_cameras()
        for tw in self._twins.values():
            tw.reset()
        self.enforce_physics_dt(self.required_dt)
        self.coordinator.on_world_reset()
        self.clear_faults()
        self.run_fresh = True
        self._write_poses()

    def on_timeline_stop(self) -> None:
        if self.coordinator:
            self._reprepare_cameras()
            for tw in self._twins.values():
                tw.reset()
            self.coordinator.on_world_reset()
            self.clear_faults()
            self._write_poses()
        self.run_fresh = True

    def apply_routine(self, enabled: bool, minutes) -> str:
        try:
            m = float(minutes)
        except (TypeError, ValueError):
            return "Routine interval must be a number of minutes."
        lo, hi = ROUTINE_MIN_RANGE
        if not (math.isfinite(m) and lo <= m <= hi):
            return f"Routine interval must be between {lo:g} and {hi:g} minutes."
        self.routine_enabled, self.routine_minutes = bool(enabled), m
        if self.coordinator:
            self.coordinator.set_routine(self.routine_enabled, m * 60.0)
        if enabled:
            return (f"Routine inspections ON: every asset is inspected every {m:g} min of running time "
                    f"(the clock stops while the simulation is paused or stopped).")
        return "Routine inspections OFF: robots only react to alarms and events."

    # ------------------------------------------------------------------ nemotron / cosmos settings
    def _remember_location(self, path: Path) -> str:
        try:
            secrets_store.remember_location(path)
        except OSError as exc:
            return (f" The location could not be remembered ({exc.strerror or exc}); "
                    f"press LOAD KEY next time.")
        env = os.environ.get("FLEET_SECRETS_FILE", "").strip()
        if env and Path(env) != path:
            return (f" NOTE: the environment variable FLEET_SECRETS_FILE ({env}) is set and takes priority over "
                    f"this location at start-up; remove it to use the remembered one.")
        return " This location is remembered: Isaac Sim reads the key from it automatically at start-up."

    def apply_credentials(self, api_key: str, base_url: str, model_id: str, cosmos_model_id: str) -> str:
        try:
            changed = self.settings.apply_nebius(api_key or None, base_url or None, model_id or None, cosmos_model_id)
        except ValueError as exc:
            return f"Not applied: {exc}"
        if self.coordinator:
            self.coordinator.refresh_credentials()
        return ("Applied for this session: " + ", ".join(changed) + ".") if changed else "Nothing changed."

    def save_credentials(self, api_key: str, base_url: str, model_id: str, cosmos_model_id: str, path_text: str) -> str:
        try:
            path = secrets_store.normalize_secrets_path(path_text)
        except ValueError as exc:
            return f"Not saved: {exc}"
        applied = self.apply_credentials(api_key, base_url, model_id, cosmos_model_id)
        if applied.startswith("Not applied"):
            return applied.replace("Not applied", "Not saved", 1)
        s = self.settings
        if not s.nebius_api_key:
            return "Not saved: there is no API key yet. Type it into the Nebius API Key field first."
        updates = {"nebius_api_key": s.nebius_api_key, "nebius_base_url": s.nebius_base_url,
                   "nemotron_model_id": s.nemotron_model_id, "cosmos_model_id": s.cosmos_model_id}
        try:
            written = secrets_store.save_secrets(updates, path)
        except OSError as exc:
            return f"Not saved: could not write {path}: {exc.strerror or exc}. Choose another folder in Key File."
        s.nebius_key_source = f"saved file {written}"
        self.key_file = Path(written)
        self._audit("credentials_saved", path=written, key=secrets_store.mask_key(s.nebius_api_key))
        return (f"Saved the API key ({secrets_store.mask_key(s.nebius_api_key)}), base URL and model ids to {written}."
                + self._remember_location(self.key_file))

    def load_credentials(self, path_text: str) -> str:
        try:
            path = secrets_store.normalize_secrets_path(path_text)
        except ValueError as exc:
            return f"Not loaded: {exc}"
        if not path.is_file():
            return f"Not loaded: there is no key file at {path}."
        try:
            self.settings.apply_saved_file(path)
        except ValueError as exc:
            return f"Not loaded: {exc}"
        self.key_file = path
        if self.coordinator:
            self.coordinator.refresh_credentials()
        self._audit("credentials_loaded", path=str(path), key=secrets_store.mask_key(self.settings.nebius_api_key))
        return (f"Loaded the API key ({secrets_store.mask_key(self.settings.nebius_api_key)}) from {path}."
                + self._remember_location(path))

    def forget_credentials(self, path_text: str) -> str:
        try:
            path = secrets_store.normalize_secrets_path(path_text)
        except ValueError as exc:
            return f"Nothing deleted: {exc}"
        try:
            deleted = secrets_store.forget_secrets(path)
        except OSError as exc:
            return f"Could not delete {path}: {exc.strerror or exc}"
        if not deleted:
            return f"Nothing to delete: there is no key file at {path}."
        self._audit("credentials_deleted", path=str(path))
        return f"Deleted {path}. The key stays active in this session until you close Isaac Sim."

    def key_file_status(self, path_text: str) -> str:
        try:
            path = secrets_store.normalize_secrets_path(path_text)
        except ValueError as exc:
            return f"Key file: {exc}."
        info = secrets_store.describe_secrets(path)
        if not info["exists"]:
            return f"Key file: {path} - not created yet (SAVE KEY creates it)."
        if not info["has_key"]:
            return f"Key file: {path} - exists but holds no API key."
        tail = ("remembered for the next start" if info["remembered"]
                else "NOT remembered yet - press SAVE KEY or LOAD KEY to remember this location")
        return f"Key file: {path} - key {info['key']} saved, {tail}."

    def start_connection_test(self) -> str:
        if self._ping_thread is not None and self._ping_thread.is_alive():
            return "A connection test is already running."
        s = self.settings
        model = self.coordinator.brain.model_id if self.coordinator else s.nemotron_model_id
        client = TokenFactoryClient(s.nebius_base_url, s.nebius_api_key, s.request_timeout_s,
                                    audit=self.coordinator.audit if self.coordinator else None)
        self.connection_result = None

        def job() -> None:
            self.connection_result = client.ping(model)

        self._ping_thread = threading.Thread(target=job, name="fleet-connection-test", daemon=True)
        self._ping_thread.start()
        return f"Testing {model} at {s.nebius_base_url} ..."

    def ask_nemotron_now(self) -> str:
        if not self.coordinator:
            return "Press LOAD first."
        return self.coordinator.force_evaluate()

    # ------------------------------------------------------------------ assets
    def asset_types(self) -> dict:
        """UI label -> type key."""
        return self.settings.catalog.labels() if self.settings.catalog else {}

    def next_asset_id(self, type_key: str) -> str:
        prefix = id_prefix(type_key or "asset")
        taken = {k.replace("-", "_") for k in self.settings.assets}
        n = 1
        while f"{prefix}_{n:02d}" in taken:
            n += 1
        return f"{prefix}-{n:02d}"

    def _robot_in_the_way(self, a: AssetConfig) -> str:
        if not self.coordinator:
            return ""
        from .fleet_config.geometry import distance_to_polygon_edge, point_in_polygon
        for rid, r in self.coordinator.fleet.robots.items():
            x, y = r.pose[0], r.pose[1]
            if point_in_polygon(x, y, a.footprint) or distance_to_polygon_edge(x, y, a.footprint) < r.radius + ROBOT_CLEARANCE:
                return rid
        return ""

    def _busy_reason(self) -> str:
        c = self.coordinator
        if c and (c.fleet.incidents or c.executor.runs):
            return "robots are working on open incidents; press STOP or RESET first"
        return ""

    def add_asset(self, type_key: str, asset_id: str, x, y, yaw_deg) -> str:
        """Returns a message for the Assets panel; it starts with 'Added' on success. Saved to the layout file."""
        try:
            return self._add_asset(type_key, asset_id, x, y, yaw_deg)
        except Exception as exc:  # noqa: BLE001 - never leave the user without an explanation
            self._audit("asset_add_failed", asset=str(asset_id), error=f"{type(exc).__name__}: {exc}",
                        trace=traceback.format_exc()[-600:])
            return f"Cannot add asset: unexpected error {type(exc).__name__}: {exc} (details in the Console)"

    def _add_asset(self, type_key, asset_id, x, y, yaw_deg) -> str:
        s = self.settings
        if s.catalog is None:
            return "Cannot add asset: asset_types.json could not be loaded (see Fleet Status)."
        entry = {"id": (asset_id or "").strip(), "type": type_key, "position": [x, y], "orientation_deg": [0, 0, yaw_deg]}
        try:
            a = asset_catalog.build_instance(AssetConfig, s.catalog, entry, s.site, s.assets)
        except asset_catalog.AssetError as exc:
            self._audit("asset_add_rejected", asset=str(asset_id), reason=str(exc))
            return f"Cannot add asset: {exc}"
        blocker = self._robot_in_the_way(a)
        if blocker:
            self._audit("asset_add_rejected", asset=a.asset_id, reason=f"robot {blocker} is standing there")
            return f"Cannot add asset: robot '{blocker}' is standing at that spot. Choose another X / Y or move the robot first."
        where, note = " It appears in the stage when you press LOAD.", ""
        if self.coordinator:
            try:
                marker, note = self._author_asset(get_current_stage(), a)
            except Exception as exc:  # noqa: BLE001
                self._audit("asset_add_failed", asset=a.asset_id, error=str(exc))
                return f"Cannot add asset: it could not be created on the stage ({exc})"
            self._asset_markers[a.asset_id] = marker
            where = f" It is on the stage at {marker.path}."
        s.assets[a.asset_id] = a
        saved = self._assets_changed()
        self._audit("asset_added", asset=a.asset_id, type=a.asset_type, position=[a.position[0], a.position[1]],
                    yaw_deg=a.orientation_deg[2])
        return (f"Added {a.label} {a.asset_id} at ({a.position[0]:.1f}, {a.position[1]:.1f}), yaw {a.orientation_deg[2]:g} deg, "
                f"robots approach from its {a.approach_side} side.{where}" + (f" NOTE: {note}." if note else "") + f" {saved}")

    def remove_asset(self, asset_id: str) -> str:
        if asset_id not in self.settings.assets:
            return "Unknown asset."
        if self.coordinator and self.coordinator.asset_in_use(asset_id):
            return f"Cannot remove {asset_id}: it has an open incident or a robot is working on it."
        if self.coordinator:
            self._asset_markers.pop(asset_id, None)
            stage_assets.remove_marker(get_current_stage(), f"{ASSET_ROOT}/{prim_name(asset_id)}")
            self.clear_faults(asset_id)
        del self.settings.assets[asset_id]
        self._audit("asset_removed", asset=asset_id)
        return f"Removed {asset_id}. " + self._assets_changed()

    def _assets_changed(self) -> str:
        self._rebuild_planner()
        if self.coordinator:
            self.coordinator.on_assets_changed()
            self._sync_gauges()
        try:
            path = asset_catalog.save_layout(self.settings.assets, robots=self._robots_for_file())
            self.settings.layout_path = path
            return f"Saved to {path}."
        except OSError as exc:
            return f"(could not save the layout: {exc})"

    def _replace_assets(self, assets: dict) -> None:
        stage = get_current_stage() if self.coordinator else None
        if stage is not None:
            for aid in list(self._asset_markers):
                stage_assets.remove_marker(stage, f"{ASSET_ROOT}/{prim_name(aid)}")
            self._asset_markers.clear()
            self.clear_faults()
        self.settings.assets.clear()
        self.settings.assets.update(assets)
        if stage is not None:
            for aid, a in assets.items():
                try:
                    self._asset_markers[aid], _ = self._author_asset(stage, a)
                except Exception as exc:  # noqa: BLE001
                    self._audit("asset_add_failed", asset=aid, error=str(exc))
        self._rebuild_planner()
        if self.coordinator:
            self.coordinator.on_assets_changed()
            self._sync_gauges()

    def _robots_for_file(self):
        """Robots to write: the ones in the scene; before LOAD (no scene yet) keep what the file has."""
        return list(self.robot_entries.values()) if self.coordinator else None

    def _save_floor(self) -> str:
        try:
            path = asset_catalog.save_layout(self.settings.assets, self.settings.layout_path or None, robots=self._robots_for_file())
        except OSError as exc:
            return f"(floor layout not saved: {exc})"
        self.settings.layout_path = path
        return f"Floor layout saved to {path}."

    def spawn_layout_robots(self, path=None) -> str:
        """Add every robot of the floor layout that is not in the scene yet (v0.5.4)."""
        if not self.coordinator:
            return ""
        robots, problems = asset_catalog.load_robots(path or self.settings.layout_path or None)
        added, failed = [], list(problems)
        for r in robots:
            if r["id"] in self.coordinator.fleet.robots:
                continue
            msg, _ = self.add_robot(r["id"], r["model"], r["mode"], r.get("usd", ""), r.get("connection", ""), r.get("map_to_world", ""),
                                    tuple(r["position"]), walking=r.get("walking", True), policy_path=r.get("policy_path", ""),
                                    env_path=r.get("env_yaml_path", ""), yaw_deg=r["yaw_deg"], save=False)
            (added if msg.startswith("Added") else failed).append(r["id"] if msg.startswith("Added") else f"{r['id']}: {msg}")
        if added or failed:
            self._audit("layout_robots", added=added, problems=failed)
        if not (added or failed):
            return ""
        return (f"Robots from the floor layout: {', '.join(added) or 'none'} added." +
                (f" Not added: {' | '.join(failed)}" if failed else ""))

    def load_layout(self, path_text: str = "") -> str:
        busy = self._busy_reason()
        if busy:
            return f"Layout not loaded: {busy}."
        if self.settings.catalog is None:
            return "Layout not loaded: asset_types.json could not be loaded."
        try:
            assets, problems, path = asset_catalog.load_layout(AssetConfig, self.settings.catalog, self.settings.site,
                                                               (path_text or "").strip().strip('"') or None)
        except asset_catalog.AssetError as exc:
            return f"Layout not loaded: {exc}"
        self._replace_assets(assets)
        self.settings.layout_path = path
        self._audit("layout_loaded", path=path, assets=sorted(assets), problems=problems)
        msg = f"Loaded {len(assets)} asset(s) from {path}."
        msg += (f" Skipped {len(problems)}: " + " | ".join(problems) if problems else "")
        robots = self.spawn_layout_robots(path)
        return msg + (f" {robots}" if robots else "")

    def save_layout(self, path_text: str = "") -> str:
        try:
            path = asset_catalog.save_layout(self.settings.assets, (path_text or "").strip().strip('"') or None,
                                             robots=self._robots_for_file())
        except OSError as exc:
            return f"Layout not saved: {exc}"
        self.settings.layout_path = path
        n = len(self.robot_entries) if self.coordinator else "the stored"
        return f"Saved {len(self.settings.assets)} asset(s) and {n} robot(s) to {path}."

    # ------------------------------------------------------------------ site map
    def load_site(self, path_text: str = "") -> str:
        busy = self._busy_reason()
        if busy:
            return f"Site map not loaded: {busy}."
        try:
            site = load_site_map((path_text or "").strip().strip('"') or None)
        except SiteMapError as exc:
            return f"Site map not loaded: {exc}"
        s = self.settings
        s.site = site
        s.geofence = site.bounds
        if self.coordinator:
            stage_assets.draw_site(get_current_stage(), site, SITE_ROOT)
        self._rebuild_planner()
        stranded = [aid for aid, a in s.assets.items() if not site.is_walkable(a.approach_pose[0], a.approach_pose[1])]
        homes = [rid for rid, r in (self.coordinator.fleet.robots.items() if self.coordinator else [])
                 if not site.is_walkable(r.home_pose[0], r.home_pose[1])]
        self._audit("site_loaded", path=site.path, site=site.describe(), unreachable_assets=stranded, homes_outside=homes)
        msg = f"Loaded site map {site.path}: {site.describe()}."
        if stranded:
            msg += f" WARNING: robots cannot reach {', '.join(stranded)} (approach point not walkable) - move or turn them."
        if homes:
            msg += f" WARNING: home of {', '.join(homes)} is not walkable."
        return msg

    def set_site_visible(self, show: bool) -> str:
        self.settings.site.show = bool(show)
        if self.coordinator:
            stage_assets.draw_site(get_current_stage(), self.settings.site, SITE_ROOT)
        return "Site map shown." if show else "Site map hidden (it still limits where robots walk)."

    def asset_ids(self) -> list:
        return list(self.settings.assets)

    def work_area(self) -> str:
        return self.settings.site.describe()

    # ------------------------------------------------------------------ fleet builder
    def rate_conflict(self) -> str:
        rates = {}
        for rid, dt in self._robot_dt.items():
            rates.setdefault(round(1.0 / dt), []).append(rid)
        if len(rates) < 2:
            return ""
        parts = "; ".join(f"{hz} Hz: {', '.join(sorted(ids))}" for hz, ids in sorted(rates.items()))
        return (f"WARNING: these walking robots need different physics rates ({parts}). One world runs at one rate "
                f"(now {1.0 / self.required_dt:.0f} Hz), so the others will fall or get stuck. Keep one rate per run: "
                f"remove the robots of the other group.")

    def add_robot(self, robot_id: str, model_key: str, mode: str, usd_path: str, connection: str, map_to_world: str,
                  spawn_xy: tuple, walking: bool = True, policy_path: str = "", env_path: str = "", yaw_deg: float = 0.0,
                  save: bool = True) -> tuple:
        if not self.coordinator:
            return "Press LOAD first.", None
        robot_id = robot_id.strip().replace(" ", "_")
        if not robot_id:
            return "Robot ID is required.", None
        if robot_id in self.coordinator.fleet.robots:
            return f"Robot '{robot_id}' already exists.", None
        why = self.settings.site.why_not(float(spawn_xy[0]), float(spawn_xy[1]))
        if why:
            return f"Cannot add robot: its home {why}. Choose a home inside the walkable area.", None
        model = MODELS[model_key]
        prim_path = f"{FLEET_ROOT}/{robot_id.replace('-', '_')}"
        try:
            yaw_deg = float(spawn_xy[2]) if len(spawn_xy) > 2 else float(yaw_deg or 0.0)
        except (TypeError, ValueError):
            return "Home Yaw must be a number of degrees.", None
        if not math.isfinite(yaw_deg):
            return "Home Yaw must be a finite number of degrees.", None
        home = (float(spawn_xy[0]), float(spawn_xy[1]), math.radians(yaw_deg))
        usd_path = usd_path.strip()
        mirror = mode == "digital_twin" and connection.strip().lower().startswith("udp:")
        use_policy = mode == "simulation" and model.legged and walking
        loaded = None
        try:
            if mirror:
                robot, visual = self._create_mirror(robot_id, model, prim_path, home, usd_path, connection, map_to_world)
            elif use_policy:
                usd_override = "" if usd_path in ("", model.usd) else resolve_asset_path(usd_path)
                body = PolicyBody(model, prim_path, *home, robot_name=robot_id, usd_path=usd_override,
                                  policy_path=resolve_asset_path(policy_path), env_path=resolve_asset_path(env_path))
                robot = create_robot(robot_id, model.robot_type, mode, home, prim_path, model=model.key,
                                     body=body, planner=self.planner)
                loaded = (body.policy_path, body.env_path)
                visual = f"{body.policy_name}"
                self._cam_paths[robot_id] = self._add_camera(f"{prim_path}/{model.base_link}")
                if body.policy_dt:
                    self._robot_dt[robot_id] = float(body.policy_dt)
                    if abs(body.policy_dt - self.required_dt) > 1e-6:
                        self.required_dt = body.policy_dt
                self.enforce_physics_dt(self.required_dt)
            else:
                robot = create_robot(robot_id, model.robot_type, mode, home, prim_path, connection.strip(),
                                     map_to_world.strip(), model=model.key, walking=False,
                                     planner=self.planner if mode == "simulation" else None)
                visual = self._spawn_puppet(prim_path, model, usd_path)
                self._cam_paths[robot_id] = self._add_camera(prim_path)
                self._xforms[robot_id] = XFormPrim(prim_path)
        except Exception as exc:  # noqa: BLE001
            self._robot_dt.pop(robot_id, None)
            self._twins.pop(robot_id, None)
            get_current_stage().RemovePrim(prim_path)
            return f"Could not create robot: {exc}", None
        self.coordinator.fleet.add(robot)
        self._prepare_camera(robot_id, model)
        self._write_poses()
        self.coordinator.audit.record("robot_added", {"robot_id": robot_id, "model": model.key, "mode": mode,
                                                      "locomotion": robot.locomotion, "visual": visual,
                                                      "policy": loaded[0] if loaded else None,
                                                      "policy_env": loaded[1] if loaded else None,
                                                      "physics_dt": self.configured_dt,
                                                      "camera": self._cam_paths.get(robot_id) or None,
                                                      "connection": connection or None})
        entry = {"id": robot_id, "model": model.key, "mode": mode, "position": [round(home[0], 4), round(home[1], 4)],
                 "yaw_deg": round(yaw_deg, 3), "walking": bool(walking)}
        if usd_path and usd_path != model.usd:
            entry["usd"] = usd_path
        for k, v in (("connection", connection), ("map_to_world", map_to_world), ("policy_path", policy_path),
                     ("env_yaml_path", env_path)):
            if (v or "").strip():
                entry[k] = v.strip()
        self.robot_entries[robot_id] = entry
        msg = f"Added {model.label} '{robot_id}' ({mode}, {visual}) at ({home[0]:.1f}, {home[1]:.1f}), heading {yaw_deg:g} deg."
        if save:
            msg += " " + self._save_floor()
        if mirror:
            msg += (" It follows the real robot once data arrives; stand the robot on its home spot and press ALIGN TWIN "
                    "(Manual Control) to line it up.")
        if use_policy:
            msg += " Press RUN - the policy starts on the first physics step and stands for 1 s before walking."
        conflict = self.rate_conflict()
        if conflict:
            self._audit("physics_rate_conflict", robots=dict(self._robot_dt), required_dt=self.required_dt)
            msg += "\n" + conflict
        return msg, loaded

    # ------------------------------------------------------------------ real robots in mirror mode (v0.5.6)
    def _create_mirror(self, robot_id, model, prim_path, home, usd_path, connection, map_to_world) -> tuple:
        if model.robot_type != "quadruped":
            raise ValueError("mirror mode is for quadrupeds (the real Go2)")
        conn = Go2UdpLink.parse_connection(connection)
        if usd_path and not usd_path.startswith("/Isaac/"):
            usd = resolve_model_path(usd_path)
        else:
            usd = resolve_asset_path(usd_path or model.usd)
        if not usd or path_exists(usd) is False:
            raise ValueError(f"twin USD not found: {usd_path or model.usd}")
        frame = Frame2D.parse(map_to_world)
        info = twin_stage.prepare_twin(get_current_stage(), prim_path, usd)
        link = Go2UdpLink(conn["port"], map_to_world=frame)
        if not link.start():
            raise RuntimeError(link.error)
        jog = JogClient(conn["jog"]) if conn["jog"] else None
        robot = MirrorRobot(robot_id, model.robot_type, home, link, prim_path, model.key, jog=jog)
        self._twins[robot_id] = TwinPose(prim_path, info["articulation_root"])
        self._audit("twin_prepared", robot_id=robot_id, usd=usd, articulation_root=info["articulation_root"],
                    graphs=info["graphs"], graphs_switched_off=info["graphs_off"],
                    retargeted=[c[1:] for c in info["retargeted"]][:10],
                    collisions_off=info["collisions_off"], gravity_off=info["gravity_off"], port=conn["port"],
                    jog=conn["jog"] or None)
        fixed = f", {len(info['retargeted'])} OmniGraph path(s) re-pointed" if info["retargeted"] else ""
        if info["graphs_off"]:
            fixed += f", {len(info['graphs_off'])} ROS / joint-control graph(s) switched off (joints come from UDP)"
        return robot, f"real-robot mirror {os.path.basename(usd)} on UDP {conn['port']}{fixed}"

    def _mirror(self, robot_id: str):
        r = self.coordinator.fleet.get(robot_id) if self.coordinator else None
        return r if isinstance(r, MirrorRobot) else None

    def align_twin(self, robot_id: str) -> str:
        """Map -> World so that where the real robot stands NOW is its home in the floor layout."""
        r = self._mirror(robot_id)
        if r is None:
            return f"{robot_id or 'That robot'} is not a mirrored real robot."
        st = r.link.state()
        if not st["have_pose"] or st["age"] is None or st["age"] > STALE_S:
            return f"{robot_id}: no position from the real robot yet ({r.note or 'no data'})."
        f = Frame2D.from_correspondence(st["map_pose"], r.home_pose)
        r.link.frame = f
        if robot_id in self.robot_entries:
            self.robot_entries[robot_id]["map_to_world"] = f.to_text()
        saved = self._save_floor()
        self._audit("twin_aligned", robot_id=robot_id, map_pose=[round(v, 3) for v in st["map_pose"]],
                    home=[round(v, 3) for v in r.home_pose], map_to_world=f.to_text(), source=st["src"])
        return (f"{robot_id} aligned: Map -> World {f.to_text()} (odometry from {st['src'] or 'the robot'}). "
                f"The robot's current spot is now its home ({r.home_pose[0]:.2f}, {r.home_pose[1]:.2f}, "
                f"{math.degrees(r.home_pose[2]):.0f} deg). {saved}")

    def jog_robot(self, robot_id: str, kind: str) -> str:
        r = self._mirror(robot_id)
        if r is None:
            return f"{robot_id or 'That robot'} is not a mirrored real robot."
        msg = r.jog(kind, self.settings.site)
        self._audit("real_robot_jog", robot_id=robot_id, jog=kind, result=msg)
        return msg

    def stop_real_robot(self, robot_id: str) -> str:
        r = self._mirror(robot_id)
        if r is None:
            return f"{robot_id or 'That robot'} is not a mirrored real robot."
        r.stop()
        self._audit("real_robot_stop", robot_id=robot_id)
        return f"{robot_id}: stop sent (zero velocity)." if r.jog_client else f"{robot_id}: no jog address - use the remote."

    def remove_robot(self, robot_id: str) -> str:
        if not self.coordinator or robot_id not in self.coordinator.fleet.robots:
            return "Unknown robot."
        self.coordinator.executor.cancel(robot_id, "removed")
        robot = self.coordinator.fleet.remove(robot_id)
        if isinstance(robot, LeggedSimRobot):
            robot.body.destroy()
        get_current_stage().RemovePrim(robot.prim_path)
        for d in (self._xforms, self._cameras, self._cam_paths, self._robot_dt, self._twins):
            d.pop(robot_id, None)
        if self._robot_dt and len(set(self._robot_dt.values())) == 1:
            self.required_dt = next(iter(self._robot_dt.values()))
            self.enforce_physics_dt(self.required_dt)
        self.robot_entries.pop(robot_id, None)
        conflict = self.rate_conflict()
        return f"Removed {robot_id}. " + self._save_floor() + (f"\n{conflict}" if conflict else "")

    def reset_robot(self, robot_id: str) -> str:
        if not self.coordinator:
            return "Press LOAD first."
        msg = self.coordinator.reset_robot(robot_id)
        self._write_poses()
        return msg

    def _spawn_puppet(self, prim_path: str, model, usd_path: str) -> str:
        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, prim_path)
        usd = resolve_asset_path(usd_path or model.usd)
        if usd:
            try:
                add_reference_to_stage(usd, f"{prim_path}/body")
                body = stage.GetPrimAtPath(f"{prim_path}/body")
                if body.IsValid() and body.GetChildren():
                    self._make_puppet(body)
                    self._z[prim_path] = model.spawn_z
                    return os.path.basename(usd) + " (kinematic)"
            except Exception:  # noqa: BLE001
                pass
            stage.RemovePrim(f"{prim_path}/body")
        sx, sy, sz = model.placeholder_size
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/body")
        cube.CreateSizeAttr(1.0)
        cube.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*model.placeholder_color)])
        self._z[prim_path] = sz / 2
        return "placeholder"

    @staticmethod
    def _make_puppet(root) -> None:
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)

    # ------------------------------------------------------------------ camera
    def _add_camera(self, parent_path: str) -> str:
        """The inspection camera on the robot, plus (by default) a small spot light pointing where it looks."""
        stage = get_current_stage()
        if not stage.GetPrimAtPath(parent_path).IsValid():
            return ""
        cam = UsdGeom.Camera.Define(stage, f"{parent_path}/inspection_cam")
        xf = UsdGeom.XformCommonAPI(cam)
        xf.SetTranslate(Gf.Vec3d(*gauge_crop.CAM_OFFSET))
        xf.SetRotate(Gf.Vec3f(90.0, 0.0, -90.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        cam.CreateFocalLengthAttr(gauge_crop.FOCAL_MM)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
        if self.settings.camera_light:
            light = UsdLux.SphereLight.Define(stage, f"{cam.GetPath()}/inspection_light")
            light.CreateRadiusAttr(0.03)
            light.CreateIntensityAttr(float(self.settings.camera_light_intensity))
            shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
            shaping.CreateShapingConeAngleAttr(45.0)
            shaping.CreateShapingConeSoftnessAttr(0.3)
        return str(cam.GetPath())

    def _prepare_camera(self, robot_id: str, model) -> None:
        """Create the sensor now so it has rendered for a while before the robot reaches an asset."""
        cam_path = self._cam_paths.get(robot_id)
        if not cam_path or getattr(model, "robot_type", "") != "quadruped":
            return
        cam = SimCameraCapture(cam_path, warmup_s=self.settings.camera_warmup_s)
        self._cameras[robot_id] = cam
        if not cam.prepare():
            self._audit("camera_not_ready", robot_id=robot_id, error=cam.error)
        elif self.camera_on_demand:
            cam.set_active(False)  # v0.5.4: rendered only when the robot is about to capture

    def set_camera_light(self, on: bool) -> str:
        self.settings.camera_light = bool(on)
        stage = get_current_stage() if self.coordinator else None
        n = 0
        for cam_path in self._cam_paths.values():
            if stage is None or not cam_path:
                continue
            p = stage.GetPrimAtPath(f"{cam_path}/inspection_light")
            if p.IsValid():
                p.SetActive(bool(on))
                n += 1
            elif on:
                light = UsdLux.SphereLight.Define(stage, f"{cam_path}/inspection_light")
                light.CreateRadiusAttr(0.03)
                light.CreateIntensityAttr(float(self.settings.camera_light_intensity))
                n += 1
        return f"Camera light {'ON' if on else 'OFF'} ({n} robot camera(s))."

    def _capture(self, robot):
        """Called by the skill executor when a robot has arrived at the asset (main thread).

        bytes / callable = a frame (or a deferred network capture); CAPTURE_RETRY = the camera is still warming up,
        the robot waits and asks again; None = no camera for this robot (or capture switched off)."""
        link = getattr(robot, "link", None)
        if link is not None and hasattr(link, "get_frame_jpeg"):
            return link.get_frame_jpeg
        cam_path = self._cam_paths.get(robot.robot_id)
        if robot.mode == "simulation" and self.save_sim_frames and cam_path:
            cam = self._cameras.get(robot.robot_id)
            if cam is None:
                cam = self._cameras[robot.robot_id] = SimCameraCapture(cam_path, warmup_s=self.settings.camera_warmup_s)
            rgb = cam.grab_rgb()
            if rgb is None:
                return CAPTURE_RETRY
            self._log_frame(robot, rgb)
            return encode_jpeg(rgb)
        return None

    def _capture_gauge(self, robot, asset_id: str):
        """v0.5.2: like _capture, but returns {"full": frame JPEG, "crop": gauge close-up JPEG or None, "info": {...}} so the
        vision model sees the dial large instead of ~2 % of the frame. Real robots: the frame is passed on uncropped."""
        link = getattr(robot, "link", None)
        if link is not None and hasattr(link, "get_frame_jpeg"):
            return link.get_frame_jpeg
        cam_path = self._cam_paths.get(robot.robot_id)
        if not (robot.mode == "simulation" and self.save_sim_frames and cam_path):
            return None
        cam = self._cameras.get(robot.robot_id)
        if cam is None:
            cam = self._cameras[robot.robot_id] = SimCameraCapture(cam_path, warmup_s=self.settings.camera_warmup_s)
        rgb = cam.grab_rgb()
        if rgb is None:
            return CAPTURE_RETRY
        self._log_frame(robot, rgb)
        full = encode_jpeg(rgb)
        cfg = gauge_crop.crop_cfg(self.cosmos_cfg)
        if not cfg.get("enabled", True) or asset_id not in self.gauges:
            return {"full": full, "crop": None, "info": {"cropped": False, "reason": "crop disabled or no gauge"}}
        try:
            crop, info = self._crop_to_gauge(robot, asset_id, rgb, cam, cfg)
        except Exception as exc:  # noqa: BLE001 - a crop problem must never lose the reading: send the full frame
            crop, info = None, {"cropped": False, "reason": f"{type(exc).__name__}: {exc}"}
        return {"full": full, "crop": encode_jpeg(crop) if crop is not None else None, "info": info}

    def _crop_to_gauge(self, robot, asset_id: str, rgb, cam, cfg: dict) -> tuple:
        h, w = rgb.shape[:2]
        rig = self.gauges[asset_id]
        centre = np.asarray(rig.position(), float)
        radius_m = float(cfg.get("dial_radius_m") or 0) or rig.dial_radius()
        # 1) Isaac Sim's own camera model (exact pose of the camera prim)
        uv_api, r_api = None, float("nan")
        pts = cam.project(np.stack([centre, centre + np.array([0.0, 0.0, radius_m])])) if hasattr(cam, "project") else None
        if pts is not None:
            uv_api, r_api = pts[0], float(np.hypot(*(pts[1] - pts[0])))
        # 2) from the robot pose and the camera mount (fallback, and logged for comparison)
        body = getattr(robot, "body", None)
        base_z = float(getattr(body, "_z", 0.0) or 0.0) or float(robot.model.spawn_z)
        pose = gauge_crop.camera_pose_from_robot(robot.pose[0], robot.pose[1], robot.pose[2], base_z)
        uv_m, r_m, depth = gauge_crop.locate(centre, radius_m, pose, w, h, gauge_crop.fx_pixels(w))
        use_api = uv_api is not None and 0 <= uv_api[0] < w and 0 <= uv_api[1] < h
        uv, r_px = (uv_api, r_api) if use_api else (uv_m, r_m)
        crop, info = gauge_crop.crop_gauge(rgb, uv, r_px, cfg)
        info.update({"method": "camera" if use_api else "robot_pose", "dial_radius_m": round(radius_m, 3),
                     "distance_m": round(depth, 2), "frame": [w, h],
                     "centre_px_camera": None if uv_api is None else [round(float(v), 1) for v in uv_api],
                     "centre_px_pose": [round(float(v), 1) for v in uv_m] if np.all(np.isfinite(uv_m)) else None})
        return crop, info

    def _target_of(self, robot_id: str) -> str:
        run = self.coordinator.executor.runs.get(robot_id) if self.coordinator else None
        return getattr(getattr(run, "assignment", None), "target_asset_id", "") or ""

    def _log_frame(self, robot, rgb) -> None:
        if not self.coordinator:
            return
        try:
            stats = frame_stats(rgb)
        except Exception:  # noqa: BLE001
            stats = {"mean_brightness": -1.0, "centre_brightness": -1.0, "dark_target": False, "almost_black": False}
        warning = None
        if stats["almost_black"]:
            warning = "frame is almost black"
        elif stats["dark_target"]:
            warning = ("the asset in the centre is almost black while the surroundings are lit: check the asset's "
                       "material and lighting (Camera Light), or look through the camera in the viewport")
        self.coordinator.audit.record("camera_frame", {"robot_id": robot.robot_id, "asset_id": self._target_of(robot.robot_id),
                                                       "shape": list(np.shape(rgb)), **stats, "warning": warning})

    def last_evidence(self) -> str:
        if not self.coordinator:
            return ""
        for entry in reversed(self.coordinator.audit.recent(80)):
            if entry.get("kind") == "fleet_event":
                paths = (entry.get("event") or {}).get("evidence") or []
                if paths:
                    return paths[-1]
        return ""

    # ------------------------------------------------------------------ physics step
    CAMERA_WAKE_M = 2.5  # a camera starts rendering (and warming up) when its robot is this close to the inspection pose

    def _camera_needed(self, robot_id: str) -> bool:
        if not self.camera_on_demand:
            return True
        run = self.coordinator.executor.runs.get(robot_id) if self.coordinator else None
        if run is None:
            self._cam_task.pop(robot_id, None)
            return False
        if self._cam_task.get(robot_id) is run:  # once on for this task it stays on (no flicker, no second warm-up)
            return True
        want = run.phase in ("capture", "gauge")
        goal = getattr(run.robot, "goal", None)
        if not want and run.phase == "navigate" and run.purpose and goal is not None:
            want = math.dist(tuple(run.robot.pose[:2]), tuple(goal[:2])) < self.CAMERA_WAKE_M
        if want:
            self._cam_task[robot_id] = run
        return want

    def _manage_idle_work(self) -> None:
        """v0.5.4: pause cameras nobody needs right now; tell walking robots whether they may hold when idle."""
        for rid, cam in self._cameras.items():
            want = self._camera_needed(rid)
            if cam.set_active(want) and self.coordinator:
                self.coordinator.audit.record("camera_on" if want else "camera_off", {"robot_id": rid})
        for r in self.coordinator.fleet.robots.values():
            if hasattr(r, "idle_hold"):
                r.idle_hold = self.idle_hold

    def set_idle_hold(self, on: bool) -> str:
        self.idle_hold = bool(on)
        if self.coordinator:
            self._manage_idle_work()
        return ("Idle walking robots stand still WITHOUT running their walking policy (resumes on the next task)." if on
                else "Every walking robot runs its policy on every step, also when idle.")

    def perf_status(self) -> str:
        p = self._perf
        if not p["steps"] and not p["fps"]:
            return "Performance: (press RUN)"
        robots = self.coordinator.fleet.robots.values() if self.coordinator else []
        holding = sum(1 for r in robots if getattr(r, "holding", False))
        walking = sum(1 for r in robots if hasattr(r, "holding"))
        cams = sum(1 for c in self._cameras.values() if c.active)
        rt = p["fps"] * (self.measured_dt or 0.0)
        return (f"Performance: {p['fps']:.0f} physics steps/s ({rt:.2f}x real time), fleet code {p['ext_ms']:.1f} ms/step | "
                f"walking policies running {walking - holding}/{walking} | cameras rendering {cams}/{len(self._cameras)}")

    def update(self, step: float) -> None:
        if not self.coordinator:
            return
        import time as _t
        t0 = _t.perf_counter()
        p = self._perf
        if p["last"] is not None:
            p["wall_s"] += t0 - p["last"]
        p["last"] = t0
        self.measured_dt = float(step)
        self._manage_idle_work()
        for cam in self._cameras.values():
            cam.advance(step)
        self.coordinator.tick(step)
        self._write_poses()
        p["ext_s"] += _t.perf_counter() - t0
        p["steps"] += 1
        if p["wall_s"] >= 2.0:
            p["fps"], p["ext_ms"] = p["steps"] / p["wall_s"], 1000.0 * p["ext_s"] / p["steps"]
            p["steps"], p["wall_s"], p["ext_s"] = 0, 0.0, 0.0
        self._since_refresh += step
        if self._since_refresh > 0.5:
            self._since_refresh = 0.0
            self._update_asset_colors()
            if self.on_refresh:
                self.on_refresh()

    def _write_poses(self) -> None:
        if not self.coordinator:
            return
        for rid, tw in self._twins.items():
            r = self.coordinator.fleet.get(rid)
            wp = getattr(r, "world_pose", None)
            if wp is not None:  # flat floor: the twin stands at the model's standing height, real roll / pitch / yaw
                tw.write(wp[0], wp[1], r.model.spawn_z, wp[3])
            joints = getattr(r, "joints", None)
            if joints is not None:  # v0.5.7: joints straight from the robot's packets (no ROS / OmniGraph)
                tw.write_joints(ISAAC_JOINT_NAMES, joints)
        for rid, xf in self._xforms.items():
            r = self.coordinator.fleet.get(rid)
            if r is None:
                continue
            x, y, yaw = r.pose
            try:
                xf.set_world_poses(positions=np.array([[x, y, self._z.get(r.prim_path, 0.0)]]), orientations=_yaw_quat(yaw))
            except Exception:  # noqa: BLE001
                pass

    def _update_asset_colors(self) -> None:
        if not self.coordinator:
            return
        stage = get_current_stage()
        for aid, marker in self._asset_markers.items():
            inc = self.coordinator.fleet.incident_for(aid)
            if inc is None:
                key = "alarm" if aid in self.faults else "ok"
            elif inc.status in (TOOLS_DISPATCHED, AWAITING_REPAIR, VERIFYING):
                key = "working"
            else:
                key = "alarm"
            try:
                stage_assets.set_status(stage, marker, COLORS[key])
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ demo controls
    def inject_fault(self, asset_id: str, kind: str, severity: str = "medium") -> str:
        """Demo: a fault appears at the asset and a sensor ALARM is raised with this priority.

        The primary reading of the asset is pushed above its normal band (by 10 / 25 / 40 % of the gauge span)."""
        if not self.coordinator:
            return "Press LOAD first."
        a = self.settings.assets.get(asset_id)
        if a is None:
            return f"Unknown asset '{asset_id}'."
        severity = severity if severity in FAULT_GAUGE_OFFSET else "medium"
        value = min(a.gauge_max, a.normal_high + FAULT_GAUGE_OFFSET[severity] * (a.gauge_max - a.gauge_min))
        if value <= a.normal_high:  # normal band reaches the top of the gauge: read at full scale
            value = a.gauge_max + 1e-6
        self.faults[asset_id] = {"kind": kind, "gauge_value": round(value, 3)}
        if kind == "leak" and asset_id not in self._puddles:
            x, y = a.approach_pose[0] * 0.3 + a.position[0] * 0.7, a.approach_pose[1] * 0.3 + a.position[1] * 0.7
            self._puddles[asset_id] = VisualCylinder(
                f"{ASSET_ROOT}/{prim_name(asset_id)}_leak", position=np.array([x, y, 0.005]),
                radius=0.35, height=0.01, color=np.array([0.1, 0.3, 1.0]))
        if kind != "leak":
            self._set_gauge(asset_id, "high")
        msg = self.coordinator.report_alarm(asset_id, "leak" if kind == "leak" else "overpressure", severity)
        self._update_asset_colors()
        return f"Injected {kind} on {a.label} {asset_id}. {msg}"

    def clear_faults(self, asset_id: str | None = None) -> None:
        for aid in [asset_id] if asset_id else list(self.faults) + list(self._puddles) + list(self.gauges):
            self._set_gauge(aid, "normal")
            self.faults.pop(aid, None)
            puddle = self._puddles.pop(aid, None)
            if puddle is not None:
                get_current_stage().RemovePrim(puddle.prim_path)
        if self.coordinator:
            self._update_asset_colors()

    def mark_repaired(self, asset_id: str) -> str:
        if not self.coordinator:
            return "Press LOAD first."
        self.clear_faults(asset_id)
        return self.coordinator.mark_repaired(asset_id)

    def mark_tools_loaded(self, asset_id: str) -> str:
        if not self.coordinator:
            return "Press LOAD first."
        return self.coordinator.mark_tools_loaded(asset_id)

    def submit_operator_photo(self, asset_id: str, jpeg: bytes) -> str:
        if not self.coordinator:
            return "Press LOAD first."
        c = self.coordinator

        def job():
            c.submit_event(c.perception.run(robot_id="operator", provenance="operator_phone", asset_id=asset_id,
                                            purpose="inspect", jpeg=jpeg, sim_fault=self.faults.get(asset_id)))
        c.pool.submit(job)
        return "Operator photo submitted (labelled provenance=operator_phone)."
