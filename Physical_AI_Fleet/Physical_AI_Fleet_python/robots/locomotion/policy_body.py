# SPDX-License-Identifier: Apache-2.0
"""RL walking-policy body for Isaac Sim (uses isaacsim.robot.policy.examples).

Follows the policy examples (go2 / quadruped / humanoid) of Isaac Sim 6.0.x:
  1. construct the policy robot - the policy class spawns the robot USD and loads policy.pt + env.yaml,
  2. on the first physics step: policy.initialize() then policy.post_reset(),
  3. every physics step: policy.forward(dt, command) with a torch tensor on the policy's device,
  4. when `robot.is_physics_tensor_entity_valid()` turns False (stop/play, world reset): initialise again.

v0.3.2 (this file) - "cannot read robot pose: AttributeError: 'Articulation' object has no attribute ..."
  * `policy.robot` is `isaacsim.core.experimental.prims.Articulation` in Isaac Sim 6.0.x: batched, warp arrays,
    plural names (`get_world_poses`, `set_world_poses`, `set_dof_positions`, `set_velocities`,
    `set_dof_position_targets`). v0.3.0/v0.3.1 called the old singular names (`get_world_pose`, `set_world_pose`,
    `set_joint_positions`, `apply_action`), so the pose could not be read (the follower never saw the robot move)
    and the stand-up / reset steps silently did nothing. All robot access now goes through
    articulation_api.ArticulationAPI, which uses the 6.x names first and the old names as a fallback, converts
    numpy / warp / torch (CPU or CUDA) as needed and reports real reasons instead of hiding them.
  * The velocity command is a torch tensor on the policy's own device (`robot._device`), like the examples.
  * Re-initialises when `is_physics_tensor_entity_valid()` is False (the examples do the same every step).
  * Once per robot the console prints which robot API / device / command type was detected.

Earlier fixes kept:
  * v0.3.1: persistent policy failure -> `body.error` (robot goes to ERROR with the reason) instead of a traceback
    on every physics step.
  * v0.3.0: robots are added to an already-loaded world, so right after initialise() the base is put back at the
    spawn pose with zero velocities and PD targets at the trained standing pose; the policy's own timing
    (sim dt, decimation) is exposed so the scenario can match the physics rate; the policy files that were
    actually loaded are recorded.
Pure helpers (construct_policy, record_policy_loads, discover_default_policy_paths) have no Isaac Sim
dependency and are unit-tested.
"""
from __future__ import annotations

import contextlib
import importlib
import inspect
import math
import re

import numpy as np

from .articulation_api import ArticulationAPI, PoseUnavailable
from .bodies import LocomotionBody
from .torch_compat import PolicyRunner, is_torch, to_numpy, zeros_like_any

POLICY_PACKAGES = ("isaacsim.robot.policy.examples.robots",)
SETTLE_S = 1.0  # zero command after (re)initialisation so the gait stabilises
MAX_INIT_ATTEMPTS = 400
POSE_FAILURES_BEFORE_ERROR = 200  # ~1 s at 200 Hz of unreadable pose -> report instead of silently freezing
REINIT_GRACE_STEPS = 20
PARK_MAX_SINK_M = 0.05  # v0.5.5: a parked robot whose base sinks more than this wakes up (policy takes over)
PARK_MAX_TILT_DEG = 8.0  # do not trust is_physics_tensor_entity_valid() right after an initialisation


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# ---------------------------------------------------------------------------------------------- lookup
def resolve_policy_module(module: str):
    errors = []
    for pkg in POLICY_PACKAGES:
        try:
            return importlib.import_module(f"{pkg}.{module}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{pkg}.{module}: {exc}")
    raise ImportError("walking policy not found - is the 'isaacsim.robot.policy.examples' extension enabled? "
                      + "; ".join(errors))


def resolve_policy_class(module: str, hint: str):
    mod = resolve_policy_module(module)
    if hint and hasattr(mod, hint):
        return getattr(mod, hint)
    found = [c for n, c in inspect.getmembers(mod, inspect.isclass)
             if n.endswith("Policy") and c.__module__ == mod.__name__]
    if found:
        return found[0]
    raise ImportError(f"{mod.__name__}: no *Policy class")


def detect_physics_engine() -> str:
    """'physx' or 'newton' (best effort; the policy class makes the final choice at load time)."""
    try:
        from isaacsim.core.simulation_manager import SimulationManager

        for fn in ("get_active_physics_engine", "get_physics_engine", "get_active_engine"):
            if hasattr(SimulationManager, fn):
                return "newton" if "newton" in str(getattr(SimulationManager, fn)()).lower() else "physx"
    except Exception:  # noqa: BLE001
        pass
    try:
        import carb

        settings = carb.settings.get_settings()
        for key in ("/physics/engine", "/isaacsim/physics/engine", "/exts/isaacsim.core.simulation_manager/engine"):
            val = settings.get(key)
            if isinstance(val, str) and val:
                return "newton" if "newton" in val.lower() else "physx"
    except Exception:  # noqa: BLE001
        pass
    return "physx"


_DIR_RE = re.compile(r'(\w*dir\w*)\s*=\s*[^\n]*?["\'](/Isaac/[^"\']+)["\']')
_FILE_RE = re.compile(r'["\'](/Isaac/[^"\']+\.(?:pt|jit|onnx|ya?ml))["\']')
_FSTR_RE = re.compile(r'\{(\w+)\}/([\w\-.]+\.(?:pt|jit|onnx|ya?ml))')


def discover_default_policy_paths(source: str, engine: str = "physx") -> tuple:
    """Read the default (policy, env) paths from a policy module's source code.

    Handles both styles used by isaacsim.robot.policy.examples:
      policy_dir = assets_root_path + "/Isaac/Samples/Policies/go2"; f"{policy_dir}/physx_policy.pt"
      assets_root_path + "/Isaac/Samples/Policies/H1_Policies/h1_policy.pt"
    Returns paths relative to the assets root, or ("", "").
    """
    dirs = {m.group(1): m.group(2) for m in _DIR_RE.finditer(source)}
    files = [m.group(1) for m in _FILE_RE.finditer(source)]
    files += [f"{dirs[m.group(1)]}/{m.group(2)}" for m in _FSTR_RE.finditer(source) if m.group(1) in dirs]

    def pick(exts):
        cands = list(dict.fromkeys(f for f in files if f.lower().endswith(exts)))
        if not cands:
            return ""
        if engine == "newton":
            pref = [f for f in cands if "newton" in f.lower()]
        else:
            pref = [f for f in cands if "physx" in f.lower()] or [f for f in cands if "newton" not in f.lower()]
        return (pref or cands)[0]

    return pick((".pt", ".jit", ".onnx")), pick((".yaml", ".yml"))


def default_policy_paths(model, engine: str | None = None) -> tuple:
    """(policy_rel, env_rel, source) for a catalog model: from the module source, else the catalog."""
    from ...fleet_config.catalog import catalog_policy_files

    engine = engine or detect_physics_engine()
    if model.policy_module:
        try:
            pt, env = discover_default_policy_paths(inspect.getsource(resolve_policy_module(model.policy_module)), engine)
            if pt and env:
                return pt, env, f"{model.policy_module}.py"
        except Exception:  # noqa: BLE001
            pass
    pt, env = catalog_policy_files(model, engine)
    return pt, env, "catalog"


# ---------------------------------------------------------------------------------------------- construction
ARG_ALIASES = {
    "prim_path": ("prim_path", "path", "robot_prim_path"),
    "name": ("name", "robot_name"),
    "position": ("position", "translation"),
    "orientation": ("orientation",),
    "usd_path": ("usd_path", "usd"),
    "policy_path": ("policy_path", "policy_file_path", "policy_file"),
    "env_path": ("env_config_path", "policy_env_path", "env_path", "env_file"),
}
CORE_ARGS = ("prim_path", "name", "position", "orientation", "usd_path")


def construct_policy(cls, candidates: dict):
    """Instantiate a policy class passing only the keyword arguments its __init__ accepts.

    Returns (instance, set of logical argument names that were passed).
    """
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        params = {}
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs, used = {}, set()
    for logical, value in candidates.items():
        aliases = ARG_ALIASES.get(logical, (logical,))
        hit = next((a for a in aliases if a in params), None)
        if hit is None and accepts_any and logical in CORE_ARGS:
            hit = logical
        if hit is not None:
            kwargs[hit] = value
            used.add(logical)
    if "prim_path" not in used:
        return cls(candidates["prim_path"], **kwargs), used | {"prim_path"}
    try:
        return cls(**kwargs), used
    except TypeError as exc:
        minimal = {k: v for k, v in kwargs.items() if k in ARG_ALIASES["prim_path"] + ARG_ALIASES["position"]}
        try:
            return cls(**minimal), {k for k in used if k in ("prim_path", "position")}
        except TypeError:
            raise TypeError(f"{cls.__name__}{inspect.signature(cls.__init__)}: {exc}") from exc


@contextlib.contextmanager
def record_policy_loads(cls):
    """Record the (policy, env) file paths passed to load_policy() while constructing a policy."""
    calls: list = []
    original = getattr(cls, "load_policy", None)
    if original is None:
        yield calls
        return
    had_own = "load_policy" in cls.__dict__

    def recorder(self, *args, **kwargs):
        pt = args[0] if args else kwargs.get("policy_file_path", kwargs.get("policy_path", ""))
        env = args[1] if len(args) > 1 else kwargs.get("policy_env_path", kwargs.get("env_path", ""))
        calls.append((str(pt or ""), str(env or "")))
        return original(self, *args, **kwargs)

    cls.load_policy = recorder
    try:
        yield calls
    finally:
        if had_own:
            cls.load_policy = original
        else:
            del cls.load_policy


def policy_timing(policy) -> tuple:
    """(sim dt, decimation) the policy was trained with, from the loaded env.yaml."""
    dt, dec = getattr(policy, "_dt", None), getattr(policy, "_decimation", None)
    if not dt:
        try:
            params = getattr(policy, "policy_env_params", None) or {}
            dt = float(params["sim"]["dt"])
            dec = int(params.get("decimation", dec or 4))
        except Exception:  # noqa: BLE001
            pass
    return (float(dt) if dt else None), (int(dec) if dec else None)


# ---------------------------------------------------------------------------------------------- body
class PolicyBody(LocomotionBody):
    def __init__(self, model, prim_path: str, x: float, y: float, yaw: float, robot_name: str,
                 usd_path: str = "", policy_path: str = "", env_path: str = "") -> None:
        self.model = model
        self.prim_path = prim_path
        self.ready = False
        self.status_note = "initialising walking policy"
        self.error = ""
        self.init_issues: list = []  # "label: error" for init/reset steps that failed (also printed once)
        self._issues_seen: set = set()
        self._runner = PolicyRunner()
        self._attempts = 0
        self._init_error = ""
        self._pose_failures = 0
        self._steps_since_init = 0
        self._announced = False
        self._settle = 0.0
        self._spawn = (x, y, yaw)
        self._pending_reset: tuple | None = None
        self._last_pose = (x, y, yaw)
        self._z, self._up = model.spawn_z, 1.0
        self._park_z = None
        self.last_wake = ""
        candidates = {"prim_path": prim_path, "name": robot_name,
                      "position": np.array([x, y, model.spawn_z]), "orientation": yaw_quat_wxyz(yaw)}
        if usd_path:
            candidates["usd_path"] = usd_path
        if model.custom_policy:
            if not (policy_path and env_path):
                raise ValueError(f"{model.label} needs a policy file (.pt) and env.yaml exported from Isaac Lab "
                                 "(Fleet Builder fields or G1_POLICY_PATH / G1_ENV_PATH)")
            from .custom_policy import IsaacLabVelocityPolicy

            self.policy = IsaacLabVelocityPolicy(policy_path=policy_path, env_path=env_path, **candidates)
            self.policy_path, self.env_path = policy_path, env_path
        else:
            cls = resolve_policy_class(model.policy_module, model.policy_class_hint)
            override = bool(policy_path and env_path)
            if override:
                candidates.update(policy_path=policy_path, env_path=env_path)
            with record_policy_loads(cls) as calls:
                self.policy, used = construct_policy(cls, candidates)
                if override and not {"policy_path", "env_path"} <= used:
                    self.policy.load_policy(policy_path, env_path)  # constructor cannot take custom files
            self.policy_path, self.env_path = calls[-1] if calls else (policy_path or "(class default)", env_path or "")
        self.policy_name = type(self.policy).__name__
        self.policy_dt, self.decimation = policy_timing(self.policy)
        self.api = ArticulationAPI(self.policy.robot)

    @property
    def command_style(self) -> str:
        """'torch' or 'numpy' once the policy has accepted a command, else ''."""
        return self._runner.style or ""

    def diagnostics(self) -> str:
        return (f"{self.policy_name} | robot API: {self.api.describe()} | command: {self.command_style or '?'} | "
                f"reset issues: {self.init_issues or 'none'} | unsupported ops: {sorted(self.api.absent) or 'none'}")

    # ------------------------------------------------------------------
    def step(self, dt: float, command: list) -> None:
        if self.holding:
            self.step_resume()
        if self.error:
            return  # latched: wait for STOP -> RUN (request_reinit) or RESET ROBOT instead of failing every step
        if not self.ready:
            self._try_init()
            return
        if self._steps_since_init > REINIT_GRACE_STEPS and self.api.is_valid() is False:
            # the physics tensor view was rebuilt (timeline stop/play, world reset): initialise again, in place
            print(f"[Physical_AI_Fleet] {self.prim_path}: physics tensors became invalid - re-initialising")
            keep = self._last_pose
            self.request_reinit()
            self._pending_reset = keep
            return
        self._steps_since_init += 1
        if self._settle > 0.0:
            self._settle -= dt
            command = [0.0, 0.0, 0.0]
        if not self._runner.forward(self.policy, dt, command) and self._runner.failed:
            self.error = f"policy.forward failed: {self._runner.last_error}"[:120]

    def hold(self, dt: float) -> None:
        """Parked (v0.5.5): do NOT run the walking network; the joints keep PD targets at the trained standing pose.
        The base height and tilt are watched (pose() is read every step before this): if it sinks more than
        PARK_MAX_SINK_M or tilts more than PARK_MAX_TILT_DEG the policy takes over at once with a clean re-stand, and
        last_wake says why. Resumes the same way when step() is called for a task."""
        if self.error or not self.ready or self._settle > 0.0:
            self.step(dt, [0.0, 0.0, 0.0])
            return
        if not self.holding:
            default_pos = getattr(self.policy, "default_pos", None)
            ok = False
            if default_pos is not None:
                ok, err = self.api.hold_joint_targets(to_numpy(default_pos, np.float32))
                if not ok and err:
                    self._note_issue("idle hold", err)
            if not ok:  # cannot hold without the policy: keep stepping it
                self.step(dt, [0.0, 0.0, 0.0])
                return
            self.holding = True
            self._park_z = self._z
            return
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, self._up))))
        sank = (self._park_z or self._z) - self._z
        if sank > PARK_MAX_SINK_M or tilt > PARK_MAX_TILT_DEG:
            self.last_wake = f"sank {sank * 100:.0f} cm, tilt {tilt:.0f} deg while parked"
            self.step(dt, [0.0, 0.0, 0.0])  # -> step_resume: clean re-stand, then the policy balances it
            return
        self._steps_since_init += 1

    def step_resume(self) -> None:
        """Leave parking with a clean re-stand where the robot is: standing joint pose, zero velocities, cleared action
        memory, then SETTLE_S of zero command before it walks."""
        if self.holding:
            self.holding = False
            self._park_z = None
            self._teleport(*self._last_pose)

    def _try_init(self) -> None:
        self._attempts += 1
        try:
            self.policy.initialize()  # the only mandatory step (sets joint defaults, gains, limits)
        except Exception as exc:  # noqa: BLE001 - physics view not ready yet
            self._init_error = f"{type(exc).__name__}: {exc}"
            if self._attempts >= MAX_INIT_ATTEMPTS:
                self.status_note = f"policy init failed: {self._init_error}"[:80] + " (press STOP then RUN)"
            return
        post_reset = getattr(self.policy, "post_reset", None)
        if callable(post_reset):
            try:
                post_reset()  # 6.x: robot.reset_to_default_state()
            except Exception as exc:  # noqa: BLE001
                self._note_issue("post_reset", f"{type(exc).__name__}: {exc}"[:120])
        default_pos = getattr(self.policy, "default_pos", None)
        if default_pos is not None and self.api.flavor != "experimental":
            # old API only: 6.x initialize() already puts the joints into the trained standing pose
            self._check("default joint state", self.api.set_default_joint_state(default_pos))
        self.ready = True
        self.status_note = ""
        self._steps_since_init = 0
        target, self._pending_reset = (self._pending_reset or self._spawn), None
        self._teleport(*target)  # standing pose + zero velocities, exactly like world.reset() in the examples
        self.policy_dt, self.decimation = policy_timing(self.policy)
        if not self._announced:
            self._announced = True
            print(f"[Physical_AI_Fleet] {self.prim_path}: {self.diagnostics()}")

    def pose(self) -> tuple:
        if not self.ready:
            return self._last_pose
        try:
            x, y, z, w, qx, qy, qz = self.api.get_pose()
        except PoseUnavailable as exc:
            self._pose_failures += 1
            if self._pose_failures == 1:
                print(f"[Physical_AI_Fleet] {self.prim_path}: cannot read the robot pose: {exc}\n"
                      f"    robot API: {self.api.describe()}")
            if self._pose_failures >= POSE_FAILURES_BEFORE_ERROR and not self.error:
                self.error = f"cannot read pose (see console): {exc}"[:110]
            return self._last_pose
        self._pose_failures = 0
        self._z = z
        self._up = 1.0 - 2.0 * (qx * qx + qy * qy)
        yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        self._last_pose = (x, y, yaw)
        return self._last_pose

    def is_fallen(self) -> bool:
        if not self.ready or self._settle > 0.0 or self.holding:  # v0.5.5: parked robots are watched by hold()
            return False
        return self._z < 0.45 * self.model.spawn_z or self._up < 0.5

    def reset_to(self, x: float, y: float, yaw: float) -> None:
        if self.error:  # operator reset: give a latched failure another chance
            self.error = ""
            self._pose_failures = 0
            self._runner.reset()
        if not self.ready:
            self._pending_reset = (x, y, yaw)
            return
        self._teleport(x, y, yaw)

    # ------------------------------------------------------------------
    def _note_issue(self, label: str, error: str) -> None:
        if label in self._issues_seen:
            return
        self._issues_seen.add(label)
        self.init_issues.append(f"{label}: {error}")
        print(f"[Physical_AI_Fleet] {self.prim_path}: {label} failed ({error})")

    def _check(self, label: str, result: tuple) -> None:
        ok, err = result
        if not ok and err:
            self._note_issue(label, err)

    def _teleport(self, x: float, y: float, yaw: float) -> None:
        """Stand the robot up at (x, y, yaw): base pose, trained joint pose, zero velocities, PD targets, clean memory."""
        self._check("set base pose", self.api.set_pose(x, y, self.model.spawn_z, yaw_quat_wxyz(yaw)))
        default_pos = getattr(self.policy, "default_pos", None)
        if default_pos is not None:
            dp = to_numpy(default_pos, np.float32)
            default_vel = getattr(self.policy, "default_vel", None)
            dv = to_numpy(default_vel, np.float32) if default_vel is not None else np.zeros_like(dp)
            self._check("set joint positions", self.api.set_joint_positions(dp))
            self._check("set joint velocities", self.api.set_joint_velocities(dv))
            self._check("hold standing joint targets", self.api.hold_joint_targets(dp))
        self._check("zero base velocity", self.api.zero_base_velocity())
        for attr in ("_previous_action", "_current_action", "action"):  # clear the policy's action memory
            val = getattr(self.policy, attr, None)
            if val is not None and (isinstance(val, np.ndarray) or is_torch(val)):
                try:
                    setattr(self.policy, attr, zeros_like_any(val))
                except Exception as exc:  # noqa: BLE001
                    self._note_issue(f"clear {attr}", f"{type(exc).__name__}: {exc}"[:120])
        if hasattr(self.policy, "_policy_counter"):
            self.policy._policy_counter = 0
        self._last_pose = (x, y, yaw)
        self._z, self._up = self.model.spawn_z, 1.0
        self._settle = SETTLE_S

    def request_reinit(self) -> None:
        self.ready = False
        self._attempts = 0
        self.error = ""
        self._pose_failures = 0
        self._steps_since_init = 0
        self._runner.reset()
        self.status_note = "re-initialising walking policy"

    def destroy(self) -> None:
        self.ready = False
