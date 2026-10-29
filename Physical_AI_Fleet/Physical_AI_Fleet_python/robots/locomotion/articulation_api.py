# SPDX-License-Identifier: Apache-2.0
"""One interface to `policy.robot` for every Isaac Sim version.

Why this exists
---------------
In Isaac Sim 6.0.x `PolicyController.robot` is `isaacsim.core.experimental.prims.Articulation` (see
`policy_controller.py`: `from isaacsim.core.experimental.prims import Articulation`). That class is *batched*
and returns *warp* arrays:

    get_world_poses()      -> (positions (N,3), orientations (N,4))      # singular get_world_pose() does NOT exist
    get_velocities()       -> (linear (N,3), angular (N,3))
    set_world_poses(...)   set_velocities(...)   set_dof_positions(...)   set_dof_velocities(...)
    set_dof_position_targets(positions=...)      reset_to_default_state()
    is_physics_tensor_entity_valid()

Older Isaac Sim (SingleArticulation) has the singular `get_world_pose` / `set_world_pose` / `set_joint_positions` /
`apply_action(ArticulationAction)` instead. v0.3.0/v0.3.1 used only the old names, so on 6.0.1:

    AttributeError: 'Articulation' object has no attribute 'get_world_pose'

and every reset step silently did nothing. This module tries the 6.x names first, the old names second,
converts array types as needed (numpy / warp / torch, CPU or CUDA) and remembers what worked. Nothing here
raises for an API difference: operations report `(ok, error)` so the caller can show a real reason.
"""
from __future__ import annotations

import numpy as np

from .torch_compat import to_numpy, to_torch

KEYWORDS = ("pose", "veloc", "dof_pos", "joint_pos", "default_state", "target", "valid", "dof_names", "device")
FORMS = ("numpy", "warp", "torch")


class PoseUnavailable(RuntimeError):
    """The articulation exposes no pose accessor we can use (message lists what it does expose)."""


def _short(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:140]


class ArticulationAPI:
    def __init__(self, robot) -> None:
        self.robot = robot
        if hasattr(robot, "get_world_poses"):
            self.flavor = "experimental"  # Isaac Sim 6.x
        elif hasattr(robot, "get_world_pose"):
            self.flavor = "legacy"  # Isaac Sim 4.x / 5.x
        else:
            self.flavor = "unknown"
        self._ok: dict = {}  # operation -> (method name, array form) that worked last time
        self.issues: dict = {}  # operation -> last error text (method existed but every recipe failed)
        self.absent: set = set()  # operations for which no method exists on this API (not an error)

    # ------------------------------------------------------------------ introspection
    @property
    def device(self) -> str | None:
        dev = getattr(self.robot, "_device", None)
        return str(dev) if dev is not None else None

    def describe(self) -> str:
        names = sorted(n for n in dir(self.robot) if not n.startswith("_") and any(k in n for k in KEYWORDS))
        return f"{type(self.robot).__name__} [{self.flavor}, device={self.device}]: " + ", ".join(names[:40])

    def is_valid(self) -> bool | None:
        """False when the physics tensor view was rebuilt (timeline stop/play, world reset): re-initialise.

        None = this API cannot tell (treated as valid)."""
        fn = getattr(self.robot, "is_physics_tensor_entity_valid", None)
        if not callable(fn):
            return None
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ reading
    def get_pose(self) -> tuple:
        """(x, y, z, qw, qx, qy, qz) as floats, from numpy / warp / torch (CPU or CUDA) results."""
        errors = []
        names = ("get_world_poses", "get_world_pose")
        remembered = self._ok.get("pose")
        if remembered:
            names = (remembered[0],) + tuple(n for n in names if n != remembered[0])
        for name in names:
            fn = getattr(self.robot, name, None)
            if not callable(fn):
                continue
            try:
                pos, quat = fn()
                p, q = to_numpy(pos).reshape(-1), to_numpy(quat).reshape(-1)
                if p.size < 3 or q.size < 4:
                    raise ValueError(f"unexpected sizes {p.size}/{q.size}")
                self._ok["pose"] = (name, "any")
                return (*(float(v) for v in p[:3]), *(float(v) for v in q[:4]))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {_short(exc)}")
        if errors:
            raise PoseUnavailable("; ".join(errors))
        raise PoseUnavailable(f"no get_world_poses / get_world_pose on {type(self.robot).__name__} "
                              f"(it has: {self.describe().split(': ', 1)[-1][:160]})")

    # ------------------------------------------------------------------ writing
    def _device_for_form(self) -> str | None:
        return self.device

    def _as(self, value, form: str):
        """numpy array -> requested array type. Non-arrays pass through."""
        if not isinstance(value, np.ndarray):
            return value
        if form == "numpy":
            return value
        if form == "warp":
            import warp as wp  # may raise ImportError: this form is then skipped

            dev = self._device_for_form()
            return wp.array(value, dtype=wp.float32, device=dev) if dev else wp.array(value, dtype=wp.float32)
        return to_torch(value, self._device_for_form())

    def try_op(self, op: str, recipes: list, required: bool = False) -> tuple:
        """Run the first recipe that works. recipes = [(method_name, builder)], builder() -> (args, kwargs) of numpy.

        Returns (ok, error). A method that does not exist on this API is not an error unless `required`.
        """
        remembered = self._ok.get(op)
        ordered = list(recipes)
        if remembered:
            ordered.sort(key=lambda r: 0 if r[0] == remembered[0] else 1)
        existing = [(name, build) for name, build in ordered if callable(getattr(self.robot, name, None))]
        if not existing:
            self.absent.add(op)
            if required:
                self.issues[op] = f"no method among {sorted({n for n, _ in recipes})}"
                return False, self.issues[op]
            return False, ""
        forms = list(FORMS)
        if remembered and remembered[1] in forms:
            forms.remove(remembered[1])
            forms.insert(0, remembered[1])
        last = ""
        for name, build in existing:
            fn = getattr(self.robot, name)
            for form in forms:
                try:
                    args, kwargs = build()
                    fn(*[self._as(a, form) for a in args], **{k: self._as(v, form) for k, v in kwargs.items()})
                except ImportError:
                    continue  # e.g. warp / torch not importable here: try the next form
                except Exception as exc:  # noqa: BLE001 - wrong arg name / shape / array type
                    last = f"{name}[{form}]: {_short(exc)}"
                    continue
                self._ok[op] = (name, form)
                self.issues.pop(op, None)
                return True, ""
        self.issues[op] = last or "no usable array type"
        return False, self.issues[op]

    # convenience operations used by PolicyBody ---------------------------------------------------
    def set_pose(self, x: float, y: float, z: float, quat_wxyz) -> tuple:
        p, q = np.array([x, y, z], dtype=np.float64), np.asarray(quat_wxyz, dtype=np.float64)
        return self.try_op("set_pose", [
            ("set_world_poses", lambda: ((), {"positions": p[None], "orientations": q[None]})),
            ("set_world_poses", lambda: ((p[None], q[None]), {})),
            ("set_world_pose", lambda: ((), {"position": p, "orientation": q})),
        ], required=True)

    def set_joint_positions(self, values) -> tuple:
        v = to_numpy(values, np.float32)
        return self.try_op("set_joint_positions", [
            ("set_dof_positions", lambda: ((v,), {})),
            ("set_dof_positions", lambda: ((v[None],), {})),
            ("set_joint_positions", lambda: ((v,), {})),
        ])

    def set_joint_velocities(self, values) -> tuple:
        v = to_numpy(values, np.float32)
        return self.try_op("set_joint_velocities", [
            ("set_dof_velocities", lambda: ((v,), {})),
            ("set_dof_velocities", lambda: ((v[None],), {})),
            ("set_joint_velocities", lambda: ((v,), {})),
        ])

    def zero_base_velocity(self) -> tuple:
        z3 = np.zeros(3, dtype=np.float32)
        ok, err = self.try_op("zero_base_velocity", [
            ("set_velocities", lambda: ((), {"linear_velocities": z3[None], "angular_velocities": z3[None]})),
            ("set_velocities", lambda: ((z3[None], z3[None]), {})),
        ])
        if ok or err:
            return ok, err
        ok1, e1 = self.try_op("zero_linear_velocity", [("set_linear_velocity", lambda: ((z3,), {}))])
        ok2, e2 = self.try_op("zero_angular_velocity", [("set_angular_velocity", lambda: ((z3,), {}))])
        return ok1 and ok2, e1 or e2

    def hold_joint_targets(self, values) -> tuple:
        v = to_numpy(values, np.float32)

        def legacy():
            from isaacsim.core.utils.types import ArticulationAction  # old API only

            return (ArticulationAction(joint_positions=v),), {}

        return self.try_op("hold_joint_targets", [
            ("set_dof_position_targets", lambda: ((), {"positions": v})),
            ("set_dof_position_targets", lambda: ((), {"positions": v[None]})),
            ("apply_action", legacy),
        ])

    def set_default_joint_state(self, values) -> tuple:
        """Legacy API only: put the joints into the trained standing pose. (6.x initialize() already did this.)"""
        v = to_numpy(values, np.float32)
        return self.try_op("set_default_joint_state", [
            ("set_joints_default_state", lambda: ((), {"positions": v})),
            ("set_joints_default_state", lambda: ((v,), {})),
        ])
