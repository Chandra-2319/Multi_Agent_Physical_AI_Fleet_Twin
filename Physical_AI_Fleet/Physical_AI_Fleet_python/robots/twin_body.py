# SPDX-License-Identifier: Apache-2.0
"""Writes the real robot's pose onto its twin in Isaac Sim every physics step (v0.5.6).
v0.5.7: also the 12 joint angles (write_joints), mapped by joint NAME onto the articulation's own DOF order; PD targets are
set to the same values so the drives hold them. No ROS / OmniGraph needed.

With an articulation (your OmniGraph USD: Articulation Controller drives the joints) the ROOT LINK is placed with the
articulation API and its velocity zeroed each step, so physics never moves the body - only the joints move, from
/go2/joint_states. Without an articulation the twin's Xform is posed. The Isaac imports are lazy (tests use fakes).
"""
from __future__ import annotations

import numpy as np

from .locomotion.articulation_api import ArticulationAPI


def _make_articulation(path: str):
    try:
        from isaacsim.core.experimental.prims import Articulation  # Isaac Sim 6.x
        return Articulation(path)
    except Exception:  # noqa: BLE001
        from isaacsim.core.prims import SingleArticulation  # older Isaac Sim
        a = SingleArticulation(path)
        a.initialize()
        return a


def _make_xform(path: str):
    from isaacsim.core.prims import XFormPrim
    return XFormPrim(path)


class TwinPose:
    def __init__(self, prim_path: str, articulation_root: str = "", make_articulation=_make_articulation,
                 make_xform=_make_xform) -> None:
        self.prim_path, self.articulation_root = prim_path, articulation_root
        self._make_art, self._make_xform = make_articulation, make_xform
        self._api = None
        self._xform = None
        self.error = ""
        self.writes = 0
        self._fails = 0

    def _target(self):
        if self.articulation_root:
            if self._api is None or self._api.is_valid() is False:
                try:
                    self._api = ArticulationAPI(self._make_art(self.articulation_root))
                except Exception as exc:  # noqa: BLE001 - physics view not ready yet: try again next step
                    self._api, self.error = None, f"articulation not ready: {type(exc).__name__}: {exc}"[:160]
            return self._api
        if self._xform is None:
            self._xform = self._make_xform(self.prim_path)
        return self._xform

    def write(self, x: float, y: float, z: float, quat_wxyz) -> bool:
        target = self._target()
        if target is None:
            return False
        q = np.asarray(quat_wxyz, dtype=np.float64)
        try:
            if isinstance(target, ArticulationAPI):
                ok, err = target.set_pose(x, y, z, q)
                if not ok:
                    raise RuntimeError(err)
                target.zero_base_velocity()
            else:
                target.set_world_poses(positions=np.array([[x, y, z]]), orientations=q[None])
        except Exception as exc:  # noqa: BLE001
            self._fails += 1
            self.error = f"{type(exc).__name__}: {exc}"[:160]
            if self._fails > 50:
                self._api = None  # rebuild the view (e.g. after STOP / PLAY)
                self._fails = 0
            return False
        self._fails, self.error = 0, ""
        self.writes += 1
        return True

    def _dof_index(self, names) -> list | None:
        """Positions of `names` in the articulation's DOF order; None if the articulation does not report its names."""
        if getattr(self, "_index", None) is not None and getattr(self, "_index_for", None) is self._api:
            return self._index
        raw = getattr(self._api.robot, "dof_names", None) if self._api is not None else None
        try:
            dof = [str(n) for n in (raw() if callable(raw) else raw)] if raw is not None else None
        except Exception:  # noqa: BLE001
            dof = None
        if not dof:
            self._index, self._index_for, self.dof_note = None, self._api, "DOF names unknown: joint order assumed"
            return None
        missing = [n for n in names if n not in dof]
        if missing:
            self._index, self._index_for = None, self._api
            self.dof_note = f"joints not in the twin USD: {missing[:4]}"
            return None
        self._index = [dof.index(n) for n in names]
        self._index_for, self.dof_note, self.dof_count = self._api, "", len(dof)
        return self._index

    def write_joints(self, names, positions) -> bool:
        """Set the twin's joints to the real robot's angles (radians, in the order of `names`)."""
        if not self.articulation_root or positions is None:
            return False
        api = self._target()
        if not isinstance(api, ArticulationAPI):
            return False
        idx = self._dof_index(list(names))
        if idx is None and getattr(self, "dof_note", "").startswith("joints not in"):
            self.joint_error = self.dof_note
            return False
        if idx is None:
            values = np.asarray(positions, dtype=np.float32)
        else:
            values = np.zeros(getattr(self, "dof_count", len(idx)), dtype=np.float32)
            current = getattr(self, "_last_full", None)
            if current is not None and len(current) == len(values):
                values[:] = current
            values[idx] = np.asarray(positions, dtype=np.float32)
        ok, err = api.set_joint_positions(values)
        if ok:
            api.hold_joint_targets(values)
            api.set_joint_velocities(np.zeros_like(values))
            self._last_full = values
            self.joint_writes = getattr(self, "joint_writes", 0) + 1
            self.joint_error = ""
        else:
            self.joint_error = err or "set_joint_positions not available"
        return ok

    def reset(self) -> None:
        self._api = None
        self._xform = None
        self._index = None
