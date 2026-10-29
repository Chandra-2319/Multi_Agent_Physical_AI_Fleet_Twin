"""Stand-ins for torch / warp / the Isaac Sim 6.0.1 policy classes (modelled on the real source files)."""
import math
import sys
import types

import numpy as np


class FakeTensor:  # type(x).__module__ == "torch"
    __module__ = "torch"

    def __init__(self, vals, device="cpu"):
        self.v = np.array(vals, dtype=np.float32)
        self.device = device

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.v.copy()

    def __array__(self, dtype=None, copy=None):
        raise TypeError("can't convert cuda tensor to numpy implicitly")


class FakeWarp:  # type(x).__module__ == "warp"
    __module__ = "warp.types"

    def __init__(self, vals):
        self.a = np.array(vals, dtype=np.float32)

    def numpy(self):
        return self.a.copy()


def install_fake_torch_warp(monkeypatch, with_warp=True):
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.as_tensor = lambda a, dtype=None, device=None: FakeTensor(a, device or "cpu")
    torch.zeros_like = lambda t: FakeTensor(np.zeros_like(t.v), t.device)
    monkeypatch.setitem(sys.modules, "torch", torch)
    if with_warp:
        warp = types.ModuleType("warp")
        warp.float32 = "float32"
        warp.array = lambda value, dtype=None, device=None: FakeWarp(value)
        monkeypatch.setitem(sys.modules, "warp", warp)
    else:
        monkeypatch.setitem(sys.modules, "warp", None)


def install_policy_module(monkeypatch, cls, module="go2", name="Go2FlatTerrainPolicy"):
    for n in ("isaacsim", "isaacsim.robot", "isaacsim.robot.policy", "isaacsim.robot.policy.examples",
              "isaacsim.robot.policy.examples.robots"):
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    mod = types.ModuleType(f"isaacsim.robot.policy.examples.robots.{module}")
    cls.__module__ = mod.__name__
    setattr(mod, name, cls)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)


# ---------------------------------------------------------------------------------------------- 6.0.1 API
class Fake6Articulation:
    """isaacsim.core.experimental.prims.Articulation: batched, plural names, warp results.

    There is NO get_world_pose / set_world_pose / set_joint_positions / apply_action here (as in 6.0.1).
    `accepts` = which array type the setters take ("numpy" | "warp" | "torch" | "any").
    """
    _device = "cuda:0"
    dof_names = [f"j{i}" for i in range(12)]

    def __init__(self, positions, orientations, accepts="any"):
        self.pos = np.array(positions, np.float32).reshape(1, 3)
        self.quat = np.array(orientations if orientations is not None else [1, 0, 0, 0], np.float32).reshape(1, 4)
        self.lin = np.zeros((1, 3), np.float32)
        self.dof_pos = np.ones((1, 12), np.float32)
        self.dof_vel = np.ones((1, 12), np.float32)
        self.targets = None
        self.valid = True
        self.accepts = accepts
        self.calls = []

    def _take(self, v, who):
        ok = {"numpy": np.ndarray, "warp": FakeWarp, "torch": FakeTensor}
        want = tuple(ok.values()) if self.accepts == "any" else (ok[self.accepts],)
        if not isinstance(v, want):
            raise TypeError(f"{who}: {type(v).__name__} not accepted")
        return v.numpy() if hasattr(v, "numpy") else np.asarray(v)

    def get_world_poses(self, indices=None):
        return FakeWarp(self.pos), FakeWarp(self.quat)

    def get_velocities(self, indices=None):
        return FakeWarp(self.lin), FakeWarp(np.zeros((1, 3)))

    def set_world_poses(self, positions=None, orientations=None, indices=None):
        self.pos = self._take(positions, "set_world_poses").reshape(1, 3)
        self.quat = self._take(orientations, "set_world_poses").reshape(1, 4)
        self.calls.append("set_world_poses")

    def set_velocities(self, linear_velocities=None, angular_velocities=None, indices=None):
        self.lin = self._take(linear_velocities, "set_velocities").reshape(1, 3)
        self.calls.append("set_velocities")

    def set_dof_positions(self, positions, indices=None, dof_indices=None):
        self.dof_pos = self._take(positions, "set_dof_positions").reshape(1, -1)
        self.calls.append("set_dof_positions")

    def set_dof_velocities(self, velocities, indices=None, dof_indices=None):
        self.dof_vel = self._take(velocities, "set_dof_velocities").reshape(1, -1)

    def set_dof_position_targets(self, positions=None, indices=None, dof_indices=None):
        self.targets = self._take(positions, "set_dof_position_targets").reshape(1, -1)
        self.calls.append("set_dof_position_targets")

    def get_dof_positions(self, indices=None, dof_indices=None):
        return FakeWarp(self.dof_pos)

    def get_dof_velocities(self, indices=None, dof_indices=None):
        return FakeWarp(self.dof_vel)

    def reset_to_default_state(self):
        self.calls.append("reset_to_default_state")

    def is_physics_tensor_entity_valid(self):
        return self.valid


class Fake6Go2:
    """Structure of go2.py / PolicyController in Isaac Sim 6.0.1 (observation from the robot, torch command)."""
    fail_first = 1
    accepts = "any"
    last = None

    def __init__(self, prim_path, root_path=None, usd_path=None, position=None, orientation=None,
                 policy_path=None, env_config_path=None):
        self.robot = Fake6Articulation(position, orientation, Fake6Go2.accepts)
        Fake6Go2.last = self
        self.load_policy(policy_path or "/Isaac/Samples/Policies/go2/physx_policy.pt",
                         env_config_path or "/Isaac/Samples/Policies/go2/physx_env.yaml")
        self._previous_action = None
        self._current_action = None
        self._policy_counter = 0
        self.forwards = []

    def load_policy(self, policy_file_path, policy_env_path):
        self._decimation, self._dt = 4, 0.005

    def initialize(self):
        if Fake6Go2.fail_first > 0:
            Fake6Go2.fail_first -= 1
            raise RuntimeError("physics view not ready")
        self.default_pos = FakeTensor(np.linspace(-0.5, 0.5, 12), "cuda:0")
        self.default_vel = FakeTensor(np.zeros(12), "cuda:0")

    def post_reset(self):
        self.robot.reset_to_default_state()

    def forward(self, dt, command):
        if not isinstance(command, FakeTensor):  # obs[9:12] = command  (obs is a torch tensor)
            raise TypeError("can't assign a numpy.ndarray to a torch.FloatTensor")
        r = self.robot
        pos, quat = r.get_world_poses()           # observation reads these every decimated step
        r.get_velocities()
        self.forwards.append(command.v.copy())
        vx, vy, wz = (float(c) for c in command.v)
        w, _x, _y, z = r.quat[0]
        yaw = 2 * math.atan2(z, w) + wz * dt     # perfect velocity tracking: the base really moves
        c, s = math.cos(yaw), math.sin(yaw)
        r.pos[0, 0] += (c * vx - s * vy) * dt
        r.pos[0, 1] += (s * vx + c * vy) * dt
        r.quat = np.array([[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]], np.float32)
        if self._previous_action is None:
            self._previous_action = FakeTensor(np.zeros(12), "cuda:0")
        if self._policy_counter % self._decimation == 0:
            r.targets = self.default_pos.v.reshape(1, -1).copy()   # policy's own call: always warp in the real API
            r.calls.append('policy:set_dof_position_targets')
        self._policy_counter += 1
