"""Policy construction, default policy-path discovery and the stand-up sequence (no Isaac Sim needed)."""
import math
import sys
import types

import numpy as np
import pytest

from Physical_AI_Fleet_python.fleet_config.catalog import MODELS, catalog_policy_files
from Physical_AI_Fleet_python.robots.locomotion import policy_body as pb
from Physical_AI_Fleet_python.robots.locomotion.policy_body import (
    construct_policy, discover_default_policy_paths, record_policy_loads,
)

CAND = {"prim_path": "/World/Fleet/h1_1", "name": "h1-1", "position": np.zeros(3),
        "orientation": np.array([1.0, 0, 0, 0])}

GO2_SRC = '''
class Go2FlatTerrainPolicy(PolicyController):
    def __init__(self, prim_path, root_path=None, usd_path=None, position=None, orientation=None,
                 policy_path=None, env_config_path=None):
        assets_root_path = get_assets_root_path()
        policy_dir = assets_root_path + "/Isaac/Samples/Policies/go2"
        is_newton = active_engine == "newton"
        if policy_path is None:
            policy_path = f"{policy_dir}/newton_policy.pt" if is_newton else f"{policy_dir}/physx_policy.pt"
        if env_config_path is None:
            env_config_path = f"{policy_dir}/newton_env.yaml" if is_newton else f"{policy_dir}/physx_env.yaml"
        self.load_policy(policy_path, env_config_path)
'''
H1_SRC = '''
        self.load_policy(
            assets_root_path + "/Isaac/Samples/Policies/H1_Policies/h1_policy.pt",
            assets_root_path + "/Isaac/Samples/Policies/H1_Policies/h1_env.yaml",
        )
'''


# ---------------------------------------------------------------- path discovery
def test_discover_go2_physx_and_newton():
    assert discover_default_policy_paths(GO2_SRC, "physx") == (
        "/Isaac/Samples/Policies/go2/physx_policy.pt", "/Isaac/Samples/Policies/go2/physx_env.yaml")
    assert discover_default_policy_paths(GO2_SRC, "newton") == (
        "/Isaac/Samples/Policies/go2/newton_policy.pt", "/Isaac/Samples/Policies/go2/newton_env.yaml")


def test_discover_literal_paths():
    assert discover_default_policy_paths(H1_SRC) == (
        "/Isaac/Samples/Policies/H1_Policies/h1_policy.pt", "/Isaac/Samples/Policies/H1_Policies/h1_env.yaml")


def test_catalog_fallback_matches_go2_file():
    assert catalog_policy_files(MODELS["go2"], "physx")[0].endswith("go2/physx_policy.pt")
    assert catalog_policy_files(MODELS["go2"], "newton")[1].endswith("go2/newton_env.yaml")


# ---------------------------------------------------------------- construction
class Controller:
    def load_policy(self, policy_file_path, policy_env_path):
        self.loaded = (policy_file_path, policy_env_path)


class NoName(Controller):
    def __init__(self, prim_path, root_path=None, usd_path=None, position=None, orientation=None):
        self.load_policy("/x/h1_policy.pt", "/x/h1_env.yaml")


class Go2Like(Controller):
    def __init__(self, prim_path, usd_path=None, position=None, orientation=None, policy_path=None, env_config_path=None):
        self.load_policy(policy_path or "/d/physx_policy.pt", env_config_path or "/d/physx_env.yaml")


class NoOrientation(Controller):
    def __init__(self, prim_path, name="spot", usd_path=None, position=None):
        self.load_policy("/s/spot_policy.pt", "/s/spot_env.yaml")


def test_no_name_keyword_and_recording():
    with record_policy_loads(NoName) as calls:
        obj, used = construct_policy(NoName, CAND)
    assert "name" not in used and "orientation" in used
    assert calls == [("/x/h1_policy.pt", "/x/h1_env.yaml")]
    assert "load_policy" not in NoName.__dict__  # patch removed


def test_override_passed_to_constructor():
    cand = {**CAND, "policy_path": "/mine/p.pt", "env_path": "/mine/e.yaml"}
    with record_policy_loads(Go2Like) as calls:
        obj, used = construct_policy(Go2Like, cand)
    assert {"policy_path", "env_path"} <= used and calls[-1] == ("/mine/p.pt", "/mine/e.yaml")


def test_no_orientation_flags_pending_heading():
    _, used = construct_policy(NoOrientation, CAND)
    assert "orientation" not in used


def test_unknown_signature_reports_it():
    class Weird:
        def __init__(self, a, b, c):
            pass

    with pytest.raises(TypeError, match="Weird"):
        construct_policy(Weird, CAND)


# ---------------------------------------------------------------- stand-up sequence (fake policy package)
class FakeArt:
    def __init__(self):
        self.p, self.q = np.zeros(3), np.array([1.0, 0, 0, 0])
        self.joints, self.jvel, self.targets = np.zeros(4), np.ones(4), None

    def get_world_pose(self):
        return self.p.copy(), self.q.copy()

    def set_world_pose(self, position, orientation):
        self.p, self.q = np.array(position, float), np.array(orientation, float)

    def set_joint_positions(self, x):
        self.joints = np.array(x)

    def set_joint_velocities(self, x):
        self.jvel = np.array(x)

    def set_joints_default_state(self, positions):
        self.default = positions


class FakeH1(Controller):
    fail_first = 2

    def __init__(self, prim_path, root_path=None, usd_path=None, position=None, orientation=None):
        self.robot = FakeArt()
        self.robot.p = np.array(position, float)
        self.load_policy("/Isaac/Samples/Policies/H1_Policies/h1_policy.pt", "/Isaac/Samples/Policies/H1_Policies/h1_env.yaml")
        self._dt, self._decimation = 0.005, 4
        self.forwards = []

    def initialize(self):
        if FakeH1.fail_first > 0:
            FakeH1.fail_first -= 1
            raise RuntimeError("physics view not ready")
        self.default_pos = np.array([-0.4, 0.8, -0.4, 0.8])

    def forward(self, dt, cmd):
        self.forwards.append(np.array(cmd))


@pytest.fixture
def fake_policy_pkg(monkeypatch):
    for name in ("isaacsim", "isaacsim.robot", "isaacsim.robot.policy", "isaacsim.robot.policy.examples",
                 "isaacsim.robot.policy.examples.robots"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    mod = types.ModuleType("isaacsim.robot.policy.examples.robots.h1")
    FakeH1.__module__ = mod.__name__
    mod.H1FlatTerrainPolicy = FakeH1
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    FakeH1.fail_first = 2
    return FakeH1


def test_policy_body_stands_up_in_default_pose(fake_policy_pkg):
    body = pb.PolicyBody(MODELS["h1"], "/World/Fleet/h1_1", 1.0, -3.0, math.pi / 2, robot_name="h1-1")
    assert body.policy_path.endswith("h1_policy.pt") and body.env_path.endswith("h1_env.yaml")
    assert body.policy_dt == 0.005 and body.decimation == 4
    robot = body.policy.robot
    robot.p[2] = 0.3  # it sagged while physics was not ready yet
    for _ in range(3):
        body.step(0.005, [0.5, 0, 0])
    assert body.ready
    assert np.allclose(robot.joints, body.policy.default_pos)  # standing pose, not the USD zero pose
    assert np.allclose(robot.jvel, 0) and abs(robot.p[2] - MODELS["h1"].spawn_z) < 1e-9
    assert abs(body.pose()[2] - math.pi / 2) < 1e-6
    for _ in range(10):
        body.step(0.005, [0.5, 0, 0.2])
    assert all(np.allclose(c, 0) for c in body.policy.forwards)  # settles with zero command first
    body._settle = 0.0
    body.step(0.005, [0.5, 0, 0.2])
    assert np.allclose(body.policy.forwards[-1], [0.5, 0, 0.2])
