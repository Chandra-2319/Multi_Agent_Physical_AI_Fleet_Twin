"""Policy command handling: torch tensors on the policy's device, numpy fallback, latching failures.

(v0.3.1 tests, updated for v0.3.2: the command is built on `robot._device`; legacy-API robots still work.)
"""
import math
import sys

import numpy as np
import pytest
from fakes import FakeTensor, install_fake_torch_warp, install_policy_module

from Physical_AI_Fleet_python.fleet_config.catalog import MODELS
from Physical_AI_Fleet_python.robots.legged_robot import LeggedSimRobot
from Physical_AI_Fleet_python.robots.locomotion import policy_body as pb
from Physical_AI_Fleet_python.robots.locomotion.bodies import UnicycleBody
from Physical_AI_Fleet_python.robots.locomotion.planner import GridPlanner
from Physical_AI_Fleet_python.robots.locomotion.torch_compat import (
    MAX_CONSECUTIVE_FAILURES, PolicyRunner, call_variants, to_numpy, zeros_like_any,
)

DT = 1 / 200


@pytest.fixture()
def fake_torch(monkeypatch):
    install_fake_torch_warp(monkeypatch, with_warp=False)


class TorchArt:
    """Legacy-API articulation that returns / accepts torch tensors only (numpy raises)."""

    def __init__(self, pos):
        self.p, self.q = FakeTensor(pos), FakeTensor([1, 0, 0, 0])
        self.joints = self.jvel = None

    def get_world_pose(self):
        return self.p, self.q

    def set_world_pose(self, position, orientation):
        if not isinstance(position, FakeTensor):
            raise TypeError("expected torch tensor")
        self.p, self.q = position, orientation

    def set_joint_positions(self, x):
        if not isinstance(x, FakeTensor):
            raise TypeError("expected torch tensor")
        self.joints = x

    def set_joint_velocities(self, x):
        if not isinstance(x, FakeTensor):
            raise TypeError("expected torch tensor")
        self.jvel = x


class TorchH1:
    fail_first = 1

    def __init__(self, prim_path, root_path=None, usd_path=None, position=None, orientation=None):
        self.robot = TorchArt(position)
        self._dt, self._decimation = 0.005, 4
        self.forwards = []
        self._previous_action = FakeTensor(np.ones(4))

    def load_policy(self, a, b):
        pass

    def initialize(self):
        if TorchH1.fail_first > 0:
            TorchH1.fail_first -= 1
            raise RuntimeError("physics view not ready")
        self.default_pos = FakeTensor([-0.4, 0.8, -0.4, 0.8])

    def forward(self, dt, command):
        if not isinstance(command, FakeTensor):  # = `obs[9:12] = command` on a torch.FloatTensor
            raise TypeError("can't assign a numpy.ndarray to a torch.FloatTensor")
        self.forwards.append(command.v.copy())


def make_body(monkeypatch, cls=TorchH1):
    install_policy_module(monkeypatch, cls, "h1", "H1FlatTerrainPolicy")
    cls.fail_first = 1
    return pb.PolicyBody(MODELS["h1"], "/World/Fleet/h1_1", 1.0, -3.0, math.pi / 2, robot_name="h1-1")


def run_steps(body, n, command=(0.5, 0.0, 0.2)):
    for _ in range(n):
        body.step(DT, list(command))


def test_command_reaches_policy_as_tensor(monkeypatch, fake_torch):
    body = make_body(monkeypatch)
    run_steps(body, 2)
    assert body.ready and not body.error
    body._settle = 0.0
    run_steps(body, 5)
    assert body.command_style == "torch" and not body.error
    assert np.allclose(body.policy.forwards[-1], [0.5, 0.0, 0.2])


def test_old_behaviour_would_have_failed(monkeypatch, fake_torch):
    body = make_body(monkeypatch)
    with pytest.raises(TypeError, match="numpy.ndarray to a torch.FloatTensor"):
        body.policy.forward(DT, np.array([0.5, 0, 0.2]))


def test_legacy_api_stand_up_and_pose_with_tensors(monkeypatch, fake_torch):
    body = make_body(monkeypatch)
    run_steps(body, 2)
    robot = body.policy.robot
    assert body.api.flavor == "legacy" and body.init_issues == [], body.init_issues
    assert np.allclose(robot.joints.v, [-0.4, 0.8, -0.4, 0.8]) and np.allclose(robot.jvel.v, 0)
    x, y, yaw = body.pose()
    assert (round(x, 3), round(y, 3)) == (1.0, -3.0) and abs(yaw - math.pi / 2) < 1e-5


class NumpyH1(TorchH1):  # older Isaac Sim: numpy only
    def forward(self, dt, command):
        if not isinstance(command, np.ndarray):
            raise TypeError("numpy command required")
        self.forwards.append(np.array(command))


class NumpyArt:
    def __init__(self, pos):
        self.p, self.q = np.array(pos, float), np.array([1.0, 0, 0, 0])

    def get_world_pose(self):
        return self.p.copy(), self.q.copy()

    def set_world_pose(self, position, orientation):
        self.p, self.q = np.array(position, float), np.array(orientation, float)

    def set_joint_positions(self, x):
        self.j = np.array(x)

    def set_joint_velocities(self, x):
        self.jv = np.array(x)


class NumpyOnly(NumpyH1):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.robot = NumpyArt(k["position"])

    def initialize(self):
        self.default_pos = np.array([-0.4, 0.8, -0.4, 0.8])


def test_numpy_only_policy_still_works(monkeypatch, fake_torch):
    body = make_body(monkeypatch, NumpyOnly)
    run_steps(body, 2)
    body._settle = 0.0
    run_steps(body, 3)
    assert body.command_style == "numpy" and not body.error and body.init_issues == []


class AlwaysBroken(TorchH1):
    def forward(self, dt, command):
        self.calls = getattr(self, "calls", 0) + 1
        raise RuntimeError("boom")


def test_persistent_failure_latches_then_recovers(monkeypatch, fake_torch, capsys):
    body = make_body(monkeypatch, AlwaysBroken)
    run_steps(body, 2)
    body._settle = 0.0
    run_steps(body, MAX_CONSECUTIVE_FAILURES + 5)
    assert "boom" in body.error and body.error.startswith("policy.forward failed")
    calls = body.policy.calls
    run_steps(body, 50)
    assert body.policy.calls == calls
    assert capsys.readouterr().out.count("policy.forward failed") == 1
    body.request_reinit()
    assert body.error == "" and not body.ready


def test_failing_policy_puts_robot_in_error_without_raising(monkeypatch, fake_torch):
    body = make_body(monkeypatch, AlwaysBroken)
    robot = LeggedSimRobot("h1-1", "humanoid", (1.0, -3.0, math.pi / 2), body, GridPlanner((-15, -15, 15, 15)), model="h1")
    other = LeggedSimRobot("go2-1", "quadruped", (0, 0, 0), UnicycleBody(0, 0, 0), GridPlanner((-15, -15, 15, 15)), model="go2")
    robot.go_to(3.0, 0.0, 0.0, "x")
    for _ in range(MAX_CONSECUTIVE_FAILURES + 20):
        robot.update(DT)
        other.update(DT)
    assert robot.status == "error" and robot.note.startswith("policy.forward failed")
    assert robot.goal is None and other.status == "idle"
    robot.reset_to_home()
    assert robot.status == "idle" and body.error == ""


def test_helpers_with_tensors(fake_torch):
    t = FakeTensor([1, 2, 3])
    assert to_numpy(t).dtype == np.float64 and np.allclose(to_numpy(t), [1, 2, 3])
    assert isinstance(zeros_like_any(t), FakeTensor) and not zeros_like_any(t).v.any()
    ok, err = call_variants(TorchArt([0, 0, 0]), "set_joint_positions", np.zeros(4))
    assert ok and err == ""
    assert call_variants(object(), "missing", 1) == (False, "")


def test_runner_builds_the_command_on_the_policy_device(monkeypatch, fake_torch):
    seen = []

    class P:
        class robot:
            _device = "cuda:0"

        def forward(self, dt, c):
            seen.append(c)

    assert PolicyRunner().forward(P(), DT, [0.1, 0, 0])
    assert isinstance(seen[0], FakeTensor) and seen[0].device == "cuda:0"


def test_runner_without_torch_falls_back_to_numpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    seen = []

    class P:
        def forward(self, dt, c):
            if not isinstance(c, np.ndarray):
                raise TypeError("numpy required")
            seen.append(c)

    r = PolicyRunner()
    assert r.forward(P(), DT, [0.1, 0, 0]) and r.style == "numpy" and len(seen) == 1
