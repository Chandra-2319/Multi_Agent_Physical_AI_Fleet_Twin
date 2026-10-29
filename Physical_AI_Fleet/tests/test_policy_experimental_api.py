"""Isaac Sim 6.0.1: policy.robot is the experimental Articulation (plural names, warp arrays).

Reproduces the reported Fleet Status error
    cannot read robot pose: AttributeError: 'Articulation' object has no attribute 'get_world_pose'
and checks that robots now read their pose, stand up, walk with pose feedback and recover.
"""
import math
import sys
import types

import numpy as np
import pytest
from fakes import Fake6Articulation, Fake6Go2, FakeTensor, install_fake_torch_warp, install_policy_module

from Physical_AI_Fleet_python.fleet_config.catalog import MODELS
from Physical_AI_Fleet_python.robots.legged_robot import LeggedSimRobot
from Physical_AI_Fleet_python.robots.locomotion import policy_body as pb
from Physical_AI_Fleet_python.robots.locomotion.articulation_api import ArticulationAPI, PoseUnavailable
from Physical_AI_Fleet_python.robots.locomotion.planner import GridPlanner

DT = 1 / 200


@pytest.fixture()
def sim(monkeypatch):
    install_fake_torch_warp(monkeypatch)
    install_policy_module(monkeypatch, Fake6Go2)
    Fake6Go2.fail_first, Fake6Go2.accepts = 1, "any"
    return Fake6Go2


def make_body(x=1.0, y=-3.0, yaw=math.pi / 2):
    return pb.PolicyBody(MODELS["go2"], "/World/Fleet/go2_1", x, y, yaw, robot_name="go2-1")


def run(body, n, cmd=(0.0, 0.0, 0.0)):
    for _ in range(n):
        body.step(DT, list(cmd))


# ------------------------------------------------------------------ the reported error
def test_old_singular_pose_call_fails_exactly_like_the_report():
    art = Fake6Articulation([0, 0, 0.4], None)
    with pytest.raises(AttributeError, match="object has no attribute 'get_world_pose'"):
        art.get_world_pose()  # what v0.3.0 / v0.3.1 called


def test_pose_is_read_from_get_world_poses(sim):
    body = make_body()
    run(body, 3)
    assert body.ready and not body.error and body.api.flavor == "experimental"
    x, y, yaw = body.pose()
    assert (round(x, 3), round(y, 3)) == (1.0, -3.0) and abs(yaw - math.pi / 2) < 1e-4
    for _ in range(400):                     # > POSE_FAILURES_BEFORE_ERROR reads: must never latch an error
        body.pose()
    assert not body.error


def test_stand_up_uses_experimental_api_and_nothing_fails_silently(sim):
    body = make_body()
    run(body, 3)
    art = body.policy.robot
    assert body.init_issues == [], body.init_issues
    for call in ("set_world_poses", "set_dof_positions", "set_velocities", "set_dof_position_targets"):
        assert call in art.calls, call
    assert np.allclose(art.targets, np.linspace(-0.5, 0.5, 12), atol=1e-6)      # PD targets at the standing pose
    assert np.allclose(art.dof_vel, 0) and np.allclose(art.lin, 0)
    assert abs(art.pos[0, 2] - MODELS["go2"].spawn_z) < 1e-6                     # base put back at the spawn height


@pytest.mark.parametrize("accepted", ["numpy", "warp", "torch"])
def test_setters_adapt_to_the_array_type_the_api_accepts(sim, accepted):
    Fake6Go2.accepts = accepted
    body = make_body()
    run(body, 3)
    assert body.init_issues == [], (accepted, body.init_issues)
    body._settle = 0.0
    run(body, 8)                              # the policy itself must keep running with no failures
    assert body.command_style == "torch" and body._runner.failures == 0 and not body.error
    assert abs(body.policy.robot.pos[0, 2] - MODELS["go2"].spawn_z) < 1e-6


# ------------------------------------------------------------------ walking with pose feedback
def test_robot_walks_to_goal_using_the_measured_pose(sim):
    body = make_body(0.0, -3.0, 0.0)
    robot = LeggedSimRobot("go2-1", "quadruped", (0.0, -3.0, 0.0), body, GridPlanner((-15, -15, 15, 15)), model="go2")
    robot.go_to(2.0, -1.0, 0.0, "motor")
    for _ in range(200 * 40):
        robot.update(DT)
        if robot.status == "idle" and robot.goal is None and body.ready and body._settle <= 0:
            break
    assert robot.status == "idle" and not body.error, (robot.status, robot.note)
    assert math.dist(robot.pose[:2], (2.0, -1.0)) < 0.35                       # really moved, and we saw it
    assert body.command_style == "torch" and body.init_issues == []
    assert any(abs(f[0]) > 0.05 for f in body.policy.forwards)                  # forward commands reached the policy


def test_reset_robot_puts_it_back_home(sim):
    body = make_body(0.0, -3.0, 0.0)
    robot = LeggedSimRobot("go2-1", "quadruped", (0.0, -3.0, 0.0), body, GridPlanner((-15, -15, 15, 15)), model="go2")
    robot.go_to(2.0, -1.0, 0.0)
    for _ in range(200 * 6):
        robot.update(DT)
    assert math.dist(robot.pose[:2], (0.0, -3.0)) > 0.3
    robot.reset_to_home()
    robot.update(DT)
    assert robot.status == "idle" and math.dist(robot.pose[:2], (0.0, -3.0)) < 0.05
    assert np.allclose(body.policy.robot.lin, 0)


# ------------------------------------------------------------------ re-initialisation (stop / play)
def test_invalid_physics_tensors_trigger_in_place_reinit(sim):
    body = make_body(0.0, -3.0, 0.0)
    run(body, 60)
    assert body.ready
    body.policy.robot.pos[0, :2] = [2.0, 1.0]
    body.pose()                               # LeggedSimRobot.update() always reads the pose before stepping
    body.policy.robot.valid = False
    run(body, 1)
    assert not body.ready and "re-initialising" in body.status_note
    body.policy.robot.valid = True            # physics view rebuilt
    run(body, 3)
    assert body.ready and not body.error
    x, y, _ = body.pose()
    assert math.dist((x, y), (2.0, 1.0)) < 0.05   # re-placed where it was, not teleported to the spawn


# ------------------------------------------------------------------ clear failure reporting
class NoPoseArticulation(Fake6Articulation):
    def get_world_poses(self, indices=None):
        raise AttributeError("'Articulation' object has no attribute 'whatever'")


def test_unreadable_pose_is_reported_clearly_once(sim, capsys):
    body = make_body()
    run(body, 3)
    body.api.robot = NoPoseArticulation([1, -3, 0.4], None)
    robot = LeggedSimRobot("go2-1", "quadruped", (1.0, -3.0, 0.0), body, GridPlanner((-15, -15, 15, 15)), model="go2")
    for _ in range(pb.POSE_FAILURES_BEFORE_ERROR + 5):
        robot.update(DT)                      # must never raise
    assert robot.status == "error" and "cannot read pose" in robot.note and "see console" in robot.note
    out = capsys.readouterr().out
    assert out.count("cannot read the robot pose") == 1 and "robot API:" in out


def test_api_without_any_pose_method_lists_what_it_has():
    class Odd:
        _device = "cpu"
        def get_velocities(self): ...
        def set_dof_positions(self, p): ...
    api = ArticulationAPI(Odd())
    assert api.flavor == "unknown"
    with pytest.raises(PoseUnavailable, match="no get_world_poses / get_world_pose on Odd"):
        api.get_pose()
    assert "get_velocities" in api.describe()


# ------------------------------------------------------------------ G1 (custom policy) reuses NVIDIA's H1 wrapper
def test_custom_policy_forwards_user_files_to_the_h1_wrapper(monkeypatch):
    captured = {}

    class H1(object):
        def __init__(self, prim_path, root_path=None, usd_path=None, position=None, orientation=None,
                     policy_path=None, env_config_path=None):
            captured.update(prim_path=prim_path, usd_path=usd_path, policy_path=policy_path, env_config_path=env_config_path)

    for n in ("isaacsim", "isaacsim.robot", "isaacsim.robot.policy", "isaacsim.robot.policy.examples",
              "isaacsim.robot.policy.examples.robots"):
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    h1mod = types.ModuleType("isaacsim.robot.policy.examples.robots.h1")
    h1mod.H1FlatTerrainPolicy = H1
    monkeypatch.setitem(sys.modules, h1mod.__name__, h1mod)
    storage = types.ModuleType("isaacsim.storage.native")
    storage.get_assets_root_path = lambda: "https://assets"
    monkeypatch.setitem(sys.modules, "isaacsim.storage", types.ModuleType("isaacsim.storage"))
    monkeypatch.setitem(sys.modules, "isaacsim.storage.native", storage)
    sys.modules.pop("Physical_AI_Fleet_python.robots.locomotion.custom_policy", None)
    from Physical_AI_Fleet_python.robots.locomotion.custom_policy import IsaacLabVelocityPolicy

    IsaacLabVelocityPolicy(prim_path="/World/Fleet/g1_1", policy_path="/me/policy.pt", env_path="/me/env.yaml",
                           name="g1-1", position=np.zeros(3), orientation=np.array([1, 0, 0, 0]))
    assert captured == {"prim_path": "/World/Fleet/g1_1", "usd_path": "https://assets/Isaac/Robots/Unitree/G1/g1.usd",
                        "policy_path": "/me/policy.pt", "env_config_path": "/me/env.yaml"}
