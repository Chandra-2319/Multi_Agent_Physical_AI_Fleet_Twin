"""Motion planning + path following for the walking robots (UnicycleBody stands in for the policy)."""
import math
import time

import pytest

from Physical_AI_Fleet_python.fleet_config.catalog import MODELS
from Physical_AI_Fleet_python.fleet_config.settings import Settings
from Physical_AI_Fleet_python.robots.legged_robot import LeggedSimRobot
from Physical_AI_Fleet_python.robots.locomotion.bodies import UnicycleBody
from Physical_AI_Fleet_python.robots.locomotion.planner import GridPlanner, RectObstacle
from Physical_AI_Fleet_python.robots.robot_factory import make_planner
from Physical_AI_Fleet_python.skills.executor import standoff_pose

DT = 1 / 200
LEGGED = [k for k, m in MODELS.items() if m.legged]


def walk(robot, seconds=90.0, check=None):
    t = 0.0
    while t < seconds and robot.goal is not None:
        robot.update(DT)
        if check:
            check(robot)
        t += DT
    return t


def test_planner_routes_around_obstacle():
    p = GridPlanner((-10, -10, 10, 10))
    p.set_obstacles([RectObstacle(0, 0, 1.0, 3.0)])
    path = p.plan((-4, 0), (4, 0), 0.4)
    assert path and len(path) > 2
    for a, b in zip(path, path[1:]):
        assert p.segment_free(a, b, 0.4) or b == path[-1]


def test_planner_straight_when_clear():
    p = GridPlanner((-10, -10, 10, 10))
    assert p.plan((0, 0), (3, 1), 0.4) == [(0.0, 0.0), (3.0, 1.0)]


def test_planner_handles_large_environment_fast():
    p = GridPlanner((-15, -15, 15, 15))
    shelves = [RectObstacle(x, y, 0.4, 1.5, f"s{x}{y}") for x in range(-12, 13, 3) for y in (-8, 0, 8)]
    clutter = [RectObstacle(-14 + 0.01 * i, 14, 0.05, 0.05) for i in range(3000)]  # many tiny prims
    p.set_obstacles(shelves + clutter)
    t0 = time.time()
    path = p.plan((-13.5, -12), (13.5, 12), 0.6)
    assert path and time.time() - t0 < 5.0
    for a, b in zip(path[1:-1], path[2:-1]):
        assert p.segment_free(a, b, 0.6)


@pytest.mark.parametrize("model", LEGGED)
def test_robot_walks_to_standoff_without_hitting_motor(model):
    s = Settings()
    planner = make_planner(s)
    m = MODELS[model]
    r = LeggedSimRobot("r", m.robot_type, (6.0, 2.0, math.pi), UnicycleBody(6.0, 2.0, math.pi), planner, model=model)
    asset = s.assets["motor-01"]
    goal = standoff_pose(asset, r.radius, 1.1)
    ax, ay = asset.position[:2]
    hx, hy = asset.half_extents

    def no_collision(rb):
        x, y, _ = rb.pose
        assert not (abs(x - ax) < hx + rb.radius * 0.8 and abs(y - ay) < hy + rb.radius * 0.8), "walked into the motor"

    r.go_to(*goal, name="motor-01")
    t = walk(r, check=no_collision)
    assert r.status == "idle" and r.goal is None, (r.status, r.note, t)
    assert math.dist(r.pose[:2], goal[:2]) <= m.goal_tol + 0.05
    assert abs((r.pose[2] - goal[2] + math.pi) % (2 * math.pi) - math.pi) < 0.2


@pytest.mark.parametrize("model", LEGGED)
def test_commands_stay_in_policy_limits(model):
    m = MODELS[model]
    r = LeggedSimRobot("r", m.robot_type, (0, -3, 0), UnicycleBody(0, -3, 0), make_planner(Settings()), model=model)
    r.go_to(-4.0, 4.0, 1.0)
    prev = [0.0, 0.0, 0.0]

    def limits(rb):
        nonlocal prev
        vx, vy, wz = rb.last_command
        assert -1e-9 <= vx <= m.max_vx + 1e-9 and abs(vy) <= m.max_vy + 1e-9 and abs(wz) <= m.max_wz + 1e-9
        assert abs(vx - prev[0]) <= m.accel * DT + 1e-9
        prev = rb.last_command

    walk(r, check=limits)
    assert r.status == "idle"


def test_humanoid_turns_in_place_before_walking():
    r = LeggedSimRobot("h", "humanoid", (0, 0, 0), UnicycleBody(0, 0, 0), make_planner(Settings()), model="h1")
    r.go_to(-3.0, 0.0, math.pi)
    for _ in range(40):
        r.update(DT)
        assert r.last_command[0] == 0.0
    walk(r)
    assert r.status == "idle"


def test_fall_reported_and_reset_recovers():
    body = UnicycleBody(0, -3, 0)
    r = LeggedSimRobot("g", "quadruped", (0, -3, 0), body, make_planner(Settings()), model="go2")
    r.go_to(2.0, -3.0, 0.0)
    for _ in range(100):
        r.update(DT)
    body.fallen = True
    r.update(DT)
    assert r.status == "error" and "fallen" in r.note and r.goal is None
    r.reset_to_home()
    r.update(DT)
    assert r.status == "idle" and r.pose[:2] == [0, -3]


def test_unreachable_goal_is_error():
    r = LeggedSimRobot("g", "quadruped", (0, -3, 0), UnicycleBody(0, -3, 0), make_planner(Settings()), model="go2")
    r.go_to(100.0, 100.0, 0.0)
    assert r.status == "error" and "no path" in r.note
