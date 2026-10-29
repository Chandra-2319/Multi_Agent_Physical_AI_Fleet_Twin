# SPDX-License-Identifier: Apache-2.0
"""Create fleet members from UI input (model x mode).

v0.4.4: the planner covers the site map's walkable polygon (no-go zones blocked) and assets block their exact,
possibly rotated, footprint.
"""
from __future__ import annotations
from ..fleet_config.catalog import MODES, model_for
from .amr_adapter import MiRRestLink
from .frames import Frame2D
from .g1_adapter import G1Link
from .go2_adapter import Go2BridgeLink
from .legged_robot import LeggedSimRobot
from .locomotion.bodies import UnicycleBody
from .locomotion.planner import GridPlanner, PolygonObstacle, RectObstacle
from .sim_robot import SimRobot
from .twin_robot import TwinRobot
LINKS = {"quadruped": Go2BridgeLink.from_connection, "amr": MiRRestLink.from_connection, "humanoid": G1Link}


def asset_obstacles(settings) -> list:
    out = []
    for aid, a in settings.assets.items():
        fp = getattr(a, "footprint", None)
        out.append(PolygonObstacle(tuple(fp), aid) if fp else RectObstacle(a.position[0], a.position[1], *a.half_extents, aid))
    return out


def configure_site(planner: GridPlanner, settings) -> None:
    site = getattr(settings, "site", None)
    if site is not None:
        planner.set_site(site.walkable, [z.polygon for z in site.no_go])


def make_planner(settings, extra_obstacles=()) -> GridPlanner:
    site = getattr(settings, "site", None)
    planner = GridPlanner(site.bounds if site is not None else settings.geofence, resolution=0.2)
    configure_site(planner, settings)
    planner.set_obstacles(asset_obstacles(settings) + list(extra_obstacles))
    return planner


def create_robot(robot_id: str, robot_type: str, mode: str, home_pose: tuple, prim_path: str = "",
                 connection: str = "", map_to_world: str = "", model: str = "", body=None, planner=None,
                 walking: bool = True):
    """walking=True: legged sim robots walk with `body` (RL policy in Isaac Sim, UnicycleBody headless).
    walking=False: kinematic glide (fallback when policy assets are unavailable)."""
    m = model_for(model or robot_type)
    robot_type = robot_type or m.robot_type
    if m.robot_type != robot_type:
        raise ValueError(f"model {m.key} is a {m.robot_type}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "simulation":
        if m.legged and walking:
            if planner is None:
                raise ValueError("legged robots need a motion planner")
            body = body or UnicycleBody(*home_pose)
            loco = "unicycle" if isinstance(body, UnicycleBody) else "rl_policy"
            return LeggedSimRobot(robot_id, robot_type, home_pose, body, planner, prim_path, m.key, loco)
        return SimRobot(robot_id, robot_type, mode, home_pose, prim_path, m.key, planner=planner)  # v0.5.4: AMR plans too
    if not connection:
        raise ValueError("Digital Twin mode needs a connection (host:port or URL)")
    link = LINKS[robot_type](connection, Frame2D.parse(map_to_world))
    return TwinRobot(robot_id, robot_type, home_pose, link, prim_path, m.key)
