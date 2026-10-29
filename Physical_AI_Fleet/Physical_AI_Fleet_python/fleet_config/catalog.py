# SPDX-License-Identifier: Apache-2.0
"""Locked robot model, capability and action catalogue.

Nemotron may only choose a skill from SKILLS, for a robot whose type supports it. No skill
carries velocity, joint, torque or trajectory parameters: low-level motion is generated locally
(motion planner -> path follower -> RL walking policy) inside the robot adapter.
"""
from __future__ import annotations

from dataclasses import dataclass

ROBOT_TYPES = ("quadruped", "humanoid", "amr")
MODES = ("simulation", "digital_twin")
PROVENANCE = ("real", "simulated", "operator_phone")


@dataclass(frozen=True)
class RobotModel:
    key: str
    label: str
    robot_type: str
    usd: str  # relative to the Isaac assets root (puppets / twins; policies spawn their own USD)
    legged: bool
    policy_module: str = ""  # isaacsim.robot.policy.examples.robots.<module>
    policy_class_hint: str = ""  # preferred class name; any *Policy class in the module is accepted
    custom_policy: bool = False  # needs user-supplied policy.pt + env.yaml (e.g. G1 from Isaac Lab)
    spawn_z: float = 0.0  # base height when spawned
    radius: float = 0.4  # footprint radius for planning / standoff
    max_vx: float = 0.5  # m/s, conservative limits inside the policy's training range
    max_vy: float = 0.0
    max_wz: float = 0.8  # rad/s
    accel: float = 0.6  # m/s^2 command ramp (keeps the gait stable)
    turn_in_place_rad: float = 0.6  # heading error above which the robot turns on the spot
    goal_tol: float = 0.25
    base_link: str = ""  # child link that carries the inspection camera
    placeholder_size: tuple = (0.6, 0.3, 0.4)
    placeholder_color: tuple = (0.5, 0.5, 0.5)
    # Fallback default policy files ((engine, policy.pt, env.yaml), relative to the assets root). The real
    # defaults are read from the policy module source at runtime and confirmed when the policy loads.
    policy_files: tuple = ()


MODELS: dict[str, RobotModel] = {
    m.key: m
    for m in (
        RobotModel("go2", "Quadruped - Unitree Go2", "quadruped", "/Isaac/Robots/Unitree/Go2/go2.usd", True,
                   "go2", "Go2FlatTerrainPolicy", spawn_z=0.40, radius=0.40, max_vx=0.8, max_vy=0.3, max_wz=1.0,
                   accel=0.8, turn_in_place_rad=0.7, goal_tol=0.2, base_link="base",
                   placeholder_size=(0.7, 0.3, 0.4), placeholder_color=(0.1, 0.6, 1.0),
                   policy_files=(("physx", "/Isaac/Samples/Policies/go2/physx_policy.pt",
                                  "/Isaac/Samples/Policies/go2/physx_env.yaml"),
                                 ("newton", "/Isaac/Samples/Policies/go2/newton_policy.pt",
                                  "/Isaac/Samples/Policies/go2/newton_env.yaml"))),
        RobotModel("spot", "Quadruped - Boston Dynamics Spot", "quadruped", "/Isaac/Robots/BostonDynamics/spot/spot.usd",
                   True, "spot", "SpotFlatTerrainPolicy", spawn_z=0.80, radius=0.60, max_vx=1.0, max_vy=0.4,
                   max_wz=1.0, accel=0.8, turn_in_place_rad=0.7, goal_tol=0.25, base_link="body",
                   placeholder_size=(1.1, 0.5, 0.6), placeholder_color=(1.0, 0.85, 0.1),
                   policy_files=(("physx", "/Isaac/Samples/Policies/Spot_Policies/spot_policy.pt",
                                  "/Isaac/Samples/Policies/Spot_Policies/spot_env.yaml"),)),
        RobotModel("h1", "Humanoid - Unitree H1", "humanoid", "/Isaac/Robots/Unitree/H1/h1.usd", True,
                   "h1", "H1FlatTerrainPolicy", spawn_z=1.05, radius=0.40, max_vx=0.5, max_vy=0.0, max_wz=0.5,
                   accel=0.3, turn_in_place_rad=0.5, goal_tol=0.3, base_link="torso_link",
                   placeholder_size=(0.4, 0.3, 1.7), placeholder_color=(1.0, 0.6, 0.1),
                   policy_files=(("physx", "/Isaac/Samples/Policies/H1_Policies/h1_policy.pt",
                                  "/Isaac/Samples/Policies/H1_Policies/h1_env.yaml"),)),
        RobotModel("g1", "Humanoid - Unitree G1 (custom policy)", "humanoid", "/Isaac/Robots/Unitree/G1/g1.usd", True,
                   custom_policy=True, spawn_z=0.80, radius=0.35, max_vx=0.5, max_vy=0.0, max_wz=0.5, accel=0.3,
                   turn_in_place_rad=0.5, goal_tol=0.3, base_link="torso_link",
                   placeholder_size=(0.4, 0.3, 1.3), placeholder_color=(0.9, 0.4, 0.1)),
        RobotModel("amr", "AMR - Nova Carter / MiR", "amr", "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd", False,
                   spawn_z=0.0, radius=0.5, max_vx=1.0, max_wz=1.5, goal_tol=0.15,
                   placeholder_size=(0.9, 0.6, 0.35), placeholder_color=(0.3, 0.9, 0.3)),
    )
}
DEFAULT_MODEL = {"quadruped": "go2", "humanoid": "h1", "amr": "amr"}


def model_for(key_or_type: str) -> RobotModel:
    if key_or_type in MODELS:
        return MODELS[key_or_type]
    if key_or_type in DEFAULT_MODEL:
        return MODELS[DEFAULT_MODEL[key_or_type]]
    raise ValueError(f"unknown robot model/type '{key_or_type}'")


def catalog_policy_files(model: RobotModel, engine: str = "physx") -> tuple:
    """(policy_rel, env_rel) from the catalog fallback, preferring the active physics engine."""
    for eng, pt, env in model.policy_files:
        if eng == engine:
            return pt, env
    return (model.policy_files[0][1], model.policy_files[0][2]) if model.policy_files else ("", "")


@dataclass(frozen=True)
class SkillSpec:
    name: str
    capability: str
    risk: str  # "low" | "high" (high needs operator approval)
    description: str
    needs_target: bool = True


SKILLS: dict[str, SkillSpec] = {
    s.name: s
    for s in (
        SkillSpec("inspect_asset", "inspect_visual", "low", "Walk to the asset approach pose and capture camera evidence."),
        SkillSpec("read_gauge", "inspect_gauge", "low", "Walk close to the asset, frame the gauge and read it."),
        SkillSpec("reinspect_asset", "inspect_visual", "low", "Independent second inspection to confirm a finding."),
        SkillSpec("deliver_tools", "carry_payload", "high", "Carry the tool kit to the asset (AMR)."),
        SkillSpec("pick_tool", "manipulate", "high", "Pick a tool from the rack (humanoid, stretch goal)."),
        SkillSpec("hold_position", "navigate", "low", "Stop and hold the current position.", needs_target=False),
        SkillSpec("return_home", "navigate", "low", "Return to the home pose.", needs_target=False),
    )
}

CAPABILITIES: dict[str, tuple] = {
    "quadruped": ("navigate", "inspect_visual", "inspect_gauge"),
    "humanoid": ("navigate", "inspect_visual", "manipulate"),
    "amr": ("navigate", "carry_payload"),
}

FORBIDDEN_KEYS = {
    "velocity", "vel", "cmd_vel", "linear", "angular", "twist", "joint", "joints",
    "joint_positions", "joint_velocities", "torque", "effort", "trajectory", "waypoints", "pwm",
}


def capabilities_for(robot_type: str) -> tuple:
    return CAPABILITIES.get(robot_type, ())


def robot_can(robot_type: str, skill: str) -> bool:
    spec = SKILLS.get(skill)
    return bool(spec) and spec.capability in capabilities_for(robot_type)


def catalog_for_prompt() -> list[dict]:
    return [{"skill": s.name, "capability": s.capability, "risk": s.risk, "description": s.description}
            for s in SKILLS.values()]
