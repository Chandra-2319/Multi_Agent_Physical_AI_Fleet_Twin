# SPDX-License-Identifier: Apache-2.0
"""Custom humanoid walking policy (Unitree G1 or any robot trained with the Isaac Lab flat-terrain velocity task).

v0.3.2: rewritten for Isaac Sim 6.0.x. The previous version re-implemented the observation with the removed
`SingleArticulation` API (`get_linear_velocity`, `ArticulationAction`, `initialize(physics_sim_view=...)`) and would
fail exactly like the pose read did. The Isaac Lab velocity task has the same layout the shipped H1 wrapper uses:

    base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) | velocity_command(3) |
    joint_pos - default(N) | joint_vel - default(N) | previous_action(N)         (N = number of joints)

so this class simply reuses `H1FlatTerrainPolicy` (NVIDIA's own, 6.x-compatible code: observation from
`get_world_poses` / `get_velocities` / `get_dof_positions`, actions through `set_dof_position_targets`) with YOUR
policy.pt + env.yaml and robot USD instead of the H1 defaults.

Requirements for the exported policy: TorchScript `policy.pt` + its `env.yaml`, trained on the same USD (identical
joint order), flat-terrain velocity task WITHOUT height-scan observations, action = joint-position offsets.
`action_scale` is read from env.yaml by the wrapper (`policy_env_params.get("action_scale", 0.5)`); if your env.yaml
does not expose it the default 0.5 is used - check this against your training config.
"""
from __future__ import annotations

try:
    from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy
except ImportError:  # package exports it at the top level too
    from isaacsim.robot.policy.examples.robots import H1FlatTerrainPolicy

DEFAULT_G1_USD = "/Isaac/Robots/Unitree/G1/g1.usd"


class IsaacLabVelocityPolicy(H1FlatTerrainPolicy):
    def __init__(self, prim_path: str, policy_path: str, env_path: str, name: str = "g1", usd_path: str | None = None,
                 position=None, orientation=None, root_path: str | None = None) -> None:
        if not usd_path:
            from isaacsim.storage.native import get_assets_root_path

            usd_path = get_assets_root_path() + DEFAULT_G1_USD
        super().__init__(
            prim_path=prim_path, root_path=root_path, usd_path=usd_path, position=position, orientation=orientation,
            policy_path=policy_path, env_config_path=env_path,
        )
