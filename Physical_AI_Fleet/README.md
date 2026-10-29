# Physical_AI_Fleet — Isaac Sim extension (v0.3.0)

Event-driven, multi-robot inspection and response. Build a fleet from a simple UI and run it in
**Simulation** or **Digital Twin** mode, optionally inside your own environment USD.

| Model | Type | Simulation locomotion | Default policy (shown in the UI) |
|---|---|---|---|
| Unitree Go2 | Quadruped | RL policy (`go2.py`) | `/Isaac/Samples/Policies/go2/physx_policy.pt` + `physx_env.yaml` (newton_* on Newton) |
| Boston Dynamics Spot | Quadruped | RL policy (`spot.py`) | read from `spot.py` at runtime |
| Unitree H1 | Humanoid | RL policy (`h1.py`) | read from `h1.py` at runtime |
| Unitree G1 | Humanoid | your Isaac Lab policy | browse to `policy.pt` + `env.yaml` |
| Nova Carter / MiR | AMR | kinematic | – |

Default policy files are read from the policy module source of *your* Isaac Sim version, then
replaced in the fields by the files the policy class actually loaded.

```
skill goal -> occupancy-grid A* (assets + environment + other robots) -> pure-pursuit follower
           -> smooth [vx, vy, wz] within policy limits -> RL policy.forward() @ 200 Hz -> joint targets
```

## Install

1. Copy the `Physical_AI_Fleet` folder into `Isaac_SIM_Extensions/` (replace the old one).
2. `isaac-sim.sh --ext-folder <path>/Isaac_SIM_Extensions --enable Physical_AI_Fleet`
   (depends on `isaacsim.robot.policy.examples`; policy files and robot USDs come from the assets root).
3. Optional environment variables: `.env.example`.

## Use

| Step | UI frame | Action |
|---|---|---|
| 1 | World Controls | Optional **Environment USD** (folder icon to browse, or a `/Isaac/...` path), ground plane / obstacle options, then **LOAD** |
| 2 | Fleet Builder | Pick **Robot Model** + **Mode**. Policy fields show the default files and whether they were found. **ADD ROBOT** |
| 3 | Run Fleet | **RUN**: each policy starts on the first physics step, stands still for 1 s, then walks |
| 4 | Manual Control | Dispatch skills, **STOP ALL**, **RETURN HOME**, **RESET ROBOT** |
| 5 | Demo Events | **INJECT** fault → approve `deliver_tools` → **MARK REPAIRED** → verification closes the incident |

Fleet Status shows the physics rate. If it says **MISMATCH**, walking robots will fall: press RESET or LOAD.

## Environment USD notes

- Referenced under `/World/Environment`; any physics scene inside it is deactivated (ours carries the 200 Hz step).
- Must be in metres, Z-up, floor at z = 0. Turn off *Add Ground Plane* if it has its own floor collider.
- With *Environment Obstacles* on, collision prims between 8 cm and 1.8 m become planner obstacles.
  Keep robots, the motor and its approach pose inside the geofence (`fleet_config/settings.py`, ±15 m).

## Test without Isaac Sim

```bash
pip install -r requirements.txt
python -m pytest -q tests        # 59 tests
python tools/run_headless.py     # Go2 + Spot + H1 + AMR full loop (UnicycleBody stands in for the policy)
python tools/mock_go2_bridge.py  # fake Go2 for Digital Twin testing
```

## License

Apache-2.0 (see `LICENSE`).
