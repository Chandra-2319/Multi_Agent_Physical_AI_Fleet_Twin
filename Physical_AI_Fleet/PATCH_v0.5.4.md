# Physical_AI_Fleet v0.5.4

## Install
Replace Physical_AI_Fleet_python/, config/ and tools/ with the ones in this zip. Copy inputs/cosmos.json over yours (new
timing). Keep your other inputs: on the first start your inputs/asset_layout.json is copied into inputs/floor_layout.json
(assets kept, no robots yet). Restart Isaac Sim.

## 1. Thinking always on
Both vision queries (where is the gauge, what is the pressure) always use thinking. The VLM Thinking checkbox and any
"thinking" key in ai_profiles.json / cosmos.json are gone / ignored. Timing: 45 s per call, 150 s at the asset in total.

## 2. AMR
Plans its path with the same planner (walkable area, no-go zones, assets, its own 0.5 m radius). A goal it cannot reach is
refused with "no path to ..." (delivery fails -> incident needs the operator), never driven through a zone.
After MARK REPAIRED the AMR that brought the tools drives home on a planned path; it can take a new delivery on the way.

## 3. Faster simulation (Run Fleet > Skip Policy When Idle, on by default)
- An idle walking robot (no task, standing still for 1 s) stops running its walking network and holds its standing pose
  with the joint PD targets; it resumes at once when it gets a task (also walking home after a task counts as a task).
- Robot cameras are paused and only render from 2.5 m before the inspection pose until the reading is done.
- Fleet Status: "Performance: N physics steps/s (X real time), fleet code ms/step | walking policies running a/b |
  cameras rendering c/d". Compare with the option on and off.
Check in Isaac Sim: an idle robot holding its pose must not sag or slide; if it does, switch the option off and tell me.

## 4. Floor layout (inputs/floor_layout.json)
assets as before + robots: {"id", "model", "mode", "position": [x, y], "yaw_deg", "walking", optional "connection",
"map_to_world", "policy_path", "env_yaml_path"}. Fleet Builder has Home Yaw (deg). ADD/REMOVE ROBOT and ADD/REMOVE ASSET save
the file; SAVE FLOOR LAYOUT writes it; LOAD (scene) and LOAD FLOOR LAYOUT add the robots in it.

## Tests
python -m pytest -q tests -> 359 passed (347 existing + 12 new). Not tested: Isaac Sim itself (idle hold stability,
Camera.pause/resume in your Isaac Sim version, real FPS gain).
