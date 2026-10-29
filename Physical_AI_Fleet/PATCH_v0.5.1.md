# Physical_AI_Fleet v0.5.1

## Install
Replace Physical_AI_Fleet_python/ and config/ with the ones in this zip, and copy inputs/asset_types.json, inputs/gauges.json
and inputs/cosmos.json over yours (your asset_layout.json, site_map.json and robots_actions.json are not touched).
Keep your Assets/ folder. Restart Isaac Sim.

## 1. Keys and models (AI Models panel)
Provider NVIDIA / Nebius -> the six fields fill from ~/.physical_ai_fleet/ai_profiles.json (example:
fleet_config/ai_profiles.example.json). Type the keys (or put them straight into the JSON), press SAVE. LOAD = fill + use,
APPLY = use without saving, FORGET = delete that provider's profile. TEST NEMOTRON / TEST VLM (choose a gauge image first).

## 2. Gauge inside the motor USD
In asset_types.json, electric_motor:
  "usd": "Assets/Motor.usd", "usd_scale": 0.5, "usd_rotate_deg": [90, 0, 0],
  "gauge": {"model": "digiten_1p6mpa", "needle_prim": ""}
- Put the needle's path from the Stage panel of Motor.usd into needle_prim (with or without the default prim name).
  Empty = the first prim named "Needle_3" (gauges.json needle_name).
- Change usd_scale / usd_rotate_deg there while trying the model; nothing is scaled automatically any more.
- Calibrate angle_at_min_deg / angle_at_max_deg in gauges.json: LOAD, select the needle under /World/Assets/motor_01/model,
  change "Rotate Z:fleetNeedle" until it points at 0, then at 1.6.
- The robot stands in front of the needle position, camera_distance_m (0.7) from the dial, facing it.

## 3. VLM timing
Thinking off by default (your test: 23-95 s per image with thinking on; total_time_s is 60). Raise total_time_s and
call_timeout_s in inputs/cosmos.json if you switch thinking on.

## Tests
python -m pytest -q tests  -> 316 passed (291 existing + 25 new; needs usd-core for the USD tests).
Not tested: Isaac Sim itself, the real Motor.usd, the live NVIDIA / Nebius endpoints.
