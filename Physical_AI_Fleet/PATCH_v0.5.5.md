# Physical_AI_Fleet v0.5.5 - robots that do not fall when idle, no more stuck pairs, simulation-time limits

## Install
Replace Physical_AI_Fleet_python/, config/ and tools/ with the ones in this zip. inputs/gauges.json is your file with only
high_value changed to 1.55 (your calibration -98 / -360 and normal_value 0.6 are kept): copy it over yours, or change that
one value yourself. Nothing else in inputs/ changes. Restart Isaac Sim.

## What changed
1. Parking (Run Fleet > Skip Policy When Idle, on by default): only at home (within 0.35 m), upright, no task, after 3 s
   standing still. Watched while parked: sinking > 5 cm or tilting > 8 deg wakes it at once with a clean re-stand
   (audit: robot_woke with the reason). After 2 such wakes in a run that robot is not parked again. A new task also wakes
   it with a clean re-stand + 1 s settle. A robot standing at an asset (e.g. waiting for the vision model) never parks.
2. Time limits in SIMULATION time: a whole task 900 s; walking to the target 60 s + 3 x distance / top speed (Spot,
   25 m: 135 s); "no progress" after 10 s with turning counted as progress. Slow physics no longer cuts tasks off.
3. Never left on a spot: after a failed task (timeout, stuck, navigation timeout) the robot walks home. Stuck (not fallen)
   = released once and sent home (robot_recovering); stuck again = waits for RESET ROBOT. Before a robot is sent to a spot,
   idle robots within 1.2 m of it are sent home (spot_cleared). Fallen robots still need RESET ROBOT.
4. Cameras stay on from switch-on until the task ends; re-created after STOP / RESET; log entries camera_on / camera_off.
5. high_value 1.55.

## Check in Isaac Sim
- Leave Spots idle at home for a few minutes: no robot_error; robot_woke entries are fine (they show a catch, not a fall).
- Fleet Status -> Performance: "walking policies running" drops for parked robots.
- Inject faults on two assets far apart: no "timeout" on long walks; after a failure the robot walks home.

## Tests
python -m pytest -q tests -> 374 passed (358 existing incl. 3 updated for the new rules + 16 new).
Not tested: Isaac Sim itself (how fast a parked Spot sags, Camera re-creation after STOP in your version).
