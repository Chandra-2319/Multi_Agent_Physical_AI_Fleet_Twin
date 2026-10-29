# Changelog
## [0.5.7] - 2026-10-30
### Changed - no ROS 2 on the PC
- The extension listens on UDP 9870 (go2_udp_sender.py's port) and sets the twin's 12 joints itself: Unitree -> Isaac
  order and the same smoothing as go2_udp_receiver.py, written by joint name with matching PD targets. go2_udp_receiver.py,
  ROS 2 on the PC and the OmniGraph joint pipeline are no longer needed.
- OmniGraphs in the twin USD with ROS 2 nodes or an Articulation Controller are switched off on load (they would fight the
  extension); other graphs are kept; a plain Go2 USD works.
- Fleet Status shows the packet rate and joint data age. A second program on the port is reported (no SO_REUSEADDR).
### Fixed
- Closing the twin link frees the UDP port at once (waits for the reader thread), so LOAD right after works.
## [0.5.6] - 2026-10-29
### Added - Digital Twin phase 1: the real Go2 mirrored in Isaac Sim
- Mode Digital Twin + Connection `udp:<port>[;jog=<ip>:<port>]` adds a MirrorRobot: your twin USD (with its OmniGraph) is
  referenced under /World/Fleet/<id>; string prim paths in the graph are re-pointed, collisions and gravity switched off,
  physics scenes inside the file deactivated. Its body follows the robot's odometry (forwarded by go2_udp_receiver.py to
  UDP 9871) through Map -> World every physics step; joints stay driven by your OmniGraph.
- ALIGN TWIN (Manual Control): Map -> World from where the robot stands now = its home; saved to floor_layout.json.
- JOG buttons (15 cm / 15 deg pulses, then stop) and STOP REAL ROBOT, through go2_vel_server.py.
- The fleet never tasks a mirror (snapshot + gate); simulated robots treat it as an obstacle; it is never "cleared" from
  a spot or teleported. Fleet Status shows data age and the odometry source.
- robot_side/: go2_udp_sender.py (+ odometry: /uslam/localization/odom, /utlidar/robot_odom, /sportmodestate; 100 Hz),
  go2_udp_receiver.py (+ forward to 9871), go2_vel_server.py (+ 0.5 s watchdog, allowed clients, 'stop'),
  go2_keyboard_control.py (repeats at 10 Hz), README.md.
- floor_layout.json robots may carry "usd".

## [0.5.5] - 2026-10-28
### Fixed
- Robots park (walking policy off) only at home, upright, idle 3 s; sink / tilt while parked wakes them with a clean
  re-stand (robot_woke); fall check off while parked.
- Task / walk / no-progress limits in simulation time (900 s, 60 s + 3 x distance / speed, 10 s; turning counts).
- Failed task -> walk home; stuck robot released once; idle robots cleared from another robot's target spot.
- Cameras stay on for the whole task, re-created after STOP / RESET; camera_on / camera_off.
- gauges.json high_value 1.55.
## [0.5.4] - 2026-10-28
### Changed
- Vision-model thinking is always on for both queries (finding the gauge and reading it); set in the code
  (perception/gauge_crop.py THINKING), not in the UI, ai_profiles.json or cosmos.json. Timing: detect.timeout_s 45,
  call_timeout_s 45, total_time_s 150.
- The AMR plans its path like the walking robots: inside the walkable area, clear of no-go zones and assets; an unreachable
  goal is refused ("no path"). After MARK REPAIRED the AMR that delivered the tools drives home (available while driving).
- inputs/floor_layout.json replaces asset_layout.json and holds assets AND robots (model, mode, home position, home yaw,
  connection, policy files). Robots are added on LOAD and by LOAD FLOOR LAYOUT; adding / removing robots or assets saves it.
  An existing inputs/asset_layout.json is migrated once. Fleet Builder has Home Yaw (deg).
### Added
- Faster simulation: "Skip Policy When Idle" (Run Fleet, on by default) - an idle walking robot holds its standing pose
  without running its walking network after 1 s; robot cameras render only when a capture is near (2.5 m from the
  inspection pose). Fleet Status shows physics steps/s, real-time factor, fleet code ms/step, policies and cameras active.
## [0.5.3] - 2026-10-27
### Added
- The vision model finds the pressure gauge in the image itself (bounding box) before reading it; the frame is cropped to that
  box. No robot / asset / gauge positions needed, so the same logic works for a real quadruped (cosmos.json "detect").
  crop.source ["vlm", "projection"]: the model's box first, the projected box (simulation) as fallback, then the full frame.
  In the simulation both boxes are logged (gauge_detect iou_with_projection).
- TEST VLM runs the same find-box -> crop -> read steps and writes <image>_vlmtest_box.jpg / _vlmtest_crop.jpg.
- tools/test_gauge_bbox.py: try the box formats (norm1000 / norm1 / pixels) on your images outside Isaac Sim.
## [0.5.2] - 2026-10-27
### Fixed
- Gauge misread in the simulation (high read as normal): the dial was ~1.5 % of the 1280x720 frame, so the vision model
  could not see the needle. The frame is now cropped to the gauge (projected from its world position with Isaac Sim's camera,
  fallback: robot pose + camera mount) and enlarged to 768 px before it is sent. Both images are saved (*_crop.jpg = what
  the model saw); a gauge_crop audit entry records the method and pixel position.
- Prompt: HIGH is judged by the needle's direction into the red sector, not by the tip touching the arc.
- Timing for thinking mode (measured 8-15 s per image): call_timeout_s 45, total_time_s 120, max_tokens 8192 (thinking used
  up to ~3800 tokens; at 4096 a reply could be cut off). VLM thinking is on by default.
## [0.5.1] - 2026-10-26
### Changed
- The pressure gauge is part of the asset USD: asset_types.json "gauge": {"model", "needle_prim"}; only the needle is turned.
  The separate /World/Gauges model, its mount and scale settings are gone. gauges.json only holds needle calibration.
- Asset models load exactly as authored; usd_rotate_deg and usd_scale (asset_types.json, overridable per layout entry) are
  applied as given. usd_auto_fit: true restores the automatic units / up-axis fit.
- AI Models panel: Provider (NVIDIA / Nebius); Nemotron URL / model / key; VLM URL / model / key / thinking; Profile File;
  LOAD / SAVE / FORGET / APPLY; TEST NEMOTRON; TEST VLM. Keys live in ~/.physical_ai_fleet/ai_profiles.json.
- Gauge reading uses the profile's VLM (default nvidia/nemotron-3.5-super-vl-120b-a12b), one call per image, temperature 0,
  thinking via chat_template_kwargs; per-model invoke URLs (/vlm/, /cosmos/, /chat/completions) are used as-is.

## [0.3.0] - 2026-10-15
### Fixed
- H1 (and other walking robots) falling right after RUN:
  - after the policy initialises, the robot is put in the trained standing pose (base at spawn height,
    joints at the policy's default pose, zero velocities, PD targets at that pose), as world.reset()
    does in the policy examples; previously robots added after LOAD started from the USD zero pose;
  - optional init steps are isolated so an API difference cannot leave the robot uncontrolled;
  - the physics step is set explicitly to the policy's env.yaml `sim.dt` (200 Hz) after LOAD and on every
    robot add; the measured step is shown in Fleet Status with a MISMATCH warning;
  - 1 s standing settle before walking; gentler H1 command limits.
### Added
- Policy (.pt) / Policy env.yaml fields show the default files of the selected model (read from the
  policy module source, e.g. go2.py physx/newton files) with found / NOT FOUND, and are updated with the
  files actually loaded. Edit them (browse) to run your own policy.
- Environment USD (browse) in World Controls, referenced under /World/Environment on LOAD; options for the
  ground plane and for deriving planner obstacles from the environment's collision geometry.
- Occupancy-grid planner (fast with large environments).

## [0.2.1] - 2026-10-15
### Fixed
- `H1FlatTerrainPolicy.__init__() got an unexpected keyword argument 'name'` (signature-aware construction).

## [0.2.0] - 2026-10-15
### Added
- Walking robots (Go2, Spot, H1, custom G1), Spot model, planner + path follower, fall detection, RESET ROBOT.

## [0.1.0] - 2026-10-14
- Fleet Builder, schemas, Cosmos/Nemotron pipeline, safety gate, executor, verifier, audit log.
