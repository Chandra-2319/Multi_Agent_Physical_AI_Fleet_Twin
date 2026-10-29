# Physical_AI_Fleet v0.5.6 - the real Go2 mirrored in Isaac Sim (includes v0.5.5)

## Install
Replace Physical_AI_Fleet_python/, config/, tools/ and add robot_side/. Your inputs/ are not touched (only
inputs/gauges.json high_value 1.55 from v0.5.5, if you have not changed it already). Restart Isaac Sim.

## Set up the real Go2: robot_side/README.md (step by step)
1. Go2: copy robot_side/go2_udp_sender.py and go2_vel_server.py over yours; set ALLOWED_CLIENTS (your PC's IP) in
   go2_vel_server.py; `source /unitree/module/graph_pid_ws/install/setup.bash`; start both.
2. PC: run robot_side/go2_udp_receiver.py instead of yours (it also forwards to UDP 9871).
3. Twin USD -> <extension>/Assets/Go2_twin.usd; remove any OmniGraph node that moves the body from /go2/odom.
4. Fleet Builder: Go2, Digital Twin, USD Assets/Go2_twin.usd, Connection udp:9871;jog=10.85.82.106:9999, home X/Y/Yaw,
   ADD ROBOT, RUN. Stand the robot on that home, facing the yaw, select it in Manual Control -> ALIGN TWIN.
5. Walk it with the remote: the twin follows. Fleet Status: "real-mirror idle (...) data 0.0s from /utlidar/robot_odom".

## Check in Isaac Sim (not testable here)
- The twin stays put when gravity / physics run and its joints still move from /go2/joint_states (articulation root is
  placed each physics step, collisions and gravity off).
- Audit "twin_prepared": articulation_root found, retargeted paths, graphs = 1.
- Height: the twin stands at the Go2 standing height (flat floor); roll / pitch / yaw come from the robot.

## Tests
python -m pytest -q tests -> 394 passed (374 of v0.5.5 + 20 new: UDP / TCP on localhost, the robot-side helpers,
OmniGraph path re-pointing on a real USD stage, align / save / reload, jog rules). Not tested: ROS 2, the real robot,
Isaac Sim physics with your twin USD.
