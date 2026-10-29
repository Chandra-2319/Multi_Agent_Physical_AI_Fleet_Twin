# Real Go2 <-> Isaac Sim (Physical_AI_Fleet v0.5.7: no ROS on the PC)

    Go2  go2_udp_sender.py  --UDP 9870-->  PC  Isaac Sim + this extension (body pose AND joints, Map -> World)
    PC   extension JOG / go2_keyboard_control.py  --TCP 9999-->  Go2  go2_vel_server.py  --> /api/sport/request

Nothing runs on the PC except Isaac Sim: go2_udp_receiver.py is NOT needed any more (stop it - the extension listens on
UDP 9870 itself and reports "cannot listen on UDP port 9870" if the receiver still holds the port). No ROS 2 on the PC.
go2_keyboard_control.py is plain Python (no ROS) if you want keyboard driving.

## On the Go2 (unchanged since v0.5.6)
Always source the workspace first - it holds the unitree_go messages. Foxy has no `--once`: use `ros2 topic echo <topic>`.

    source /unitree/module/graph_pid_ws/install/setup.bash
    python3 ~/go2_udp_sender.py                  # terminal 1 - PC_IP inside must be the Isaac Sim PC, PORT 9870
    python3 ~/go2_vel_server.py                  # terminal 2 - set ALLOWED_CLIENTS to your PC's IP first

## The twin USD
Any Go2 USD with an articulation, e.g. your Go2_twin.usd in <extension>/Assets/. Graphs in it that use ROS 2 nodes or an
Articulation Controller are switched off when it is loaded (audit "twin_prepared" -> graphs_switched_off): the extension
sets the 12 joints itself, by joint name (FL_hip_joint ... RR_calf_joint), from the robot's packets - reordered from
Unitree to Isaac order and smoothed exactly like go2_udp_receiver.py did. Other graphs are left alone.

## In the extension
Fleet Builder: Go2, Digital Twin, USD Path Assets/Go2_twin.usd, Connection `udp:9870;jog=<robot ip>:9999`, Home X/Y/Yaw,
ADD ROBOT, RUN. Stand the robot on that home facing the home yaw, Manual Control -> ALIGN TWIN.
Fleet Status: "real-mirror idle (...) data 0.0s 100/s, joints 0.0s from /utlidar/robot_odom".
floor_layout.json entries with "udp:9871" from v0.5.6: change to "udp:9870" (9871 only worked through the receiver).

## Safety
* The fleet never gives the mirror a task (not a candidate; the safety gate refuses it). Simulated robots avoid it.
* JOG = one 15 cm step (0.25 m/s, 0.6 s) or a 15 deg turn, then zero; refused when the twin is offline or the end
  point is outside the walkable area / in a no-go zone. STOP REAL ROBOT sends zero at once.
* go2_vel_server.py stops the robot if commands stop for 0.5 s while it moves. The hand remote always overrides.
* Keep the robot password out of scripts: use an SSH key (ssh-copy-id unitree@<robot>).
