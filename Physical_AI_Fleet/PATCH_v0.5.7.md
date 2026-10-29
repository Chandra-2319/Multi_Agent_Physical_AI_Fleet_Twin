# Physical_AI_Fleet v0.5.7 - real Go2 twin without ROS on the PC

## Install
Replace Physical_AI_Fleet_python/, config/, tools/ and robot_side/. Your inputs/ are not touched - except: in
inputs/floor_layout.json change the twin's "connection" from "udp:9871;..." to "udp:9870;..." (or edit it in Fleet Builder).
Restart Isaac Sim.

## Go2 side: nothing to change
go2_udp_sender.py and go2_vel_server.py are the same as in v0.5.6. Keep running them (PC_IP = the Isaac Sim PC, port 9870).

## PC side
Stop go2_udp_receiver.py - not needed. No ROS 2 on the PC. Isaac Sim's own ROS 2 bridge is not used for the twin.

## Check in Isaac Sim
- Audit "twin_prepared": graphs_switched_off lists your ActionGraph (ROS2 Subscribe Joint State / Articulation Controller).
- Fleet Status: "data 0.0s 100/s, joints 0.0s". Move a leg on the real robot (or walk it): the twin's legs follow.
- If the legs do not move: audit / console for "joints not in the twin USD" (the USD's joint names differ from
  FL_hip_joint ... RR_calf_joint) - send me the joint names from the Stage panel.

## Tests
403 passed (394 of v0.5.6, 2 of them updated: the user's controller graph is now switched off; + 9 new). Not tested:
Isaac Sim physics with your twin USD, the real robot.
