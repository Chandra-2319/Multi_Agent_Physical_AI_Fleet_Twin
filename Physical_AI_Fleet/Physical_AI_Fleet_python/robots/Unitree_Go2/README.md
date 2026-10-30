# Go2 Autonomous Graph Navigation

Autonomous, goal-driven navigation for the Unitree Go2: build a 3D LiDAR map,
define a topological graph of waypoints, then send the robot to any node by number.

**Team:** <team name> | **Members:** <names> | **Hackathon:** <name>
**Demo video:** <link>

## How it works
1. **Mapping:** LIO-SAM (`lio_sam_ros2`) builds a 3D point-cloud map from a Hesai XT16 LiDAR.
2. **Graph:** the operator drops nodes (`x`) while walking; `graph_process` / `QT_Server`
   connect them into a topology (multi-floor supported).
3. **Planning:** A* over the graph (`nav2_planner_astar`), obstacle handling via costmap /
   occupancy grid, and PID path following in `local_planner`.
4. **Relocalization:** on a saved site, key `a` relocalizes on the stored map, so there is no re-mapping.
5. **Control:** C++ demo clients in `GO_2_autonomous_navigation/src/rocky_demos` talk to the stack over CycloneDDS
   (topics `rt/qt_command`, `rt/qt_add_node`, `rt/qt_add_edge`, ...).
   A watchdog stops/replans on obstacles (15 s timeout).

## Hardware / software
- Unitree Go2 (Jetson, L4T R35.3.1, aarch64) with Hesai XT16 LiDAR
- ROS 2 Foxy, CycloneDDS (`rmw_cyclonedds_cpp`), GTSAM 4, METIS, CUDA 11.4
- Onboard NIC must be named `eth0`; the robot and your laptop must be on the same network
- Prebuilt aarch64 binaries: download from the Releases page (or build from `GO_2_autonomous_navigation/src/`)

## Install (on the robot)
```bash
git clone https://github.com/vedsom/GO_2_Autonomous_navigation.git && cd GO_2_Autonomous_navigation
bash GO_2_autonomous_navigation/scripts/install_on_robot.sh
# edit ~/cyclonedds_wifi_test.xml: set Peer addresses to your machines' IPs
```
Install GTSAM and METIS into `/usr/local/lib`, then unpack the release binaries.

## Run: returning to a saved site (the hackathon demo)
Terminal 1 (`ssh unitree@<ROBOT_IP>`):
```bash
~/site_manager.sh load hackathon
export CYCLONEDDS_URI=~/cyclonedds_wifi_test.xml
source /unitree/module/graph_pid_ws/install/setup.bash
bash /unitree/module/graph_pid_ws/0_unitree_slam.sh
```
Terminal 2 (same env exports):
```bash
/unitree/lib/unitree_slam/rocky/build/demo_xt16_replan_manual_goal eth0   # single goal
/unitree/lib/unitree_slam/rocky/build/demo_xt16_priority_goal eth0        # multi-goal
```
Press `a` to relocalize (about 3 s), then enter a goal node number.

## Run: mapping a new site
Full steps are in [docs/AUTONOMOUS_STEPS.md](GO_2_autonomous_navigation/docs/AUTONOMOUS_STEPS.md). Summary:
`site_manager.sh save <current>` first, start the stack, then in the demo:
`w` (start map) -> walk the area -> `e` (end map) -> `x` at each node -> `c` (push topology) -> `q`;
Ctrl+C the stack and `site_manager.sh save <new_site>`.

## Keys
`w` start map | `e` end map | `x` add node | `c` push topology | `a` relocate |
`<n>`+Enter goal | `s` stop | `d` resume | `q` quit

## site_manager.sh
`list | current | save <name> | load <name>`. **Never press `w` before saving the live site:**
`w`/`e` overwrite `default/` with no backup.

## Troubleshooting
- Demo can't find the robot: check `CYCLONEDDS_URI` and the peers in the xml.
- A USB-Ethernet dongle can steal the `eth0` name.
- `load` refuses to run while the stack is up: Ctrl+C `0_unitree_slam.sh` first.
- Missing libs: see `GO_2_autonomous_navigation/docs/ldd_missing.txt`.

## Third-party
LIO-SAM, Nav2, Hesai LiDAR driver, CycloneDDS, GTSAM, METIS and the Unitree SDK keep their own licenses.
