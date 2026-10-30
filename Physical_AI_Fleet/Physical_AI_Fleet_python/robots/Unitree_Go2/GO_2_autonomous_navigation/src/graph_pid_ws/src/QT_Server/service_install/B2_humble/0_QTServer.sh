#!/bin/bash
sleep 5
source /opt/ros/humble/setup.bash
source /home/unitree/slam_config/cyclonedds_go2_B2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/home/unitree/slam_config/cyclonedds_go2_B2_ws/cyclonedds.xml
cd /home/unitree/graph_pid_ws
source /home/unitree/graph_pid_ws/install/setup.bash
ros2 launch QT_Server start.launch.py
