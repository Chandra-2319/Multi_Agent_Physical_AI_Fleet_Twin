#!/bin/bash
sleep 5
cd /home/unitree/graph_pid_ws
source /home/unitree/graph_pid_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/home/unitree/slam_config/cyclone_go2_B2_ws/cyclonedds.xml
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
source /opt/ros/foxy/setup.bash
ros2 launch QT_Server start.launch.py
