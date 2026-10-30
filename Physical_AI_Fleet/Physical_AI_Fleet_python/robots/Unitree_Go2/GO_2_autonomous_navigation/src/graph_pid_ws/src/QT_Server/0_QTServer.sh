#!/bin/bash
sleep 10
cd /unitree/module/graph_pid_ws
source /unitree/module/graph_pid_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/home/unitree/cyclonedds_ws/cyclonedds.xml;
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
source /opt/ros/foxy/setup.bash
ros2 launch QT_Server start.launch.py
