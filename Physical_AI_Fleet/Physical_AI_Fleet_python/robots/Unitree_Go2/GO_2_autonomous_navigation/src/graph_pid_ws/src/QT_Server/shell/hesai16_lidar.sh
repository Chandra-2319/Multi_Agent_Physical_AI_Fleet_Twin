#!/bin/bash

current_path=$(dirname "$0")
echo "current path: $current_path"
set -e
cd $current_path
cd ../../..

source install/setup.bash
{
	ros2 launch hesai_lidar hesai_lidar_launch.py
}&
sleep 7
