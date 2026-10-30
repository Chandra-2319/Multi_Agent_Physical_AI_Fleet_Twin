#!/bin/bash
robot_type=$1
current_path=$(dirname "$0")
echo "current path: $current_path"
set -e
cd $current_path
cd ../../..
source install/setup.bash

if [ -n "$robot_type" ]; then
    if [ "$robot_type" = "B2" ]; then
    	echo "robot type : B2"
    	{
    		ros2 launch rslidar_sdk start.launch.py
    	}&
    	{
    		ros2 launch dog_control dogcontrol.launch.py
    	}&
    elif [ "$robot_type" = "Go2" ]; then
    	echo "robot type : Go2"
    	{
    		ros2 launch dog_control dogcontrol.launch.py
    	}&
    else
    	echo "Invalid robot type value"
    	exit 1
    fi
else
    echo "Invalid robot type value"
    exit 1
fi
