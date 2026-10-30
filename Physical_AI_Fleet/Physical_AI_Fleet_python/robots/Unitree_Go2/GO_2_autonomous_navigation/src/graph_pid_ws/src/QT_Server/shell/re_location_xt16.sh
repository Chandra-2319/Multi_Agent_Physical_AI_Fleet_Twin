#!/bin/bash
current_path=$(dirname "$0")
echo "current path: $current_path"
set -e
cd $current_path
cd ../../..
source install/setup.bash
{
ros2 run lidar_passwd lidar_passwd

}&


{
ros2 launch QT_Server relio_qt_xt16.launch.py
}&

