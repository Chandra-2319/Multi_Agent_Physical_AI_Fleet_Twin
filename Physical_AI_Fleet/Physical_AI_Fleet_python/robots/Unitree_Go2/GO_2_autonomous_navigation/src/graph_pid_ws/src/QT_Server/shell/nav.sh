#!/bin/bash
current_path=$(dirname "$0")
echo "current path: $current_path"
set -e
cd $current_path
cd ../../..
source install/setup.bash
{
ros2 launch QT_Server control_all.launch.py
}&

{
ros2 launch QT_Server graph_star_multifloor.launch.py

}&
sleep 2

