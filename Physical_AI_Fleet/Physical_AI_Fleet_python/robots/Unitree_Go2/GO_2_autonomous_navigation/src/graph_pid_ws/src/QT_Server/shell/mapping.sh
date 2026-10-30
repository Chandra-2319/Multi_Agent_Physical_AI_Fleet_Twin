#!/bin/bash

current_path=$(dirname "$0")
echo "current path: $current_path"
set -e
cd $current_path
cd ../../..

source install/setup.bash

{
ros2 launch QT_Server lio_mapping_qt.launch.py
}&

{
ros2 launch QT_Server occmaping.launch.py

}&
