import os
import sys

import launch
import launch_ros.actions
from launch.actions import IncludeLaunchDescription,TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

#QtSrever路径
# unitree_slam_interfaces_path = os.path.join(os.path.abspath('.'), 'src', 'QT_Server')
unitree_slam_interfaces_path = os.path.join(os.path.abspath('.'), 'config_files', 'QT_Server_config')

def generate_launch_description():
    #QtSrever
    unitree_slam_interfaces_param = launch.substitutions.LaunchConfiguration(
        'unitree_slam_interfaces_config_dir',
        default=os.path.join(
            unitree_slam_interfaces_path,  
            'configs',
            'params.yaml'))   
    unitree_slam_interfaces_node = launch_ros.actions.Node(
        package='QT_Server',
        executable='unitree_slam',
        #prefix = ['xterm -e gdb -ex run --args'],
        parameters=[{unitree_slam_interfaces_param}],
        output='screen'
        )
    rs_lidar_cheak = launch_ros.actions.Node(
        package="QT_Server",
        executable="rs_lidar_cheak",
        parameters=[{unitree_slam_interfaces_param}],
        output='screen'
        )
    # dog_control = IncludeLaunchDescription(
    #   PythonLaunchDescriptionSource([os.path.join(
    #      get_package_share_directory('dog_control'), 'launch'),
    #      '/dogcontrol.launch.py'])
    # )

    return launch.LaunchDescription([
            unitree_slam_interfaces_node,
            rs_lidar_cheak,
            # dog_control,
            ])
