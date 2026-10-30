import os

import launch_ros.actions
from launch.actions import IncludeLaunchDescription,TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration

#rviz_config = os.path.join(get_package_share_directory('task'), 'rviz2', 'build_map.rviz')
rviz_config = os.path.join(os.path.abspath('.'), 'src', 'task', 'rviz2', 'build_map.rviz')
print(rviz_config)

def generate_launch_description():
    
    dog_control = launch_ros.actions.Node(
        package='dog_control',
        executable='dog_control_B1_one',
        name='dog_control',
        output='screen'
        )

    #rslidar = launch_ros.actions.Node(
    #    package='rslidar_sdk',
    #    executable='rslidar_sdk_node',
    #    name='rslidar_sdk_node',
    #    output='screen'
    #    )
    
    rslidar = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('hesai_lidar'), 'launch'),
         '/hesai_lidar_launch.py'])
      )

    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        )

    # lio_sam_and_occ = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([os.path.join(
    #         get_package_share_directory('lio_sam_ros2'), 'launch'),
    #         '/lio_mapping.launch.py']))
    #add by zzr
    occ_grid_map = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('occ_grid_mapping'), 'launch'),
         '/mapping.launch.py'])
      )

    lio_sam_and_occ = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(os.path.abspath('.'), 'src', 'lio_sam_ros2', 'launch', 'lio_mapping.launch.py'))
        )   

    return LaunchDescription([
        # dog_control,
        # rslidar,
        rviz_node,
        occ_grid_map,
        TimerAction(period=2.0,actions=[lio_sam_and_occ])
    ])