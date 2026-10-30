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


    lio_sam_and_occ = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(os.path.abspath('.'), 'src', 'lio_sam_ros2', 'launch', 'lio_mapping_qt_xt16.launch.py'))
        )   
    # occ_grid_map = IncludeLaunchDescription(
    #   PythonLaunchDescriptionSource([os.path.join(
    #      get_package_share_directory('multimap_server'), 'launch'),
    #      '/occmaping.launch.py'])
    #   )
    return LaunchDescription([
        # dog_control,
        # rslidar,
        # rviz_node,
        # occ_grid_map,
        TimerAction(period=2.0,actions=[lio_sam_and_occ])
    ])
