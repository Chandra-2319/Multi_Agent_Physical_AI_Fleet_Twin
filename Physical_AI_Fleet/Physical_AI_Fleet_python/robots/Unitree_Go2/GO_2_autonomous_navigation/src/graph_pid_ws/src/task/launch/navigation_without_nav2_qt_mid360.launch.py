import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,TimerAction
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node



def generate_launch_description():
    fishbot_navigation2_dir = get_package_share_directory('task')
    # nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    map_yaml_path = LaunchConfiguration('map',default=os.path.join(os.path.abspath('.'), 'src','task','maps','map','default.yaml'))
    # nav2_param_path = LaunchConfiguration('params_file',default=os.path.join(fishbot_navigation2_dir,'config','robot.yaml'))

    rviz_config_dir = os.path.join(os.path.abspath('.'), 'src', 'task', 'rviz2', 'costmap.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time',default_value=use_sim_time,description='Use simulation (Gazebo) clock if true'),
        DeclareLaunchArgument('map',default_value=map_yaml_path,description='Full path to map file to load'),
        # DeclareLaunchArgument('params_file',default_value=nav2_param_path,description='Full path to param file to load'),

        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([nav2_bringup_dir,'/launch','/bringup_launch.py']),
        #     launch_arguments={
        #         'map': map_yaml_path,
        #         'use_sim_time': use_sim_time,
        #         'params_file': nav2_param_path}.items(),
        # ),
        

       
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([os.path.join(
        #         get_package_share_directory('hesai_lidar'), 'launch'),
        #                                    '/hesai_lidar_launch.py'])
        # ),

        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([os.path.join(
        #         get_package_share_directory('multimap_server'), 'launch'),
        #                                     '/load_multimap.launch.py'])
        # ),                      
        #Node(
        #    package='rslidar_sdk',
        #    executable='rslidar_sdk_node',
        #    name='rslidar_sdk_node',
        #    output='screen'
        #),

        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([os.path.join(
        #         get_package_share_directory('lio_sam_ros2'), 'launch'),
        #         '/relio.launch.py'])
        # ),

        TimerAction(period=2.0,actions= [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('lio_sam_ros2'), 'launch'),
                    '/relio_qt_mid360.launch.py'])
            )
        ]),
        

        TimerAction(period=2.0,actions= [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('nav2_costmap'), 'launch'),
                    '/nav2_costmap_inflation_layer.launch.py'])
            )
        ]),

        Node(
            package="lio_sam_ros2",
            executable="lio_sam_ros2_IniPoseFromText",
            output='screen'
        ),
    ])
