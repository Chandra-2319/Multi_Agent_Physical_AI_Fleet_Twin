#开启多楼层拓扑图
import os
import sys
import launch
import launch_ros.actions
from launch.actions import IncludeLaunchDescription,TimerAction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

graph_process_path = os.path.join(os.path.abspath('.'), 'config_files', 'graph_process_config')

#####
#开始加载导航
###
def generate_launch_description():
 

    # mapping_param_dir = launch.substitutions.LaunchConfiguration(
    #     'mapping_param_dir',
    #     default=os.path.join(
    #         get_package_share_directory('lio_sam_ros2'),
    #         'config',
    #         'params.yaml'))

    # rviz_config = os.path.join(os.path.abspath('.'), 'src', 'task', 'rviz2', 'bulid_map.rviz')

    # use_sim_time = launch.substitutions.LaunchConfiguration('use_sim_time', default= 'False' )

    graph_config_dir = launch.substitutions.LaunchConfiguration(
        'graph_config_dir',
        default=os.path.join(
            graph_process_path,  
            # get_package_share_directory('pid_tracing'),    # 获取 功能包 名称
            'graph_params.yaml')) 


    map_management = launch_ros.actions.Node(
        package='graph_process',
        executable='map_management_multifloor',
        #parameters=[{mapping_param_dir},{'use_sim_time':use_sim_time}],
        # prefix = [' -e gdb -ex run --args'],
        parameters=[graph_config_dir],
        output='screen'
        )

    main_process = launch_ros.actions.Node(
        package='graph_process',
        executable='main_process_multifloor',
        # prefix = ['deepin-terminal -e gdb -ex run --args'],
        parameters=[graph_config_dir],
        output='screen'
        )

    path_management = launch_ros.actions.Node(
        package='graph_process',
        executable='path_management_multifloor',
        #parameters=[{mapping_param_dir},{'use_sim_time': use_sim_time}],
        # prefix = [' -e gdb -ex run --args'],
        parameters=[graph_config_dir],
        output='screen'
        )

    graph_visual = launch_ros.actions.Node(
        package='graph_process',
        executable='graph_visual_multifloor',
        #parameters=[{mapping_param_dir},{'use_sim_time': use_sim_time}],
        output='screen'
        )
    graph_pub = launch_ros.actions.Node(
        package='graph_process',
        executable='pub_graph',
        #parameters=[{mapping_param_dir},{'use_sim_time':use_sim_time}],
        output='screen'
        )
    # TimerAction(period=2.0,actions= [
    #         # IncludeLaunchDescription(
    #         #     PythonLaunchDescriptionSource([os.path.join(
    #         #         get_package_share_directory('nav2_costmap'), 'launch'),
    #         #         '/nav2_costmap_inflation_layer.launch.py'])
    #         # )
    #     ]),
        

    return launch.LaunchDescription([
        # launch.actions.DeclareLaunchArgument(
        #     'mapping_param_dir',
        #     default_value=mapping_param_dir,
        #     description='Full path to mapping parameter file to load'),
        path_management,
        main_process,
        map_management,
        graph_visual,
        # graph_pub,
        # featurex,   
        # map_opt,
        # tf_static,
        # occ_grid_map,
        #rviz_node,
            ])
