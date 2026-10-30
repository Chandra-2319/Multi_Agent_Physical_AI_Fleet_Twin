# 导入库
import os
import sys

import launch
import launch_ros.actions
from launch.actions import IncludeLaunchDescription,TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

local_planner_path = os.path.join(os.path.abspath('.'), 'config_files', 'local_planner_config')
empty_map_path = os.path.join(os.path.abspath('.'), 'config_files', 'empty_map_config')
merger_map_path = os.path.join(os.path.abspath('.'), 'config_files', 'merger_map_config')
# 定义函数名称为：generate_launch_description
def generate_launch_description():

# params
    merger_map_param_dir = launch.substitutions.LaunchConfiguration(
        'merger_map_param_config_dir',
        default=os.path.join(
            merger_map_path,  
            'params.yaml'))   

    empty_params = launch.substitutions.LaunchConfiguration(
        'empty_map_config_dir',
        default=os.path.join(
            empty_map_path,  
            'empty_map_params.yaml')) 

    local_planner_config_dir = launch.substitutions.LaunchConfiguration(
        'local_planner_config_dir',
        default=os.path.join(
            local_planner_path,  
            'params.yaml'))    

# node

    path_planner_node = launch_ros.actions.Node(
        package='path_planner',
        executable='path_planner_node',
        # output='screen'
        )

    fusion_odom = launch_ros.actions.Node(
        package='unitree_kalman',
        executable='fusion_odom',
        output='screen'
        )    

    merger_map_node = launch_ros.actions.Node(
        package='merger_map',
        executable='merger_map_node',
        parameters=[{merger_map_param_dir}],
        output='screen'
        )

    empty_params_node = launch_ros.actions.Node(
        package='empty_map',
        executable='empty_map_node',
        parameters=[{empty_params}],
        output='screen'
    )

    # 创建Actions.Node对象，标明node所在位置
    local_planner = launch_ros.actions.Node(
        package="local_planner",     # 功能包
        executable="local_planner_control",  # 程序名
        parameters=[local_planner_config_dir],   # 读取参数
        output='screen'
        )

    control_log = launch_ros.actions.Node(
        package="local_planner",     # 功能包
        executable="log_demo",  # 程序名
        parameters=[local_planner_config_dir],   # 读取参数
        output='screen'
        )

    to_charge = launch_ros.actions.Node(
        package="local_planner",     # 功能包
        executable="to_charge",  # 程序名
        parameters=[local_planner_config_dir],   # 读取参数
        output='screen'
        )

    remote_add_point = launch_ros.actions.Node(
        package="local_planner",     # 功能包
        executable="remote_add_point",  # 程序名
        output='screen'
        )

    control_web = launch_ros.actions.Node(
        package="local_planner",     # 功能包
        executable="control_web",  # 程序名
        output='screen'
        )
# 启动
    return launch.LaunchDescription([

            path_planner_node,
            #fusion_odom,
            merger_map_node,
            empty_params_node,
            local_planner,
            # control_log,
            # to_charge,
            # remote_add_point,
            # control_web,
            ])
