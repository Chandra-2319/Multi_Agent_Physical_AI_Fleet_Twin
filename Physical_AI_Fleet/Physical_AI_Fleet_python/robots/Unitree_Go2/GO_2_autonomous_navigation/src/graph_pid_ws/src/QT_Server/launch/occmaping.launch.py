import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    occmapping_params = launch.substitutions.LaunchConfiguration(
        'occmapping_params',
        default=os.path.join(
            get_package_share_directory('multimap_server'),
            'config',
            'build_map_params.yaml'))


    creat_occgrid_map_node = launch_ros.actions.Node(
        package='multimap_server',
        executable='creat_occgrid_map',
        parameters=[{occmapping_params}],
        #prefix=['xterm -e gdb -ex run --args'], #xterm -e
        output='screen'
    )


    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            'merger_map_param_dir',
            default_value=occmapping_params,
            description='Full path to mapping parameter file to load'),
        creat_occgrid_map_node,
    ])
