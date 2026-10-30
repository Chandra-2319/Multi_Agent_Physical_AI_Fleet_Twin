import os
import sys
import launch
import launch_ros.actions
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,TimerAction
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def my_get_path():
    # # 参数是 map_name:=test
    # my_map_name = sys.argv[4]
    # mapname = my_map_name[10:]
    # return os.path.join(os.path.abspath('.'),'src','task','maps','pcd',mapname)
    argc = len(sys.argv)

    if argc < 5:
        mapname = 'default'+ '/'

    else:
        # 参数是 map_name:=test
        my_map_name = sys.argv[4]
        mapname = my_map_name[10:] + '/'

    return os.path.join(os.path.abspath('.'),'config_files','lio_sam_config','maps','pcd',mapname)
    

def generate_launch_description():

    map_name_dir = LaunchConfiguration(
        'map_name_dir',
        default=os.path.join(
            my_get_path()
            )
    )

    rviz_config = os.path.join(os.path.abspath('.'), 'src', 'task', 'rviz2', 're_localization.rviz')

    use_sim_time = launch.substitutions.LaunchConfiguration('use_sim_time', default= 'False' )

    mapping_param_dir = launch.substitutions.LaunchConfiguration(
        'mapping_param_dir',
        default=os.path.join(
            'config_files',
            'lio_sam_config',
            'params_xt16.yaml'))

    dogodom = launch_ros.actions.Node(
        package='lio_sam_ros2',
        executable='lio_sam_ros2_dogOdomForReloc',
        parameters=[{mapping_param_dir},{'use_sim_time':use_sim_time}],
        output='screen'
        )

    imu_pre = launch_ros.actions.Node(
        package='lio_sam_ros2',
        executable='lio_sam_ros2_imuPreintegration',
        parameters=[{mapping_param_dir},{'use_sim_time':use_sim_time}],
        output='screen'
        )

    imagep = launch_ros.actions.Node(
        package='lio_sam_ros2',
        executable='lio_sam_ros2_imageProjection',
        parameters=[{mapping_param_dir},{'use_sim_time':use_sim_time}],
        output='screen'
        )

    featurex = launch_ros.actions.Node(
        package='lio_sam_ros2',
        executable='lio_sam_ros2_featureExtraction',
        parameters=[{mapping_param_dir},{'use_sim_time':use_sim_time}],
        output='screen'
        )

    global_loc = launch_ros.actions.Node(
        package='lio_sam_ros2',
        executable='lio_sam_ros2_globalLocalize',      
        parameters=[{mapping_param_dir},{'savePCDDirectory': map_name_dir},{'use_sim_time': use_sim_time}],
        output='screen'
        )

    listen_tf = launch_ros.actions.Node(
        package='lio_sam_ros2',
        executable='lio_sam_ros2_listenTF',
        parameters=[{mapping_param_dir},{'use_sim_time': use_sim_time}],
        output='screen'
        )

    logrecord= launch_ros.actions.Node(
        package='lio_sam_ros2',
        executable='lio_sam_ros2_logRecord',
        parameters=[{mapping_param_dir},{'use_sim_time': use_sim_time}],
        output='screen'
        )

    # tf_static = launch_ros.actions.Node(#lz:该静态变换改在程序中
    #         package='tf2_ros',
    #         executable='static_transform_publisher',
    #         parameters=[{'use_sim_time': use_sim_time}],
    #         arguments = ['0', '0', '0', '0', '0', '0', 'base_link', 'rslidar']
    #     )
        
    scinitial = launch_ros.actions.Node(
            package='lio_sam_ros2',
            executable='lio_sam_ros2_scinitial',
            parameters=[{mapping_param_dir},{'use_sim_time': use_sim_time}],
            output='screen'
        )
    # rviz_node = launch_ros.actions.Node(
    #         package='rviz2',
    #         executable='rviz2',
    #         name='rviz2',
    #         arguments=['-d', rviz_config],
    #     )

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            'mapping_param_dir',
            default_value=mapping_param_dir,
            description='Full path to mapping parameter file to load'),
        dogodom,
        imu_pre,
        imagep,
        # featurex,   
        global_loc,
        # listen_tf,
        # logrecord,
        #tf_static,
        scinitial,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                get_package_share_directory('nav2_costmap'), 'launch'),
                '/nav2_costmap_inflation_layer.launch.py'])
        ),
        
        Node(
            package="lio_sam_ros2",
            executable="lio_sam_ros2_IniPoseFromText",
            output='screen'
        ),
        #rviz_node,
            ])
