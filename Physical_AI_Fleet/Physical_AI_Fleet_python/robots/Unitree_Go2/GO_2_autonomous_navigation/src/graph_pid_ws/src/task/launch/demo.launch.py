import launch_ros.actions
from launch import LaunchDescription

def generate_launch_description():
    
    communication_interface = launch_ros.actions.Node(
        package='task',
        executable='communication_interface',
        name='communication_interface',
        output='screen'
        )

    return LaunchDescription([
        communication_interface,
    ])