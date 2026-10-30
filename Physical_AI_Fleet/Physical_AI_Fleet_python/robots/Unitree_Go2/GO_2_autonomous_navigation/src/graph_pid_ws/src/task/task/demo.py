import os
import json
import psutil
import signal
import subprocess

navigation_node = ["dog_control_B1",
                   
                   "hesai_lidar_node",
                   
                   "lio_sam_ros2_dogOdomForReloc",
                   "lio_sam_ros2_imuPreintegration",
                   "lio_sam_ros2_imageProjection",
                   "lio_sam_ros2_featureExtraction",
                   "lio_sam_ros2_globalLocalize",
                   "static_transform_publisher",
                   "lio_sam_ros2_scinitial",
                   "lio_sam_ros2_listenTF",
                   
                   # "template_match",
                   
                   "map_server",
                   "lifecycle_manager",
                   "controller_server",
                   "planner_server",
                   "recoveries_server",
                   "bt_navigator",
                   "waypoint_follower"]

def check_nodes(node_lists):
    all_node_name = []
    for proc in psutil.process_iter(['pid', 'name']):
        all_node_name.append(proc.info['name'])
            
    return all(item in all_node_name for item in node_lists)
        
def kill_nodes(node_lists):
    for node in node_lists:
        os.system('pkill -f "{}"'.format(node))
    
    return not(check_nodes(node_lists))

kill_nodes(navigation_node)
    
    