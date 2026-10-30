import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf_transformations                                 
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from std_msgs.msg import String,Float64,Bool
from geometry_msgs.msg import PoseStamped,PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from custom_interface.msg import RechargeInfo,VirtualWall,NavigationToPose,NavState

import os
import json
import psutil
import signal
import subprocess
import time

mapping_node = ["dog_control_B1_one",
                
                
                "hesai_lidar_node",
                
                "lio_sam_ros2_dogOdomForMapping",
                "lio_sam_ros2_imuPreintegration",
                "lio_sam_ros2_imageProjection",
                "lio_sam_ros2_featureExtraction",
                "lio_sam_ros2_mapOptmization",
                "static_transform_publisher",
                
                "CloudToScan",
                "mapgrid"]

navigation_node = ["dog_control_B1_one",
                   
                   "hesai_lidar_node",
                   
                   "lio_sam_ros2_dogOdomForReloc",
                   "lio_sam_ros2_imuPreintegration",
                   "lio_sam_ros2_imageProjection",
                   "lio_sam_ros2_featureExtraction",
                   "lio_sam_ros2_globalLocalize",
                   "static_transform_publisher",
                   "lio_sam_ros2_scinitial",
                   "lio_sam_ros2_listenTF",
                   "lio_sam_ros2_logRecord",
                   
                   #"template_match",
                   
                   "virtual_wall",
                   "map_server",
                   "lifecycle_manager",
                   "controller_server",
                   "planner_server",
                   "recoveries_server",
                   "bt_navigator",
                   "waypoint_follower",
                  
                   "navigation_to_p"
                  ]


class CommunicationInterface(Node):

    def __init__(self):
        super().__init__('communication_interface')
        
        self.robotState_publisher_ = self.create_publisher(String,'robot_state',10)
        self.robotLocal_publisher_ = self.create_publisher(String,'robot_local',10)
        self.autoInitPosition_publisher_ = self.create_publisher(String,'InitializedType',10)
        self.initPosition_publisher_ = self.create_publisher(PoseWithCovarianceStamped,'initialpose',10)
        self.virtualWall_publisher_ = self.create_publisher(VirtualWall,'VirtualWall',10)
        self.reCharge_publisher_ = self.create_publisher(RechargeInfo,'recharge_info',10)
        self.savePcd_publisher_ = self.create_publisher(Bool,'lio_sam_ros2/savepcd/savepcd_flag',10)
        self.reChargeSaveMap_publisher_ = self.create_publisher(String,'save_flag',10)
        self.navToPose_publisher_ = self.create_publisher(NavigationToPose,'nav_to_pose',10)
        self.paramFlag_publisher_ = self.create_publisher(String,'key_point_set',10)

        self.robotCommand_subscription_ = self.create_subscription(String,'robot_command',self.robotCommand_callback,10)
        self.odometry_subscription_ = self.create_subscription(Odometry,'dogcentertomap',self.odometry_callback,10)
        self.credibility_subcription_ = self.create_subscription(Float64,'temp_score',self.credibility_callback,10)
        self.initLocalitionState_subcription_ = self.create_subscription(Bool,'lio_sam_ros2/mapping/InitialFlag',self.initLocalitionState_callback,10)
        self.reChargeState_subcription_ = self.create_subscription(String,'pubendflag',self.reChargeState_callback,10)
        self.navState_subcription_ = self.create_subscription(NavState,"nav_state",self.navState_callback,10)
        self.localizationState_subcription_ = self.create_subscription(Bool,"lio_sam_ros2/mapping/TFStateFlag",self.localizationState_callback,10)

        self.exec_state_ = self.create_timer(1.0,self.execState_callback)

        self.credibility_value = 0.0
        self.command_states = "true"
        self.init_state = "true"
        self.localization_states = "true"
        self.nav_states = "true"
        self.recharge_state = "true" 
        
    def robotCommand_callback(self, msg):        
        data = json.loads(msg.data)

        if data['command'] == 'StartMapping':
            self.get_logger().info('Start Mapping.')
            self.start_mapping()
        
        elif data['command'] == 'CancelMapping':
            self.get_logger().info('Cancel Mapping.')
            self.cancel_mapping()
        
        elif data['command'] == 'CloseMapping':
            self.get_logger().info('Close Mapping.')
            self.close_mapping()
        
        elif data['command'] == 'StartNavigation':
            self.get_logger().info('Start Navigation.')
            self.start_navigation()
        
        elif data['command'] == 'CloseNavigation':
            self.get_logger().info('Close Navigation.')
            self.close_navigaton()

        elif data['command'] == 'AutoInitPose':
            self.get_logger().info('Auto Init Position')
            self.auto_init_position()

        elif data['command'] == 'InitPose':
            self.get_logger().info('Init Position.')
            self.init_position(data['pose_x'],data['pose_y'],data['pose_z'],data['quat_x'],data['quat_y'],data['quat_z'],data['quat_w'])
            
        elif data['command'] == 'NavPose':
            self.get_logger().info('Navigation Position.')
            self.nav_position(data['uuid'],data['pose_x'],data['pose_y'],data['pose_z'],data['quat_x'],data['quat_y'],data['quat_z'],data['quat_w'])

        elif data['command'] == 'cancelNav':
            self.get_logger().info('Cancel Navigation.')

        elif data['command'] == 'reChargeMap':
            self.get_logger().info('ReCharge Map.')
            self.re_charge_map(data['map_name'])
            
        elif data['command'] == 'reChargeCancel':
            self.get_logger().info('ReCharge Cancel.')
            self.re_charge_cancel()

        elif data['command'] == 'reChargeSaveMap':
            self.get_logger().info('ReCharge Save Map.')
            self.re_charge_save_map()

        elif data['command'] == 'reCharge':
            self.get_logger().info('recharge.')
            command = 'start'
            map_name = data['map_name']
            self.re_charge(command,map_name,data['pose_x'],data['pose_y'],data['pose_z'],data['quat_x'],data['quat_y'],data['quat_z'],data['quat_w'])

        elif data['command'] == 'addWall':
            self.get_logger().info('addWall.')
            self.add_wall(data['frame_start_x'],data['frame_start_y'],data['frame_end_x'],data['frame_end_y'],data['threshold'])
        elif data['command'] == 'delWall':
            self.get_logger().info('delWall.')
            self.del_wall(data['frame_start_x'],data['frame_start_y'],data['frame_end_x'],data['frame_end_y'],data['threshold'])

        elif data['command'] == 'reStart':
            self.get_logger().info('reStart.')
            self.re_start()

        command_state_data = {}

        command_state_data['uuid'] = data['uuid']
        command_state_data['state'] = self.command_states

        command_state_msg = String()
        command_state_msg.data = str(json.dumps(command_state_data))
        self.robotState_publisher_.publish(command_state_msg)

        self.command_states = "true"

    def odometry_callback(self,msg):
        robot_pose_data = {}

        robot_pose_data['uuid'] = "robot_pose"
        robot_pose_data['state'] = self.credibility_value
        robot_pose_data['pose_x'] = msg.pose.pose.position.x
        robot_pose_data['pose_y'] = msg.pose.pose.position.y
        robot_pose_data['pose_z'] = msg.pose.pose.position.z
        robot_pose_data['quat_x'] = msg.pose.pose.orientation.x 
        robot_pose_data['quat_y'] = msg.pose.pose.orientation.y 
        robot_pose_data['quat_z'] = msg.pose.pose.orientation.z 
        robot_pose_data['quat_w'] = msg.pose.pose.orientation.w 

        robot_pose_msg = String()
        robot_pose_msg.data = str(json.dumps(robot_pose_data))        
        self.robotState_publisher_.publish(robot_pose_msg)

    def credibility_callback(self,msg):
        self.credibility_value = str(msg.data)
        
    def localizationState_callback(self,msg):
        if msg.data == True:
            self.localization_states = "true"
        else:
            self.localization_states = "false"

        command_state_data = {}

        command_state_data['uuid'] = "robot_local"
        command_state_data['state'] = self.localization_states

        command_state_msg = String()
        command_state_msg.data = str(json.dumps(command_state_data))
        self.robotState_publisher_.publish(command_state_msg)

    def execState_callback(self):
        command_state_data = {}

        command_state_data['uuid'] = "robot_exec"
        command_state_data['mapping_state'] = "false"
        command_state_data['nav_state'] = "false"
        
        if self.check_nodes(mapping_node) == True:
            command_state_data['mapping_state'] = "true"
        else:
            command_state_data['mapping_state'] = "false"
            
        if self.check_nodes(navigation_node) == True:
            command_state_data['nav_state'] = "true"
        else:
            command_state_data['nav_state'] = "false"
        
        command_state_msg = String()
        command_state_msg.data = str(json.dumps(command_state_data))
        self.robotState_publisher_.publish(command_state_msg)
        
    def initLocalitionState_callback(self,msg):
        command_state_data = {}

        command_state_data['uuid'] = "robot_initState"
        command_state_data['state'] = "init_true"
        
        if msg.data == True:
            command_state_data['state'] = "init_true"
        else:
            command_state_data['state'] = "init_false"

        command_state_msg = String()
        command_state_msg.data = str(json.dumps(command_state_data))
        self.robotState_publisher_.publish(command_state_msg)

        self.init_state = "true"
        
    def reChargeState_callback(self,msg):
        command_state_data = {}

        command_state_data['uuid'] = "robot_reChargeState"
        command_state_data['state'] = "recharge_true"
        
        if msg.data == "success":
            command_state_data['state'] = "recharge_true"
        else:
            command_state_data['state'] = "recharge_false"
        
        time.sleep(2)
        os.system('pkill -f template_match')
        
        command_state_msg = String()
        command_state_msg.data = str(json.dumps(command_state_data))
        self.robotState_publisher_.publish(command_state_msg)

        self.recharge_state = "true"

    def node_kill(self,name):
        for proc in psutil.process_iter(['pid', 'name']):
            #print(proc.info)        
            if proc.info['name'] == name:
                pid = (proc.info['pid'])
                os.kill(pid, signal.SIGKILL) 

    def check_nodes(self,node_lists):
        all_node_name = []
        for proc in psutil.process_iter(['pid', 'name']):
            all_node_name.append(proc.info['name'])
            
        #self.get_logger().info(f'Could not transform {all_node_name}')
        return all(item in all_node_name for item in node_lists)
        
    def kill_nodes(self,node_lists):
        for node in node_lists:
            os.system('pkill -f "{}"'.format(node))
    
        return not(self.check_nodes(node_lists))

    def start_mapping(self):
        self.get_logger().info("Start Mapping.")
        subprocess.Popen(["ros2","launch","task","mapping.launch.py"])
        time.sleep(5)
        if self.check_nodes(mapping_node) == True:
            self.command_states = "true"
        else:
            self.command_states = "false"

    def cancel_mapping(self):
        self.get_logger().info('Cancel Mapping.')
        if self.kill_nodes(mapping_node) == True:
            self.command_states = "true"
        else:
            self.command_states = "false"

    def close_mapping(self):
        self.get_logger().info('Close Mapping.')
        subprocess.Popen(["ros2","run","map_server_py","map_saver_py"])
        
        savePcd_msg = Bool()
        savePcd_msg.data = True
        self.savePcd_publisher_.publish(savePcd_msg)
        
        time.sleep(5)
        if self.kill_nodes(mapping_node) == True:
            self.command_states = "true"
        else:
            self.command_states = "false"

    def start_navigation(self):
        self.get_logger().info('Start Navigation.')
        subprocess.Popen(["ros2","launch","task","navigation.launch.py"])
        subprocess.Popen(["ros2","run","task","navigation_to_pose"])
        subprocess.Popen(["ros2","run","virtual_wall","virtual_wall"])
        time.sleep(5)
        if self.check_nodes(navigation_node) == True:
            self.command_states = "true"
        else:
            self.command_states = "false"

    def close_navigaton(self):
        self.get_logger().info('Close Navigation.')
        if self.kill_nodes(navigation_node) == True:
            self.command_states = "true"
        else:
            self.command_states = "false"

    def auto_init_position(self):
        autoInitPose_msg = String()
        autoInitPose_msg.data = 'ScanContext'
        self.autoInitPosition_publisher_.publish(autoInitPose_msg)

    def init_position(self,pose_x,pose_y,pose_z,quat_x,quat_y,quat_z,quat_w):
        autoInitPose_msg = String()
        autoInitPose_msg.data = 'Manual'
        self.autoInitPosition_publisher_.publish(autoInitPose_msg)
        
        initialpose_msg = PoseWithCovarianceStamped()

        initialpose_msg.header.frame_id = 'map'
        initialpose_msg.header.stamp = self.get_clock().now().to_msg()

        initialpose_msg.pose.pose.position.x = float(pose_x)
        initialpose_msg.pose.pose.position.y = float(pose_y)
        initialpose_msg.pose.pose.position.z = float(pose_z)
        initialpose_msg.pose.pose.orientation.x = float(quat_x)
        initialpose_msg.pose.pose.orientation.y = float(quat_y)
        initialpose_msg.pose.pose.orientation.z = float(quat_z)
        initialpose_msg.pose.pose.orientation.w = float(quat_w)

        initialpose_msg.pose.covariance = [0.0] * 36

        self.initPosition_publisher_.publish(initialpose_msg)
        
    def nav_position(self,uuid,pose_x,pose_y,pose_z,quat_x,quat_y,quat_z,quat_w):
        navigationToPose_data = NavigationToPose()

        navigationToPose_data.id = uuid
        navigationToPose_data.pose_x = pose_x
        navigationToPose_data.pose_y = pose_y
        navigationToPose_data.pose_z = pose_z
        navigationToPose_data.quat_x = quat_x
        navigationToPose_data.quat_y = quat_y
        navigationToPose_data.quat_z = quat_z
        navigationToPose_data.quat_w = quat_w

        self.navToPose_publisher_.publish(navigationToPose_data)

    def navState_callback(self,msg):
        param_data = String()
        param_data.data = 'out'
        self.paramFlag_publisher_.publish(param_data)
        
        nav_state_data = {}

        nav_state_data['uuid'] = msg.id
        nav_state_data['state'] = msg.state

        nav_state_msg = String()
        nav_state_msg.data = str(json.dumps(nav_state_data))
        self.robotState_publisher_.publish(nav_state_msg)

        self.nav_states = "true"

    def re_charge_map(self,map_name):
        subprocess.Popen(["ros2","launch","template_matching","save_pcd.launch.py","map_name:="+map_name])
        
    def re_charge_cancel(self):
        os.system('pkill -f save_pcd')
    
    def re_charge_save_map(self):
        reChargeSaveMap_data = String()
        reChargeSaveMap_data.data = "save"
        self.reChargeSaveMap_publisher_.publish(reChargeSaveMap_data)
        
        time.sleep(2)
        os.system('pkill -f save_pcd')
        
    def re_charge(self,command,map_name,pose_x,pose_y,pose_z,quat_x,quat_y,quat_z,quat_w):
        param_data = String()
        param_data.data = 'in'
        self.paramFlag_publisher_.publish(param_data)
        
        subprocess.Popen(["ros2","launch","template_matching","template_matching.launch.py","map_name:="+map_name])
        time.sleep(1)
        
        recharge_info_msg = RechargeInfo()
        
        recharge_info_msg.start_flag = command
        recharge_info_msg.target_pose.position.x = float(pose_x)
        recharge_info_msg.target_pose.position.y = float(pose_y)
        recharge_info_msg.target_pose.position.z = float(pose_z)
        recharge_info_msg.target_pose.orientation.x = float(quat_x)
        recharge_info_msg.target_pose.orientation.y = float(quat_y)
        recharge_info_msg.target_pose.orientation.z = float(quat_z)
        recharge_info_msg.target_pose.orientation.w = float(quat_w)
        
        self.reCharge_publisher_.publish(recharge_info_msg)
        
    def add_wall(self,frame_start_x,frame_start_y,frame_end_x,frame_end_y,threshold):
        virtual_wall_msg = VirtualWall()

        virtual_wall_msg.wall_type = 100
        virtual_wall_msg.frame_start_x = float(frame_start_x)
        virtual_wall_msg.frame_start_y = float(frame_start_y)
        virtual_wall_msg.frame_end_x = float(frame_end_x)
        virtual_wall_msg.frame_end_y = float(frame_end_y)
        virtual_wall_msg.threshold = int(threshold)

        self.virtualWall_publisher_.publish(virtual_wall_msg)

    def del_wall(self,frame_start_x,frame_start_y,frame_end_x,frame_end_y,threshold):
        virtual_wall_msg = VirtualWall()

        virtual_wall_msg.wall_type = 0
        virtual_wall_msg.frame_start_x = frame_start_x
        virtual_wall_msg.frame_start_y = frame_start_y
        virtual_wall_msg.frame_end_x = frame_end_x
        virtual_wall_msg.frame_end_y = frame_end_y
        virtual_wall_msg.threshold = threshold

        self.virtualWall_publisher_.publish(virtual_wall_msg)

    def re_start(self):
        os.system('reboot')

def main(args=None):
    rclpy.init(args=args)

    communication_interface = CommunicationInterface()

    rclpy.spin(communication_interface)

    communication_interface.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
