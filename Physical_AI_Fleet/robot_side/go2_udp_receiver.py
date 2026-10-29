#!/usr/bin/env python3
"""OPTIONAL since v0.5.7 - the extension reads the Go2 packets itself; only run this if another ROS 2 program on the PC
needs /go2/joint_states (then point the extension to udp:9871). Needs ROS 2 on the PC.

PC side: Go2 state -> ROS 2 topics for your OmniGraph, AND a copy for the Physical_AI_Fleet extension. v0.5.6

Only change from your original: every packet received on UDP 9870 is also forwarded unchanged to FORWARD
(127.0.0.1:9871), where the extension's Go2 twin listens (only one program can read port 9870).
Disable the OmniGraph node that moves the robot BODY from /go2/odom: the extension places the body in the world
(Map -> World); your graph keeps driving the JOINTS from /go2/joint_states.
"""
import json
import socket

PORT = 9870
FORWARD = [('127.0.0.1', 9871)]
ALPHA = 0.4
ISAAC_JOINT_NAMES = [
    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
    'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint',
]
# Unitree motor order: 0:FR 1:FR 2:FR / 3:FL 4:FL 5:FL / 6:RR 7:RR 8:RR / 9:RL 10:RL 11:RL
REMAP = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]


def forward(sock, data, targets=None):
    for addr in (targets or FORWARD):
        try:
            sock.sendto(data, addr)
        except OSError:
            pass


def main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState, Imu
    from nav_msgs.msg import Odometry

    class UDPReceiver(Node):
        def __init__(self):
            super().__init__('go2_udp_receiver')
            self.js_pub = self.create_publisher(JointState, '/go2/joint_states', 10)
            self.imu_pub = self.create_publisher(Imu, '/go2/imu', 10)
            self.odom_pub = self.create_publisher(Odometry, '/go2/odom', 10)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('0.0.0.0', PORT))
            self.sock.setblocking(False)
            self.fwd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.smooth_q = None
            self.smooth_dq = None
            self.create_timer(0.002, self.poll)
            self.get_logger().info('go2_udp_receiver v0.5.6 - port %d, forwarding to %s' % (PORT, FORWARD))

        def poll(self):
            data = None
            while True:
                try:
                    data, _ = self.sock.recvfrom(65535)
                except BlockingIOError:
                    break
                forward(self.fwd, data)
            if data is None:
                if self.smooth_q is not None:
                    self._publish_joints(self.smooth_q, self.smooth_dq)
                return
            try:
                d = json.loads(data.decode())
            except Exception:
                return
            now = self.get_clock().now().to_msg()
            raw_q = [d['q'][i] for i in REMAP]
            raw_dq = [d['dq'][i] for i in REMAP]
            if self.smooth_q is None:
                self.smooth_q, self.smooth_dq = raw_q, raw_dq
            else:
                self.smooth_q = [ALPHA * r + (1 - ALPHA) * s for r, s in zip(raw_q, self.smooth_q)]
                self.smooth_dq = [ALPHA * r + (1 - ALPHA) * s for r, s in zip(raw_dq, self.smooth_dq)]
            self._publish_joints(self.smooth_q, self.smooth_dq)
            imu = Imu()
            imu.header.stamp, imu.header.frame_id = now, 'imu_link'
            q = d['quat']
            imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z = (float(v) for v in q)
            g = d['gyro']
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = (float(v) for v in g)
            a = d['acc']
            imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = (float(v) for v in a)
            self.imu_pub.publish(imu)
            if d.get('odom'):
                o = d['odom']
                m = Odometry()
                m.header.stamp, m.header.frame_id, m.child_frame_id = now, 'odom', 'base_link'
                m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = float(o['px']), float(o['py']), float(o['pz'])
                m.pose.pose.orientation.x, m.pose.pose.orientation.y = float(o['qx']), float(o['qy'])
                m.pose.pose.orientation.z, m.pose.pose.orientation.w = float(o['qz']), float(o['qw'])
                self.odom_pub.publish(m)

        def _publish_joints(self, q, dq):
            js = JointState()
            js.header.stamp, js.header.frame_id = self.get_clock().now().to_msg(), 'base'
            js.name, js.position, js.velocity, js.effort = ISAAC_JOINT_NAMES, q, dq, [0.0] * 12
            self.js_pub.publish(js)

    rclpy.init()
    rclpy.spin(UDPReceiver())


if __name__ == '__main__':
    main()
