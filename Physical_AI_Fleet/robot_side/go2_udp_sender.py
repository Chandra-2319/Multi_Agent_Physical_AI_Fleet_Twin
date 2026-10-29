#!/usr/bin/env python3
"""Go2 -> PC state stream. Runs ON THE GO2 (ROS 2 Foxy, Python 3.8). v0.5.6 (Physical_AI_Fleet)

What changed from your original:
  * also sends "odom": the robot's own position estimate, so the twin in Isaac Sim moves with the real robot.
    Sources, first fresh one wins (and is kept while it stays fresh, so the twin does not jump between origins):
        /uslam/localization/odom   SLAM localisation in your saved map (when the map navigation is running)
        /utlidar/robot_odom        lidar odometry (nav_msgs/Odometry, always on)
        /sportmodestate            sport-mode estimate (unitree_go/SportModeState)
    The topic type is read from ROS at run time; a topic is only used if it is Odometry or SportModeState.
  * sends at SEND_HZ (100) instead of every /lowstate message (~500 Hz).

Run (the unitree_go messages are in this workspace - without it 'ros2 topic echo' fails with No module named 'unitree_go'):
    source /unitree/module/graph_pid_ws/install/setup.bash
    python3 ~/go2_udp_sender.py
"""
import json
import socket
import time

PC_IP = '192.168.2.202'   # the PC running go2_udp_receiver.py and Isaac Sim
PORT = 9870
SEND_HZ = 100.0
ODOM_TOPICS = ['/uslam/localization/odom', '/utlidar/robot_odom', '/sportmodestate']
ODOM_FRESH_S = 0.5
ODOMETRY_TYPE = 'nav_msgs/msg/Odometry'
SPORT_TYPE = 'unitree_go/msg/SportModeState'


def odom_from_odometry(msg):
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    return {'px': float(p.x), 'py': float(p.y), 'pz': float(p.z),
            'qw': float(q.w), 'qx': float(q.x), 'qy': float(q.y), 'qz': float(q.z)}


def odom_from_sport(msg):
    q = msg.imu_state.quaternion  # Unitree order: w, x, y, z
    return {'px': float(msg.position[0]), 'py': float(msg.position[1]), 'pz': float(msg.position[2]),
            'qw': float(q[0]), 'qx': float(q[1]), 'qy': float(q[2]), 'qz': float(q[3])}


def pick_odom(latest, now, current=None, order=None, fresh=ODOM_FRESH_S):
    """latest: {topic: (t, odom_dict)}. Keep `current` while it is fresh, else the first fresh topic in `order`.
    Returns (topic, odom_dict_with_src) or (None, None)."""
    order = order or ODOM_TOPICS
    if current in latest and now - latest[current][0] <= fresh:
        topic = current
    else:
        topic = next((t for t in order if t in latest and now - latest[t][0] <= fresh), None)
    if topic is None:
        return None, None
    odom = dict(latest[topic][1])
    odom['src'] = topic
    return topic, odom


def build_packet(motor_state, imu_state, odom):
    data = {
        'q': [float(motor_state[i].q) for i in range(12)],
        'dq': [float(motor_state[i].dq) for i in range(12)],
        'tau': [float(motor_state[i].tau_est) for i in range(12)],
        'quat': [float(x) for x in imu_state.quaternion],
        'gyro': [float(x) for x in imu_state.gyroscope],
        'acc': [float(x) for x in imu_state.accelerometer],
    }
    if odom is not None:
        data['odom'] = odom
    return json.dumps(data).encode()


def main():
    import rclpy
    from rclpy.node import Node
    from unitree_go.msg import LowState
    try:
        from unitree_go.msg import SportModeState
    except ImportError:
        SportModeState = None
    from nav_msgs.msg import Odometry

    class UDPSender(Node):
        def __init__(self):
            super().__init__('go2_udp_sender')
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.latest, self.current, self.subs = {}, None, {}
            self.last_send = 0.0
            self.create_subscription(LowState, '/lowstate', self.cb, 10)
            self.create_timer(2.0, self.discover)
            self.discover()
            self.get_logger().info('go2_udp_sender v0.5.6 -> %s:%d at %.0f Hz' % (PC_IP, PORT, SEND_HZ))

        def discover(self):
            types = dict((n, t) for n, t in self.get_topic_names_and_types())
            for topic in ODOM_TOPICS:
                if topic in self.subs or topic not in types:
                    continue
                if ODOMETRY_TYPE in types[topic]:
                    self.subs[topic] = self.create_subscription(Odometry, topic, self.make_cb(topic, odom_from_odometry), 10)
                elif SPORT_TYPE in types[topic] and SportModeState is not None:
                    self.subs[topic] = self.create_subscription(SportModeState, topic, self.make_cb(topic, odom_from_sport), 10)
                else:
                    continue
                self.get_logger().info('odometry source available: %s (%s)' % (topic, types[topic][0]))

        def make_cb(self, topic, convert):
            def cb(msg):
                self.latest[topic] = (time.time(), convert(msg))
            return cb

        def cb(self, msg):
            now = time.time()
            if now - self.last_send < 1.0 / SEND_HZ:
                return
            self.last_send = now
            topic, odom = pick_odom(self.latest, now, self.current)
            if topic != self.current:
                self.get_logger().info('odometry source now: %s' % topic)
                self.current = topic
            self.sock.sendto(build_packet(msg.motor_state, msg.imu_state, odom), (PC_IP, PORT))

    rclpy.init()
    rclpy.spin(UDPSender())


if __name__ == '__main__':
    main()
