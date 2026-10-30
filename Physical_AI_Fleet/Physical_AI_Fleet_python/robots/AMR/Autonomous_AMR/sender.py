import roslibpy, socket, json, math, time

ISAAC_IP, PORT = "10.85.61.200", 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

ros = roslibpy.Ros(host='192.168.12.20', port=9090)
ros.run()
print("connected:", ros.is_connected)

def on_odom(m):
    p = m['pose']['pose']['position']
    q = m['pose']['pose']['orientation']
    yaw = math.atan2(2*(q['w']*q['z'] + q['x']*q['y']),
                     1 - 2*(q['y']**2 + q['z']**2))
    sock.sendto(json.dumps({"x": p['x'], "y": p['y'], "yaw": yaw}).encode(),
                (ISAAC_IP, PORT))

roslibpy.Topic(ros, '/odom', 'nav_msgs/Odometry').subscribe(on_odom)
while True:
    time.sleep(1)
