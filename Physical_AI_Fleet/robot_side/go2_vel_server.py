#!/usr/bin/env python3
"""Velocity commands -> Go2 sport mode. Runs ON THE GO2 (ROS 2 Foxy, Python 3.8). v0.5.6

What changed from your original:
  * WATCHDOG: if no command arrives for WATCHDOG_S (0.5 s) while the robot was told to move, zero velocity is sent.
    A crashed PC script, a lost network or a stuck jog can no longer leave the robot walking.
    So the sender must repeat a moving command at ~10 Hz (go2_keyboard_control.py and the extension's JOG do).
  * ALLOWED_CLIENTS: only these IPs may drive the robot (empty = anyone, with a warning).
  * a connection may carry several lines; the line 'stop' = zero velocity.
Protocol unchanged: "vx,vy,vyaw\\n" -> "OK\\n".

    source /unitree/module/graph_pid_ws/install/setup.bash
    python3 ~/go2_vel_server.py
"""
import json
import socket
import threading
import time

LISTEN_PORT = 9999
MAX_LIN = 0.6
MAX_ANG = 1.2
WATCHDOG_S = 0.5
ALLOWED_CLIENTS = []   # e.g. ['10.85.82.50'] = the PC's address on the robot network


def parse_command(line):
    """'vx,vy,vyaw' or 'stop' -> clamped (vx, vy, vyaw); None if malformed."""
    text = (line or '').strip().lower()
    if text == 'stop':
        return 0.0, 0.0, 0.0
    parts = text.split(',')
    if len(parts) != 3:
        return None
    try:
        vx, vy, w = (float(p) for p in parts)
    except ValueError:
        return None
    if any(v != v or v in (float('inf'), float('-inf')) for v in (vx, vy, w)):
        return None
    clamp = lambda v, m: max(-m, min(m, v))  # noqa: E731
    return clamp(vx, MAX_LIN), clamp(vy, MAX_LIN), clamp(w, MAX_ANG)


def client_allowed(ip, allowed=None):
    allowed = ALLOWED_CLIENTS if allowed is None else allowed
    return not allowed or ip in allowed


class Watchdog:
    """moving + no command for timeout_s -> the caller must send zero (once)."""

    def __init__(self, timeout_s=WATCHDOG_S, clock=time.time):
        self.timeout_s, self.clock = timeout_s, clock
        self.last, self.moving = clock(), False
        self.lock = threading.Lock()

    def command(self, vx, vy, w):
        with self.lock:
            self.last = self.clock()
            self.moving = any(abs(v) > 1e-6 for v in (vx, vy, w))

    def expired(self):
        with self.lock:
            if self.moving and self.clock() - self.last > self.timeout_s:
                self.moving = False
                return True
            return False


def main():
    import rclpy
    from rclpy.node import Node
    from unitree_api.msg import Request

    class VelNode(Node):
        def __init__(self):
            super().__init__('go2_vel_server')
            self.pub = self.create_publisher(Request, '/api/sport/request', 10)
            self.dog = Watchdog()
            self.create_timer(0.1, self.check)

        def move(self, vx, vy, w):
            msg = Request()
            msg.header.identity.api_id = 1008  # Move
            msg.parameter = json.dumps({'x': vx, 'y': vy, 'z': w})
            self.pub.publish(msg)

        def command(self, vx, vy, w):
            self.dog.command(vx, vy, w)
            self.move(vx, vy, w)

        def check(self):
            if self.dog.expired():
                self.move(0.0, 0.0, 0.0)
                print('WATCHDOG: no command for %.1f s - stopped' % WATCHDOG_S)

    rclpy.init()
    node = VelNode()

    def handle(conn, ip):
        try:
            conn.settimeout(2.0)
            buf = b''
            while True:
                chunk = conn.recv(256)
                if not chunk:
                    break
                buf += chunk
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    cmd = parse_command(line.decode(errors='replace'))
                    if cmd is None:
                        conn.send(b'ERR\n')
                        continue
                    node.command(*cmd)
                    conn.send(b'OK\n')
        except (OSError, socket.timeout):
            pass
        finally:
            conn.close()

    def serve():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('0.0.0.0', LISTEN_PORT))
        srv.listen(5)
        if not ALLOWED_CLIENTS:
            print('WARNING: ALLOWED_CLIENTS is empty - anyone on the network can drive the robot')
        print('go2_vel_server v0.5.6 on port %d, watchdog %.1f s' % (LISTEN_PORT, WATCHDOG_S))
        while True:
            conn, addr = srv.accept()
            if not client_allowed(addr[0]):
                print('refused %s' % addr[0])
                conn.close()
                continue
            threading.Thread(target=handle, args=(conn, addr[0]), daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
