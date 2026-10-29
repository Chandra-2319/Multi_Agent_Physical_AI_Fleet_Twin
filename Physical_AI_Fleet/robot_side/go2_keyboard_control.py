#!/usr/bin/env python3
"""PC keyboard -> Go2 (replaces go2_isaac_control.py). v0.5.6
The chosen command is repeated at 10 Hz until SPACE / another key, because go2_vel_server.py now stops the robot when
commands stop for 0.5 s (watchdog). W/S forward/back, A/D strafe, Q/E turn, SPACE stop, Ctrl+C exit (sends stop).
"""
import socket
import sys
import threading
import time

ROBOT_IP = '10.85.83.123'
ROBOT_PORT = 9999
VX_SPEED, VY_SPEED, YAW_SPEED = 0.4, 0.3, 0.6
RATE_HZ = 10.0
KEYS = {'w': (VX_SPEED, 0, 0, 'FORWARD'), 's': (-VX_SPEED, 0, 0, 'BACKWARD'), 'a': (0, VY_SPEED, 0, 'LEFT'),
        'd': (0, -VY_SPEED, 0, 'RIGHT'), 'q': (0, 0, YAW_SPEED, 'TURN L'), 'e': (0, 0, -YAW_SPEED, 'TURN R'),
        ' ': (0, 0, 0, 'STOP')}


def send_tcp(vx, vy, vyaw):
    try:
        s = socket.create_connection((ROBOT_IP, ROBOT_PORT), timeout=0.6)
        s.send(('%.3f,%.3f,%.3f\n' % (vx, vy, vyaw)).encode())
        s.recv(64)
        s.close()
        return True
    except OSError as e:
        print('\n[warn] %s' % e, end='\r')
        return False


def main():
    import termios
    import tty
    cmd = [0.0, 0.0, 0.0]
    alive = [True]

    def repeat():
        while alive[0]:
            if any(cmd):
                send_tcp(*cmd)
            time.sleep(1.0 / RATE_HZ)
    threading.Thread(target=repeat, daemon=True).start()
    print("W/S=forward/back  A/D=strafe  Q/E=turn  SPACE=stop  Ctrl+C=exit")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            key = sys.stdin.read(1)
            if key == '\x03':
                break
            k = KEYS.get(key.lower())
            if k is None:
                continue
            cmd[:] = k[:3]
            send_tcp(*cmd)
            print('  >> %-10s vx=%.1f vy=%.1f vyaw=%.1f   ' % (k[3], cmd[0], cmd[1], cmd[2]), end='\r')
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        alive[0] = False
        send_tcp(0.0, 0.0, 0.0)
        print('\n  Stopped.')


if __name__ == '__main__':
    main()
