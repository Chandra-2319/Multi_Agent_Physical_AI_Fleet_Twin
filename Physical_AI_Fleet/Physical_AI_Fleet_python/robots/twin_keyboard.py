# Physical_AI_Fleet_python/robots/twin_keyboard.py  (v0.5.7) - hold WASD/QE in the viewport, release = stop
import socket, threading, time, carb.input, omni.appwindow

KEYS = {"W": (0.4, 0, 0), "S": (-0.4, 0, 0), "A": (0, 0.3, 0), "D": (0, -0.3, 0), "Q": (0, 0, 0.6), "E": (0, 0, -0.6)}

class TwinKeyboard:
    def __init__(self, host="10.85.83.123", port=9999, hz=10.0):
        self.addr, self.period = (host, port), 1.0 / hz
        self.cmd, self.held, self.last_error, self._run = (0.0, 0.0, 0.0), set(), "", True
        self._input = carb.input.acquire_input_interface()
        self._kb = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._kb, self._on_key)
        threading.Thread(target=self._loop, name="go2-keys", daemon=True).start()

    def _send(self, vx, vy, wz):
        try:
            with socket.create_connection(self.addr, timeout=0.4) as s:
                s.sendall(f"{vx:.3f},{vy:.3f},{wz:.3f}\n".encode()); s.recv(64)
            self.last_error = ""
        except OSError as e:
            self.last_error = str(e)

    def _on_key(self, ev, *a, **k):
        name = ev.input.name if hasattr(ev.input, "name") else str(ev.input).split(".")[-1]
        if ev.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name == "SPACE":
                self.held.clear()
            elif name in KEYS:
                self.held.add(name)
        elif ev.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.held.discard(name)
        v = [sum(KEYS[n][i] for n in self.held) for i in range(3)]  # W+Q = forward while turning
        was_moving, self.cmd = any(self.cmd), tuple(v)
        if was_moving and not any(v):
            threading.Thread(target=self._send, args=(0.0, 0.0, 0.0), daemon=True).start()
        return True

    def _loop(self):
        while self._run:
            if any(self.cmd):
                self._send(*self.cmd)   # repeat at 10 Hz - keeps go2_vel_server's 0.5 s watchdog fed
            time.sleep(self.period)

    def shutdown(self):
        self._run, self.held, self.cmd = False, set(), (0.0, 0.0, 0.0)
        self._input.unsubscribe_to_keyboard_events(self._kb, self._sub)
        self._send(0.0, 0.0, 0.0)
