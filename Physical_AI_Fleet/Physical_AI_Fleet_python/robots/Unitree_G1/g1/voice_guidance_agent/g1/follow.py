import sys, time, json, socket, subprocess
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

IFACE = sys.argv[1]
LIVE = "--live" in sys.argv
TARGET_H = float(sys.argv[sys.argv.index("--target-h") + 1]) if "--target-h" in sys.argv else 0.55  # feet y-position
MAX_SECS = 10**9
DET = "http://10.85.82.137:8003/detect"
MAX_V, MAX_W = 0.3, 0.3
HOLD_SECS = 1.0
last_cmd = (0.0, 0.0)
ARRIVE_SECS, SEARCH_SECS, FIRST_LOCK_SECS = 6.0, 3.0, 8.0

ChannelFactoryInitialize(0, IFACE)
loco = LocoClient(); loco.SetTimeout(10.0); loco.Init()
def stop():
    for _ in range(3): loco.Move(0, 0, 0); time.sleep(0.05)
    loco.StopMove()

u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
u.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
u.bind(("127.0.0.1", 9100)); u.setblocking(False)
trk = subprocess.Popen(["env", "PYTHONNOUSERSITE=1", "/usr/bin/python3", "/home/unitree/tracker.py", DET])

def latest():
    pkt = None
    try:
        while True: pkt = json.loads(u.recv(4096))
    except BlockingIOError: pass
    return pkt

start = last_seen = time.time(); seen_once = False; last_dir = 1; arrive_since = None; code = 0
print("LIVE" if LIVE else "DRY RUN (no motion)", "target", TARGET_H)
try:
    while True:
        time.sleep(0.1); now = time.time()
        if now - start > MAX_SECS: print("max time"); code = 3; break
        pkt = latest()
        vx = w = 0.0
        if pkt and pkt.get("found") and now - pkt["t"] < 1.2:
            seen_once, last_seen = True, now
            err = pkt["cx"] - 0.5
            last_dir = -1 if err > 0 else 1
            w = 0.0 if abs(err) < 0.08 else max(-MAX_W, min(MAX_W, -0.8 * err))
            ratio = (TARGET_H - pkt["yb"]) / TARGET_H
            vx = max(0.0, min(MAX_V, 0.4 * ratio)) if ratio > 0.1 and abs(err) < 0.3 else 0.0
            if ratio <= 0.1 and abs(err) < 0.15:
                arrive_since = arrive_since or now
                if now - arrive_since > ARRIVE_SECS: print("arrived"); code = 0; break
            else: arrive_since = None
            print("yb=%.2f cx=%.2f score=%.2f n=%d -> vx=%.2f w=%.2f" % (pkt["yb"], pkt["cx"], pkt["score"], pkt["n"], vx, w))
        else:
            limit = SEARCH_SECS if seen_once else FIRST_LOCK_SECS
            if now - last_seen > limit: print("lost"); code = 2; break
            if seen_once and now - last_seen < HOLD_SECS:
                vx, w = last_cmd
            elif seen_once:
                w = last_dir * 0.2
            print("searching...")
        last_cmd = (vx, w)
        if LIVE: loco.Move(vx, 0, w)
except KeyboardInterrupt:
    code = 130
finally:
    stop(); trk.terminate()
sys.exit(code)
