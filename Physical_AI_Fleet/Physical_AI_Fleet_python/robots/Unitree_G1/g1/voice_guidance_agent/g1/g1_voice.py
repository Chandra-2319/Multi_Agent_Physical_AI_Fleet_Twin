import sys, socket, struct, wave, time, subprocess, requests
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

SERVER = "http://10.85.82.137:8000"
GROUP, PORT, IFACE = "239.168.123.161", 5555, "192.168.123.164"
MAX_V, MAX_W = 0.2, 0.3
ROUTE = [("fwd", 3), ("left", 4), ("fwd", 2)]   # ~30 cm per fwd burst, ~25 deg per turn burst
REST_HANDS = (0.0, 0.0)                          # (right, left); 0.0 closed, 1.0 open
TARGET_H = 0.60                                  # follow-mode stop position (calibrate on robot)

ChannelFactoryInitialize(0, sys.argv[1])
c = AudioClient(); c.SetTimeout(10.0); c.Init(); c.SetVolume(100)
loco = LocoClient(); loco.SetTimeout(10.0); loco.Init()
arm = G1ArmActionClient(); arm.SetTimeout(10.0); arm.Init()
hand_pub = ChannelPublisher("rt/inspire/cmd", MotorCmds_); hand_pub.Init()

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("", PORT))
s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
             struct.pack("4s4s", socket.inet_aton(GROUP), socket.inet_aton(IFACE)))
s.settimeout(1)

VISUAL = ("what is this", "what's this", "look at", "what am i holding", "is this normal",
          "what do you see", "can you see", "read this", "identify", "what is that")

def say(t):
    c.TtsMaker(t, 1)

def record(sec):
    s.settimeout(0.1)
    try:
        while True: s.recvfrom(4096)
    except socket.timeout: pass
    s.settimeout(1)
    buf, end = b"", time.time() + sec
    while time.time() < end:
        try: buf += s.recvfrom(4096)[0]
        except socket.timeout: pass
    w = wave.open("/tmp/q.wav", "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(buf); w.close()

def set_hands(right, left, secs=1.2):
    m = MotorCmds_(); m.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]
    end = time.time() + secs
    while time.time() < end:
        for i in range(6):
            m.cmds[i].q = right
            m.cmds[6 + i].q = left
        hand_pub.Write(m)
        time.sleep(0.02)

def gesture(name, secs=3.0):
    set_hands(1.0, 1.0)
    arm.ExecuteAction(action_map.get(name))
    time.sleep(secs)
    arm.ExecuteAction(action_map.get("release arm"))
    time.sleep(1.0)
    set_hands(*REST_HANDS)

def stop():
    for _ in range(3):
        loco.Move(0, 0, 0); time.sleep(0.05)
    loco.StopMove()

def burst(vx, vy, w, secs=1.5):
    vx = max(-MAX_V, min(MAX_V, vx)); vy = max(-MAX_V, min(MAX_V, vy)); w = max(-MAX_W, min(MAX_W, w))
    end = time.time() + secs
    try:
        while time.time() < end:
            loco.Move(vx, vy, w); time.sleep(0.1)
    finally:
        stop()
    time.sleep(0.4)

STEP = {"fwd": (MAX_V, 0, 0), "back": (-0.15, 0, 0), "left": (0, 0, MAX_W), "right": (0, 0, -MAX_W)}

def run_route():
    input("Route: clear floor ahead, e-stop in hand. Press Enter to start (Ctrl+C aborts)...")
    say("Following the route to the fault location.")
    try:
        for action, n in ROUTE:
            for _ in range(n):
                burst(*STEP[action])
    except KeyboardInterrupt:
        stop(); print("route aborted"); return
    say("At the fault location. Say the problem when ready.")

def run_follow():
    say("Okay, I will follow you. Stand in front of me, alone.")
    time.sleep(5)
    rc = subprocess.call([sys.executable, "/home/unitree/follow.py", sys.argv[1], "--live", "--target-h", str(TARGET_H)])
    if rc == 0: say("We have arrived at the fault location. Tell me the problem.")
    elif rc == 2: say("I lost you. Please come back in front of me.")
    else: say("Following stopped.")

def has(t, words):
    return any(w in t for w in words)

def handle_command(t):
    if has(t, ("stop", "halt")):
        stop(); print("[stop]"); return True
    if has(t, ("go to the fault", "take me to the fault", "fault location")):
        run_route(); return True
    if has(t, ("follow me",)):
        run_follow(); return True
    if has(t, ("come forward", "go forward", "move forward", "walk forward")):
        burst(*STEP["fwd"]); return True
    if has(t, ("go back", "step back", "move back", "walk back")):
        burst(*STEP["back"]); return True
    if has(t, ("turn left",)):
        burst(*STEP["left"]); return True
    if has(t, ("turn right",)):
        burst(*STEP["right"]); return True
    if has(t, ("high five",)):
        gesture("high five"); return True
    if has(t, ("clap",)):
        gesture("clap", 4); return True
    if has(t, ("shake hand",)):
        gesture("shake hand"); return True
    if has(t, ("wave", "say hello")):
        gesture("high wave", 4); return True
    return False

def ask_with_image():
    subprocess.run(["env", "PYTHONNOUSERSITE=1", "/usr/bin/python3",
                    "/home/unitree/grab_view.py", "/tmp/view.jpg"], check=True)
    with open("/tmp/q.wav", "rb") as a, open("/tmp/view.jpg", "rb") as i:
        return requests.post(SERVER + "/ask_image", files={"audio": a, "image": i}, timeout=60).json()

say("Rig assistant ready. Tell me the problem.")
try:
    gesture("high wave", 4)
except Exception as e:
    print("greeting gesture failed:", e)

while True:
    input("Press Enter, then speak for 5 seconds...")
    record(5)
    try:
        with open("/tmp/q.wav", "rb") as a:
            heard = requests.post(SERVER + "/transcribe", files={"audio": a}, timeout=30).json()["heard"]
        t = heard.lower()
        print("Heard:", heard)
        if handle_command(t):
            continue
        if has(t, VISUAL):
            try:
                r = ask_with_image()
            except Exception as e:
                print("camera/vision error:", e)
                say("Sorry, I could not see that."); continue
        else:
            with open("/tmp/q.wav", "rb") as a:
                r = requests.post(SERVER + "/ask", files={"audio": a}, timeout=30).json()
    except Exception as e:
        print("server error:", e)
        say("Sorry, I lost connection."); continue
    print("Reply:", r["reply"])
    say(r["reply"])
    if "repair is complete" in r["reply"].lower() or "repair complete" in r["reply"].lower():
        time.sleep(3)
        gesture("high five")
