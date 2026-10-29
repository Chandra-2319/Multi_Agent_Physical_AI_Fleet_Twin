# Multi_Agent_Physical_AI_Fleet_Twin
# Physical_AI_Fleet – Unitree Go2 Digital Twin in Isaac Sim

A real Unitree Go2 walks around a room, and a copy of it (the **digital twin**) moves exactly the same way in NVIDIA Isaac Sim.
From Isaac Sim you can also nudge ("jog") the real robot, mark no-go zones, and plan inspection routes.

This guide assumes you have **never used the extension before**. Follow it top to bottom.

---

## 1. How it works (one picture)

```
 ┌──────────── Unitree Go2 (robot) ────────────┐              ┌──────────────── PC (Isaac Sim) ─────────────────┐
 │ go2_udp_sender.py   joints + IMU + odometry │──UDP 9870──▶ │ go2_udp_receiver.py ──ROS 2──▶ /go2/joint_states │
 │                                             │              │        │ (copy)                → legs (OmniGraph)│
 │                                             │              │        └──UDP 127.0.0.1:9871──▶ extension      │
 │ go2_vel_server.py   ◀── "vx,vy,vyaw" ───────│◀─TCP 9999─── │ Jog buttons / WASD keys       → body position  │
 └─────────────────────────────────────────────┘              └──────────────────────────────────────────────────┘
```

| Port | Protocol | From → To | Purpose |
|---|---|---|---|
| 9870 | UDP | Go2 → PC | Robot state, 100 Hz |
| 9871 | UDP (local only) | receiver → extension | Copy of the same data for the twin |
| 9999 | TCP | PC → Go2 | Velocity commands (jog) |

---

## 2. Hardware

### 2.1 PC that runs Isaac Sim
Source: NVIDIA Isaac Sim requirements (https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).

| Part | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04 / 24.04 (Linux strongly preferred, ROS 2 parts are Linux) | Ubuntu 22.04 |
| CPU | Intel Core i7 7th gen / AMD Ryzen 5, 4 cores | i7 9th gen / Ryzen 7, 8+ cores |
| RAM | 32 GB | 64 GB |
| GPU | NVIDIA **RTX** with RT cores, 16 GB VRAM (e.g. GeForce RTX 4080) | RTX 5080 / RTX PRO class |
| Storage | 50 GB free on SSD | 500 GB+ NVMe SSD |
| NVIDIA driver | The version listed by NVIDIA for your Isaac Sim release (check with `nvidia-smi`) | Latest production branch |
| Network | Wired Ethernet or 5 GHz Wi-Fi on the same network as the robot | Wired Ethernet |

Notes:
* Data-centre GPUs without RT cores (A100, H100) do **not** work.
* Cards with < 16 GB VRAM may start but drop FPS badly; that makes the twin stutter.
* Internet is needed the first time (Isaac Sim downloads robot/environment assets).
* Run NVIDIA's *Isaac Sim Compatibility Checker* once to confirm the PC is OK.

### 2.2 Robot
| Item | Requirement |
|---|---|
| Robot | Unitree **Go2 EDU** (needs on-board computer + ROS 2 access; AIR/PRO do not give you this) |
| On-board software | Ubuntu with **ROS 2 Foxy** and the `unitree_go` message package (factory image) |
| Battery | Charged; robot standing, in **normal (sport) mode** – not AI mode |
| Space | Clear floor area matching your site map; a person with the **remote / E-stop** at all times |

### 2.3 Network
* PC and Go2 must reach each other by IP (`ping` both ways).
* Example used in our lab: **PC = 192.168.2.202**, **Go2 = 10.85.83.123**. Replace with yours.
* Firewall on the PC: allow UDP 9870 (`sudo ufw allow 9870/udp` if ufw is active).

---

## 3. Software

### 3.1 On the PC
| Software | Version | Why |
|---|---|---|
| NVIDIA Isaac Sim | **5.x** (extension uses the `isaacsim.*` 5.x API; Python 3.11 inside Isaac) | Simulation + twin |
| ROS 2 | Humble (Ubuntu 22.04) or Jazzy (Ubuntu 24.04) | Runs `go2_udp_receiver.py`, feeds the legs |
| Isaac Sim ROS 2 Bridge | Enabled in Isaac Sim (Window → Extensions → "ROS 2 Bridge") | OmniGraph reads `/go2/joint_states` |
| Python packages | None extra – extension is standard library only | – |
| Optional: AI keys | Brain (LLM) and VLM URL / model / API key | Only for the AI-planning features |

Both machines must use the **same `ROS_DOMAIN_ID`** only if you also want ROS topics to cross the network (not needed for the twin itself).

### 3.2 On the Go2
Nothing to install. Copy these files to `~/Nebius/` (or any folder) on the robot:
* `go2_udp_sender.py`
* `go2_vel_server.py`

### 3.3 Files in the bundle
| File | Where it runs | What it does |
|---|---|---|
| `robot_side/go2_udp_sender.py` | Go2 | Streams robot state to the PC |
| `robot_side/go2_vel_server.py` | Go2 | Receives jog/velocity commands, 0.5 s safety watchdog |
| `robot_side/go2_udp_receiver.py` | PC | Turns the stream into ROS 2 topics and forwards a copy to the extension |
| `robot_side/go2_keyboard_control.py` | PC terminal | Optional: drive with W/A/S/D/Q/E, SPACE = stop |
| `Physical_AI_Fleet_python/` | Isaac Sim | The extension |
| `inputs/site_map.json` | Isaac Sim | Walkable area + no-go zones (**metres**) |
| `inputs/floor_layout.json` | Isaac Sim | Robots and assets, saved Map→World alignment |

---

## 4. One-time setup

### 4.1 Set the IP addresses (most common mistake!)
| File | Setting | Set to |
|---|---|---|
| `go2_udp_sender.py` (on Go2) | `PC_IP` | PC address, e.g. `192.168.2.202` |
| `go2_keyboard_control.py` (PC) | `ROBOT_IP` | Go2 address, e.g. `10.85.83.123` |
| Extension → Connection field / `floor_layout.json` → `"connection"` | `udp:9871;jog=<GO2_IP>:9999` | e.g. `udp:9871;jog=10.85.83.123:9999` |
| `go2_vel_server.py` (on Go2) | `ALLOWED_CLIENTS` | Empty (any) or the PC's IP |

Also in `go2_udp_sender.py`, fix one odometry source so the twin never jumps:
```python
ODOM_TOPICS = ['/utlidar/robot_odom']
```

### 4.2 Install the extension
1. Copy the bundle folder somewhere permanent, e.g. `~/isaac_ext/Physical_AI_Fleet`.
2. Start Isaac Sim → **Window → Extensions** → ☰ (settings) → **Extension Search Paths** → add `~/isaac_ext`.
3. Search **Physical_AI_Fleet** → toggle **Enabled** (tick *Autoload* if you want it every time).
4. The extension window appears; it shows the folder used for input files (`Input files: …`).

### 4.3 Site map
Put your `site_map.json` in that input folder. Rules:
* All coordinates in **metres** (divide CAD centimetres by 100).
* `walkable_area` must contain every no-go zone.
* Polygons must not cross themselves; every zone needs its own `id`.

### 4.4 Legs (OmniGraph)
Your Go2 USD needs a graph: **ROS 2 Subscribe JointState (`/go2/joint_states`) → Articulation Controller**.
**Disable** any graph node that moves the robot *body* from `/go2/odom` – the extension places the body.

---

## 5. Every-day start-up (in this order)

**On the Go2** (`ssh unitree@<GO2_IP>`), two terminals:
```bash
source /opt/ros/foxy/setup.bash
source /unitree/module/graph_pid_ws/install/setup.bash
python3 ~/Nebius/go2_udp_sender.py        # terminal 1 – run ONLY ONE copy
python3 ~/Nebius/go2_vel_server.py        # terminal 2 – prints "on port 9999, watchdog 0.5 s"
```
Make sure the robot is **standing** and in **normal mode** (not AI mode).

**On the PC:**
```bash
source /opt/ros/humble/setup.bash
python3 go2_udp_receiver.py               # leave running
```
Then in Isaac Sim:
1. Open the extension → **Load** the scene.
2. Add robot: Model **Go2**, Mode **digital_twin**, Connection `udp:9871;jog=<GO2_IP>:9999`, X / Y / Yaw = where the real robot stands on the map.
3. Press **RUN / Play**.
4. Stand the real robot exactly on that X/Y spot, facing that yaw, and press **Align Twin**. The alignment is saved to `floor_layout.json`.
5. Walk the real robot with the remote – the twin follows. Use the **Jog** buttons to nudge it (15 cm or 15° per press).

**Re-align after every robot reboot** (its odometry restarts at 0,0).

---

## 6. Quick health checks
| Check | Command (PC) | Good result |
|---|---|---|
| Robot data arrives | `sudo tcpdump -ni any udp port 9870 -c 5` | 5 packets in < 1 s |
| Port is open | `ss -ulpn \| grep 987` | receiver on 9870, `kit` on 9871 |
| Jog server reachable | `echo "0.2,0,0" \| nc -w1 <GO2_IP> 9999` | `OK` (robot steps briefly) |
| Legs | `ros2 topic hz /go2/joint_states` | ~100 Hz |

---

## 7. Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| 0 packets in tcpdump | Wrong `PC_IP` in sender, or no route | Fix `PC_IP`; `ping` from Go2 to PC |
| Twin sits at (0,0) | Not aligned yet | Stand robot on its home spot → **Align Twin** |
| Twin offset from real robot after Align | Robot not on the home spot / wrong yaw / odometry restarted | Re-place robot, re-align |
| Twin jerky when FPS drops | Low FPS, single-sample update | Install the v0.5.7 smoothing patch (`twin_smoothing.py`); lower render quality |
| Jog does nothing (log says "jog forward") | Wrong IP in Connection, robot in AI mode, or server not running | Check §4.1 and the `nc` test |
| Robot stops after ~0.5 s | Watchdog: commands must repeat at 10 Hz | Normal – hold the key / button |
| "Site map not loaded" | Zone outside walkable area, centimetres, or self-crossing polygon | See §4.3 |

---

## 8. Safety
* Someone must hold the **Unitree remote** whenever the robot can move; L2+B / damping mode is the emergency stop.
* Start with low speeds and a clear area. Jog commands are deliberately slow (0.25 m/s).
* Stop everything with **SPACE** (keyboard script) or by closing `go2_vel_server.py` (Ctrl+C).

# G1 Voice Guidance Agent (Physical AI Track)

A Unitree G1 humanoid that goes to a fault location with the worker, guides the repair step by step by voice from written SOPs, identifies objects with its camera, and reacts with gestures. All of it was tested on a real G1.

Part of the Multi_Agent_Industrial_Twin project: a Go2 inspects and raises a work order, an AMR brings the tools (digital twin), and the G1 supports the worker on site.

## What the G1 does

| Feature | How it works |
|---|---|
| Voice SOP guidance | Worker speaks, Parakeet transcribes, Nemotron answers from the SOPs one step at a time ("next", "repeat", "what tool do I need") |
| Safe refusal | Unclear speech or procedures not in the SOPs get "please say it again" or "contact the supervisor", never an invented step |
| Vision Q&A | "What is this" makes the G1 grab a live camera frame, and Cosmos Reason answers about it |
| Follow mode | "Follow me": the G1 tracks the worker with the head camera and walks behind, then announces arrival |
| Voice walking | "Turn left/right", "come forward", "go back", "go to the fault location" (scripted route) |
| Gestures | High five, clap, shake hand, wave, using preset arm actions with Inspire hands opened first |
| Repair complete | High five at the end of a procedure |

## NVIDIA open models used

| Model | Role |
|---|---|
| nvidia/parakeet-tdt-0.6b-v2 (NeMo) | Speech to text |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 (vLLM) | SOP-grounded answers |
| nvidia/Cosmos-Reason1-7B (vLLM) | Vision question answering |

Models are downloaded from Hugging Face and are not included in this repo.

## Architecture

````text
 G1 (Jetson Orin NX)                          GPU workstation
 -------------------                          ---------------
 Mic array (UDP multicast) --- 5 s WAV -----> /transcribe, /ask  (Parakeet -> Nemotron)
 Head camera (RTP multicast) - JPEG frame --> /ask_image         (Parakeet -> Cosmos)
 Head camera frames (follow) - JPEG -------> /detect             (person detector)
 Speaker (TtsMaker) <-------- reply text ---- JSON {heard, reply}
 Walking, arms, hands (Unitree SDK2)
````

The workstation runs four services (ports 8000, 8001, 8002, 8003). The G1 handles audio, camera capture, walking, arm gestures and hands, and calls the workstation over HTTP.

## Repository layout

````text
g1/voice_guidance_agent/
  server/asr_llm_server.py   # FastAPI: /transcribe /ask /ask_image /reset
  server/detect_server.py    # FastAPI: /detect (person box + colour lock)
  g1/g1_voice.py             # main loop on the robot
  g1/follow.py               # follow-mode controller (conda python)
  g1/tracker.py              # camera -> detector -> local UDP (system python)
  g1/grab_view.py            # grabs one good camera frame
  g1/hands.py                # Inspire hand open/close helper
  sops/*.md                  # repair procedures
  requirements.txt
````

## Hardware requirements

- Unitree G1 EDU with SDK access, Inspire hands, head RealSense D435i (camera is streamed by Unitree's `videohub_pc4` service as H.264 over RTP multicast `230.1.1.1:1720`)
- Workstation with an NVIDIA GPU (tested: RTX 6000 Ada, 48 GB). All three models plus the detector need roughly 40 GB at the settings below
- G1 and workstation on the same network. The G1's `eth0` (192.168.123.x) is the robot-internal network used by the SDK, and `wlan0` reaches the workstation

## Software requirements

Workstation: Ubuntu, Python 3.10, CUDA drivers, and a Hugging Face login (`hf auth login`) for the model downloads.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login
```

G1 (Jetson, Ubuntu 20.04):
- conda Python with `unitree_sdk2py` (from https://github.com/unitreerobotics/unitree_sdk2_python, installed with `pip install -e .`) and `requests`
- system Python 3.8 with GStreamer bindings (`gi`), `cv2` and `numpy` (used only for camera capture; call it with `PYTHONNOUSERSITE=1 /usr/bin/python3`)
- Inspire hand bridge service (`inspire_g1`) running

## Setup and run

### 1. Workstation: start the services in this order

```bash
# Cosmos first (needs the larger memory share)
nohup vllm serve nvidia/Cosmos-Reason1-7B --port 8002 --trust-remote-code \
  --gpu-memory-utilization 0.42 --max-model-len 4096 --max-num-seqs 4 \
  --limit-mm-per-prompt '{"image": 1}' > logs/cosmos.log 2>&1 &
until curl -s localhost:8002/health >/dev/null; do sleep 5; done

# Nemotron
nohup vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --port 8001 --trust-remote-code \
  --gpu-memory-utilization 0.40 --max-model-len 8192 --max-num-seqs 8 > logs/vllm.log 2>&1 &
until curl -s localhost:8001/health >/dev/null; do sleep 5; done

# API (Parakeet loads on start, takes about a minute) and person detector
cd g1/voice_guidance_agent
nohup uvicorn server.asr_llm_server:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
nohup uvicorn server.detect_server:app --host 0.0.0.0 --port 8003 > logs/detect.log 2>&1 &
```

Create the `logs/` folder first (`mkdir -p logs`). Check: `curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8000/transcribe` prints `422` when the API is up.

### 2. G1: check the camera, start the hand bridge, run

```bash
# camera stream check: should print around 3000 packets in 4 s
python3 - <<'PY'
import socket, struct, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('', 1720))
s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, struct.pack('4s4s', socket.inet_aton('230.1.1.1'), socket.inet_aton('192.168.123.164')))
s.settimeout(1.0); n = 0; t0 = time.time()
while time.time() - t0 < 4:
    try: s.recvfrom(65535); n += 1
    except socket.timeout: pass
print("packets in 4s:", n)
PY

python3 g1_voice.py eth0
```

Press Enter, speak for 5 seconds, and wait for the reply to finish before pressing Enter again. Copy `g1/*.py` to `/home/unitree/` on the robot (the scripts call each other there).

## Voice commands

| Say | Result |
|---|---|
| "How do I fix a valve leak" / "pump fault" / "gas alarm" / "pressure gauge" / "motor overheating" | Starts that SOP, one step per reply |
| "Next", "repeat that", "what tool do I need" | Moves through or repeats the procedure |
| "What is this" / "what am I holding" | Camera frame goes to Cosmos, answer is spoken |
| "Follow me" | Follow mode; stand still at the stop distance to finish |
| "Go to the fault location" | Scripted walking route (asks for Enter first) |
| "Turn left", "turn right", "come forward", "go back" | One short walking burst (about 30 cm or 25 degrees) |
| "High five", "clap", "shake hand", "wave" | Preset gesture with hands opened |
| "Stop" | Stops walking between commands |

## SOPs

Procedures are Markdown files in `sops/` (valve leak, pump fault, gas alarm, pressure gauge, motor overheating). The server re-reads them on every request, so adding a new file needs no restart. The step values are demo content, so replace them with real procedures before any real use.

## Calibration

- Follow stop distance: run `python3 follow.py eth0` (dry run, nothing moves), stand at the desired stop spot, read the steady `yb=` value, and pass it as `--target-h`. Walking happens only with `--live`.
- Walking: one forward burst is about 30 cm and one turn burst about 25 degrees, so edit `ROUTE` in `g1_voice.py` to fit your space.

## Known limitations

- The head camera is fixed and looks steeply downward, so follow mode tracks the worker's feet and legs and estimates distance from foot position in the image. Keep it within about 4 m.
- No obstacle avoidance, so the path must be clear. Keep bystanders out of view when starting follow mode, because it locks onto the first person it sees.
- "Stop" cannot interrupt a walking burst or follow mode. Use Ctrl+C or the e-stop.
- Speech is recorded in fixed 5-second windows, and misheard commands do nothing.
- The IPs (`10.85.82.137`, `192.168.1.22`) are hardcoded for our lab network, so edit them in `g1_voice.py`, `follow.py` and `tracker.py`.
- Pick-and-place is on the roadmap. We fine-tuned a GR00T N1.7 policy in Isaac Lab Arena, but it did not transfer to real object positions on the real robot.
- Nebius usage is shown in the simulation part of this project.

## Safety

Run with a clear floor, the e-stop in hand and a second person beside the robot. Test new motion on the harness first, and keep fingers clear of the Inspire hands.

## Troubleshooting (issues we hit)

| Problem | Fix |
|---|---|
| vLLM: "Free memory ... less than desired" | Lower `--gpu-memory-utilization`, close other GPU apps (Omniverse) |
| Nemotron: "max_num_seqs exceeds Mamba cache blocks" | Add `--max-num-seqs 8` |
| Cosmos: "No available memory for the cache blocks" | Start Cosmos first at 0.42 |
| Model download stalls | `hf auth login`, or `export HF_HUB_DISABLE_XET=1` |
| 0 camera packets | Check the camera's USB cable, then restart `videohub_pc4` with a plain `sudo kill` |
| Preset gestures leave the fingers closed | Hands are separate from the arm actions, so open them via `hands.py` |
| G1 speaks Chinese | Use speaker id 1 in `TtsMaker(text, 1)` |

# Autonomous Navigation: Run Guide

Two workflows are covered:

- **[Scenario A](#scenario-a--mapping-a-brand-new-site)**: map a new place and define its waypoint graph
- **[Scenario B](#scenario-b--returning-to-a-previously-saved-site)**: reload a saved site and navigate (the hackathon demo)

## Prerequisites

- The Go2 is powered on and reachable over SSH (`ssh unitree@<ROBOT_IP>`).
- The LiDAR is connected and the onboard NIC is named `eth0`.
- `~/site_manager.sh` and `~/cyclonedds_wifi_test.xml` are installed (see the README install section).
- You have two SSH terminals open to the robot.

> **Critical rule:** never press `w` (new mapping) before saving whatever site is currently live.
> `w` and `e` overwrite `default/` in place, with no backup and no undo.

---

## Scenario A: Mapping a brand-new site

### Terminal 1: protect the current map, then start the stack

```bash
ssh unitree@<ROBOT_IP>
```

If a site is currently loaded and you have not saved it, save it now:

```bash
~/site_manager.sh save <current_site_name>
```

Skip this only if it is already saved, or if `default/` is empty or irrelevant.

Start the stack fresh:

```bash
export CYCLONEDDS_URI=~/cyclonedds_wifi_test.xml
source /unitree/module/graph_pid_ws/install/setup.bash
bash /unitree/module/graph_pid_ws/0_unitree_slam.sh
```

### Terminal 2: run the mapping client

```bash
ssh unitree@<ROBOT_IP>
export CYCLONEDDS_URI=~/cyclonedds_wifi_test.xml
source /unitree/module/graph_pid_ws/install/setup.bash
/unitree/lib/unitree_slam/rocky/build/demo_xt16_replan_manual_goal eth0
```

At the keyboard prompt:

| Step | Key | What it does |
|---|---|---|
| 1 | `w` | Start mapping. Walk the robot around the **whole** space before continuing. |
| 2 | `e` | End mapping. Writes `GlobalMap.pcd` etc. into `default/`. |
| 3 | `x` | Add node 1 (stand at node 1's spot first). |
| 4 | `x` | Add node 2 (walk to its spot, then press `x`). Repeat for every node. |
| 5 | `c` | Push the topology to the robot (writes into `graph_map/`). |
| 6 | `q` | Close the demo program. |

### Terminal 1: stop the stack and save the new site

Press `Ctrl+C` to stop the stack, then:

```bash
~/site_manager.sh save <new_site_name>
~/site_manager.sh list        # sanity check: the new site should appear
```

The site is now archived under `~/unitree_sites/<new_site_name>/`.

---

## Scenario B: Returning to a previously saved site

### Terminal 1: load the site and start the stack

```bash
ssh unitree@<ROBOT_IP>
~/site_manager.sh load hackathon        # or any saved site, e.g. Test_01
```

Expected output:

```
Loading site '<saved_site_name>' into the live PCD + graph paths...
Done. Now start 0_unitree_slam.sh — it will boot with site '<saved_site_name>''s map and graph.
```

If the script reports that the SLAM stack is still running, press `Ctrl+C` on the running
`0_unitree_slam.sh` in the other session and retry the load.

Start the stack. It boots already using that site's map and graph:

```bash
export CYCLONEDDS_URI=~/cyclonedds_wifi_test.xml
source /unitree/module/graph_pid_ws/install/setup.bash
bash /unitree/module/graph_pid_ws/0_unitree_slam.sh
```

### Terminal 2: run a navigation client

```bash
ssh unitree@<ROBOT_IP>
export CYCLONEDDS_URI=~/cyclonedds_wifi_test.xml
source /unitree/module/graph_pid_ws/install/setup.bash
```

Pick one:

```bash
# Single goal
/unitree/lib/unitree_slam/rocky/build/demo_xt16_replan_manual_goal eth0

# Multiple goals (priority queue)
/unitree/lib/unitree_slam/rocky/build/demo_xt16_priority_goal eth0
```

Then at the prompt:

```
a                   # relocalize on the restored map (no mapping or node re-entry needed;
                    # waits ~3 s to settle, then prompts you)
Enter goal node: 2
```

The robot then navigates to the goal with the obstacle watchdog active.

---

## Keyboard reference

| Key | Action |
|---|---|
| `w` | Start mapping (**overwrites live map**) |
| `e` | End mapping |
| `x` | Add a node at the robot's current position |
| `c` | Push the topology to the robot |
| `a` | Relocalize on the loaded map |
| `<n>` + Enter | Navigate to goal node `n` |
| `s` | Stop |
| `d` | Resume |
| `q` | Quit the demo |

## `site_manager.sh` reference

| Situation | Command |
|---|---|
| List saved sites | `~/site_manager.sh list` |
| Show the last loaded or saved site | `~/site_manager.sh current` |
| Save the current live map and graph as a site | `~/site_manager.sh save <name>` |
| Restore a saved site as the live one | `~/site_manager.sh load <name>` |
