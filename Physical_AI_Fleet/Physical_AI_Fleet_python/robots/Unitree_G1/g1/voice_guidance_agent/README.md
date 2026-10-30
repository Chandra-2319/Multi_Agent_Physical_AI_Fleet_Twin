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

## Third-party and licenses

This repo is Apache-2.0. Model weights are under their own NVIDIA open model licenses. The Unitree SDK is BSD-3-Clause. The person detector uses Ultralytics YOLO, which is AGPL-3.0.
