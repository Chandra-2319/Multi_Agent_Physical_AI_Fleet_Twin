# AMR Digital Twin in NVIDIA Isaac Sim

A live digital twin of a real Autonomous Mobile Robot (AMR), modelled as a MiR200. The robot's odometry is read from ROS, streamed over UDP, and applied to the AMR prim inside an NVIDIA Isaac Sim scene, so the simulated robot mirrors the real robot's position and heading in real time.

## Overview

The real AMR navigates autonomously using its own navigation stack. This project mirrors its pose into Isaac Sim:

1. `sender.py` connects to the robot's **rosbridge** websocket, subscribes to `/odom`, converts the orientation quaternion to yaw, and sends `{x, y, yaw}` as JSON over UDP.
2. `Receiver.py` runs **inside Isaac Sim**. Every app update it drains the UDP socket, keeps the most recent packet, and sets the translation and rotation of the AMR prim.

## Architecture

```
 Real AMR                         Sender machine                    Isaac Sim machine
┌──────────────┐  rosbridge WS   ┌─────────────────┐   UDP :5005   ┌───────────────────────┐
│ ROS  /odom   │ ──────────────► │   sender.py     │ ────────────► │  Receiver.py          │
│ nav_msgs/    │   port 9090     │  quaternion →   │  JSON         │  sets pose of         │
│ Odometry     │                 │  yaw, to JSON   │  {x,y,yaw}    │  /World/mir           │
└──────────────┘                 └─────────────────┘               │  in amr_03.usd        │
                                                                   └───────────────────────┘
```

## Repository structure

```
.
├── sender.py                # ROS /odom -> UDP bridge (runs on a machine that can reach the robot)
├── Receiver.py              # UDP -> USD prim pose updater (runs inside Isaac Sim)
├── Collected_amr_03.zip     # Isaac Sim scene (collected USD), unzip before use
├── .gitignore
└── README.md
```

## Requirements

- NVIDIA Isaac Sim (version: _fill in_)
- Python 3 with `roslibpy` on the sender machine: `pip install roslibpy`
- A ROS robot exposing `/odom` (`nav_msgs/Odometry`) with **rosbridge_server** running (default port 9090)
- Network connectivity: sender -> robot (TCP 9090) and sender -> Isaac Sim machine (UDP 5005)

## Scene files

The scene ships as `Collected_amr_03.zip` (about 3 MB). It was exported with Omniverse's *Collect As*, so the scene and everything it references are bundled together. Unzip it and keep the folder structure intact so the relative references resolve:

```bash
unzip Collected_amr_03.zip
```

- `amr_03.usd`: top-level scene
- `SubUSDs/mir200.usda`: MiR200 robot model
- `SubUSDs/geometries.usd`, `materials.usda`, `physics.usda`, `physx.usda`, `mujoco.usda`, `robot.usda`, `base.usda`, `instances.usda`: geometry, materials and physics layers
- `SubUSDs/SICK_microScan3_*.usd`: SICK microScan3 laser scanner assets and scan profiles

You can also point the twin at your own AMR asset, as long as the prim path used in `Receiver.py` exists.

## Usage

### 1. Start rosbridge on the robot

```bash
roslaunch rosbridge_server rosbridge_websocket.launch   # default port 9090
```

### 2. Open the scene in Isaac Sim and run the receiver

1. Unzip `Collected_amr_03.zip`, then open `Collected_amr_03/amr_03.usd`.
2. Confirm the AMR prim path matches `PRIM_PATH` in `Receiver.py` (default `/World/mir`).
3. Open **Window > Script Editor**, load `Receiver.py`, and run it. It binds UDP port 5005 and starts updating the prim each frame.

### 3. Run the sender

Edit the addresses at the top of `sender.py`:

```python
ISAAC_IP, PORT = "<isaac-sim-machine-ip>", 5005
ros = roslibpy.Ros(host='<robot-ip>', port=9090)
```

Then:

```bash
python3 sender.py
```

You should see `connected: True`, and the AMR in Isaac Sim will follow the real robot.

## Configuration

| Setting | File | Default | Meaning |
|---|---|---|---|
| `PRIM_PATH` | `Receiver.py` | `/World/mir` | USD prim moved by the twin |
| UDP port | both | `5005` | Must match on both sides |
| `ISAAC_IP` | `sender.py` | _set locally_ | IP of the machine running Isaac Sim |
| ROS host/port | `sender.py` | _set locally_ / `9090` | rosbridge address of the robot |
| Topic | `sender.py` | `/odom` | `nav_msgs/Odometry` source |

## Message format

One UDP datagram per odometry message:

```json
{"x": 1.234, "y": -0.567, "yaw": 1.5708}
```

`x`, `y` are in metres (odom frame); `yaw` is in radians, computed from the quaternion as `atan2(2(wz + xy), 1 - 2(y² + z²))`.

## Limitations

- Pose sync is one-way (robot -> simulation). The simulation does not command the robot.
- Only planar motion is mirrored: Z is fixed at 0 and roll/pitch are ignored.
- UDP has no delivery guarantee; the receiver always applies only the latest packet.
- Poses are in the odom frame, so drift accumulates unless you send a map-frame pose (for example from AMCL).
- Network addresses are hardcoded in `sender.py`.

## Roadmap ideas

- Move IP/port/topic to command-line arguments or a config file
- Mirror map-frame pose (`/amcl_pose`) instead of raw odometry
- Sync sensors (lidar scan) and goals into the simulation
- Close the loop: send navigation goals from Isaac Sim back to the robot

## License

_Choose a license (e.g. MIT) and add a `LICENSE` file._
