"""v0.5.7: the extension reads the Go2's UDP packets itself - joints included - so the PC needs no ROS 2."""
import json
import math
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.robots import unitree_twin_link as L
from Physical_AI_Fleet_python.robots.mirror_robot import MirrorRobot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot_side"))
import go2_udp_receiver as receiver  # noqa: E402
import go2_udp_sender as sender  # noqa: E402


def packet(q, px=1.0, py=2.0):
    ms = [SimpleNamespace(q=v, dq=0.0, tau_est=0.0) for v in q]
    imu = SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0], accelerometer=[0, 0, 9.8])
    return sender.build_packet(ms, imu, {"px": px, "py": py, "pz": 0.3, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
                                         "src": "/utlidar/robot_odom"})


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Clock:
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t


# ------------------------------------------------------------------ joints exactly as go2_udp_receiver.py made them
def test_joint_order_and_smoothing_match_the_old_ros_receiver():
    assert list(L.REMAP) == receiver.REMAP and list(L.ISAAC_JOINT_NAMES) == receiver.ISAAC_JOINT_NAMES
    assert L.JOINT_ALPHA == receiver.ALPHA
    q = [i / 10 for i in range(12)]  # Unitree order: FR hip = 0.0, FL hip = 0.3, ...
    isaac = L.unitree_to_isaac(q)
    assert isaac[0] == 0.3 and isaac[1] == 0.0 and isaac[2] == 0.9 and isaac[3] == 0.6  # FL, FR, RL, RR hips
    assert L.smooth(None, [1.0]) == [1.0] and L.smooth([0.0], [1.0]) == pytest.approx([0.4])


def test_link_smooths_joints_and_reports_the_packet_rate():
    c = Clock()
    link = L.Go2UdpLink(0, clock=c)
    for k in range(21):
        c.t += 0.01  # 100 packets per second
        link.feed(packet([0.0] * 12 if k == 0 else [1.0] * 12), t=c.t)
    st = link.state()
    assert st["joints_isaac"] == pytest.approx([1 - 0.6 ** 20] * 12) and st["joint_age"] == 0.0
    assert st["rate_hz"] == pytest.approx(100.0, rel=0.02) and len(st["joints"]) == 12  # raw Unitree order kept too


def test_mirror_hands_out_joints_only_while_they_are_fresh():
    c = Clock()
    link = L.Go2UdpLink(0, clock=c)
    r = MirrorRobot("go2-real", "quadruped", (0, 0, 0), link, "/World/Fleet/go2_real", "go2")
    r.update(0.01)
    assert r.joints is None
    link.feed(packet([0.5] * 12), t=c.t)
    r.update(0.01)
    assert r.joints == pytest.approx([0.5] * 12) and "joints 0.0s" in r.summary()
    c.t += 2.0
    r.update(0.01)
    assert r.joints is None and r.status == "offline"  # stale: the twin is not posed with old angles


def test_default_port_is_the_senders_port_and_the_old_9871_still_parses():
    assert L.DEFAULT_PORT == sender.PORT == 9870 and L.Go2UdpLink().port == 9870
    assert L.Go2UdpLink.parse_connection("udp:9870;jog=10.85.82.106:9999")["port"] == 9870
    assert L.Go2UdpLink.parse_connection("udp:9871")["port"] == 9871


# ------------------------------------------------------------------ real UDP on localhost, like the Go2 sending to the PC
def test_end_to_end_over_udp_from_the_unchanged_sender_format():
    port = free_port()
    link = L.Go2UdpLink(port, host="127.0.0.1")
    assert link.start()
    try:
        r = MirrorRobot("go2-real", "quadruped", (0, 0, 0), link, "/x", "go2")
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        q = [0.1 * i for i in range(12)]
        for _ in range(100):
            tx.sendto(packet(q, px=3.0, py=-1.0), ("127.0.0.1", port))
            time.sleep(0.01)
            r.update(0.01)
            if r.joints is not None and r.status != "offline":
                break
        assert r.pose[:2] == pytest.approx([3.0, -1.0]) and r.joints == pytest.approx(L.unitree_to_isaac(q))
    finally:
        link.close()


def test_a_second_reader_on_the_port_fails_loudly_and_close_frees_the_port(monkeypatch):
    port = free_port()
    a = L.Go2UdpLink(port, host="127.0.0.1")
    assert a.start()
    monkeypatch.setattr(L, "DEFAULT_PORT", port)  # as if this were 9870
    b = L.Go2UdpLink(port, host="127.0.0.1")
    assert b.start() is False and "cannot listen on UDP port" in b.error and "go2_udp_receiver.py" in b.error
    a.close()
    c = L.Go2UdpLink(port, host="127.0.0.1")
    assert c.start(), c.error  # LOAD again right after: the port is free at once
    c.close()


# ------------------------------------------------------------------ joints onto the twin, by name
def test_joints_are_written_by_name_with_matching_pd_targets():
    from fakes import Fake6Articulation
    from Physical_AI_Fleet_python.robots.twin_body import TwinPose
    art = Fake6Articulation([0, 0, 0], None)
    order = list(reversed(L.ISAAC_JOINT_NAMES))  # the USD lists its joints in some other order
    art.dof_names = order
    tp = TwinPose("/World/Fleet/go2_real", "/World/Fleet/go2_real/Go2", make_articulation=lambda p: art)
    q = [0.01 * i for i in range(12)]
    assert tp.write_joints(L.ISAAC_JOINT_NAMES, q) and tp.joint_error == ""
    for name, v in zip(L.ISAAC_JOINT_NAMES, q):
        assert art.dof_pos[0, order.index(name)] == pytest.approx(v) and art.targets[0, order.index(name)] == pytest.approx(v)
    assert np.allclose(art.dof_vel, 0)
    art.dof_names = ["a"] * 12
    tp.reset()
    assert tp.write_joints(L.ISAAC_JOINT_NAMES, q) is False and "joints not in the twin USD" in tp.joint_error
    assert TwinPose("/x", "").write_joints(L.ISAAC_JOINT_NAMES, q) is False  # no articulation: nothing to drive


def test_without_dof_names_the_isaac_order_is_assumed():
    from fakes import Fake6Articulation
    from Physical_AI_Fleet_python.robots.twin_body import TwinPose
    art = Fake6Articulation([0, 0, 0], None)
    art.dof_names = None
    tp = TwinPose("/x", "/x/Go2", make_articulation=lambda p: art)
    assert tp.write_joints(L.ISAAC_JOINT_NAMES, [0.2] * 12) and np.allclose(art.dof_pos, 0.2)
    assert "assumed" in tp.dof_note


# ------------------------------------------------------------------ the twin USD
pxr = pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


def twin(path, nodes):
    st = Usd.Stage.CreateNew(str(path))
    st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(st, "/World/Go2").GetPrim())
    for gname, ntypes in nodes.items():
        st.DefinePrim(f"/World/{gname}", "OmniGraph")
        for i, t in enumerate(ntypes):
            st.DefinePrim(f"/World/{gname}/n{i}", "OmniGraphNode").CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set(t)
    st.Save()
    return str(path)


def test_ros_and_controller_graphs_are_switched_off_others_kept(tmp_path):
    from Physical_AI_Fleet_python import twin_stage
    usd = twin(tmp_path / "t.usda", {
        "ActionGraph": ["omni.graph.action.OnPlaybackTick", "isaacsim.ros2.bridge.ROS2Context",
                        "isaacsim.ros2.bridge.ROS2SubscribeJointState", "isaacsim.core.nodes.IsaacArticulationController"],
        "OdomGraph": ["isaacsim.ros2.bridge.ROS2Subscriber"],
        "LightsGraph": ["omni.graph.action.OnTick", "omni.graph.nodes.WritePrimAttribute"]})
    st = Usd.Stage.CreateInMemory()
    info = twin_stage.prepare_twin(st, "/World/Fleet/go2_real", usd)
    off = {g["graph"]: g["nodes"] for g in info["graphs_off"]}
    assert set(off) == {"/World/Fleet/go2_real/ActionGraph", "/World/Fleet/go2_real/OdomGraph"} and info["graphs"] == 3
    assert "isaacsim.core.nodes.IsaacArticulationController" in off["/World/Fleet/go2_real/ActionGraph"]
    assert not st.GetPrimAtPath("/World/Fleet/go2_real/ActionGraph").IsActive()
    assert st.GetPrimAtPath("/World/Fleet/go2_real/LightsGraph").IsActive()
    assert info["articulation_root"] == "/World/Fleet/go2_real/Go2"
    plain = twin(tmp_path / "plain.usda", {})  # a plain Go2 USD works as the twin
    info2 = twin_stage.prepare_twin(st, "/World/Fleet/go2_b", plain)
    assert info2["graphs"] == 0 and info2["graphs_off"] == [] and info2["articulation_root"].endswith("/Go2")
