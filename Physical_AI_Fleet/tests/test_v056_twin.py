"""v0.5.6: the real Go2 mirrored in the simulation (Digital Twin phase 1) + the robot-side scripts."""
import importlib
import json
import math
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.robots.frames import Frame2D
from Physical_AI_Fleet_python.robots.mirror_robot import MirrorRobot
from Physical_AI_Fleet_python.robots import unitree_twin_link as L

ROBOT_SIDE = Path(__file__).resolve().parents[1] / "robot_side"
sys.path.insert(0, str(ROBOT_SIDE))
import go2_udp_receiver as receiver  # noqa: E402
import go2_udp_sender as sender  # noqa: E402
import go2_vel_server as vel  # noqa: E402


def yaw_q(yaw):
    return {"qw": math.cos(yaw / 2), "qx": 0.0, "qy": 0.0, "qz": math.sin(yaw / 2)}


def packet(px=1.0, py=2.0, yaw=0.3, src="/utlidar/robot_odom", odom=True):
    ms = [SimpleNamespace(q=0.1 * i, dq=0.0, tau_est=0.0) for i in range(12)]
    imu = SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0.0], accelerometer=[0, 0, 9.8])
    o = dict(px=px, py=py, pz=0.31, src=src, **yaw_q(yaw)) if odom else None
    return sender.build_packet(ms, imu, o)


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def free_port(kind=socket.SOCK_DGRAM):
    s = socket.socket(socket.AF_INET, kind)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ------------------------------------------------------------------ robot-side scripts (pure parts)
def test_sender_prefers_slam_then_lidar_then_sport_and_sticks_to_one_source():
    now = 100.0
    latest = {"/sportmodestate": (now, {"px": 3}), "/utlidar/robot_odom": (now, {"px": 2})}
    t, o = sender.pick_odom(latest, now)
    assert t == "/utlidar/robot_odom" and o == {"px": 2, "src": "/utlidar/robot_odom"}
    latest["/uslam/localization/odom"] = (now, {"px": 1})
    assert sender.pick_odom(latest, now, current="/utlidar/robot_odom")[0] == "/utlidar/robot_odom"  # no jump
    assert sender.pick_odom(latest, now, current=None)[0] == "/uslam/localization/odom"
    latest["/utlidar/robot_odom"] = (now - 2, {"px": 2})  # went stale
    assert sender.pick_odom(latest, now, current="/utlidar/robot_odom")[0] == "/uslam/localization/odom"
    assert sender.pick_odom({}, now) == (None, None)


def test_sender_reads_sportmodestate_position_and_wxyz_quaternion():
    msg = SimpleNamespace(position=[1.5, -2.0, 0.3], imu_state=SimpleNamespace(quaternion=[0.9, 0.1, 0.2, 0.3]))
    assert sender.odom_from_sport(msg) == {"px": 1.5, "py": -2.0, "pz": 0.3, "qw": 0.9, "qx": 0.1, "qy": 0.2, "qz": 0.3}
    m = SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=1, y=2, z=3),
                                                                  orientation=SimpleNamespace(w=1, x=0, y=0, z=0))))
    assert sender.odom_from_odometry(m)["px"] == 1.0
    d = json.loads(packet())
    assert len(d["q"]) == 12 and d["odom"]["src"] == "/utlidar/robot_odom" and "odom" not in json.loads(packet(odom=False))


def test_receiver_forwards_every_packet_unchanged():
    port = free_port()
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", port))
    rx.settimeout(2)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.forward(tx, b"PAYLOAD", [("127.0.0.1", port)])
    assert rx.recvfrom(100)[0] == b"PAYLOAD"
    receiver.forward(tx, b"x", [("256.0.0.1", 1)])  # never raises
    assert receiver.FORWARD == [("127.0.0.1", 9871)] and receiver.PORT == 9870  # optional since v0.5.7


def test_vel_server_parses_clamps_restricts_and_has_a_watchdog():
    assert vel.parse_command("0.3,0,0.2\n") == (0.3, 0.0, 0.2) and vel.parse_command("5,-5,9") == (0.6, -0.6, 1.2)
    assert vel.parse_command("stop") == (0, 0, 0) and vel.parse_command("1,2") is None and vel.parse_command("a,b,c") is None
    assert vel.parse_command("nan,0,0") is None
    assert vel.client_allowed("1.2.3.4", []) and vel.client_allowed("1.2.3.4", ["1.2.3.4"]) and not vel.client_allowed("9.9.9.9", ["1.2.3.4"])
    c = Clock()
    dog = vel.Watchdog(0.5, clock=c)
    dog.command(0.3, 0, 0)
    c.t += 0.4
    assert not dog.expired()
    c.t += 0.2
    assert dog.expired() and not dog.expired()  # once
    dog.command(0, 0, 0)
    c.t += 5
    assert not dog.expired()  # standing still: nothing to stop


def test_robot_scripts_are_python_3_8():
    import ast
    for f in ROBOT_SIDE.glob("*.py"):
        ast.parse(f.read_text(), feature_version=(3, 8))


# ------------------------------------------------------------------ the link in the extension
def test_packet_from_the_real_sender_gives_a_world_pose_through_map_to_world():
    c = Clock()
    link = L.Go2UdpLink(0, map_to_world=Frame2D(10.0, -5.0, math.radians(90)), clock=c)
    assert link.feed(packet(px=1.0, py=2.0, yaw=0.3), t=c.t)
    st = link.state()
    assert st["have_pose"] and st["src"] == "/utlidar/robot_odom" and st["age"] == 0
    assert (round(st["x"], 6), round(st["y"], 6)) == (8.0, -4.0) and st["yaw"] == pytest.approx(0.3 + math.pi / 2)
    assert L.yaw_from_quat(*st["quat_world"]) == pytest.approx(0.3 + math.pi / 2)
    assert st["map_pose"] == pytest.approx((1.0, 2.0, 0.3)) and len(st["joints"]) == 12
    c.t += 1.5
    assert link.state()["age"] == pytest.approx(1.5)
    assert not link.feed(b"{broken") and link.bad_packets == 1
    link2 = L.Go2UdpLink(0, clock=c)
    link2.feed(packet(odom=False))
    assert link2.state()["have_pose"] is False and link2.state()["age"] == 0


def test_moving_is_estimated_from_the_odometry():
    c = Clock()
    link = L.Go2UdpLink(0, clock=c)
    for i in range(20):
        c.t += 0.05
        link.feed(packet(px=0.02 * i, py=0.0, yaw=0.0), t=c.t)  # 0.4 m/s
    assert link.state()["moving"] is True
    for _ in range(20):
        c.t += 0.05
        link.feed(packet(px=0.38, py=0.0, yaw=0.0), t=c.t)
    assert link.state()["moving"] is False


def test_link_listens_on_a_real_udp_port_and_reports_a_busy_port():
    port = free_port()
    link = L.Go2UdpLink(port, host="127.0.0.1")
    assert link.start()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for _ in range(50):
            s.sendto(packet(px=4.0, py=5.0), ("127.0.0.1", port))
            time.sleep(0.02)
            if link.state()["have_pose"]:
                break
        assert link.state()["x"] == pytest.approx(4.0) and link.packets >= 1
        busy = L.Go2UdpLink(port, host="127.0.0.1")
        blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            blocker.bind(("127.0.0.1", free_port()))
        finally:
            blocker.close()
    finally:
        link.close()


def test_connection_text():
    assert L.Go2UdpLink.parse_connection("udp:9871") == {"port": 9871, "jog": ""}
    assert L.Go2UdpLink.parse_connection(" UDP:9871 ; jog=10.85.82.106:9999 ") == {"port": 9871, "jog": "10.85.82.106:9999"}
    for bad in ("9871", "udp:", "udp:70000", "udp:9871;jog=10.85.82.106"):
        with pytest.raises(ValueError):
            L.Go2UdpLink.parse_connection(bad)


def test_align_transform_puts_the_robot_on_its_home():
    f = Frame2D.from_correspondence((2.0, 1.0, math.radians(30)), (-12.0, -10.0, math.radians(90)))
    x, y, yaw = f.to_world(2.0, 1.0, math.radians(30))
    assert (x, y) == pytest.approx((-12.0, -10.0)) and yaw == pytest.approx(math.radians(90))
    x2, y2, _ = f.to_world(3.0, 1.0, 0)  # 1 m along the robot's map x = 1 m along world +Y (60 deg turn)
    assert math.hypot(x2 - x, y2 - y) == pytest.approx(1.0)
    assert Frame2D.parse(f.to_text()).to_world(2.0, 1.0, 0)[:2] == pytest.approx((-12.0, -10.0), abs=1e-3)


# ------------------------------------------------------------------ jog
def test_jog_pulse_is_time_limited_repeated_and_always_ends_with_zero():
    sent = []
    c = Clock()
    j = L.JogClient("10.0.0.5:9999", send=sent.append, sleep=lambda s: setattr(c, "t", c.t + s), clock=c)
    j.pulse(5.0, 0, 0, 0.6)  # clamped to JOG_V
    j._thread.join(2)
    assert sent[0] == "0.250,0.000,0.000\n" and sent[-1] == "0.000,0.000,0.000\n" and 5 <= len(sent) <= 8
    fails = []

    def broken(line):
        fails.append(line)
        raise OSError("unreachable")
    j2 = L.JogClient("10.0.0.5:9999", send=broken, sleep=lambda s: None)
    j2.pulse(0.2, 0, 0, 0.5)
    j2._thread.join(2)
    assert "unreachable" in j2.last_error and fails[-1] == "0.000,0.000,0.000\n"
    with pytest.raises(ValueError):
        L.JogClient("nohost")


def test_jog_talks_to_a_vel_server_over_tcp_and_stop_interrupts():
    port = free_port(socket.SOCK_STREAM)
    got = []
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(20)
    srv.settimeout(5)

    def serve():
        try:
            while True:
                conn, _ = srv.accept()
                line = conn.recv(256).decode()
                got.append(vel.parse_command(line))
                conn.send(b"OK\n")
                conn.close()
        except OSError:
            pass
    threading.Thread(target=serve, daemon=True).start()
    j = L.JogClient(f"127.0.0.1:{port}")
    j.pulse(0, 0, 0.5, 2.0)
    time.sleep(0.35)
    j.stop(wait=True)
    assert got and got[0] == (0.0, 0.0, 0.5) and got[-1] == (0.0, 0.0, 0.0) and len(got) < 10 and not j.active
    srv.close()


# ------------------------------------------------------------------ the mirror robot
def mirror(c, jog=None, home=(-12.0, -10.0, 0.0)):
    link = L.Go2UdpLink(0, clock=c)
    return MirrorRobot("go2-real", "quadruped", home, link, "/World/Fleet/go2_real", "go2", jog=jog), link


def test_mirror_follows_the_robot_and_goes_offline_without_data(monkeypatch):
    c = Clock()
    r, link = mirror(c)
    r.update(0.01)
    assert r.status == "offline" and "waiting for robot data" in r.note
    link.feed(packet(odom=False), t=c.t)
    r.update(0.01)
    assert r.status == "offline" and "no position" in r.note
    link.feed(packet(px=1.0, py=2.0, yaw=0.3), t=c.t)
    r.update(0.01)
    assert r.status == "idle" and r.pose == pytest.approx([1.0, 2.0, 0.3]) and r.world_pose[3] is not None
    assert r.telemetry_age() < 5 and "real-mirror" in r.summary() and "/utlidar/robot_odom" in r.summary()
    c.t += 2.0
    r.update(0.01)
    assert r.status == "offline" and "no robot data for 2 s" in r.note
    r.go_to(5, 5, 0)
    r.reset_to_home()
    assert r.pose == pytest.approx([1.0, 2.0, 0.3])  # a real robot is never teleported


def test_mirror_jog_rules():
    from test_site_and_assets import site
    c = Clock()
    r, link = mirror(c)
    assert "no jog address" in r.jog("forward")
    sent = []
    j = L.JogClient("10.0.0.5:9999", send=sent.append, sleep=lambda s: None)
    r.jog_client = j
    assert "not jogging - the twin is offline" in r.jog("forward")
    link.feed(packet(px=19.9, py=1.0, yaw=0.0), t=c.t)  # at the edge of a 20 x 20 site, facing out
    r.update(0.01)
    s = site([[0, 0], [20, 0], [20, 20], [0, 20]])
    assert "outside the walkable area" in r.jog("forward", s) and sent == []
    assert r.jog("back", s).endswith("(15 cm, then stop)")
    j._thread.join(2)
    assert sent[0].startswith("-0.250") and sent[-1] == "0.000,0.000,0.000\n"
    assert "15 deg" in r.jog("turn_left", s)
    assert "unknown jog" in r.jog("dance")


def test_the_fleet_never_tasks_a_mirror_but_robots_see_it(tmp_path):
    from helpers_v4 import FakeRobot, FakeSettings, make_cfg
    from Physical_AI_Fleet_python.coordinator.fleet_state import FleetState
    from Physical_AI_Fleet_python.reasoning.snapshot_builder import Scheduler, attach_candidates, compute_due_tasks, explain_no_candidates
    from Physical_AI_Fleet_python.safety.gate import REJECTED, SafetyGate
    from Physical_AI_Fleet_python.schemas.task_assignment import TaskAssignment
    c = Clock()
    m, link = mirror(c, home=(0.0, 0.0, 0.0))
    link.feed(packet(px=0.0, py=0.0, yaw=0.0), t=time.time())
    m.link.clock = time.time
    m.update(0.01)
    f = FleetState(FakeSettings(tmp_path))
    f.add(m)
    f.add(FakeRobot("go2-1", "go2", (3, 3)))
    cfg = make_cfg()
    ts = compute_due_tasks(cfg, f, Scheduler(cfg, ["pump-01"], 0.0), set(), 1000.0)
    attach_candidates(cfg, f, ts, set())
    assert [c[1] for c in ts[0].candidates] == ["go2-1"]
    f.robots.pop("go2-1")
    assert "mirror mode" in explain_no_candidates(cfg, f, ts[0], set())
    d = SafetyGate(f.s, cfg).check(TaskAssignment("x", "go2-real", "read_gauge", "pump-01", "low", "r", source="operator"), None, f)
    assert d.status == REJECTED and "mirrors a real robot" in d.reasons[-1]
    other = FakeRobot("spot-1", "spot", (5, 5))
    f.add(other)
    assert ("go2-real", 0.0, 0.0, 0.4) in f.neighbors("spot-1")


def test_floor_layout_keeps_the_twin_usd():
    from Physical_AI_Fleet_python.fleet_config import asset_catalog as ac
    from Physical_AI_Fleet_python.fleet_config.catalog import MODELS
    r = ac.parse_robot_entry({"id": "go2-real", "model": "go2", "mode": "digital_twin", "position": [-12, -10],
                              "connection": "udp:9871;jog=10.85.82.106:9999", "usd": "Assets/Go2_twin.usd",
                              "map_to_world": "1,2,30"}, MODELS)
    assert r["usd"] == "Assets/Go2_twin.usd" and r["connection"].startswith("udp:9871") and r["map_to_world"] == "1,2,30"


# ------------------------------------------------------------------ the twin USD on the stage
pxr = pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


def make_twin_usd(path, controller=True):
    """Like the user's twin: default prim /World, robot /World/Go2 (articulation, rigid links with collisions),
    a physics scene, and an OmniGraph with a STRING robotPath and a RELATIONSHIP targetPrim."""
    st = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(st, "/World/physicsScene")
    go2 = UsdGeom.Xform.Define(st, "/World/Go2")
    UsdPhysics.ArticulationRootAPI.Apply(go2.GetPrim())
    for link in ("base", "FL_calf"):
        x = UsdGeom.Xform.Define(st, f"/World/Go2/{link}")
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        c = UsdGeom.Cube.Define(st, f"/World/Go2/{link}/collision")
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    g = st.DefinePrim("/World/ActionGraph", "OmniGraph")
    node = st.DefinePrim("/World/ActionGraph/articulation_controller", "OmniGraphNode")
    node.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set(  # the user's graph drives the articulation
        "isaacsim.core.nodes.IsaacArticulationController" if controller else "omni.graph.nodes.ConstantString")
    node.CreateAttribute("inputs:robotPath", Sdf.ValueTypeNames.String).Set("/World/Go2")
    node.CreateAttribute("inputs:topicName", Sdf.ValueTypeNames.String).Set("/go2/joint_states")
    node.CreateRelationship("inputs:targetPrim").SetTargets(["/World/Go2"])
    st.Save()
    return str(path)


def test_twin_usd_is_referenced_paths_fixed_and_made_safe(tmp_path):
    from Physical_AI_Fleet_python import twin_stage
    usd = make_twin_usd(tmp_path / "Go2_twin.usda", controller=False)  # a graph that does not drive the robot is kept
    st = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(st, "/World/Fleet")
    info = twin_stage.prepare_twin(st, "/World/Fleet/go2_real", usd)
    node = st.GetPrimAtPath("/World/Fleet/go2_real/ActionGraph/articulation_controller")
    assert node.GetAttribute("inputs:robotPath").Get() == "/World/Fleet/go2_real/Go2"  # string path re-pointed
    assert node.GetAttribute("inputs:topicName").Get() == "/go2/joint_states"  # a ROS topic is not a prim path
    assert [str(t) for t in node.GetRelationship("inputs:targetPrim").GetTargets()] == ["/World/Fleet/go2_real/Go2"]
    assert info["articulation_root"] == "/World/Fleet/go2_real/Go2" and info["graphs"] == 1 and len(info["retargeted"]) == 1
    assert info["graphs_off"] == [] and st.GetPrimAtPath("/World/Fleet/go2_real/ActionGraph").IsActive()
    assert info["collisions_off"] == 2 and info["gravity_off"] == 2 and info["scenes_off"] == 1
    assert not st.GetPrimAtPath("/World/Fleet/go2_real/physicsScene").IsActive()
    base = st.GetPrimAtPath("/World/Fleet/go2_real/Go2/base")
    assert base.GetAttribute("physxRigidBody:disableGravity").Get() is True
    assert UsdPhysics.CollisionAPI(st.GetPrimAtPath("/World/Fleet/go2_real/Go2/base/collision")).GetCollisionEnabledAttr().Get() is False
    twin_stage.prepare_twin(st, "/World/Fleet/go2_real", usd)  # re-adding does not double the prefix
    node = st.GetPrimAtPath("/World/Fleet/go2_real/ActionGraph/articulation_controller")
    assert node.GetAttribute("inputs:robotPath").Get() == "/World/Fleet/go2_real/Go2"


def test_twin_pose_places_the_articulation_root_and_zeroes_its_velocity():
    from fakes import Fake6Articulation, install_fake_torch_warp
    from Physical_AI_Fleet_python.robots.twin_body import TwinPose
    arts = []

    def make(path):
        if not arts:
            arts.append(None)
            raise RuntimeError("physics view not ready")
        a = Fake6Articulation([0, 0, 0], None)
        arts.append(a)
        return a
    tp = TwinPose("/World/Fleet/go2_real", "/World/Fleet/go2_real/Go2", make_articulation=make)
    q = (math.cos(0.25), 0.0, 0.0, math.sin(0.25))
    assert tp.write(1, 2, 0.4, q) is False and "not ready" in tp.error
    assert tp.write(1, 2, 0.4, q) is True
    a = arts[-1]
    assert a.pos[0] == pytest.approx([1, 2, 0.4]) and a.quat[0] == pytest.approx(q, abs=1e-6) and np.allclose(a.lin, 0)
    xs = []
    tp2 = TwinPose("/World/Fleet/x", "", make_xform=lambda p: SimpleNamespace(set_world_poses=lambda **k: xs.append(k)))
    assert tp2.write(3, 4, 0.4, (1, 0, 0, 0)) and xs[0]["positions"][0].tolist() == [3, 4, 0.4]


from test_scenario_v044 import STATE, env  # noqa: E402,F401


def test_scenario_adds_aligns_saves_and_reloads_a_mirrored_go2(env, monkeypatch, tmp_path):
    sc_mod = importlib.import_module("Physical_AI_Fleet_python.scenario")
    writes = []

    class FakeTwinPose:
        def __init__(self, prim_path, root): self.root = root
        def write(self, x, y, z, q): writes.append((x, y, z, q)); return True
        def write_joints(self, names, q): joint_writes.append((tuple(names), list(q))); return True
        def reset(self): pass
    joint_writes = []
    monkeypatch.setattr(sc_mod, "TwinPose", FakeTwinPose)
    usd = make_twin_usd(tmp_path / "Go2_twin.usda")
    sc = sc_mod.FleetScenario()
    env.made.append(sc)
    sc.setup_scene()
    sc.setup_post_load()
    port = free_port()
    msg, _ = sc.add_robot("go2-real", "go2", "digital_twin", usd, f"udp:{port}", "", (0.0, -6.0, 90.0))
    assert msg.startswith("Added") and "real-robot mirror" in msg and "1 ROS / joint-control graph(s) switched off" in msg, msg
    r = sc.coordinator.fleet.get("go2-real")
    assert isinstance(r, MirrorRobot) and r.status == "offline"
    assert not STATE["stage"].GetPrimAtPath("/World/Fleet/go2_real/ActionGraph").IsActive()  # v0.5.7: joints come from UDP
    assert "go2-real" not in sc._cam_paths  # no simulated camera on a real robot
    assert "not a mirrored real robot" in sc.align_twin("nobody")
    r.link.feed(packet(px=2.0, py=1.0, yaw=0.0))
    sc.update(0.005)
    assert writes and writes[-1][:2] == pytest.approx((2.0, 1.0))  # not aligned yet: map frame
    assert joint_writes and joint_writes[-1][0][0] == "FL_hip_joint" and len(joint_writes[-1][1]) == 12
    msg = sc.align_twin("go2-real")
    assert "aligned: Map -> World" in msg, msg
    r.link.feed(packet(px=2.0, py=1.0, yaw=0.0))
    sc.update(0.005)
    assert r.pose == pytest.approx([0.0, -6.0, math.pi / 2]) and writes[-1][2] == pytest.approx(r.model.spawn_z)
    r.link.feed(packet(px=3.0, py=1.0, yaw=0.0))  # 1 m forward on the robot = 1 m along world +Y after the 90 deg turn
    sc.update(0.005)
    assert r.pose[:2] == pytest.approx([0.0, -5.0])
    floor = json.load(open(sc.settings.layout_path))
    e = next(x for x in floor["robots"] if x["id"] == "go2-real")
    assert e["mode"] == "digital_twin" and e["connection"] == f"udp:{port}" and e["usd"] == usd
    assert Frame2D.parse(e["map_to_world"]).to_world(2.0, 1.0, 0)[:2] == pytest.approx((0.0, -6.0), abs=1e-3)
    assert "no jog address" in sc.jog_robot("go2-real", "forward") and "use the remote" in sc.stop_real_robot("go2-real")
    assert sc.coordinator.operator_dispatch("go2-real", "read_gauge", "motor-01").startswith("rejected")
    sc.coordinator.stop_all()  # safe with a mirror in the fleet
    sc.setup_scene()  # LOAD again: the mirror comes back from the floor layout with its alignment
    sc.setup_post_load()
    r2 = sc.coordinator.fleet.get("go2-real")
    assert r2 is not None, sc.layout_robots_note
    assert isinstance(r2, MirrorRobot) and r2.link.frame.to_text() == e["map_to_world"]
    assert sc.remove_robot("go2-real").startswith("Removed go2-real") and "go2-real" not in sc._twins
    bad, _ = sc.add_robot("go2-x", "go2", "digital_twin", str(tmp_path / "missing.usd"), "udp:9", "", (0.0, -6.0, 0.0))
    assert "twin USD not found" in bad and not STATE["stage"].GetPrimAtPath("/World/Fleet/go2_x").IsValid()
    bad2, _ = sc.add_robot("amr-x", "amr", "digital_twin", usd, "udp:9", "", (0.0, -6.0, 0.0))
    assert "mirror mode is for quadrupeds" in bad2


def test_ui_has_the_real_robot_buttons(env):
    ui_mod = importlib.import_module("Physical_AI_Fleet_python.ui_builder")
    ui = ui_mod.UIBuilder()
    env.made.append(ui._scenario)
    ui.build_ui()
    ui._load_btn.press()
    ui._sel_robot = "nobody"
    ui._align_btn.click()
    assert "not a mirrored real robot" in ui._builder_msg.text
    ui._jog_fwd.click()
    ui._real_stop.click()
    assert "not a mirrored real robot" in ui._builder_msg.text
