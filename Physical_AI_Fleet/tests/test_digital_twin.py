"""Digital Twin mode against the fake Go2 bridge (real TCP, no hardware)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from mock_go2_bridge import serve  # noqa: E402

from Physical_AI_Fleet_python.coordinator.coordinator import FleetCoordinator  # noqa: E402
from Physical_AI_Fleet_python.robots.robot_factory import create_robot  # noqa: E402


def test_twin_go2_follows_real_robot_and_inspects():
    server, _fake = serve(9123, block=False)
    try:
        c = FleetCoordinator()
        robot = create_robot("go2-real", "quadruped", "digital_twin", (0, -3, 0), connection="127.0.0.1:9123", model="go2")
        c.fleet.add(robot)
        deadline = time.time() + 3
        while robot.status == "offline" and time.time() < deadline:
            c.tick(0.05)
            time.sleep(0.05)
        assert robot.status == "idle" and robot.provenance == "real"
        assert "dispatched" in c.operator_dispatch("go2-real", "inspect_asset", "motor-01")
        deadline = time.time() + 25
        while time.time() < deadline and "go2-real" in c.executor.runs:
            c.tick(0.05)
            time.sleep(0.05)
        assert "go2-real" not in c.executor.runs
        from Physical_AI_Fleet_python.skills.executor import standoff_pose
        asset = c.s.assets["motor-01"]
        gx, gy, _ = standoff_pose(asset, robot.radius, ((asset.approach_pose[0] - asset.position[0]) ** 2 + (asset.approach_pose[1] - asset.position[1]) ** 2) ** 0.5)
        assert abs(robot.pose[0] - gx) < 0.45 and abs(robot.pose[1] - gy) < 0.45  # v0.4.4 layout: motor approach from -X
        assert "fleet_event" in [e["kind"] for e in c.audit.recent(100)]
        c.shutdown()
    finally:
        server.shutdown()
