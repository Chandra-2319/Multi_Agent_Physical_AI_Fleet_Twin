# SPDX-License-Identifier: Apache-2.0
"""Run the whole detect -> assign -> act -> verify loop without Isaac Sim.

Legged robots use the same planner + path follower as in Isaac Sim, driving a UnicycleBody
instead of the RL walking policy. High-risk approvals are auto-approved here (and logged).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Physical_AI_Fleet_python.coordinator.coordinator import FleetCoordinator  # noqa: E402
from Physical_AI_Fleet_python.fleet_config.catalog import model_for  # noqa: E402
from Physical_AI_Fleet_python.robots.robot_factory import create_robot, make_planner  # noqa: E402
from Physical_AI_Fleet_python.schemas.fleet_event import FleetEvent  # noqa: E402

FAULTS: dict = {}


def main(max_sim_s: float = 400.0, dt: float = 1 / 200, verbose: bool = True) -> int:
    coord = FleetCoordinator(fault_fn=lambda a: FAULTS.get(a))
    planner = make_planner(coord.s)
    for rid, model, home in (("go2-1", "go2", (0.0, -3.0, 0.0)), ("spot-1", "spot", (2.0, -3.0, 0.0)),
                             ("h1-1", "h1", (-2.0, -3.0, 0.0)), ("amr-1", "amr", (-4.0, -3.0, 0.0))):
        coord.fleet.add(create_robot(rid, model_for(model).robot_type, "simulation", home, model=model, planner=planner))
    if verbose:
        coord.audit.add_listener(lambda e: print(coord.audit.one_line(e)))
    FAULTS["motor-01"] = {"kind": "overpressure", "gauge_value": 8.6}
    coord.submit_event(FleetEvent(robot_id="sensor-gw", provenance="simulated", asset_id="motor-01", event_type="anomaly",
                                  observation="Pressure sensor threshold exceeded.", confidence=0.7))
    t, repaired = 0.0, False
    while t < max_sim_s:
        coord.tick(dt)
        for a in coord.approvals.pending():
            coord.approve(a.assignment_id, True)
        inc = coord.fleet.incident_for("motor-01")
        if inc and inc.status == "awaiting_repair" and not repaired:
            FAULTS.pop("motor-01")
            coord.mark_repaired("motor-01")
            repaired = True
        if repaired and inc is None and not coord.busy and all(r.status == "idle" for r in coord.fleet.robots.values()):
            print(f"\nDONE in {t:.1f}s sim time. Closed: {[i.incident_id for i in coord.fleet.closed]}")
            print("\n".join(coord.status_lines()))
            coord.shutdown()
            return 0
        t += dt
    print("\nTIMEOUT\n" + "\n".join(coord.status_lines()))
    coord.shutdown()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
