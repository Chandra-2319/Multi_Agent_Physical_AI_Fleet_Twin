"""End-to-end plumbing: schedule/event triggers -> Nemotron (fake) -> gate -> executor (fake) -> results -> repair loop."""
import json

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.coordinator.fleet_state import (AWAITING_REPAIR, CONFIRMED, NEEDS_OPERATOR, OPEN,
                                                               TOOLS_DISPATCHED, TOOLS_PICKUP, VERIFYING)
from helpers_v4 import (FakeAsset, FakeNemotron, FakeRobot, FakeSettings, make_cfg, make_coordinator, out_of_limits, pump,
                        pump_until, unreadable, within_limits)


def fleet(coord, *robots):
    for r in robots:
        coord.fleet.add(r)


def started(coord):
    return coord.executor.started


def run_inspection(coord, clock, robot_id="go2-1"):
    assert pump_until(coord, clock, lambda: robot_id in coord.executor.runs)


def enable_g1(raw):
    raw["robot_models"]["g1"].update(available=True)
    raw["robot_models"]["g1"]["locomotion"] = {"policy_path": "/p/g1.pt", "env_yaml_path": "/p/g1_env.yaml"}


def test_first_tick_assigns_the_routine_inspection_via_nemotron(monkeypatch, tmp_path):
    nem = FakeNemotron()
    coord, clock = make_coordinator(monkeypatch, tmp_path, nemotron=nem)
    fleet(coord, FakeRobot("go2-1", "go2", (0, 0)), FakeRobot("go2-2", "go2", (9, 9)), FakeRobot("amr-1", "amr"))
    assert pump_until(coord, clock, lambda: started(coord))
    assert started(coord) == [("read_gauge", "go2-1", "pump-01", "")] and len(nem.calls) == 1
    files = sorted(p.name for p in (tmp_path / "nemotron").iterdir())
    assert len(files) == 2 and any(f.endswith("_decision.json") for f in files)
    assert coord.brain.last.source == "nemotron" and "BRAIN [nemotron]" in "\n".join(coord.status_lines())
    assert coord.audit.last("gate_decision")["status"] == "approved"


def test_exactly_one_call_in_flight_and_none_while_nothing_is_due(monkeypatch, tmp_path):
    nem = FakeNemotron()
    coord, clock = make_coordinator(monkeypatch, tmp_path, nemotron=nem)
    fleet(coord, FakeRobot("go2-1", "go2"))
    pump_until(coord, clock, lambda: started(coord))
    pump(coord, clock, steps=200)
    assert len(nem.calls) == 1


def test_no_call_when_inspections_are_not_due(monkeypatch, tmp_path):
    nem = FakeNemotron()
    coord, clock = make_coordinator(monkeypatch, tmp_path, nemotron=nem, cfg=make_cfg(lambda r: r["schedule"].update(initial_due=False)))
    coord.scheduler.last["pump-01"] = coord._run_s
    fleet(coord, FakeRobot("go2-1", "go2"))
    pump(coord, clock, steps=100)
    assert nem.calls == []
    coord._run_s += 900  # 15 minutes of RUNNING time later
    assert pump_until(coord, clock, lambda: nem.calls)


def test_auto_dispatch_off_and_nemotron_switch(monkeypatch, tmp_path):
    nem = FakeNemotron()
    coord, clock = make_coordinator(monkeypatch, tmp_path, nemotron=nem)
    coord.auto_dispatch = False
    fleet(coord, FakeRobot("go2-1", "go2"))
    pump(coord, clock, steps=30)
    assert nem.calls == [] and started(coord) == []
    coord.auto_dispatch = True
    coord.assigner.use_nemotron = False
    assert pump_until(coord, clock, lambda: started(coord))
    assert nem.calls == [] and coord.brain.last.source == "rule_based"


def test_robot_added_later_does_not_wait_for_the_periodic_timer(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path, nemotron=FakeNemotron())
    pump(coord, clock, steps=3)
    fleet(coord, FakeRobot("go2-1", "go2"))
    assert pump_until(coord, clock, lambda: started(coord), max_steps=15)


def test_full_repair_chain_without_g1_uses_the_operator_tools_button(monkeypatch, tmp_path):
    nem = FakeNemotron()
    coord, clock = make_coordinator(monkeypatch, tmp_path, nemotron=nem)
    fleet(coord, FakeRobot("go2-1", "go2", (0, 0)), FakeRobot("go2-2", "go2", (9, 9)), FakeRobot("amr-1", "amr", (1, 1)))
    run_inspection(coord, clock)
    coord.executor.finish("go2-1", event=out_of_limits("pump-01", "go2-1"))
    pump(coord, clock, steps=2)
    inc = coord.fleet.incident_for("pump-01")
    assert inc.status == CONFIRMED and inc.last_inspector == "go2-1"
    pump(coord, clock, steps=15)
    assert "task_waiting" in coord.audit.kinds() and inc.waiting_note.startswith("waiting: no available robot model") and len(nem.calls) == 1
    assert "AMR will deliver" in coord.mark_tools_loaded("pump-01")
    assert pump_until(coord, clock, lambda: "amr-1" in coord.executor.runs)
    assert started(coord)[-1] == ("deliver_tools", "amr-1", "pump-01", "") and inc.status == TOOLS_DISPATCHED and len(nem.calls) == 1
    coord.executor.finish("amr-1")
    pump(coord, clock, steps=2)
    assert inc.status == AWAITING_REPAIR
    assert "re-inspection" in coord.mark_repaired("pump-01") and inc.status == VERIFYING
    assert pump_until(coord, clock, lambda: "go2-2" in coord.executor.runs)
    assert started(coord)[-1] == ("reinspect_asset", "go2-2", "pump-01", "verify") and len(nem.calls) == 2
    coord.executor.finish("go2-2", event=within_limits("pump-01", "go2-2"))
    pump(coord, clock, steps=2)
    assert coord.fleet.incident_for("pump-01") is None and "incident_closed" in coord.audit.kinds()
    assert coord.scheduler.last["pump-01"] is not None


def test_still_out_of_limits_after_repair_triggers_the_repair_again(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path)
    fleet(coord, FakeRobot("go2-1", "go2", (0, 0)), FakeRobot("go2-2", "go2", (9, 9)), FakeRobot("amr-1", "amr"))
    inc = coord.fleet.open_incident("pump-01")
    inc.status, inc.last_inspector = VERIFYING, "go2-1"
    assert pump_until(coord, clock, lambda: "go2-2" in coord.executor.runs)
    coord.executor.finish("go2-2", event=out_of_limits("pump-01", "go2-2"))
    pump(coord, clock, steps=2)
    assert inc.status == CONFIRMED and any("still out of limits" in h for h in inc.history)


def test_unreadable_gauge_is_reinspected_then_escalated(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path)
    fleet(coord, FakeRobot("go2-1", "go2", (0, 0)), FakeRobot("go2-2", "go2", (9, 9)))
    run_inspection(coord, clock)
    coord.executor.finish("go2-1", event=unreadable("pump-01", "go2-1"))
    pump(coord, clock, steps=2)
    inc = coord.fleet.incident_for("pump-01")
    assert inc.status == OPEN and inc.needs_reinspect and inc.unreadable == 1
    assert pump_until(coord, clock, lambda: "go2-2" in coord.executor.runs) and started(coord)[-1][0] == "reinspect_asset"
    coord.executor.finish("go2-2", event=unreadable("pump-01", "go2-2"))
    pump(coord, clock, steps=2)
    assert inc.status == NEEDS_OPERATOR and inc.unreadable == 2
    n = len(started(coord))
    pump(coord, clock, steps=100)
    assert len(started(coord)) == n


def test_false_alarm_reinspection_closes_the_incident(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path)
    fleet(coord, FakeRobot("go2-1", "go2", (0, 0)))
    coord.fleet.open_incident("pump-01").needs_reinspect = True
    assert pump_until(coord, clock, lambda: "go2-1" in coord.executor.runs)
    coord.executor.finish("go2-1", event=within_limits("pump-01", "go2-1"))
    pump(coord, clock, steps=2)
    assert coord.fleet.incident_for("pump-01") is None


def test_with_g1_enabled_pickup_needs_approval_then_the_amr_follows_automatically(monkeypatch, tmp_path):
    nem = FakeNemotron()
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(enable_g1), nemotron=nem)
    fleet(coord, FakeRobot("go2-1", "go2"), FakeRobot("g1-1", "g1", (2, 2)), FakeRobot("amr-1", "amr", (1, 0)))
    run_inspection(coord, clock)
    coord.executor.finish("go2-1", event=out_of_limits("pump-01", "go2-1"))
    assert pump_until(coord, clock, lambda: any(m.get("state") == "pending" for m in coord._meta.values()))
    assert started(coord)[-1][0] == "read_gauge"
    aid = next(k for k, m in coord._meta.items() if m["state"] == "pending")
    assert coord._meta[aid]["action"] == "pickup_tool" and coord._meta[aid]["robot"] == "g1-1" and "g1-1" in coord._busy_robots()
    assert coord.approve(aid, True).startswith("dispatched pick_tool")
    inc = coord.fleet.incident_for("pump-01")
    assert inc.status == TOOLS_PICKUP
    coord.executor.finish("g1-1")
    assert pump_until(coord, clock, lambda: "amr-1" in coord.executor.runs)
    assert "tools_loaded" in coord.audit.kinds() and inc.status == TOOLS_DISPATCHED
    assert started(coord)[-1][:2] == ("deliver_tools", "amr-1") and len(nem.calls) == 2


def test_operator_rejecting_the_pickup_escalates_and_blocks_the_pair(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(enable_g1))
    fleet(coord, FakeRobot("g1-1", "g1", (2, 2)))
    coord.fleet.open_incident("pump-01").status = CONFIRMED
    assert pump_until(coord, clock, lambda: coord._meta)
    aid = next(iter(coord._meta))
    assert coord.approve(aid, False) == "rejected by operator"
    assert coord.fleet.incident_for("pump-01").status == NEEDS_OPERATOR and not coord._meta
    assert ("pickup_tool:pump-01", "g1-1") in coord._blocked


def test_pending_approval_expires(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(enable_g1))
    fleet(coord, FakeRobot("g1-1", "g1"))
    coord.fleet.open_incident("pump-01").status = CONFIRMED
    assert pump_until(coord, clock, lambda: coord._meta)
    clock.advance(301)
    pump(coord, clock, steps=2)
    assert not coord._meta and "approval_expired" in coord.audit.kinds()


def test_real_go2_needs_operator_approval_for_every_action(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path)
    fleet(coord, FakeRobot("real-go2", "go2", mode="digital_twin"))
    assert pump_until(coord, clock, lambda: coord._meta)
    assert started(coord) == [] and coord.audit.last("gate_decision")["status"] == "pending_approval"
    snap = json.loads(coord.brain.client.calls[0][1]["content"])
    assert snap["robots"][0]["provenance"] == "real" and snap["mode"] == "digital_twin"
    assert coord.approve(next(iter(coord._meta)), True).startswith("dispatched read_gauge")


def test_garbage_from_nemotron_still_dispatches_via_the_rule_fallback(monkeypatch, tmp_path):
    nem = FakeNemotron("garbage")
    coord, clock = make_coordinator(monkeypatch, tmp_path, nemotron=nem)
    fleet(coord, FakeRobot("go2-1", "go2"))
    assert pump_until(coord, clock, lambda: started(coord))
    assert coord.brain.last.source == "rule_based" and len(nem.calls) == 2 and "assigner_fallback" in coord.audit.kinds()


def test_stale_decision_is_dropped(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(lambda r: r["nemotron"].update(snapshot_valid_for_s=30)))
    fleet(coord, FakeRobot("go2-1", "go2"))
    clock.advance(1)
    coord.tick(0.005)
    assert coord._brain_future is not None
    coord._brain_future.result(timeout=10)
    clock.advance(45)
    coord.tick(0.005)
    assert started(coord) == [] and coord._recent[-1]["outcome"] == "stale" and "decision_dropped" in coord.audit.kinds()


def test_decision_is_dropped_when_the_robot_became_busy_meanwhile(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path)
    r = FakeRobot("go2-1", "go2")
    fleet(coord, r)
    clock.advance(1)
    coord.tick(0.005)
    coord._brain_future.result(timeout=10)
    r.status = "busy"
    coord.tick(0.005)
    assert started(coord) == [] and coord._recent[-1]["outcome"] == "state_changed"


def test_gate_rejection_blocks_the_pair_so_it_is_not_offered_again(monkeypatch, tmp_path):
    s = FakeSettings(tmp_path, {"pump-01": FakeAsset("pump-01", (50, 0), (49, 0, 0))})
    nem = FakeNemotron()
    coord, clock = make_coordinator(monkeypatch, tmp_path, settings=s, nemotron=nem)
    fleet(coord, FakeRobot("go2-1", "go2"))
    assert pump_until(coord, clock, lambda: coord._recent)
    assert coord._recent[-1]["outcome"] == "rejected" and ("inspect:pump-01", "go2-1") in coord._blocked
    n = len(nem.calls)
    pump(coord, clock, steps=40)
    assert len(nem.calls) == n
    clock.advance(130)
    assert pump_until(coord, clock, lambda: len(nem.calls) > n)


def test_task_failure_blocks_that_robot_for_a_while(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path, nemotron=FakeNemotron())
    fleet(coord, FakeRobot("go2-1", "go2"), FakeRobot("go2-2", "go2", (9, 9)))
    run_inspection(coord, clock)
    coord.executor.finish("go2-1", error="robot fell")
    pump(coord, clock, steps=2)
    assert ("inspect:pump-01", "go2-1") in coord._blocked and coord.scheduler.last["pump-01"] is None
    assert pump_until(coord, clock, lambda: "go2-2" in coord.executor.runs)


def test_next_due_task_is_chained_after_min_interval_not_after_the_periodic_timer(monkeypatch, tmp_path):
    assets = {"pump-01": FakeAsset("pump-01", (5, 0), (4.0, 0.0, 0.0)), "pump-02": FakeAsset("pump-02", (5, 6), (4.0, 6.0, 0.0))}
    nem = FakeNemotron()
    times = []
    orig = nem.chat
    def chat(model, messages, **kw):
        times.append(coord._clock())
        return orig(model, messages, **kw)
    nem.chat = chat
    coord, clock = make_coordinator(monkeypatch, tmp_path, settings=FakeSettings(tmp_path, assets), nemotron=nem)
    fleet(coord, FakeRobot("go2-1", "go2", (0, 0)), FakeRobot("go2-2", "go2", (0, 6)))
    assert pump_until(coord, clock, lambda: len(started(coord)) == 2, max_steps=40)
    assert len(times) == 2 and {s[2] for s in started(coord)} == {"pump-01", "pump-02"} and 5.0 - 1e-9 <= times[1] - times[0] < 30.0


def test_world_reset_and_stop_all_clear_pending_work(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path)
    fleet(coord, FakeRobot("go2-1", "go2"))
    run_inspection(coord, clock)
    assert coord._meta
    coord.stop_all()
    assert not coord._meta and not coord.executor.runs
    coord.on_world_reset()
    assert not coord._meta


def test_operator_dispatch_still_works_and_marks_the_robot_busy_for_nemotron(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path, cfg=make_cfg(lambda r: r["schedule"].update(initial_due=False)))
    coord.scheduler.last["pump-01"] = coord._run_s
    fleet(coord, FakeRobot("go2-1", "go2"))
    assert coord.operator_dispatch("go2-1", "read_gauge", "pump-01").startswith("dispatched")
    assert "go2-1" in coord._busy_robots() and "pump-01" in coord._busy_assets()
    coord.executor.finish("go2-1")
    assert not coord._meta and coord.operator_dispatch("go2-1", "deliver_tools", "pump-01").startswith("rejected")


def test_status_and_mark_buttons_have_safe_answers(monkeypatch, tmp_path):
    coord, clock = make_coordinator(monkeypatch, tmp_path)
    assert coord.mark_repaired("pump-01") == "no open incident" and coord.mark_tools_loaded("pump-01") == "no open incident"
    coord.fleet.open_incident("pump-01").status = AWAITING_REPAIR
    assert "cannot be marked" in coord.mark_tools_loaded("pump-01") and coord.status_lines()[0].startswith("(no robots")
