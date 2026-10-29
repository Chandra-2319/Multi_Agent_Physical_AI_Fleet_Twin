import pytest

from Physical_AI_Fleet_python.schemas.fleet_event import FleetEvent
from Physical_AI_Fleet_python.schemas.task_assignment import TaskAssignment
from Physical_AI_Fleet_python.schemas.validate import SchemaError, extract_json_object

GOOD_EVENT = {"robot_id": "go2-a", "provenance": "real", "asset_id": "motor-01", "event_type": "gauge_reading",
              "observation": "needle high", "confidence": 0.8,
              "gauge": {"value": 8.1, "unit": "bar", "in_range": False, "confidence": 0.8, "roi_px": 120}}
GOOD_ASSIGN = {"event_id": "EV-1", "robot_id": "go2-a", "skill": "reinspect_asset", "target_asset_id": "motor-01",
               "priority": "medium", "rationale": "confirm"}


def test_event_roundtrip():
    ev = FleetEvent.from_dict(GOOD_EVENT)
    assert ev.actionable
    assert FleetEvent.from_dict(ev.to_dict()).event_id == ev.event_id


@pytest.mark.parametrize("patch", [
    {"provenance": "camera"}, {"event_type": "explode"}, {"confidence": 1.5}, {"confidence": True},
    {"extra": 1}, {"cmd_vel": [1, 0]}, {"gauge": "high"},
    {"gauge": {"value": 1, "unit": "bar", "in_range": True, "confidence": 0.5, "joint": 1}},
])
def test_event_rejects_malformed(patch):
    with pytest.raises(SchemaError):
        FleetEvent.from_dict({**GOOD_EVENT, **patch})


def test_assignment_ok():
    assert TaskAssignment.from_model_dict(GOOD_ASSIGN, source="nemotron").risk == "low"


@pytest.mark.parametrize("patch", [
    {"skill": "walk_fast"}, {"velocity": 2.0}, {"target_asset_id": None}, {"priority": "urgent"}, {"rationale": "x" * 500},
])
def test_assignment_rejects(patch):
    with pytest.raises(SchemaError):
        TaskAssignment.from_model_dict({**GOOD_ASSIGN, **patch}, source="nemotron")


def test_extract_json_from_model_reply():
    assert extract_json_object("<think>reasoning {x}</think>\n```json\n{\"a\": 1}\n```") == {"a": 1}
    with pytest.raises(SchemaError):
        extract_json_object("no json here")
