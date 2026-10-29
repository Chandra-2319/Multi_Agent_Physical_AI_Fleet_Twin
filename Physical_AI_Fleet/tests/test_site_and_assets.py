"""Site map (walkable polygon + no-go zones), equipment types and the asset layout. Pure Python."""
import json
import math

import pytest

from isolation import isolated_user_files  # noqa: F401
from Physical_AI_Fleet_python.fleet_config import asset_catalog as ac
from Physical_AI_Fleet_python.fleet_config.geometry import (as_polygon, convex_overlap, point_in_polygon, rotated_rect,
                                                            signed_area, triangulate)
from Physical_AI_Fleet_python.fleet_config.settings import AssetConfig, Settings
from Physical_AI_Fleet_python.fleet_config.site_map import SiteMapError, load_site_map, parse_site_map

L_SHAPE = [[0, 0], [20, 0], [20, 6], [6, 6], [6, 20], [0, 20]]


def site(walkable=L_SHAPE, zones=()):
    return parse_site_map({"schema_version": "1.0", "walkable_area": {"polygon": walkable},
                           "no_go_zones": [{"id": f"z{i}", "label": f"zone {i}", "polygon": z} for i, z in enumerate(zones)]})


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("pts,match", [([[0, 0], [1, 1]], "at least 3"), ([[0, 0], [1, 0], [2, 0]], "enclose"),
                                       ([[0, 0], [2, 2], [2, 0], [0, 2]], "crosses itself"), ([[0, 0], [1, "a"], [1, 1]], "numbers"),
                                       ([[0, 0], [1], [1, 1]], r"\[x, y\]")])
def test_bad_polygons_are_explained(pts, match):
    with pytest.raises(ValueError, match=match):
        as_polygon(pts)


def test_polygon_helpers():
    poly = as_polygon(L_SHAPE + [L_SHAPE[0]])  # a closing point is accepted
    assert len(poly) == 6 and signed_area(poly) == pytest.approx(20 * 6 + 6 * 14)
    assert point_in_polygon(3, 15, poly) and not point_in_polygon(15, 15, poly)
    tris = triangulate(poly)
    assert len(tris) == 4 and sum(abs(signed_area([poly[i] for i in t])) for t in tris) == pytest.approx(signed_area(poly))
    r = rotated_rect(0, 0, 2, 1, math.pi / 2)
    assert max(p[0] for p in r) == pytest.approx(1) and max(p[1] for p in r) == pytest.approx(2)
    assert convex_overlap(rotated_rect(0, 0, 1, 1, 0.3), rotated_rect(1.5, 0, 1, 1, 0)) and not convex_overlap(
        rotated_rect(0, 0, 1, 1, 0), rotated_rect(3, 0, 1, 1, 0))


# ------------------------------------------------------------------ site map
def test_shipped_site_map_and_queries():
    s = load_site_map()
    assert s.bounds == (-15, -15, 15, 15) and [z.zone_id for z in s.no_go] == ["flare-exclusion"]
    assert s.is_walkable(0, 0) and not s.is_walkable(-12, 12) and not s.is_walkable(20, 0)
    assert "no-go zone 'Flare radiation zone (example)'" in s.why_not(-12, 12) and "outside the walkable area" in s.why_not(20, 0)
    assert s.why_not(0, 0) == "" and not s.is_walkable(14.9, 0, clearance=0.3)


@pytest.mark.parametrize("raw,match", [
    ({"schema_version": "2.0"}, "schema_version"), ({"schema_version": "1.0"}, "walkable_area.polygon is required"),
    ({"schema_version": "1.0", "walkable_area": {"polygon": [[0, 0], [1, 0]]}}, "at least 3"),
    ({"schema_version": "1.0", "walkable_area": {"polygon": L_SHAPE}, "no_go_zones": [{"id": "a", "polygon": [[50, 50], [51, 50], [51, 51]]}]}, "completely outside"),
])
def test_bad_site_maps_are_explained(raw, match):
    with pytest.raises(SiteMapError, match=match):
        parse_site_map(raw)


def test_duplicate_zone_ids_load_both_zones_with_a_warning():
    s = parse_site_map({"schema_version": "1.0", "walkable_area": {"polygon": [[-15, -15], [15, -15], [15, 15], [-15, 15]]},
                        "no_go_zones": [{"id": "flare-exclusion", "label": "Flare radiation zone", "polygon": [[-14, 4], [-9, 4], [-9, 9], [-14, 9]]},
                                        {"id": "flare-exclusion", "label": "Flare zone", "polygon": [[4, 4], [4, 14], [9, 14], [14, 9], [9, 4]]}]})
    assert [z.zone_id for z in s.no_go] == ["flare-exclusion", "flare-exclusion-2"] and len(s.no_go[1].polygon) == 5
    assert not s.is_walkable(6, 10) and not s.is_walkable(-12, 6) and s.is_walkable(0, 0)
    assert len(s.warnings) == 1 and "used twice" in s.warnings[0] and "flare-exclusion-2" in s.warnings[0]


def test_site_map_file_errors_name_the_line(tmp_path):
    p = tmp_path / "s.json"
    p.write_text('{\n "schema_version": "1.0",\n oops\n}')
    with pytest.raises(SiteMapError, match="line 3"):
        load_site_map(p)
    with pytest.raises(SiteMapError, match="not found"):
        load_site_map(tmp_path / "missing.json")


# ------------------------------------------------------------------ equipment types and layout
def test_shipped_catalog_has_oil_and_gas_equipment_with_their_own_readings():
    cat = ac.load_catalog()
    assert {"electric_motor", "centrifugal_pump", "reciprocating_compressor", "gate_valve", "ball_valve", "control_valve",
            "pressure_safety_valve", "shell_tube_heat_exchanger", "three_phase_separator", "pressure_vessel", "storage_tank",
            "wellhead_christmas_tree", "pig_launcher"} == set(cat.types)
    assert cat.types["storage_tank"].primary == "level" and cat.types["storage_tank"].readings[0]["unit"] == "%"
    assert [r["name"] for r in cat.types["centrifugal_pump"].readings] == ["discharge_pressure", "suction_pressure", "bearing_temperature"]
    assert "Centrifugal pump (centrifugal_pump)" in cat.labels()


def test_shipped_layout_loads_with_rotation_and_type_specific_limits():
    s = Settings()
    assert list(s.assets) == ["motor-01", "pump-01", "valve-01"] and s.load_problems == []
    m, p, v = s.assets["motor-01"], s.assets["pump-01"], s.assets["valve-01"]
    assert (m.unit, m.normal_low, m.normal_high) == ("bar", 2, 6) and m.approach_pose[:2] == pytest.approx((2.5, 2.0))
    assert p.half_extents == pytest.approx((0.4, 0.8)) and p.approach_pose[:2] == pytest.approx((9.5, -4.0))  # turned 90 deg: -Y face points to +X
    assert math.cos(p.approach_pose[2]) == pytest.approx(-1.0) and (p.normal_low, p.normal_high) == (6, 14)
    assert v.approach_pose[0] > v.position[0] and v.primary == "upstream_pressure"


def entry(**over):
    e = {"id": "pump-09", "type": "centrifugal_pump", "position": [0, 0], "orientation_deg": [0, 0, 0]}
    e.update(over)
    return e


def build(e, s=None, existing=None):
    return ac.build_instance(AssetConfig, ac.load_catalog(), e, s or site(), existing or {})


def test_any_orientation_turns_footprint_and_approach_side():
    a = build(entry(position=[3, 3], orientation_deg=[0, 0, 45]))
    assert len(a.footprint) == 4 and a.orientation_deg == (0, 0, 45)
    d = math.dist(a.approach_pose[:2], a.position[:2])
    assert d == pytest.approx(0.4 + 1.1) and a.approach_pose[0] > 3 and a.approach_pose[1] < 3  # -Y local turned by 45 deg
    e = entry(position=[3, 3], yaw_deg=180, scale=2)
    e.pop("orientation_deg")
    b = build(e)
    assert b.local_half == pytest.approx((1.6, 0.8)) and b.approach_pose[1] == pytest.approx(3 + 0.8 + 1.1)
    assert b.height == pytest.approx(2.0)


@pytest.mark.parametrize("over,match", [
    ({"type": "spaceship"}, "unknown type 'spaceship'"), ({"id": "9x"}, "start with a letter"), ({"position": [1]}, "position"),
    ({"scale": 0}, "scale must be positive"), ({"approach_side": "up"}, "approach_side"),
    ({"readings": {"nonsense": {"max": 1}}}, "not a reading"), ({"readings": {"discharge_pressure": {"normal_high": 99}}}, "normal_high <= max"),
    ({"primary": "flow"}, "primary 'flow'"),
])
def test_bad_layout_entries_are_explained(over, match):
    with pytest.raises(ac.AssetError, match=match):
        build(entry(**{"position": [3, 3], **over}))


def test_approach_point_must_be_walkable_and_outside_no_go():
    s = site(zones=[[[1, 10], [5, 10], [5, 14], [1, 14]]])
    with pytest.raises(ac.AssetError, match=r"inside no-go zone 'zone 0'"):
        build(entry(position=[3, 9], orientation_deg=[0, 0, 180]), s)  # -Y local -> +Y world: approach at y 10.5
    with pytest.raises(ac.AssetError, match="outside the walkable area"):
        build(entry(position=[15, 15]), s)
    assert build(entry(position=[3, 9]), s).approach_pose[1] < 9  # facing the other way is fine


def test_overlaps_and_duplicates_are_refused():
    a = build(entry(id="pump-01", position=[3, 3]))
    with pytest.raises(ac.AssetError, match="already exists"):
        build(entry(id="pump_01", position=[10, 3]), existing={"pump-01": a})
    with pytest.raises(ac.AssetError, match="overlaps asset 'pump-01'"):
        build(entry(id="pump-02", position=[3.5, 3.2], orientation_deg=[0, 0, 30]), existing={"pump-01": a})


def test_reading_overrides_and_primary_switch():
    a = build(entry(position=[3, 3], primary="bearing_temperature", readings={"bearing_temperature": {"normal_high": 70}}))
    assert (a.unit, a.normal_high, a.gauge_max) == ("degC", 70, 150)


def test_layout_roundtrip_skips_bad_entries_and_resolves_relative_usd(tmp_path):
    (tmp_path / "models").mkdir()
    p = tmp_path / "layout.json"
    p.write_text(json.dumps({"schema_version": "1.0", "assets": [
        entry(id="pump-01", position=[3, 3], usd="models/pump.usd", scale=[1, 1, 1.5]),
        entry(id="bad", type="nope"), entry(id="tank-01", type="storage_tank", position=[12, 3], orientation_deg=[0, 0, 0])]}))
    assets, problems, path = ac.load_layout(AssetConfig, ac.load_catalog(), site(), p)
    assert list(assets) == ["pump-01", "tank-01"] and len(problems) == 1 and "unknown type 'nope'" in problems[0]
    assert assets["pump-01"].usd == str(tmp_path / "models" / "pump.usd")
    out = ac.save_layout(assets, tmp_path / "out.json")
    again, problems2, _ = ac.load_layout(AssetConfig, ac.load_catalog(), site(), out)
    assert problems2 == [] and again["pump-01"].scale == (1, 1, 1.5) and again["tank-01"].approach_pose == assets["tank-01"].approach_pose


def test_old_home_folder_files_no_longer_override_the_inputs_folder(isolated_user_files):
    old = isolated_user_files / ".physical_ai_fleet" / "assets.json"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text(json.dumps({"assets": [{"asset_id": "motor-03", "position": [0.0, -4.0, 0.4], "gauge_min": 0, "gauge_max": 10,
                                           "normal_low": 2, "normal_high": 6, "unit": "bar"}]}))
    (isolated_user_files / ".physical_ai_fleet" / "site_map.json").write_text("{broken")
    s = Settings()
    assert list(s.assets) == ["motor-01", "pump-01", "valve-01"] and s.load_problems == []  # inputs/ is the only source


def test_broken_json_files_never_stop_the_extension(monkeypatch, tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("{nope")
    monkeypatch.setenv("FLEET_SITE_MAP", str(bad))
    monkeypatch.setenv("FLEET_ASSET_LAYOUT", str(bad))
    s = Settings()
    assert s.site.bounds == (-15, -15, 15, 15) and s.assets == {} and len(s.load_problems) == 2
