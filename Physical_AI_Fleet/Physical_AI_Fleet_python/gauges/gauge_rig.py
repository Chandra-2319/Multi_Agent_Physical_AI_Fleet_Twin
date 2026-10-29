# SPDX-License-Identifier: Apache-2.0
"""The pressure gauge that is part of an asset's own USD (v0.5.1).

The gauge is modelled inside e.g. Assets/Motor.usd, so its size, position and orientation come with the asset and nothing
has to be placed or scaled per asset type. The fleet only needs to find the needle prim and turn it:

  /World/Assets/<id>/model/.../Needle_3   + xformOp:rotate<axis>:fleetNeedle  (appended: turns about the needle's own pivot)

needle_prim (asset_types.json "gauge") is a path inside the asset USD as shown in its Stage panel, with or without the file's
default prim name; if it is empty or not found, the first prim named gauges.json models.<m>.needle_name is used.
The needle's world position (= the dial centre) tells the robot where to stand. Only `pxr` is needed.
"""
from __future__ import annotations

from pxr import Usd, UsdGeom

from .gauge_math import value_to_angle, zone_of

NEEDLE_OP = "fleetNeedle"


class GaugeError(RuntimeError):
    pass


def needle_rel_path(needle_prim: str, default_prim: str = "") -> str:
    p = (needle_prim or "").strip().replace("\\", "/")
    parts = [x for x in p.split("/") if x]
    if parts and default_prim and parts[0] == default_prim:
        parts = parts[1:]
    return "/".join(parts)


def _default_prim_name(model_prim) -> str:
    """The referenced/payloaded file's default prim name, if the composed model has a single child with that name."""
    for spec in model_prim.GetPrimStack():
        for arc in list(spec.payloadList.GetAddedOrExplicitItems()) + list(spec.referenceList.GetAddedOrExplicitItems()):
            try:
                from pxr import Sdf
                layer = Sdf.Layer.FindOrOpen(arc.assetPath)
                if layer is not None and layer.defaultPrim:
                    return layer.defaultPrim
            except Exception:  # noqa: BLE001
                pass
    return ""


class EmbeddedGauge:
    def __init__(self, stage, asset, spec: dict, model_path: str, needle_prim: str = ""):
        self.stage, self.asset, self.model, self.model_path = stage, asset, spec, model_path
        self.asset_id = asset.asset_id
        self.needle_prim = needle_prim
        self.value = float(spec["normal_value"])
        self.needle_path = ""
        self._op = None
        self.notes: list = []

    def build(self) -> "EmbeddedGauge":
        root = self.stage.GetPrimAtPath(self.model_path)
        if not root.IsValid():
            raise GaugeError(f"{self.asset_id}: no model prim at {self.model_path} (is 'usd' set for this asset type?)")
        rel = needle_rel_path(self.needle_prim, _default_prim_name(root))
        prim = self.stage.GetPrimAtPath(f"{self.model_path}/{rel}") if rel else None
        if prim is None or not prim.IsValid():
            name = rel.rsplit("/", 1)[-1] if rel else str(self.model.get("needle_name", "") or "")
            hits = [p for p in Usd.PrimRange(root) if name and p.GetName() == name]
            if not hits:
                raise GaugeError(f"{self.asset_id}: needle prim not found in the asset USD (looked for "
                                 f"'{self.needle_prim or name}' under {self.model_path}); set gauge.needle_prim in "
                                 f"asset_types.json to the needle's path from the Stage panel")
            prim = hits[0]
            if len(hits) > 1:
                self.notes.append(f"{len(hits)} prims named '{name}', using {prim.GetPath()}")
        self.needle_path = str(prim.GetPath())
        xf = UsdGeom.Xformable(prim)
        if not xf:
            raise GaugeError(f"{self.asset_id}: {self.needle_path} is not transformable (pick the needle Xform)")
        axis = str(self.model.get("needle_axis", "Z")).upper()
        name = f"xformOp:rotate{axis}:{NEEDLE_OP}"
        self._op = next((op for op in xf.GetOrderedXformOps() if op.GetOpName() == name), None)
        if self._op is None:
            self._op = {"X": xf.AddRotateXOp, "Y": xf.AddRotateYOp, "Z": xf.AddRotateZOp}[axis](opSuffix=NEEDLE_OP)
        self.set_value(self.value)
        x, y, z = self.position()
        self.notes.append(f"needle {self.needle_path} at ({x:.2f}, {y:.2f}, {z:.2f})")
        return self

    def position(self) -> tuple:
        """World position of the needle pivot (= dial centre), with the needle op's own rotation not affecting it."""
        m = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.needle_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = m.ExtractTranslation()  # stage units = the world metres the planner uses (asset positions are written as-is)
        return float(t[0]), float(t[1]), float(t[2])

    def dial_radius(self) -> float:
        """Dial radius in world metres: gauges.json dial_radius_m, else estimated from the needle's size (a needle is about
        one dial radius long, pivot to tip plus its tail). Cached. v0.5.2"""
        if getattr(self, "_radius", None):
            return self._radius
        r = float(self.model.get("dial_radius_m", 0) or 0)
        if r <= 0:
            try:
                rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]) \
                    .ComputeWorldBound(self.stage.GetPrimAtPath(self.needle_path)).ComputeAlignedRange()
                r = 0.0 if rng.IsEmpty() else float(max(rng.GetMax() - rng.GetMin()))
            except Exception:  # noqa: BLE001
                r = 0.0
            r = r if r > 0 else 0.06
        self._radius = min(max(r, 0.02), 1.0)
        return self._radius

    def set_value(self, value: float) -> float:
        self.value = float(value)
        angle = value_to_angle(self.value, self.model)
        if self._op is not None:
            self._op.Set(float(angle))
        return angle

    def set_state(self, state: str) -> float:
        return self.set_value(self.model["high_value"] if state == "high" else self.model["normal_value"])

    def truth(self) -> str:
        return zone_of(self.value, self.model)

    def remove(self) -> None:
        """The gauge belongs to the asset: removing the asset removes it. Only the added needle op is cleared."""
        p = self.stage.GetPrimAtPath(self.needle_path) if self.needle_path else None
        if p is not None and p.IsValid():
            attr = p.GetAttribute(f"xformOp:rotate{str(self.model.get('needle_axis', 'Z')).upper()}:{NEEDLE_OP}")
            if attr:
                attr.Set(0.0)
