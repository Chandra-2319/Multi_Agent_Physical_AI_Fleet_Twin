# SPDX-License-Identifier: Apache-2.0
"""Assets and the site map authored directly on the USD stage. Only `pxr` is needed (no Isaac Sim modules).

/World/Assets/<id>               Xform: translate (position), rotateXYZ (orientation_deg), scale
    model                        the equipment USD loaded as a PAYLOAD (scaled by the type's usd_scale)
    placeholder                  when there is no USD: a box of footprint x height with a UsdPreviewSurface material
    status                       a small sphere above the asset; its colour shows ok / alarm / work in progress
/World/Site/Walkable             the walkable-area outline (green curve)
/World/Site/NoGo/<id>            each no-go zone: translucent red fill + red outline
Materials are real UsdPreviewSurface shaders (not only displayColor) so every renderer shades them the same way.
v0.5.0: the payload could be fitted to the stage from the file's metersPerUnit / up axis.
v0.5.1: by default the model is loaded EXACTLY as authored, then usd_rotate_deg and usd_scale from asset_types.json (or the
layout entry) are applied to /World/Assets/<id>/model. usd_auto_fit: true brings back the automatic fit.
"""
from __future__ import annotations

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from .fleet_config.geometry import triangulate

LOOKS = "/World/Looks/Fleet"


class StageMarker:
    """Handle for a stage-authored asset: its prim path (and the path of its status beacon)."""

    def __init__(self, path: str) -> None:
        self.path = path

    @property
    def status_path(self) -> str:
        return f"{self.path}/status"


def _color(rgb) -> "Vt.Vec3fArray":
    return Vt.Vec3fArray([Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))])


def _token(text: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in str(text))
    return out if out and not out[0].isdigit() else f"_{out}"


def preview_material(stage, path: str, color, opacity: float = 1.0, emissive=None):
    """A UsdPreviewSurface material (created or updated)."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in color[:3]]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in emissive[:3]]))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(mat)


def marker_exists(stage, path: str) -> bool:
    return stage is not None and stage.GetPrimAtPath(path).IsValid()


def create_asset_prim(stage, path: str, asset, usd_path: str = "", status_color=(0.2, 0.8, 0.2)) -> StageMarker:
    """Author one asset. Raises RuntimeError if the prim cannot be created. An old prim at `path` is replaced."""
    if stage is None:
        raise RuntimeError("there is no open USD stage")
    parent = path.rsplit("/", 1)[0]
    if parent and not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Xform.Define(stage, parent)
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
    root = UsdGeom.Xform.Define(stage, path)
    prim = root.GetPrim()
    if not prim.IsValid():
        raise RuntimeError(f"could not define a prim at {path}")
    sx, sy, sz = (float(v) for v in asset.scale)
    xf = UsdGeom.XformCommonAPI(prim)
    xf.SetTranslate(Gf.Vec3d(*[float(v) for v in asset.position]))
    xf.SetRotate(Gf.Vec3f(*[float(v) for v in asset.orientation_deg]), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    xf.SetScale(Gf.Vec3f(sx, sy, sz))
    prim.CreateAttribute("fleet:asset_id", Sdf.ValueTypeNames.String).Set(asset.asset_id)
    prim.CreateAttribute("fleet:asset_type", Sdf.ValueTypeNames.String).Set(asset.asset_type)
    lhx, lhy = asset.local_half or asset.half_extents
    w, d, h = 2 * lhx / sx, 2 * lhy / sy, asset.height / sz  # unscaled sizes (the root scale applies on top)
    tkey = _token(asset.asset_type)
    if usd_path:
        model = UsdGeom.Xform.Define(stage, f"{path}/model").GetPrim()
        model.GetPayloads().AddPayload(usd_path)
        mxf = UsdGeom.XformCommonAPI(model)
        if getattr(asset, "usd_auto_fit", False):
            fit = model_fit(stage, usd_path, float(asset.usd_scale), getattr(asset, "usd_up_axis", ""))
            rot, sc = (fit["rotate_x_deg"], 0.0, 0.0), fit["scale"]
        else:
            rot, sc = tuple(float(v) for v in getattr(asset, "usd_rotate_deg", (0.0, 0.0, 0.0))), float(asset.usd_scale)
            fit = {"mode": "as given", "rotate_deg": list(rot), "scale": sc}
        if any(rot):
            mxf.SetRotate(Gf.Vec3f(*rot), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        if sc != 1.0:
            mxf.SetScale(Gf.Vec3f(sc, sc, sc))
        prim.CreateAttribute("fleet:model_fit", Sdf.ValueTypeNames.String).Set(str(fit))
    else:
        box = UsdGeom.Cube.Define(stage, f"{path}/placeholder")
        box.CreateSizeAttr(1.0)
        bxf = UsdGeom.XformCommonAPI(box.GetPrim())
        bxf.SetTranslate(Gf.Vec3d(0.0, 0.0, h / 2))
        bxf.SetScale(Gf.Vec3f(w, d, h))
        box.CreateDisplayColorAttr(_color(asset.color))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        _bind(box.GetPrim(), preview_material(stage, f"{LOOKS}/type_{tkey}", asset.color))
    beacon = UsdGeom.Sphere.Define(stage, f"{path}/status")
    beacon.CreateRadiusAttr(0.12)
    UsdGeom.XformCommonAPI(beacon.GetPrim()).SetTranslate(Gf.Vec3d(0.0, 0.0, h + 0.3 / sz))
    beacon.CreateDisplayColorAttr(_color(status_color))
    _bind(beacon.GetPrim(), preview_material(stage, f"{LOOKS}/status_{_token(asset.asset_id)}", status_color,
                                             emissive=status_color))
    return StageMarker(path)


def file_units(usd_path: str) -> tuple:
    """(metersPerUnit, upAxis) authored in a USD file; (1.0, 'Z') when unknown or unreadable."""
    try:
        layer = Sdf.Layer.FindOrOpen(usd_path)
    except Exception:  # noqa: BLE001
        layer = None
    if layer is None:
        return 1.0, "Z"
    root = layer.pseudoRoot
    mpu = float(root.GetInfo("metersPerUnit")) if root.HasInfo("metersPerUnit") else 1.0
    up = str(root.GetInfo("upAxis")) if root.HasInfo("upAxis") else "Z"
    return (mpu if mpu > 0 else 1.0), (up.upper() or "Z")


def model_fit(stage, usd_path: str, extra_scale: float = 1.0, up_override: str = "") -> dict:
    """Scale / rotation that make a payload sit in this stage's units and up axis."""
    mpu, up = file_units(usd_path)
    up = (up_override or up).upper()
    stage_mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    stage_up = str(UsdGeom.GetStageUpAxis(stage) or "Z").upper()
    rot = 90.0 if (up == "Y" and stage_up == "Z") else (-90.0 if (up == "Z" and stage_up == "Y") else 0.0)
    return {"file_meters_per_unit": mpu, "file_up_axis": up, "scale": round(mpu / stage_mpu * float(extra_scale), 9),
            "rotate_x_deg": rot}


def set_status(stage, marker: StageMarker, color) -> bool:
    if stage is None or not stage.GetPrimAtPath(marker.status_path).IsValid():
        return False
    UsdGeom.Gprim(stage.GetPrimAtPath(marker.status_path)).GetDisplayColorAttr().Set(_color(color))
    aid = stage.GetPrimAtPath(marker.path).GetAttribute("fleet:asset_id").Get() or marker.path.rsplit("/", 1)[-1]
    shader = UsdShade.Shader.Get(stage, f"{LOOKS}/status_{_token(aid)}/Shader")
    if shader:
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(*[float(c) for c in color[:3]]))
        if shader.GetInput("emissiveColor"):
            shader.GetInput("emissiveColor").Set(Gf.Vec3f(*[float(c) for c in color[:3]]))
    return True


def remove_marker(stage, path: str) -> bool:
    """True if a prim was removed."""
    if stage is None or not stage.GetPrimAtPath(path).IsValid():
        return False
    return bool(stage.RemovePrim(path))


# ------------------------------------------------------------------ the site map
def _outline(stage, path: str, poly, color, z: float, width: float):
    curve = UsdGeom.BasisCurves.Define(stage, path)
    pts = [Gf.Vec3f(float(x), float(y), z) for x, y in poly] + [Gf.Vec3f(float(poly[0][0]), float(poly[0][1]), z)]
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreatePointsAttr(pts)
    curve.CreateCurveVertexCountsAttr([len(pts)])
    curve.CreateWidthsAttr([width] * len(pts))
    curve.CreateDisplayColorAttr(_color(color))
    return curve


def draw_site(stage, site, root: str = "/World/Site") -> dict:
    """(Re)draw the walkable outline and the no-go zones. Visual only: no colliders. Returns prim counts."""
    if stage is None:
        raise RuntimeError("there is no open USD stage")
    if stage.GetPrimAtPath(root).IsValid():
        stage.RemovePrim(root)
    UsdGeom.Xform.Define(stage, root)
    if not site.show:
        return {"walkable": 0, "no_go": 0}
    green, red = (0.1, 0.85, 0.2), (0.95, 0.1, 0.1)
    _outline(stage, f"{root}/Walkable", site.walkable, green, 0.02, 0.08)
    UsdGeom.Xform.Define(stage, f"{root}/NoGo")
    fill = preview_material(stage, f"{LOOKS}/no_go", red, opacity=0.35)
    for z in site.no_go:
        zp = f"{root}/NoGo/{_token(z.zone_id)}"
        UsdGeom.Xform.Define(stage, zp)
        mesh = UsdGeom.Mesh.Define(stage, f"{zp}/fill")
        tris = triangulate(list(z.polygon))
        mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), 0.015) for x, y in z.polygon])
        mesh.CreateFaceVertexCountsAttr([3] * len(tris))
        mesh.CreateFaceVertexIndicesAttr([i for t in tris for i in t])
        mesh.CreateDisplayColorAttr(_color(red))
        mesh.CreateDisplayOpacityAttr([0.35])
        mesh.CreateDoubleSidedAttr(True)
        _bind(mesh.GetPrim(), fill)
        _outline(stage, f"{zp}/outline", z.polygon, red, 0.03, 0.06)
        stage.GetPrimAtPath(zp).CreateAttribute("fleet:label", Sdf.ValueTypeNames.String).Set(z.label or z.zone_id)
    return {"walkable": 1, "no_go": len(site.no_go)}
