# SPDX-License-Identifier: Apache-2.0
"""Puts the twin USD of a real robot (with its OmniGraph) on the stage under /World/Fleet/<id> (v0.5.6). Only `pxr`.

  * The file is referenced, so its OmniGraph comes along. Relationship targets (e.g. Articulation Controller
    'Target Prim') are remapped by USD automatically; STRING paths (e.g. 'Robot Path' = "/World/Go2") are not, so every
    string / token attribute under the twin that names a prim of the original file is rewritten to the new location.
  * Physics scenes inside the file are deactivated (the extension's scene carries the physics rate).
  * Collisions are switched off and gravity is disabled on every rigid body: the twin is placed by the real robot's pose
    each physics step and must never push the simulated robots or fall between two updates.
  * v0.5.7: the extension sets the joints itself (from the robot's UDP packets). An OmniGraph in the file that uses ROS 2
    nodes or an Articulation Controller would fight it (and needs ROS on the PC), so such graphs are DEACTIVATED.
    Other graphs are left alone. A plain Go2 USD (no graph at all) works as the twin.
"""
from __future__ import annotations

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


def default_prim_path(usd_path: str) -> str:
    layer = Sdf.Layer.FindOrOpen(usd_path)
    if layer is None:
        raise RuntimeError(f"cannot open twin USD {usd_path}")
    name = layer.defaultPrim or (layer.rootPrims[0].name if layer.rootPrims else "")
    if not name:
        raise RuntimeError(f"{usd_path} has no prims")
    return "/" + name


def retarget_string_paths(stage, root_path: str, old_prefix: str, new_prefix: str) -> list:
    """Rewrite string / token attribute values that start with old_prefix (as a path) to new_prefix. Returns changes."""
    changed = []
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        for attr in prim.GetAttributes():
            tn = attr.GetTypeName()
            if tn not in (Sdf.ValueTypeNames.String, Sdf.ValueTypeNames.Token):
                continue
            val = attr.Get()
            if not isinstance(val, str) or not val.startswith("/"):
                continue
            if val.startswith(new_prefix + "/") or val == new_prefix:
                continue
            if val == old_prefix or val.startswith(old_prefix + "/"):
                new = new_prefix + val[len(old_prefix):]
                attr.Set(new)
                changed.append((str(attr.GetPath()), val, new))
    return changed


CONFLICTING_NODE_HINTS = ("ros2", "ros_", "articulationcontroller", "articulation_controller")


def graph_conflicts(graph_prim) -> list:
    """Node types / names in an OmniGraph that would drive the robot themselves (ROS 2 nodes, articulation controller)."""
    hits = []
    for p in Usd.PrimRange(graph_prim):
        if p == graph_prim:
            continue
        attr = p.GetAttribute("node:type")
        ntype = str(attr.Get() or "") if attr and attr.HasAuthoredValue() else ""
        text = (ntype or p.GetName()).lower().replace(".", "")  # the node type decides; the name only if there is none
        if any(h in text for h in CONFLICTING_NODE_HINTS):
            hits.append(ntype or p.GetName())
    return hits


def prepare_twin(stage, root_path: str, usd_path: str) -> dict:
    """Reference the twin USD at root_path and make it safe to pose. Returns what was done."""
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)
    prim = stage.GetPrimAtPath(root_path)
    prim.GetReferences().AddReference(usd_path)
    old = default_prim_path(usd_path)
    info = {"usd": usd_path, "original_root": old, "retargeted": retarget_string_paths(stage, root_path, old, root_path),
            "articulation_root": "", "collisions_off": 0, "gravity_off": 0, "scenes_off": 0, "graphs": 0,
            "graphs_off": []}
    graphs = [p for p in Usd.PrimRange(prim) if p.GetTypeName() == "OmniGraph"]
    info["graphs"] = len(graphs)
    to_switch = [(str(g.GetPath()), graph_conflicts(g)) for g in graphs]
    for path, hits in to_switch:  # deactivate only after looking at all of them (children of a deactivated prim expire)
        if hits:
            stage.GetPrimAtPath(path).SetActive(False)
            info["graphs_off"].append({"graph": path, "nodes": sorted(set(hits))[:6]})
    for p in Usd.PrimRange(prim):  # inactive (switched-off) graphs are skipped by PrimRange
        tname = p.GetTypeName()
        if p.IsA(UsdPhysics.Scene):
            p.SetActive(False)
            info["scenes_off"] += 1
            continue
        if p.HasAPI(UsdPhysics.ArticulationRootAPI) and not info["articulation_root"]:
            info["articulation_root"] = str(p.GetPath())
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
            info["collisions_off"] += 1
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            try:
                p.AddAppliedSchema("PhysxRigidBodyAPI")
            except Exception:  # noqa: BLE001
                pass
            p.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)
            info["gravity_off"] += 1
    return info
