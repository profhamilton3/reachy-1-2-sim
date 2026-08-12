"""
R12-402 / R12-503: SceneDocument -> MJCF fragment for the native MuJoCo backend.

Compiles validated scene objects (see scenes/scene.schema.json) into MJCF
``<body>`` elements.  Field names follow the real schema:

    geometry: {kind, size|radius|height}
    pose:     {position, orientation_wxyz | rpy}
    material: {rgba}
    physics:  {dynamic, collision, mass, friction, restitution}
    tracked:  bool

Static objects (physics.dynamic == false, the default) are welded to the world.
Dynamic objects (physics.dynamic == true) get a <freejoint> and mass, so the
native backend simulates and streams their pose (R12-503).

Object collision channel matches reachy_1_2.xml:  contype=4 conaffinity=7
(collides with world, robot and other objects; see R12-500 collision model).

Coordinate convention: scene position [x,y,z] m, quaternion [w,x,y,z].  MuJoCo
pos="x y z" quat="w x y z" — same order.  rpy is converted to a wxyz quaternion.
"""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Optional, Tuple

# Object contact channel (see reachy_1_2.xml R12-500 collision model).
_OBJ_CONTYPE = 4
_OBJ_CONAFFINITY = 7


class SceneCompilerError(Exception):
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_scene(scene_doc: Mapping[str, Any]) -> str:
    """Return a standalone ``<mujoco>`` document with all scene objects."""
    asset_xml, body_xml = compile_scene_body_fragment(scene_doc)
    asset_block = f"  <asset>\n    {asset_xml}\n  </asset>\n" if asset_xml else ""
    return (
        '<mujoco model="scene_objects">\n'
        + asset_block
        + "  <worldbody>\n"
        + body_xml
        + "\n  </worldbody>\n"
        + "</mujoco>\n"
    )


def compile_scene_body_fragment(scene_doc: Mapping[str, Any]) -> Tuple[str, str]:
    """Return (asset_xml, body_xml) for embedding in reachy_1_2.xml."""
    objects = scene_doc.get("objects") or []
    body_parts: List[str] = []
    asset_parts: List[str] = []
    for obj in objects:
        body_xml, asset_xml = _compile_object(obj)
        body_parts.append(body_xml)
        if asset_xml:
            asset_parts.append(asset_xml)
    return "\n".join(asset_parts), "\n".join(body_parts)


def tracked_object_ids(scene_doc: Mapping[str, Any]) -> List[str]:
    """IDs of objects whose pose should be streamed (tracked == true)."""
    return [
        str(o.get("id"))
        for o in (scene_doc.get("objects") or [])
        if o.get("tracked")
    ]


def dynamic_object_ids(scene_doc: Mapping[str, Any]) -> List[str]:
    """IDs of objects simulated with a freejoint (physics.dynamic == true)."""
    ids = []
    for o in (scene_doc.get("objects") or []):
        phys = o.get("physics") or {}
        if phys.get("dynamic"):
            ids.append(str(o.get("id")))
    return ids


# ---------------------------------------------------------------------------
# Per-object compiler
# ---------------------------------------------------------------------------

def _compile_object(obj: Mapping[str, Any]) -> Tuple[str, str]:
    obj_id = obj.get("id") or "unknown"
    pose = obj.get("pose") or {}
    geo = obj.get("geometry") or {}
    material = obj.get("material") or {}
    physics = obj.get("physics") or {}

    pos_str = _pos_str(pose.get("position") or [0, 0, 0])
    quat_str = _quat_str(_pose_quat(pose, obj_id))
    rgba_str = _rgba_str(material.get("rgba") or [0.7, 0.7, 0.7, 1.0])

    dynamic = bool(physics.get("dynamic"))
    collision = physics.get("collision", True)

    geom_xml, asset_xml = _compile_geometry(
        geo, obj_id, rgba_str, physics, dynamic, collision
    )

    free = "\n      <freejoint/>" if dynamic else ""
    body_xml = (
        f'    <body name="{_xml_attr(obj_id)}" pos="{pos_str}" quat="{quat_str}">'
        f"{free}\n"
        f"      {geom_xml}\n"
        f"    </body>"
    )
    return body_xml, asset_xml


def _compile_geometry(
    geo: Mapping[str, Any],
    obj_id: str,
    rgba_str: str,
    physics: Mapping[str, Any],
    dynamic: bool,
    collision: bool,
) -> Tuple[str, str]:
    kind = geo.get("kind") or "box"
    asset_xml = ""

    if kind == "box":
        size = geo.get("size") or [0.1, 0.1, 0.1]
        if len(size) != 3:
            raise SceneCompilerError(
                f"Object '{obj_id}': box.size must have 3 elements"
            )
        # scene size = full extents; MJCF box size = half-extents.
        s = " ".join(f"{v/2:.6f}" for v in size)
        shape = f'type="box" size="{s}"'

    elif kind == "sphere":
        r = float(geo.get("radius") or 0.05)
        shape = f'type="sphere" size="{r:.6f}"'

    elif kind == "cylinder":
        r = float(geo.get("radius") or 0.05)
        h = float(geo.get("height") or 0.1)
        shape = f'type="cylinder" size="{r:.6f} {h/2:.6f}"'

    elif kind == "mesh":
        path = str(geo.get("path") or geo.get("uri") or "")
        name = _safe_name(obj_id)
        asset_xml = f'<mesh name="{name}" file="{_xml_attr(path)}"/>'
        shape = f'type="mesh" mesh="{name}"'

    else:
        raise SceneCompilerError(
            f"Object '{obj_id}': unknown geometry kind '{kind}'"
        )

    attrs = [shape, f'rgba="{rgba_str}"']

    # Contact channel + friction only when collidable.
    if collision:
        attrs.append(f'contype="{_OBJ_CONTYPE}" conaffinity="{_OBJ_CONAFFINITY}"')
        fr = physics.get("friction")
        if isinstance(fr, (list, tuple)) and len(fr) >= 1:
            fr_str = " ".join(f"{float(v):.4f}" for v in fr[:3])
            attrs.append(f'friction="{fr_str}"')
    else:
        attrs.append('contype="0" conaffinity="0"')

    # Mass for dynamic objects (static bodies' mass is irrelevant).
    if dynamic:
        mass = physics.get("mass")
        if mass is not None:
            attrs.append(f'mass="{float(mass):.6f}"')
    if "restitution" in physics:
        # MuJoCo maps restitution via solref; expose as a comment-safe attr.
        attrs.append(f'solref="0.01 1"')

    return f'<geom {" ".join(attrs)}/>', asset_xml


# ---------------------------------------------------------------------------
# Pose / formatting helpers
# ---------------------------------------------------------------------------

def _pose_quat(pose: Mapping[str, Any], obj_id: str) -> Tuple[float, float, float, float]:
    if "orientation_wxyz" in pose:
        q = tuple(float(v) for v in pose["orientation_wxyz"])
        if len(q) != 4:
            raise SceneCompilerError(f"Object '{obj_id}': orientation_wxyz needs 4 values")
        return q
    if "rpy" in pose:
        rpy = pose["rpy"]
        if len(rpy) != 3:
            raise SceneCompilerError(f"Object '{obj_id}': rpy needs 3 values")
        return _rpy_to_wxyz(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    return (1.0, 0.0, 0.0, 0.0)


def _rpy_to_wxyz(roll: float, pitch: float, yaw: float
                 ) -> Tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2),  math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2),   math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _pos_str(pos: Any) -> str:
    if not isinstance(pos, (list, tuple)) or len(pos) != 3:
        return "0 0 0"
    return " ".join(f"{float(v):.6f}" for v in pos)


def _quat_str(q: Any) -> str:
    if not isinstance(q, (list, tuple)) or len(q) != 4:
        return "1 0 0 0"
    return " ".join(f"{float(v):.6f}" for v in q)


def _rgba_str(color: Any) -> str:
    if isinstance(color, (list, tuple)):
        if len(color) == 3:
            r, g, b = color
            return f"{float(r):.3f} {float(g):.3f} {float(b):.3f} 1.000"
        if len(color) == 4:
            r, g, b, a = color
            return f"{float(r):.3f} {float(g):.3f} {float(b):.3f} {float(a):.3f}"
    return "0.700 0.700 0.700 1.000"


def _xml_attr(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
