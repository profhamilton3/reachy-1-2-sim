"""
R12-402: YAML SceneDocument → MJCF XML fragment.

The scene compiler translates validated SceneDocument dicts (from scene_loader)
into an MJCF <worldbody> fragment that the native MuJoCo server can inject into
the running model via mjSpec (MuJoCo 3.x) or written as a standalone include.

For the MVP, the compiler produces a static XML string that represents all
scene objects.  Dynamic injection (hot-reload via mjSpec) is a follow-on.

Supported geometry kinds
------------------------
  box       → <geom type="box"     size="sx sy sz"/>
  sphere    → <geom type="sphere"  size="r"/>
  cylinder  → <geom type="cylinder" size="r h"/> (h = half-height)
  mesh      → <geom type="mesh" mesh="<asset_name>"/>  (asset must be registered)

Coordinate convention
---------------------
  Scene YAML: position [x,y,z] metres, quaternion [w,x,y,z] (schema order).
  MuJoCo:     pos="x y z",  quat="w x y z"  — same order, no conversion needed.
"""

from __future__ import annotations

import math
import textwrap
from typing import Any, Dict, List, Mapping, Optional, Tuple


class SceneCompilerError(Exception):
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_scene(scene_doc: Mapping[str, Any]) -> str:
    """Return an MJCF XML string containing all scene objects as <body> elements.

    The returned string is a complete ``<mujoco>`` document so it can be
    validated standalone or included via ``<include file="…"/>``.
    """
    objects = scene_doc.get("objects") or []
    body_lines: List[str] = []
    asset_lines: List[str] = []

    for obj in objects:
        body_xml, asset_xml = _compile_object(obj)
        body_lines.append(body_xml)
        if asset_xml:
            asset_lines.append(asset_xml)

    asset_block = ""
    if asset_lines:
        inner = "\n    ".join(asset_lines)
        asset_block = f"  <asset>\n    {inner}\n  </asset>\n"

    body_block = "\n".join(body_lines)
    return (
        '<mujoco model="scene_objects">\n'
        + asset_block
        + "  <worldbody>\n"
        + body_block
        + "\n  </worldbody>\n"
        + "</mujoco>\n"
    )


def compile_scene_body_fragment(scene_doc: Mapping[str, Any]) -> Tuple[str, str]:
    """Return (asset_xml, body_xml) strings suitable for embedding in reachy_1_2.xml."""
    objects = scene_doc.get("objects") or []
    body_parts: List[str] = []
    asset_parts: List[str] = []

    for obj in objects:
        body_xml, asset_xml = _compile_object(obj)
        body_parts.append(body_xml)
        if asset_xml:
            asset_parts.append(asset_xml)

    return "\n".join(asset_parts), "\n".join(body_parts)


# ---------------------------------------------------------------------------
# Per-object compiler
# ---------------------------------------------------------------------------

def _compile_object(obj: Mapping[str, Any]) -> Tuple[str, str]:
    """Return (body_xml, asset_xml) for a single scene object."""
    obj_id = obj.get("id") or "unknown"
    pose = obj.get("pose") or {}
    geo = obj.get("geometry") or {}
    visual = obj.get("visual") or {}

    pos_str = _pos_str(pose.get("position") or [0, 0, 0])
    quat_str = _quat_str(pose.get("quaternion") or [1, 0, 0, 0])

    rgba_str = _rgba_str(visual.get("color") or [0.7, 0.7, 0.7, 1.0])

    geom_xml, asset_xml = _compile_geometry(geo, obj_id, rgba_str)

    body_xml = (
        f'    <body name="{_xml_attr(obj_id)}" pos="{pos_str}" quat="{quat_str}">\n'
        f"      {geom_xml}\n"
        f"    </body>"
    )
    return body_xml, asset_xml


def _compile_geometry(
    geo: Mapping[str, Any], obj_id: str, rgba_str: str
) -> Tuple[str, str]:
    kind = geo.get("kind") or "box"
    asset_xml = ""

    if kind == "box":
        dims = geo.get("dimensions") or [0.1, 0.1, 0.1]
        if len(dims) != 3:
            raise SceneCompilerError(
                f"Object '{obj_id}': box.dimensions must have 3 elements"
            )
        # MJCF box size = half-extents
        sx, sy, sz = dims[0] / 2, dims[1] / 2, dims[2] / 2
        geom_xml = (
            f'<geom type="box" size="{sx:.6f} {sy:.6f} {sz:.6f}" '
            f'rgba="{rgba_str}"/>'
        )

    elif kind == "sphere":
        r = float(geo.get("radius") or 0.05)
        geom_xml = f'<geom type="sphere" size="{r:.6f}" rgba="{rgba_str}"/>'

    elif kind == "cylinder":
        r = float(geo.get("radius") or 0.05)
        h = float(geo.get("height") or 0.1)
        geom_xml = (
            f'<geom type="cylinder" size="{r:.6f} {h/2:.6f}" '
            f'rgba="{rgba_str}"/>'
        )

    elif kind == "mesh":
        path = str(geo.get("path") or "")
        asset_name = _safe_name(obj_id)
        asset_xml = f'<mesh name="{asset_name}" file="{_xml_attr(path)}"/>'
        geom_xml = f'<geom type="mesh" mesh="{asset_name}" rgba="{rgba_str}"/>'

    else:
        raise SceneCompilerError(
            f"Object '{obj_id}': unknown geometry kind '{kind}'"
        )

    return geom_xml, asset_xml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos_str(pos: Any) -> str:
    if not isinstance(pos, (list, tuple)) or len(pos) != 3:
        return "0 0 0"
    return " ".join(f"{v:.6f}" for v in pos)


def _quat_str(q: Any) -> str:
    """Scene YAML quaternion is [w,x,y,z]; MuJoCo quat="w x y z"."""
    if not isinstance(q, (list, tuple)) or len(q) != 4:
        return "1 0 0 0"
    return " ".join(f"{v:.6f}" for v in q)


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
