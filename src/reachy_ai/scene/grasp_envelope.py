"""The outer envelope a grasp planner may use for a compound object.

A part with a through-hole is `kind: compound`: several boxes around the hole.
A pinch closes on its OUTER faces, so for grasping it is the box or cylinder
those faces make.  That is true only if the outer surface really is solid, all
the way round and over the height, so it is DERIVED from the parts and checked,
never assumed from a name or a class:

  box       every point just inside the bounding rectangle's perimeter is
            covered by a part that runs the object's full height;
  cylinder  the same for a circle of the bounding box's radius, on a square
            footprint, with nothing standing more than `ROUND_PROTRUSION` out
            beyond it.  The envelope's radius is the farthest part corner, so
            it ENCLOSES the solid (the sim's 16-sided ring stands 0.48 mm proud
            of its flats).

Anything else (an L, a notched face, a plate on a stem, a tipped part) has no
envelope, and the planner refuses it with the reason.

Only the object being grasped is seen this way (`SceneModel.grasp_target`).  As
an obstacle a compound stays its solid bounding box.
"""

from __future__ import annotations

import dataclasses
import math

#: How far inside the outline the surface is probed, and how finely along it.
INSET_M = 0.00005
STEP_M = 0.00025
#: A part counts as full height if it stops no more than this short of either
#: end (the PART set's recessed pieces stop 0.1 mm short).
HEIGHT_TOL_M = 0.0005
#: A round outline's parts may stand this far (fraction of the radius) beyond
#: it: a 16-sided ring circumscribed on the radius reaches 1.96 %.
ROUND_PROTRUSION = 0.025
_SQUARE_TOL_M = 1e-5


class NoGraspEnvelope(ValueError):
    """This compound is not a box or a cylinder from outside; the message says why."""


def _footprints(parts):
    """(contains(x, y), corner radius, z low, z high) of each upright part."""
    from .awareness import _scene_io
    rotation_matrix = _scene_io().rotation_matrix
    out = []
    for index, part in enumerate(parts):
        rot = rotation_matrix(part["quat_wxyz"])
        if abs(rot[2][2] - 1.0) > 1e-6:
            raise NoGraspEnvelope(f"part {index} is not upright (turned about more than z)")
        px, py, pz = part["pos"]
        if part["kind"] == "box":
            hx, hy, hz = (v / 2.0 for v in part["size"])
            c, s = rot[0][0], rot[1][0]

            def contains(x, y, px=px, py=py, hx=hx, hy=hy, c=c, s=s):
                u, v = c * (x - px) + s * (y - py), -s * (x - px) + c * (y - py)
                return abs(u) <= hx + 1e-9 and abs(v) <= hy + 1e-9

            reach = max(math.hypot(px + c * a - s * b, py + s * a + c * b)
                        for a in (-hx, hx) for b in (-hy, hy))
        else:
            radius, hz = part["radius"], part["length"] / 2.0

            def contains(x, y, px=px, py=py, radius=radius):
                return math.hypot(x - px, y - py) <= radius + 1e-9

            reach = math.hypot(px, py) + radius
        out.append((contains, reach, pz - hz, pz + hz))
    return out


def _rectangle(sx, sy):
    hx, hy = sx / 2.0 - INSET_M, sy / 2.0 - INSET_M
    nx, ny = int(2 * hx / STEP_M) + 2, int(2 * hy / STEP_M) + 2
    for i in range(nx):
        x = -hx + 2 * hx * i / (nx - 1)
        yield x, -hy
        yield x, hy
    for j in range(ny):
        y = -hy + 2 * hy * j / (ny - 1)
        yield -hx, y
        yield hx, y


def _circle(radius):
    r = radius - INSET_M
    n = int(2 * math.pi * r / STEP_M) + 2
    for i in range(n):
        a = 2 * math.pi * i / n
        yield r * math.cos(a), r * math.sin(a)


def outline_of(parts, size):
    """("box" | "cylinder", (x, y, z) full extents) of a compound's outer surface."""
    sx, sy, sz = size
    prints = _footprints(parts)
    full = [contains for contains, _reach, lo, hi in prints
            if lo <= -sz / 2.0 + HEIGHT_TOL_M and hi >= sz / 2.0 - HEIGHT_TOL_M]

    def solid(points):
        return all(any(contains(x, y) for contains in full) for x, y in points)

    if solid(_rectangle(sx, sy)):
        return "box", (sx, sy, sz)
    if abs(sx - sy) < _SQUARE_TOL_M and solid(_circle(sx / 2.0)):
        reach = max(r for _c, r, _lo, _hi in prints)
        if reach <= sx / 2.0 * (1.0 + ROUND_PROTRUSION):
            return "cylinder", (2.0 * reach, 2.0 * reach, sz)
    raise NoGraspEnvelope(
        "its outline is not solid over the full height as a "
        f"{1000 * sx:.1f} x {1000 * sy:.1f} mm box or as a cylinder, so there is no "
        "outer envelope to grasp it by")


def grasp_envelope(obj):
    """`obj` as a grasp planner may treat it when it is the thing being grasped.

    Anything but a compound is returned as itself.  A compound comes back as a
    copy of kind "box" or "cylinder" with `envelope_of="compound"`, or raises
    `NoGraspEnvelope`.
    """
    if obj.kind != "compound":
        return obj
    if not obj.parts:
        raise NoGraspEnvelope("the scene model holds no parts for it")
    kind, size = outline_of(obj.parts, obj.size)
    return dataclasses.replace(obj, kind=kind, size=size, envelope_of="compound")
