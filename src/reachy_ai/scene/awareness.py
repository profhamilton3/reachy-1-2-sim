"""
Preloaded scene awareness for Reachy 1.2 (mimics vision by loading a known scene).

The kinematic simulation backend has no physics — it will happily drive the arm
straight through the table because it has no notion of the table at all.  This
module gives the motion layer that missing knowledge: where the table surface
is, where each object rests, and whether a proposed gripper point would collide
with a static obstacle (the table) or dip below its surface.

Coordinates are in the scene ``frame_id`` (``pedestal``), which coincides with
the robot world frame.  The right-arm FK/IK frame maps to this frame by adding
(0, 0, 1.0) — the torso sits 1 m above the pedestal origin.  See
``ARM_BASE_IN_WORLD``.

All geometry is parsed directly from the scene YAML so this module has no ROS,
MuJoCo, or reachy_sdk dependency and is unit-testable on any host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

# Right-arm FK/IK frame origin expressed in world/pedestal coordinates.
ARM_BASE_IN_WORLD: Tuple[float, float, float] = (0.0, 0.0, 1.0)

XYZ = Tuple[float, float, float]


@dataclass(frozen=True)
class SceneObject:
    """A single scene object with enough geometry to plan a grasp."""
    id: str
    kind: str                      # box | cylinder | sphere | capsule | plane
    center: XYZ                    # world position of the object centre
    size: Tuple[float, float, float]  # full extents (box) or (2r, 2r, length)
    dynamic: bool
    tracked: bool
    tags: Tuple[str, ...] = ()
    semantic_class: Optional[str] = None
    quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
    # Interactive control metadata (None for plain objects).
    control_type: Optional[str] = None      # button | switch | lever
    articulation: Optional[dict] = None      # {joint, axis, range, handle_offset, ...}

    @property
    def is_control(self) -> bool:
        return self.control_type is not None

    @property
    def half_height(self) -> float:
        return self.size[2] / 2.0

    @property
    def top_z(self) -> float:
        return self.center[2] + self.half_height

    @property
    def bottom_z(self) -> float:
        return self.center[2] - self.half_height

    @property
    def footprint_radius(self) -> float:
        """Conservative XY radius for overlap checks."""
        return max(self.size[0], self.size[1]) / 2.0


@dataclass(frozen=True)
class CollisionViolation:
    """Reported when a proposed gripper point is unsafe."""
    kind: str          # "below_table" | "inside_object"
    obstacle_id: str
    point: XYZ
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.obstacle_id}: {self.detail} at {self.point}"


def _center_of(pose: dict) -> XYZ:
    pos = pose.get("position", [0.0, 0.0, 0.0])
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _quat_of(pose: dict) -> Tuple[float, float, float, float]:
    """World orientation (w,x,y,z) from a scene pose (orientation_wxyz or rpy)."""
    if "orientation_wxyz" in pose:
        q = [float(v) for v in pose["orientation_wxyz"]]
        return (q[0], q[1], q[2], q[3])
    if "rpy" in pose:
        r, p, y = (float(v) for v in pose["rpy"])
        cr, sr = math.cos(r / 2), math.sin(r / 2)
        cp, sp = math.cos(p / 2), math.sin(p / 2)
        cy, sy = math.cos(y / 2), math.sin(y / 2)
        return (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    return (1.0, 0.0, 0.0, 0.0)


def _rotate(q: Tuple[float, float, float, float], v: XYZ) -> XYZ:
    """Rotate vector v by unit quaternion q (w,x,y,z)."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(q_vec, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v' = v + w*t + cross(q_vec, t)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _cross(a: XYZ, b: XYZ) -> XYZ:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit(v: XYZ) -> XYZ:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


@dataclass(frozen=True)
class ControlTarget:
    """How the gripper should actuate one control (see SceneModel.control_target)."""
    id: str
    control_type: str            # button | switch | lever
    point: XYZ                   # cap centre (button) or handle tip (hinge)
    actuate_dir: XYZ             # unit world direction to turn the control ON
    off_dir: XYZ                 # unit world direction to turn it OFF
    preferred_arm: str           # right | left | either


def _full_extents(geo: dict) -> Tuple[float, float, float]:
    kind = geo.get("kind", "box")
    if kind == "box":
        sx, sy, sz = geo.get("size", [0.1, 0.1, 0.1])
        return float(sx), float(sy), float(sz)
    if kind == "cylinder":
        r = float(geo.get("radius", 0.05))
        h = float(geo.get("length", 0.1))
        return 2 * r, 2 * r, h
    if kind == "sphere":
        r = float(geo.get("radius", 0.05))
        return 2 * r, 2 * r, 2 * r
    if kind == "capsule":
        r = float(geo.get("radius", 0.05))
        l = float(geo.get("length", 0.1))
        return 2 * r, 2 * r, l + 2 * r
    if kind == "plane":
        sx, sy, _ = geo.get("size", [1.0, 1.0, 0.0])
        return float(sx), float(sy), 0.01
    sx, sy, sz = geo.get("size", [0.1, 0.1, 0.1])
    return float(sx), float(sy), float(sz)


class SceneModel:
    """Loaded knowledge of one tabletop scene."""

    # Default vertical clearance the gripper keeps above the table while
    # traversing (metres).  Grasp/place motions descend below this on purpose.
    CARRY_CLEARANCE = 0.10
    HOVER_CLEARANCE = 0.05

    def __init__(
        self,
        frame_id: str,
        objects: Sequence[SceneObject],
        table_id: Optional[str],
    ) -> None:
        self.frame_id = frame_id
        self._objects: Dict[str, SceneObject] = {o.id: o for o in objects}
        self._table_id = table_id

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "SceneModel":
        with open(path) as f:
            doc = yaml.safe_load(f)
        return cls.from_doc(doc)

    @classmethod
    def from_doc(cls, doc: dict) -> "SceneModel":
        frame_id = doc.get("frame_id", "pedestal")
        objects: List[SceneObject] = []
        table_id: Optional[str] = None
        for raw in doc.get("objects", []):
            geo = raw.get("geometry", {})
            phys = raw.get("physics", {})
            tags = tuple(raw.get("tags", []) or [])
            inter = raw.get("interactive") or {}
            obj = SceneObject(
                id=raw["id"],
                kind=geo.get("kind", "box"),
                center=_center_of(raw.get("pose", {})),
                size=_full_extents(geo),
                dynamic=bool(phys.get("dynamic", False)),
                tracked=bool(raw.get("tracked", False)),
                tags=tags,
                semantic_class=raw.get("semantic_class"),
                quat=_quat_of(raw.get("pose", {})),
                control_type=inter.get("type"),
                articulation=raw.get("articulation"),
            )
            objects.append(obj)
            if table_id is None and (
                "tabletop" in tags
                or raw.get("semantic_class", "").startswith("furniture.table")
            ):
                table_id = obj.id
        return cls(frame_id, objects, table_id)

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def objects(self) -> Dict[str, SceneObject]:
        return dict(self._objects)

    def get(self, object_id: str) -> SceneObject:
        return self._objects[object_id]

    def manipulable_ids(self) -> List[str]:
        """Tracked, dynamic objects (grasp targets), in scene order."""
        return [o.id for o in self._objects.values() if o.tracked and o.dynamic]

    @property
    def table(self) -> Optional[SceneObject]:
        return self._objects.get(self._table_id) if self._table_id else None

    @property
    def table_surface_z(self) -> float:
        """World z of the tabletop surface (top of the table box)."""
        t = self.table
        if t is None:
            return 0.0
        return t.top_z

    def static_obstacles(self) -> List[SceneObject]:
        # Controls are small and intentionally contacted by the gripper, so they
        # are NOT planning obstacles; large fixtures (console/table) are.
        return [
            o for o in self._objects.values()
            if not o.dynamic and not o.is_control
        ]

    def panel_obstacles(self) -> List[SceneObject]:
        """Console/panel fixture boxes the arm should route around."""
        return [
            o for o in self._objects.values()
            if not o.is_control and (
                "fixture" in o.tags or "panel" in o.tags
                or (o.semantic_class or "").startswith("furniture.")
            )
        ]

    # ── Interactive controls ───────────────────────────────────────────────────

    def controls(self) -> List[SceneObject]:
        """All operable controls (buttons/switches/levers), in scene order."""
        return [o for o in self._objects.values() if o.is_control]

    def preferred_arm(self, object_id: str, margin: float = 0.03) -> str:
        """'right' (y<0), 'left' (y>0), or 'either' (near the midline)."""
        y = self._objects[object_id].center[1]
        if y < -margin:
            return "right"
        if y > margin:
            return "left"
        return "either"

    def controls_for_arm(self, side: str, margin: float = 0.03) -> List[str]:
        """Control ids that ``side`` ('right'|'left') should operate.

        'either' (near-midline) controls are included for both sides.
        """
        out = []
        for o in self.controls():
            pref = self.preferred_arm(o.id, margin)
            if pref == side or pref == "either":
                out.append(o.id)
        return out

    def control_target(self, object_id: str) -> "ControlTarget":
        """Where and how the gripper actuates a control.

        * button — ``point`` is the cap centre; ``actuate_dir`` is the world
          press direction (into the button, along its slide axis); ``off_dir``
          is the reverse (release/out).
        * switch / lever — ``point`` is the handle tip (centre + rotated
          handle_offset); ``actuate_dir`` is the tangential push that turns it
          ON; ``off_dir`` the tangent that turns it OFF.
        """
        o = self._objects[object_id]
        art = o.articulation or {}
        axis_local = tuple(float(v) for v in (art.get("axis") or [0.0, 0.0, 1.0]))
        if o.control_type == "button":
            # Press travels along -axis_world (slide range is negative).
            axis_world = _rotate(o.quat, axis_local)
            press = _unit((-axis_world[0], -axis_world[1], -axis_world[2]))
            point = o.center
            return ControlTarget(
                id=o.id, control_type="button", point=point,
                actuate_dir=press, off_dir=(-press[0], -press[1], -press[2]),
                preferred_arm=self.preferred_arm(o.id),
            )
        # Hinge control: handle tip and tangential push directions.
        off = tuple(float(v) for v in (art.get("handle_offset") or [0.0, 0.0, 0.0]))
        r_world = _rotate(o.quat, off)                      # pivot -> tip
        tip = (o.center[0] + r_world[0], o.center[1] + r_world[1], o.center[2] + r_world[2])
        axis_world = _rotate(o.quat, axis_local)
        # Tangent that increases the joint angle (toward ON) = axis × r.
        tangent = _unit(_cross(axis_world, r_world))
        return ControlTarget(
            id=o.id, control_type=o.control_type, point=tip,
            actuate_dir=tangent, off_dir=(-tangent[0], -tangent[1], -tangent[2]),
            preferred_arm=self.preferred_arm(o.id),
        )

    # ── Grasp / place geometry ────────────────────────────────────────────────

    def grasp_point(self, object_id: str) -> XYZ:
        """World point the gripper FK origin should reach to grasp the object
        (its centre — the object marker is attached here during carry)."""
        c = self._objects[object_id].center
        return (c[0], c[1], c[2])

    def hover_point(self, object_id: str, clearance: Optional[float] = None) -> XYZ:
        """A point directly above the object, clear of its top."""
        o = self._objects[object_id]
        clr = self.HOVER_CLEARANCE if clearance is None else clearance
        return (o.center[0], o.center[1], o.top_z + clr)

    def carry_z(self) -> float:
        """Safe gripper height for traversing above the table."""
        return self.table_surface_z + self.CARRY_CLEARANCE

    def rest_point(self, xy: Tuple[float, float], object_id: str) -> XYZ:
        """World point to release ``object_id`` so it rests on the table at xy
        (gripper FK origin = object centre when its base touches the surface)."""
        o = self._objects[object_id]
        return (xy[0], xy[1], self.table_surface_z + o.half_height)

    # ── Collision awareness ───────────────────────────────────────────────────

    def check_point(
        self,
        point: XYZ,
        surface_margin: float = 0.005,
        ignore: Sequence[str] = (),
    ) -> Optional[CollisionViolation]:
        """Return a CollisionViolation if a gripper point is unsafe, else None.

        Two checks:
          * below_table  — point is under the tabletop surface within its XY
            footprint (the arm would drive through the table).
          * inside_object — point is inside the bounding volume of a *static*
            obstacle other than the table.
        """
        x, y, z = point
        t = self.table
        if t is not None and t.id not in ignore:
            hx, hy = t.size[0] / 2.0, t.size[1] / 2.0
            within_xy = (
                abs(x - t.center[0]) <= hx and abs(y - t.center[1]) <= hy
            )
            if within_xy and z < self.table_surface_z - surface_margin:
                return CollisionViolation(
                    "below_table", t.id, point,
                    f"z={z:.3f} below surface {self.table_surface_z:.3f}",
                )
        for obs in self.static_obstacles():
            if obs.id == self._table_id or obs.id in ignore:
                continue
            if (
                abs(x - obs.center[0]) <= obs.size[0] / 2.0
                and abs(y - obs.center[1]) <= obs.size[1] / 2.0
                and abs(z - obs.center[2]) <= obs.size[2] / 2.0
            ):
                return CollisionViolation(
                    "inside_object", obs.id, point,
                    f"inside static obstacle bounds",
                )
        return None

    def validate_path(
        self,
        points: Sequence[XYZ],
        ignore: Sequence[str] = (),
        surface_margin: float = 0.005,
    ) -> List[CollisionViolation]:
        """Check every point of a planned trajectory; return all violations."""
        out: List[CollisionViolation] = []
        for p in points:
            v = self.check_point(p, surface_margin=surface_margin, ignore=ignore)
            if v is not None:
                out.append(v)
        return out
