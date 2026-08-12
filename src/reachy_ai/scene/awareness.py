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
    CARRY_CLEARANCE = 0.13
    HOVER_CLEARANCE = 0.06

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
            obj = SceneObject(
                id=raw["id"],
                kind=geo.get("kind", "box"),
                center=_center_of(raw.get("pose", {})),
                size=_full_extents(geo),
                dynamic=bool(phys.get("dynamic", False)),
                tracked=bool(raw.get("tracked", False)),
                tags=tags,
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
        return [o for o in self._objects.values() if not o.dynamic]

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
