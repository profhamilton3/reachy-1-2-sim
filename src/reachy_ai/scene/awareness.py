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
from dataclasses import dataclass, field, replace
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
    # False only for scene objects with physics.collision == false (markings,
    # placement targets, calibration decals).  "fixture" counts as colliding.
    collides: bool = True

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
class Clearance:
    """Closest approach between one arm segment and one scene object.

    ``distance`` is a true signed distance in metres: positive is air between
    the link's surface and the object's surface, negative means they overlap by
    that much.  ``point`` is where on the link the closest approach happened,
    which is what tells you *which part of the arm* is the problem.
    """
    distance: float
    object_id: str
    link: str
    point: XYZ

    def __str__(self) -> str:
        verb = "clears" if self.distance >= 0 else "OVERLAPS"
        return (f"{self.link} {verb} {self.object_id} by "
                f"{abs(self.distance) * 100:.1f} cm")


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


# ── Signed distance to a single object ────────────────────────────────────────
#
# `check_point` below answers "is this point inside something?".  That is not
# enough to keep the arm off the table: the arm is a set of long links, not a
# point, and the link that reaches an object is usually not the one being
# watched.  The motion notebook's section 4.3 sweeps the WRIST and used to push
# red_cube across the board with the FOREARM, somewhere the gripper pad never
# went.  The functions here answer the harder question — how far is a whole
# link from an object — so a move can be refused or shortened before it is
# commanded, rather than diagnosed from the wreckage afterwards.
#
# Both primitives are exact outside the solid and use the standard signed
# distance field inside it, so a report of "-1.2 cm" means a real 1.2 cm of
# interpenetration rather than an arbitrary negative number.


def _box_sdf(half: XYZ, p: XYZ) -> float:
    """Signed distance from p to an origin-centred axis-aligned box."""
    q = (abs(p[0]) - half[0], abs(p[1]) - half[1], abs(p[2]) - half[2])
    outside = math.sqrt(sum(max(v, 0.0) ** 2 for v in q))
    inside = min(max(q[0], q[1], q[2]), 0.0)
    return outside + inside


def _cylinder_sdf(radius: float, half_height: float, p: XYZ) -> float:
    """Signed distance from p to an origin-centred upright cylinder."""
    dr = math.hypot(p[0], p[1]) - radius
    dz = abs(p[2]) - half_height
    outside = math.hypot(max(dr, 0.0), max(dz, 0.0))
    inside = min(max(dr, dz), 0.0)
    return outside + inside


def object_sdf(obj: SceneObject, point: XYZ) -> float:
    """Signed distance from a world point to ``obj``'s bounding solid.

    Boxes are evaluated in their own frame (so a yawed box is not inflated to
    its world-axis-aligned bounds); cylinders, capsules and spheres are treated
    as upright cylinders of ``footprint_radius``.  Rounding a cylinder's ends
    into a capsule would add a phantom dome of up to ``radius`` above the
    object's real top — on a 6 cm cube that is 3 cm of imaginary height, enough
    to make an honest clearance report read as a collision.
    """
    rel = (point[0] - obj.center[0],
           point[1] - obj.center[1],
           point[2] - obj.center[2])
    w, x, y, z = obj.quat
    local = _rotate((w, -x, -y, -z), rel)          # world -> object frame
    if obj.kind == "box" or obj.kind == "plane":
        half = (obj.size[0] / 2.0, obj.size[1] / 2.0, obj.size[2] / 2.0)
        return _box_sdf(half, local)
    return _cylinder_sdf(obj.footprint_radius, obj.half_height, local)


def segment_object_distance(
    obj: SceneObject, p0: XYZ, p1: XYZ, radius: float = 0.0,
) -> Tuple[float, XYZ]:
    """Closest approach between a capsule (p0→p1, ``radius``) and ``obj``.

    Returns (signed distance, closest point on the segment axis).  The object's
    SDF is convex, so its restriction to the segment is convex too and a coarse
    scan followed by a golden-section refinement finds the true minimum — no
    sampling resolution to tune, and no minimum missed between samples.
    """
    def at(t: float) -> XYZ:
        return (p0[0] + t * (p1[0] - p0[0]),
                p0[1] + t * (p1[1] - p0[1]),
                p0[2] + t * (p1[2] - p0[2]))

    n = 8
    ts = [i / n for i in range(n + 1)]
    vals = [object_sdf(obj, at(t)) for t in ts]
    k = min(range(len(ts)), key=lambda i: vals[i])
    lo, hi = ts[max(k - 1, 0)], ts[min(k + 1, n)]
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo + (1 - inv_phi) * (hi - lo), lo + inv_phi * (hi - lo)
    fa, fb = object_sdf(obj, at(a)), object_sdf(obj, at(b))
    for _ in range(24):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = lo + (1 - inv_phi) * (hi - lo)
            fa = object_sdf(obj, at(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + inv_phi * (hi - lo)
            fb = object_sdf(obj, at(b))
    t = 0.5 * (lo + hi)
    return object_sdf(obj, at(t)) - radius, at(t)


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
                collides=phys.get("collision", True) is not False,
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
        # Non-colliding statics (tape markings, grid cells, calibration decals)
        # have no physical presence, so they are not obstacles either.
        return [
            o for o in self._objects.values()
            if not o.dynamic and not o.is_control and o.collides
        ]

    def panel_obstacles(self) -> List[SceneObject]:
        """Console/panel fixture boxes the arm should route around."""
        return [
            o for o in self._objects.values()
            if not o.is_control and o.collides and (
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

    # ── Taped grid cells ──────────────────────────────────────────────────────

    def grid_cells(self) -> List[str]:
        """IDs of the taped grid's addressable cells, sorted by id.

        Cells are non-colliding, non-dynamic markers flush with the tabletop
        (see scenes/FWDCenterLabMCC.yaml).  Scenes without a grid return [].
        """
        return sorted(
            o.id for o in self._objects.values() if "grid-cell" in o.tags
        )

    def cell_center(self, cell_id: str) -> XYZ:
        """World point at the centre of a grid cell, on the table surface.

        The z returned is the tabletop surface — not the marker's own z, which
        sits a fraction of a millimetre above it so the decal renders.  Pair
        with ``rest_point`` to place an object standing in the cell.
        """
        o = self._objects[cell_id]
        if "grid-cell" not in o.tags:
            raise KeyError(f"{cell_id!r} is not a grid cell")
        return (o.center[0], o.center[1], self.table_surface_z)

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

    # ── Keeping the model honest ──────────────────────────────────────────────

    def update_poses(self, poses: Dict[str, Sequence[float]]) -> List[str]:
        """Move objects to new world centres; returns the ids actually moved.

        A SceneModel loaded from YAML knows where its objects were *placed*.
        Objects move — the robot moves them on purpose, and until the motion
        notebook's section 4 was fixed it moved them by accident.  A clearance
        check run against a stale position is the same class of error as
        checking the gripper pad instead of the whole arm: the model and the
        world disagree, and the model wins an argument it should lose.  Feed
        this the live poses (under mujoco-remote the container mirrors tracked
        objects into /tmp/reachy_scene_overrides.json at 15 Hz) before asking
        for a clearance that a robot is about to act on.

        Unknown ids are ignored rather than raising: the live feed carries
        whatever the simulator is tracking, which need not match this scene.
        """
        moved: List[str] = []
        for oid, center in poses.items():
            obj = self._objects.get(oid)
            if obj is None:
                continue
            c = (float(center[0]), float(center[1]), float(center[2]))
            if c != obj.center:
                self._objects[oid] = replace(obj, center=c)
                moved.append(oid)
        return moved

    # ── Whole-arm clearance ───────────────────────────────────────────────────

    def obstacle_ids(
        self, include_static: bool = False, include_table: bool = False,
    ) -> List[str]:
        """Objects a moving link must not touch.

        Defaults to the manipulable objects alone — the ones that get knocked
        over.  ``static_obstacles()`` deliberately excludes them (they are grasp
        targets, not scenery), which is why nothing in the planner noticed the
        arm sweeping them off the board.

        ``include_static`` adds the rig fixtures.  It does NOT add the tabletop:
        that is a separate opt-in, because the tabletop is a SUPPORT SURFACE
        rather than an obstacle.  Working on it is the arm's whole job, and the
        poses that do are already inside it.  Measured across the motion
        notebook's placement route, whose waypoints were verified against the
        rig at 0.2 deg resolution:

            REST     hand      -2.1 cm  vs the tabletop — it is resting on it
            HOVER    upper arm +1.4 cm  vs the tabletop
            SWING_1  hand      +0.9 cm  vs rig_rail_outer_right (worst rail)

        So a guard that counts the tabletop refuses every pose that reaches
        across the board — it blocked all nine grid cells in section 4.7 of the
        notebook, none of them for a reason involving an object.  The rails are
        the opposite case: nothing on the verified route comes closer than
        9 mm, so they are a real no-go volume and worth checking.
        """
        ids = list(self.manipulable_ids())
        if include_static:
            ids += [o.id for o in self.static_obstacles()
                    if o.id not in ids and o.id != self._table_id]
        if include_table and self._table_id and self._table_id not in ids:
            ids.append(self._table_id)
        return ids

    def clearance(
        self,
        segments: Sequence[Tuple[str, XYZ, XYZ, float]],
        ids: Optional[Sequence[str]] = None,
        include_static: bool = False,
        include_table: bool = False,
    ) -> Optional[Clearance]:
        """Closest approach between any arm segment and any tracked object.

        ``segments`` are (link name, end, end, radius) capsules in world
        coordinates — see ``motion.kinematics.link_capsules``.  Returns the
        single worst Clearance, or None if there is nothing to check.
        """
        want = (list(ids) if ids is not None
                else self.obstacle_ids(include_static, include_table))
        worst: Optional[Clearance] = None
        for oid in want:
            obj = self._objects.get(oid)
            if obj is None or not obj.collides:
                continue
            for name, p0, p1, radius in segments:
                d, at = segment_object_distance(obj, p0, p1, radius)
                if worst is None or d < worst.distance:
                    worst = Clearance(d, oid, name, at)
        return worst

    def clearances(
        self,
        segments: Sequence[Tuple[str, XYZ, XYZ, float]],
        ids: Optional[Sequence[str]] = None,
        include_static: bool = False,
        include_table: bool = False,
    ) -> Dict[str, Clearance]:
        """Per-object worst clearance — the detail behind ``clearance()``."""
        want = (list(ids) if ids is not None
                else self.obstacle_ids(include_static, include_table))
        out: Dict[str, Clearance] = {}
        for oid in want:
            obj = self._objects.get(oid)
            if obj is None or not obj.collides:
                continue
            for name, p0, p1, radius in segments:
                d, at = segment_object_distance(obj, p0, p1, radius)
                if oid not in out or d < out[oid].distance:
                    out[oid] = Clearance(d, oid, name, at)
        return out
