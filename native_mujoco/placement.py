"""Runtime object placement onto the tabletop grid (R12-607, issue #38).

Puts an object on a named grid cell of a RUNNING simulation, without a model
recompile and without a restart.

WHY THIS IS POSSIBLE AT ALL, AND ONLY NOW
    MuJoCo freezes nbody/njnt/ngeom at compile time; there is no add_body().
    So "add an object" has to mean "move one that already exists", which needs
    the object's free joint to be addressable by name -- and free joints were
    emitted UNNAMED until 2026-09-02 (#37), so mj_name2id returned -1 and every
    pose write failed by doing nothing.  Naming them is what unlocked this.

    The second half is that a COMPILED model is more mutable than it looks:
    geom_size, geom_rgba and body_mass are all writable on a live model.  So a
    pool slot can be reshaped, recoloured and re-massed in place, and only a
    genuinely new mesh needs the atomic model swap that server.py still
    correctly refuses.

WHY OBJECTS PARK ON THE FLOOR AND NOT BELOW IT
    The floor is an infinite plane.  A body stowed at z = -0.5 is PENETRATING
    it and gets ejected: measured, one came back to z = +0.005 carrying
    2.4 m/s after 2000 steps.  On the floor, gravity holds it, it renders (so
    the pool is visibly available), and at that distance it is far outside the
    right arm's reach.

REACHABILITY IS READ, NOT DERIVED
    FWDCenterLabMCC measured reachability per cell with 40-restart DLS IK and
    recorded the result.  This module reports the shoulder distance -- and the
    tests assert those distances still match MCC's published table -- but takes
    reachable/not from the cell's own tags, because the IK result is the
    measurement and a sphere radius is only a proxy for it.

    Note for whoever revisits MCC: it used to state a 0.609 m maximum
    alongside a per-cell table that labelled cell_r3c3 at 0.622 m reachable
    while calling 0.649 m unreachable -- apparently self-contradictory if
    0.609 m were a hard sphere.  Issue #41 re-ran the sweep and confirmed the
    per-cell table (this module's source of truth) was right and the single
    "maximum" figure was the wrong framing: reach here is direction-dependent,
    not a sphere, so cell_r3c3's cell tags were never in doubt.  See MCC's
    header for the reproduction.  Deferring to the per-cell IK labels is what
    keeps this module from having to pick a side.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

GRID_CELL_TAG = "grid-cell"
POOL_TAG = "pool"
# Applied to a cell in the scene when IK could NOT reach it.  A tag rather than
# a schema field: tags are already free-form, so this needs no schema change and
# no compiler change, and it keeps the fact next to the geometry it describes.
UNREACHABLE_TAG = "no-right-arm"

# Right shoulder in world coordinates, and the height above the board that
# MCC's reachability sweep targeted.  Both are needed to reproduce its table.
SHOULDER_XYZ = (0.0, -0.19, 1.0)
GRASP_APPROACH_M = 0.05

# Objects are placed this far above their exact resting height, so they settle
# DOWN onto the board.  Placing at the exact height risks starting a hair
# inside it, and a penetrating contact is resolved by ejection.
_SETTLE_CLEARANCE_M = 0.001

_CELL_RE = re.compile(r"^cell_(r\d+c\d+)$")


class PlacementError(ValueError):
    """A placement that cannot be satisfied — unknown cell, unknown object,
    or a cell the right arm has been measured unable to reach."""


@dataclass(frozen=True)
class Cell:
    name: str            # "r2c2"
    object_id: str       # "cell_r2c2"
    x: float
    y: float
    top_z: float         # board surface — where an object's underside rests
    half_extent: float   # half the cell's smaller span, for occupancy tests
    reachable: bool
    shoulder_distance_m: float


@dataclass(frozen=True)
class Placement:
    """What a place/stow actually did — the record a client evaluates against."""
    object_id: str
    cell: Optional[str]
    position: Tuple[float, float, float]
    quat_wxyz: Tuple[float, float, float, float]
    yaw_deg: float
    stowed: bool

    def as_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "cell": self.cell,
            "position": list(self.position),
            "quat_wxyz": list(self.quat_wxyz),
            "yaw_deg": self.yaw_deg,
            "stowed": self.stowed,
        }


# ── Scene-side geometry (pure; no MuJoCo, runs in the offline test tier) ──────

def shoulder_distance(x: float, y: float, top_z: float) -> float:
    """Distance from the right shoulder to the grasp-approach point over a cell.

    Reproduces FWDCenterLabMCC's published table exactly; test_placement.py
    asserts that, so a change to the board geometry that silently invalidates
    the recorded reachability sweep fails the suite instead of going unnoticed.
    """
    sx, sy, sz = SHOULDER_XYZ
    return math.dist((x, y, top_z + GRASP_APPROACH_M), (sx, sy, sz))


def cells_from_scene(scene_doc: Mapping[str, Any]) -> Dict[str, Cell]:
    """Grid cells of a resolved scene document, keyed by short name ("r2c2")."""
    cells: Dict[str, Cell] = {}
    for obj in scene_doc.get("objects") or []:
        if not isinstance(obj, Mapping):
            continue
        tags = obj.get("tags") or []
        if GRID_CELL_TAG not in tags:
            continue
        m = _CELL_RE.match(str(obj.get("id", "")))
        if not m:
            continue
        pos = (obj.get("pose") or {}).get("position") or [0.0, 0.0, 0.0]
        size = (obj.get("geometry") or {}).get("size") or [0.0, 0.0, 0.0]
        # Scene sizes are FULL extents (scene_compiler halves them for MJCF),
        # so the cell's top surface is its centre plus half its thickness.
        top_z = float(pos[2]) + float(size[2]) / 2.0
        x, y = float(pos[0]), float(pos[1])
        cells[m.group(1)] = Cell(
            name=m.group(1),
            object_id=str(obj["id"]),
            x=x, y=y, top_z=top_z,
            half_extent=min(float(size[0]), float(size[1])) / 2.0,
            reachable=UNREACHABLE_TAG not in tags,
            shoulder_distance_m=shoulder_distance(x, y, top_z),
        )
    return cells


def pool_ids(scene_doc: Mapping[str, Any]) -> list:
    """Ids of objects the scene offers as placeable pool slots."""
    return [str(o["id"]) for o in (scene_doc.get("objects") or [])
            if isinstance(o, Mapping) and POOL_TAG in (o.get("tags") or [])]


def yaw_to_quat(yaw_deg: float) -> Tuple[float, float, float, float]:
    half = math.radians(yaw_deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def resolve_cell(cells: Mapping[str, Cell], cell: str,
                 allow_unreachable: bool = False) -> Cell:
    """Look a cell up by short name or full object id, and refuse an
    unreachable one by default with a message that says why."""
    key = cell[5:] if cell.startswith("cell_") else cell
    found = cells.get(key)
    if found is None:
        raise PlacementError(
            f"unknown cell {cell!r}; scene has {sorted(cells)}")
    if not found.reachable and not allow_unreachable:
        raise PlacementError(
            f"cell {found.name} is {found.shoulder_distance_m*100:.0f} cm from "
            f"the right shoulder and was measured UNREACHABLE by IK in "
            f"FWDCenterLabMCC (Siva confirmed physically).  Placing here gives "
            f"a target the arm cannot pick.  Pass allow_unreachable=True if "
            f"that is the point of the run — e.g. testing that a planner "
            f"declines rather than flails.")
    return found


# ── Runtime side (needs a compiled model) ────────────────────────────────────

try:                                    # keeps the geometry above importable
    import mujoco                       # in the offline test tier
except ImportError:                     # pragma: no cover
    mujoco = None                       # type: ignore[assignment]

import numpy as np


@dataclass
class _Slot:
    object_id: str
    body_id: int
    geom_id: int
    qpos_adr: int
    qvel_adr: int
    home_qpos: "np.ndarray"


class ObjectPlacer:
    """Places free-joint objects onto grid cells of a live model.

    Every free-joint object is placeable, not just the ones tagged `pool` —
    the tag advertises spare slots, it does not restrict the API.  Placing the
    scene's own cube on a chosen cell is the same operation.

    NOT THREAD-SAFE BY DESIGN.  It writes qpos and mutates the model, so it
    must be driven from whichever thread owns the physics — in server.py that
    is the sim thread, reached through a pending request exactly as
    zoom_command is, and for the same reason: mutating renderer or model state
    from the websocket task would race a step in progress.
    """

    def __init__(self, model, data, scene_doc: Mapping[str, Any]) -> None:
        if mujoco is None:                              # pragma: no cover
            raise RuntimeError("ObjectPlacer needs mujoco")
        self._m = model
        self._d = data
        self._cells = cells_from_scene(scene_doc)
        self._pool = pool_ids(scene_doc)
        self._slots: Dict[str, _Slot] = {}
        for jid in range(model.njnt):
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            bid = int(model.jnt_bodyid[jid])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if not name:
                continue
            adr = int(model.jnt_qposadr[jid])
            self._slots[name] = _Slot(
                object_id=name,
                body_id=bid,
                geom_id=int(model.body_geomadr[bid]),
                qpos_adr=adr,
                qvel_adr=int(model.jnt_dofadr[jid]),
                home_qpos=np.array(data.qpos[adr:adr + 7], dtype=float),
            )

    # -- introspection ------------------------------------------------------

    @property
    def cells(self) -> Dict[str, Cell]:
        return dict(self._cells)

    @property
    def placeable_ids(self) -> list:
        return sorted(self._slots)

    @property
    def pool_ids(self) -> list:
        return list(self._pool)

    def _slot(self, object_id: str) -> _Slot:
        slot = self._slots.get(object_id)
        if slot is None:
            raise PlacementError(
                f"{object_id!r} is not a placeable object; the scene's free-joint "
                f"objects are {self.placeable_ids}")
        return slot

    # -- geometry read back from the LIVE model, never from the scene doc ---
    #    so a reshaped object still lands at the right height.

    def _half_height(self, gid: int) -> float:
        t = self._m.geom_type[gid]
        s = self._m.geom_size[gid]
        G = mujoco.mjtGeom
        if t == G.mjGEOM_BOX:
            return float(s[2])
        if t == G.mjGEOM_CYLINDER:
            return float(s[1])
        if t == G.mjGEOM_CAPSULE:
            return float(s[0] + s[1])
        if t == G.mjGEOM_SPHERE:
            return float(s[0])
        return float(max(s))

    @staticmethod
    def _rbound_for(geom_type, size) -> float:
        G = mujoco.mjtGeom
        if geom_type == G.mjGEOM_BOX:
            return float(math.sqrt(sum(float(v) ** 2 for v in size[:3])))
        if geom_type == G.mjGEOM_CYLINDER:
            return float(math.hypot(float(size[0]), float(size[1])))
        if geom_type == G.mjGEOM_CAPSULE:
            return float(size[0] + size[1])
        if geom_type == G.mjGEOM_SPHERE:
            return float(size[0])
        return float(max(size))

    # -- mutation -----------------------------------------------------------

    def reshape(self, object_id: str, *, size: Optional[Sequence[float]] = None,
                radius: Optional[float] = None, length: Optional[float] = None,
                rgba: Optional[Sequence[float]] = None,
                mass: Optional[float] = None) -> None:
        """Resize / recolour / re-mass a slot on the live model.

        `size` is FULL extents and `length` a FULL length, matching the scene
        format; MuJoCo stores half, and this halves them.

        THE TRAP, AND WHY THIS METHOD EXISTS INSTEAD OF A BARE geom_size WRITE:
        geom_rbound is the broadphase bounding radius.  It is computed by the
        COMPILER and is not recomputed when geom_size changes — mj_setConst
        does not fix it either (verified: growing a half-size 0.015 -> 0.05 left
        rbound at 0.02598).  Grow a geom without updating rbound and the
        broadphase culls the pair before narrowphase ever runs, so contacts
        silently stop being generated.  That presents as the gripper passing
        through the object, with nothing in any log.
        """
        slot = self._slot(object_id)
        gid = slot.geom_id
        gtype = self._m.geom_type[gid]
        G = mujoco.mjtGeom

        # Reshaping the first geom of a multi-geom body would resize part of the
        # object and leave the rest, and half_height() would then describe
        # neither.  Refuse instead of doing half the job.
        if int(self._m.body_geomnum[slot.body_id]) != 1:
            raise PlacementError(
                f"{object_id!r} has {int(self._m.body_geomnum[slot.body_id])} "
                f"geoms; reshape only handles single-geom bodies")

        # geom_size means different things per primitive: [hx, hy, hz] for a box
        # but [radius, half-length] for a cylinder.  Writing `size` to a cylinder
        # would set its radius from hx and its half-length from hy, which is a
        # plausible-looking object of entirely the wrong shape.
        if size is not None and gtype != G.mjGEOM_BOX:
            raise PlacementError(
                f"{object_id!r} is not a box; use radius=/length= instead of size=")
        if (radius is not None or length is not None) and gtype == G.mjGEOM_BOX:
            raise PlacementError(
                f"{object_id!r} is a box; use size=[x, y, z] instead of "
                f"radius=/length=")
        if length is not None and gtype == G.mjGEOM_SPHERE:
            raise PlacementError(f"{object_id!r} is a sphere; it has no length")

        if size is not None:
            if len(size) != 3:
                raise PlacementError("size must have 3 elements (full extents)")
            self._m.geom_size[gid][:3] = [float(v) / 2.0 for v in size]
        if radius is not None:
            self._m.geom_size[gid][0] = float(radius)
        if length is not None:
            self._m.geom_size[gid][1] = float(length) / 2.0
        if size is not None or radius is not None or length is not None:
            self._m.geom_rbound[gid] = self._rbound_for(
                gtype, self._m.geom_size[gid])

        if rgba is not None:
            if len(rgba) != 4:
                raise PlacementError("rgba must have 4 elements")
            self._m.geom_rgba[gid][:] = [float(v) for v in rgba]
        if mass is not None:
            if float(mass) <= 0.0:
                raise PlacementError("mass must be > 0")
            self._m.body_mass[slot.body_id] = float(mass)

    def occupant_of(self, cell: Cell) -> Optional[str]:
        """Which placeable object, if any, is already sitting on `cell`."""
        for oid, slot in self._slots.items():
            q = self._d.qpos[slot.qpos_adr:slot.qpos_adr + 3]
            if (abs(float(q[0]) - cell.x) <= cell.half_extent
                    and abs(float(q[1]) - cell.y) <= cell.half_extent
                    and float(q[2]) > cell.top_z - 0.01):
                return oid
        return None

    def place(self, object_id: str, cell: str, yaw_deg: float = 0.0,
              allow_unreachable: bool = False,
              allow_occupied: bool = False) -> Placement:
        """Put `object_id` on `cell`, resting on the board.

        Refuses an occupied cell by default: two bodies spawned into the same
        space start deeply interpenetrating, and MuJoCo resolves that by
        launching them, which reads as a physics bug rather than as the caller's
        mistake.
        """
        slot = self._slot(object_id)
        c = resolve_cell(self._cells, cell, allow_unreachable)
        if not allow_occupied:
            held = self.occupant_of(c)
            if held is not None and held != object_id:
                raise PlacementError(
                    f"cell {c.name} already holds {held!r}; stow it first, or "
                    f"pass allow_occupied=True if a collision is the point")
        z = c.top_z + self._half_height(slot.geom_id) + _SETTLE_CLEARANCE_M
        pos = (c.x, c.y, z)
        quat = yaw_to_quat(yaw_deg)
        self._write(slot, pos, quat)
        return Placement(object_id, c.name, pos, quat, yaw_deg, stowed=False)

    def stow(self, object_id: str) -> Placement:
        """Return an object to the floor position it was compiled at."""
        slot = self._slot(object_id)
        q = slot.home_qpos
        pos = (float(q[0]), float(q[1]), float(q[2]))
        quat = (float(q[3]), float(q[4]), float(q[5]), float(q[6]))
        self._write(slot, pos, quat)
        return Placement(object_id, None, pos, quat, 0.0, stowed=True)

    def _write(self, slot: _Slot, pos, quat) -> None:
        adr = slot.qpos_adr
        self._d.qpos[adr:adr + 3] = pos
        self._d.qpos[adr + 3:adr + 7] = quat
        # Zero the velocity too: a slot that was falling, or that the arm had
        # just knocked, would otherwise arrive on the board already moving.
        self._d.qvel[slot.qvel_adr:slot.qvel_adr + 6] = 0.0
        mujoco.mj_forward(self._m, self._d)
