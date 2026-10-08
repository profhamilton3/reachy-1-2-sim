"""Jaw-gap alignment and pad clearance for an elevated, jaws-down grasp (#55).

Why this exists
---------------
Two recorded crane-style attempts on the 6 cm red_cube reached the elevated,
near palm-down posture and still never straddled the cube.  At the -65 deg
opening the open pads have about 2.4 mm of air per side around a 6 cm cube;
the realised hand sat 4-5 mm off the commanded one (shoulder pitch settling
~1.1 deg short under gravity), and a TILTED pad's lower corner reached the
cube's top edge first: the finger on the x-face attempt, the thumb on the
y-face attempt.  Every check before motion had been made against the
COMMANDED pose, and the tube hand model in ``check_arm_path`` cannot see a
millimetre-scale pad problem at all (it is a single ~11 cm capsule).

So the questions this module answers are asked of the hand that is actually
there:

  * where the two collision pads are, as oriented boxes, from MEASURED joint
    positions and the MEASURED gripper opening (``pad_boxes``);
  * how far each pad is from the object, and which side of it it is on
    (``pad_clearance``);
  * whether both pads have a clear route past the object's top edges all the
    way down to grasp depth, along a predicted realised descent
    (``route_clearance``) -- centring at the hover alone is not enough,
    because the pads are tilted and the hand's orientation drifts with height;
  * whether the pads actually straddle the object before the gripper closes
    (``straddle_check``);
  * the arm and pads against the TABLE (``table_clearance``), which the
    whole-arm preflight deliberately does not cover, and a carried object
    against the table, the rails and the other objects
    (``carried_clearance``).

Geometry is the MJCF's, not re-measured: the pads are ``r_thumb_col`` and
``r_finger_col`` exactly as ``kinematics`` already records them (and as
``tests/unit/test_grasp_alignment.py`` checks against the compiled model),
and the chain is ``kinematics.link_frames``.  Object geometry is the scene
model's.  Nothing here talks to the robot; it is numpy only, so it runs on
the host and in the container alike.

What a clearance number here is
-------------------------------
Box-box distances use the separating-axis test.  A positive value is a LOWER
BOUND on the true gap (exact when the closest features are parallel faces,
which is the case that matters: a pad beside a cube face); a negative value
is minus the smallest penetration depth over the tested axes.  It never
reports air that is not there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import kinematics as K
from ..scene.awareness import SceneModel, SceneObject, segment_object_distance

#: The collision pads, (centre, half-extents) in their own body frames, read
#: from native_mujoco/model/reachy_1_2.xml (the same values kinematics keeps
#: in _THUMB_FRAME_BOXES / _FINGER_FRAME_BOXES).
THUMB_PAD = K._THUMB_FRAME_BOXES[1]
FINGER_PAD = K._FINGER_FRAME_BOXES[1]
assert THUMB_PAD[0] == "r_thumb_col" and FINGER_PAD[0] == "r_finger_col"

#: Links checked against the table.  The hand's only collision geoms in the
#: MJCF are the two pads, so the hand is checked as those two boxes, not as
#: the tube.
TABLE_LINKS = ("upper_arm", "forearm")


# ── Oriented boxes ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Box:
    """An oriented box: ``axes`` columns are its local x, y, z in world."""
    name: str
    center: np.ndarray
    axes: np.ndarray
    half: np.ndarray

    def corners(self) -> np.ndarray:
        s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                      for sz in (-1, 1)], dtype=float)
        return self.center + (s * self.half) @ self.axes.T

    @property
    def highest_z(self) -> float:
        return float(self.center[2] + np.abs(self.axes[2]) @ self.half)

    @property
    def lowest_z(self) -> float:
        return float(self.center[2] - np.abs(self.axes[2]) @ self.half)

    def moved(self, rot: np.ndarray, trans: np.ndarray) -> "Box":
        """This box under the rigid motion x -> rot @ x + trans."""
        return Box(self.name, rot @ self.center + trans, rot @ self.axes,
                   self.half)


def obb_separation(a: Box, b: Box) -> float:
    """Signed separating-axis distance between two oriented boxes (metres).

    Positive: a lower bound on the gap.  Negative: minus the smallest overlap
    over the 15 SAT axes (the penetration depth when the boxes intersect).
    """
    t = b.center - a.center
    axes = [a.axes[:, i] for i in range(3)] + [b.axes[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            c = np.cross(a.axes[:, i], b.axes[:, j])
            n = np.linalg.norm(c)
            if n > 1e-9:
                axes.append(c / n)
    best = -math.inf
    for ax in axes:
        ra = float(np.abs(a.axes.T @ ax) @ a.half)
        rb = float(np.abs(b.axes.T @ ax) @ b.half)
        best = max(best, abs(float(t @ ax)) - ra - rb)
    return best


def _quat_to_rot(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def object_box(obj: SceneObject, center: Optional[Sequence[float]] = None,
               quat: Optional[Sequence[float]] = None) -> Box:
    """The scene object as an oriented box.

    A box is itself.  A cylinder/sphere/capsule is its bounding box (size is
    already (2r, 2r, h) in the scene model), which can only under-report air.
    ``center``/``quat`` override the scene pose with a live one.
    """
    c = np.asarray(obj.center if center is None else center, dtype=float)
    R = _quat_to_rot(obj.quat if quat is None else quat)
    return Box(obj.id, c, R, np.asarray(obj.size, dtype=float) / 2.0)


# ── The hand, from joint positions ────────────────────────────────────────────

@dataclass(frozen=True)
class HandFrames:
    wrist: np.ndarray
    thumb_rot: np.ndarray        # r_gripper_thumb frame (rides wrist roll)
    finger_rot: np.ndarray       # r_gripper_finger frame (also the gripper)
    thumb: Box                   # r_thumb_col
    finger: Box                  # r_finger_col
    thumb_face: np.ndarray       # centre of the thumb pad's inner face
    finger_face: np.ndarray      # centre of the finger pad's inner face

    @property
    def gap_mid(self) -> np.ndarray:
        """The jaw-gap centre: midway between the two inner pad faces."""
        return 0.5 * (self.thumb_face + self.finger_face)

    @property
    def jaw(self) -> np.ndarray:
        """Unit vector from the thumb's inner face to the finger's."""
        v = self.finger_face - self.thumb_face
        return v / np.linalg.norm(v)

    @property
    def approach(self) -> np.ndarray:
        """Unit vector down the hand (the thumb frame's -Z)."""
        return -self.thumb_rot[:, 2]

    @property
    def tilt_from_vertical_deg(self) -> float:
        return math.degrees(math.acos(max(-1.0, min(1.0, -self.approach[2]))))


def hand_frames(joints: Sequence[float], gripper_deg: float) -> HandFrames:
    """The two collision pads, in world, for a right-arm pose.

    ``joints`` are the seven arm joints (deg) and ``gripper_deg`` the gripper
    angle (deg) -- pass MEASURED values to ask about the hand that is there.
    The chain is ``kinematics.link_frames`` term for term; the pad offsets are
    the MJCF's (r_wrist2hand -> r_gripper_thumb -> r_gripper_finger).
    """
    q = np.radians(np.asarray(list(joints)[:7], dtype=float))
    _s, _e, wrist, _R = K.link_frames(joints, "right")
    R = K._roty(q[0]) @ K._rotx(q[1]) @ K._rotz(q[2])
    R = R @ K._roty(q[3]) @ K._rotz(q[4])
    r_pitch = R @ K._roty(q[5])
    r_thumb = r_pitch @ K._rotx(q[6])
    thumb_origin = wrist + r_pitch @ K._THUMB_ORIGIN_OFFSET
    r_finger = r_thumb @ K._rotx(math.radians(float(gripper_deg)))
    finger_origin = thumb_origin + r_thumb @ K._FINGER_HINGE_OFFSET

    _n, tc, th = THUMB_PAD
    _n, fc, fh = FINGER_PAD
    tc, th, fc, fh = (np.asarray(v, dtype=float) for v in (tc, th, fc, fh))
    thumb = Box("thumb_pad", thumb_origin + r_thumb @ tc, r_thumb, th)
    finger = Box("finger_pad", finger_origin + r_finger @ fc, r_finger, fh)
    # Inner faces: the thumb pad faces -y (toward the finger, which hangs at
    # y = -0.037); the finger pad faces +y (back toward the thumb).
    thumb_face = thumb_origin + r_thumb @ (tc - np.array([0.0, th[1], 0.0]))
    finger_face = finger_origin + r_finger @ (fc + np.array([0.0, fh[1], 0.0]))
    return HandFrames(wrist, r_thumb, r_finger, thumb, finger, thumb_face,
                      finger_face)


def hand_rotation(joints: Sequence[float]) -> np.ndarray:
    """The IK/FK frame's rotation (r_wrist2hand) -- what inverse_kinematics
    takes as its target orientation."""
    return K.link_frames(joints, "right")[3]


# ── Alignment of the jaw with an object ───────────────────────────────────────

@dataclass(frozen=True)
class PadClearance:
    """Where the open jaws are relative to one box-shaped object.

    ``face_normal`` is the object's horizontal axis closest to the jaw, signed
    to point from the thumb's side to the finger's.  ``thumb_side_m`` /
    ``finger_side_m`` are how far the WHOLE pad lies outside the object's face
    along that normal (negative: some corner is inside the face plane) -- the
    lateral room the pad needs to pass the top edge.  ``thumb_m`` /
    ``finger_m`` are the full box-box clearances.
    """
    thumb_m: float
    finger_m: float
    thumb_side_m: float
    finger_side_m: float
    gap_offset_normal_m: float     # gap centre - object centre, along the normal
    gap_offset_along_m: float      # ... along the face (the other horizontal axis)
    gap_above_center_m: float      # gap centre height above the object centre
    thumb_center_below_top_m: float
    finger_center_below_top_m: float
    jaw_face_angle_deg: float      # jaw vs the face normal, in the horizontal
    tilt_from_vertical_deg: float
    face_normal: Tuple[float, float, float]
    face_along: Tuple[float, float, float]

    @property
    def min_m(self) -> float:
        return min(self.thumb_m, self.finger_m)

    def as_dict(self) -> Dict[str, float]:
        mm = lambda v: round(1000.0 * v, 2)
        return {
            "thumb_mm": mm(self.thumb_m), "finger_mm": mm(self.finger_m),
            "thumb_side_mm": mm(self.thumb_side_m),
            "finger_side_mm": mm(self.finger_side_m),
            "gap_offset_normal_mm": mm(self.gap_offset_normal_m),
            "gap_offset_along_mm": mm(self.gap_offset_along_m),
            "gap_above_center_mm": mm(self.gap_above_center_m),
            "thumb_center_below_top_mm": mm(self.thumb_center_below_top_m),
            "finger_center_below_top_mm": mm(self.finger_center_below_top_m),
            "jaw_face_angle_deg": round(self.jaw_face_angle_deg, 2),
            "tilt_from_vertical_deg": round(self.tilt_from_vertical_deg, 2),
        }


def face_axes(obj: Box, jaw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """(normal, along, index): the object's horizontal axis nearest the jaw,
    signed thumb -> finger, the other horizontal axis, and the normal's index
    into ``obj.axes``/``obj.half``."""
    horiz = [i for i in range(3) if abs(obj.axes[2, i]) < 0.7]
    i = max(horiz, key=lambda k: abs(float(obj.axes[:, k] @ jaw)))
    n = obj.axes[:, i] * (1.0 if float(obj.axes[:, i] @ jaw) >= 0 else -1.0)
    j = next(k for k in horiz if k != i)
    return n, obj.axes[:, j], i


def pad_clearance(hand: HandFrames, obj: Box) -> PadClearance:
    n, m, i = face_axes(obj, hand.jaw)
    h_n = float(obj.half[i])
    c = obj.center
    thumb_proj = (hand.thumb.corners() - c) @ n
    finger_proj = (hand.finger.corners() - c) @ n
    top = float(c[2] + np.abs(obj.axes[2]) @ obj.half)
    jh = hand.jaw - hand.jaw[2] * np.array([0.0, 0.0, 1.0])
    jh = jh / (np.linalg.norm(jh) or 1.0)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(jh @ n))))))
    g = hand.gap_mid - c
    return PadClearance(
        thumb_m=obb_separation(hand.thumb, obj),
        finger_m=obb_separation(hand.finger, obj),
        thumb_side_m=-float(thumb_proj.max()) - h_n,
        finger_side_m=float(finger_proj.min()) - h_n,
        gap_offset_normal_m=float(g @ n),
        gap_offset_along_m=float(g @ m),
        gap_above_center_m=float(g[2]),
        thumb_center_below_top_m=top - float(hand.thumb.center[2]),
        finger_center_below_top_m=top - float(hand.finger.center[2]),
        jaw_face_angle_deg=ang,
        tilt_from_vertical_deg=hand.tilt_from_vertical_deg,
        face_normal=tuple(float(v) for v in n),
        face_along=tuple(float(v) for v in m),
    )


# ── The tracking-error model and the insertion route ──────────────────────────

def joint_offset(measured: Sequence[float], commanded: Sequence[float]) -> np.ndarray:
    """measured - commanded, the seven arm joints (deg).

    The recorded crane attempts held this nearly constant over a 9 cm
    descent (shoulder pitch 1.06-1.12 deg short at every rung), so the
    realised pose for a nearby commanded one is predicted as
    ``commanded + offset``; ``route_clearance`` is only ever asked about
    poses close to where the offset was last measured, and the prediction's
    own recorded error is what the caller's tolerance is made of.
    """
    return (np.asarray(list(measured)[:7], dtype=float)
            - np.asarray(list(commanded)[:7], dtype=float))


@dataclass(frozen=True)
class RouteReport:
    thumb_m: float               # worst thumb clearance anywhere on the route
    finger_m: float
    thumb_at: int                # sample index of each worst
    finger_at: int
    end: PadClearance            # at the last sample (grasp depth)
    samples: int

    @property
    def min_m(self) -> float:
        return min(self.thumb_m, self.finger_m)

    def as_dict(self) -> Dict[str, object]:
        return {"route_thumb_mm": round(1000 * self.thumb_m, 2),
                "route_finger_mm": round(1000 * self.finger_m, 2),
                "route_thumb_at": self.thumb_at,
                "route_finger_at": self.finger_at,
                "samples": self.samples, "end": self.end.as_dict()}


def densify(waypoints: Sequence[Sequence[float]], max_step_deg: float = 0.25
            ) -> List[List[float]]:
    """The joint-space lines between waypoints, sampled <= max_step_deg."""
    out: List[List[float]] = [list(waypoints[0])[:7]]
    for a, b in zip(waypoints, waypoints[1:]):
        n = max(2, math.ceil(max(abs(float(y) - float(x)) for x, y in
                                 zip(list(a)[:7], list(b)[:7])) / max_step_deg) + 1)
        out += [list(p) for p in K.joint_path(a, b, n)[1:]]
    return out


def route_clearance(route: Sequence[Sequence[float]], gripper_deg: float,
                    obj: Box, offset: Optional[np.ndarray] = None,
                    max_step_deg: float = 0.25) -> RouteReport:
    """Worst thumb and finger clearance to ``obj`` along a commanded route.

    ``route`` is the commanded joint waypoints, top to bottom; the lines
    between them are sampled at ``max_step_deg`` (0.25 deg moves a pad by
    well under 1 mm at this reach).  ``offset`` (measured - commanded) turns
    each commanded pose into the predicted realised one.
    """
    off = np.zeros(7) if offset is None else np.asarray(offset, dtype=float)
    pts = densify(route, max_step_deg)
    tw, fw, ti, fi = math.inf, math.inf, -1, -1
    end = None
    for k, q in enumerate(pts):
        hand = hand_frames(np.asarray(q) + off, gripper_deg)
        t = obb_separation(hand.thumb, obj)
        f = obb_separation(hand.finger, obj)
        if t < tw:
            tw, ti = t, k
        if f < fw:
            fw, fi = f, k
        if k == len(pts) - 1:
            end = pad_clearance(hand, obj)
    return RouteReport(tw, fw, ti, fi, end, len(pts))


def centring_shift(end: PadClearance, route: RouteReport,
                   target_gap_above_center_m: float) -> np.ndarray:
    """World shift to apply to the commanded jaw-gap target.

    Along the face normal: equalise the two pads' worst ROUTE clearances
    (moving +n gives the finger more room and the thumb less, one for one).
    Along the face: put the gap centre on the object's centre line.
    Vertically: bring the realised gap centre to the planned grasp height.
    """
    n = np.asarray(end.face_normal)
    m = np.asarray(end.face_along)
    s_n = 0.5 * (route.thumb_m - route.finger_m)
    s_m = -end.gap_offset_along_m
    s_z = target_gap_above_center_m - end.gap_above_center_m
    return s_n * n + s_m * m + np.array([0.0, 0.0, s_z])


# ── Straddle: may the gripper close? ──────────────────────────────────────────

@dataclass(frozen=True)
class StraddleCheck:
    ok: bool
    failed: Tuple[str, ...]
    clearance: PadClearance

    def as_dict(self) -> Dict[str, object]:
        return {"ok": self.ok, "failed": list(self.failed),
                **self.clearance.as_dict()}


def straddle_check(hand: HandFrames, obj: Box, table_z: float,
                   min_side_m: float, max_jaw_face_deg: float) -> StraddleCheck:
    """Do the open pads straddle the object, ready to close on two faces?

    Every condition is geometric, evaluated on the MEASURED hand:
      * each pad lies wholly outside its face (``*_side_m >= min_side_m``)
        and clear of the object (box clearance > 0) -- on OPPOSITE faces by
        construction of ``face_axes``;
      * each pad's centre is below the object's top, so closing presses pad
        on face rather than pad on edge;
      * the jaw is within ``max_jaw_face_deg`` of the face normal;
      * neither pad is in the table.
    """
    pc = pad_clearance(hand, obj)
    failed = []
    if pc.thumb_side_m < min_side_m or pc.thumb_m <= 0.0:
        failed.append("thumb_not_outside_face")
    if pc.finger_side_m < min_side_m or pc.finger_m <= 0.0:
        failed.append("finger_not_outside_face")
    if pc.thumb_center_below_top_m <= 0.0:
        failed.append("thumb_above_top")
    if pc.finger_center_below_top_m <= 0.0:
        failed.append("finger_above_top")
    if pc.jaw_face_angle_deg > max_jaw_face_deg:
        failed.append("jaw_not_on_faces")
    if min(hand.thumb.lowest_z, hand.finger.lowest_z) < table_z:
        failed.append("pad_in_table")
    return StraddleCheck(not failed, tuple(failed), pc)


# ── The table, and a carried object ───────────────────────────────────────────

@dataclass(frozen=True)
class TableReport:
    worst_m: float
    link: str
    index: int
    by_link: Dict[str, float]


def table_clearance(scene: SceneModel, seq: Iterable[Sequence[float]],
                    gripper_deg: float) -> TableReport:
    """Upper arm and forearm capsules (exact capsule-vs-box) and both pad
    boxes (lowest corner) against the tabletop, worst over ``seq``."""
    table = scene.table
    if table is None:
        raise ValueError("scene has no table")
    top = scene.table_surface_z
    by: Dict[str, float] = {}
    worst = (math.inf, "", -1)
    for k, q in enumerate(seq):
        caps = K.link_capsules(q, "right", gripper_deg, hand="tube")
        for name, p0, p1, r in caps:
            if name not in TABLE_LINKS:
                continue
            d, _at = segment_object_distance(table, p0, p1, r)
            by[name] = min(by.get(name, math.inf), d)
            if d < worst[0]:
                worst = (d, name, k)
        hand = hand_frames(q, gripper_deg)
        for box in (hand.thumb, hand.finger):
            d = box.lowest_z - top
            by[box.name] = min(by.get(box.name, math.inf), d)
            if d < worst[0]:
                worst = (d, box.name, k)
    return TableReport(worst[0], worst[1], worst[2], by)


def grasp_transform(hand: HandFrames, obj: Box) -> Tuple[np.ndarray, np.ndarray]:
    """The object's pose in the thumb frame at the moment of closure."""
    R = hand.thumb_rot
    return R.T @ obj.axes, R.T @ (obj.center - hand.thumb.center)


def carried_box(hand: HandFrames, obj: Box,
                rel: Tuple[np.ndarray, np.ndarray]) -> Box:
    """``obj`` rigidly following the thumb frame (``rel`` from grasp_transform)."""
    R = hand.thumb_rot
    rot_rel, t_rel = rel
    return Box(obj.name, hand.thumb.center + R @ t_rel, R @ rot_rel, obj.half)


@dataclass(frozen=True)
class CarriedReport:
    table_m: float               # lowest corner above the table surface
    table_at: int
    others_m: float              # vs every other colliding object, worst
    other_id: str
    others_at: int

    def as_dict(self) -> Dict[str, object]:
        return {"table_mm": round(1000 * self.table_m, 2),
                "table_at": self.table_at,
                "others_mm": round(1000 * self.others_m, 2),
                "other_id": self.other_id, "others_at": self.others_at}


def carried_clearance(scene: SceneModel, object_id: str,
                      seq: Sequence[Sequence[float]], gripper_deg: float,
                      obj: Box, rel: Tuple[np.ndarray, np.ndarray],
                      max_step_deg: float = 0.5) -> CarriedReport:
    """A carried object along a commanded route: against the tabletop, and
    against every other colliding scene object (manipulables and rig
    fixtures; cylinders as their bounding boxes).  The table is reported, not
    judged: the caller decides where contact is intended."""
    others = [o for o in scene.objects.values()
              if o.collides and o.id != object_id and o.id != (
                  scene.table.id if scene.table is not None else None)
              and "grid-cell" not in o.tags]
    boxes = [object_box(o) for o in others]
    top = scene.table_surface_z
    tm, ti, om, oid, oi = math.inf, -1, math.inf, "", -1
    for k, q in enumerate(densify(seq, max_step_deg)):
        b = carried_box(hand_frames(q, gripper_deg), obj, rel)
        if b.lowest_z - top < tm:
            tm, ti = b.lowest_z - top, k
        for o in boxes:
            d = obb_separation(b, o)
            if d < om:
                om, oid, oi = d, o.name, k
    return CarriedReport(tm, ti, om, oid, oi)
