"""Elevated, jaws-down ("crane") pick with measured-pose alignment (#55).

What it is for
--------------
The pick in ``pick_place_live`` approaches with whatever orientation FAST IK
finds first, which on this rig is a low, near-sideways hand.  The crane pick
keeps the elbow and forearm high and brings the open jaws down over a
box-shaped object with the pads on two opposing faces.  Two recorded attempts
at that posture (2026-10-06, Reachy-Lab ``trial-2026-10-06-crane-pick*``)
reached it and still never straddled the cube: the realised hand sat 4-5 mm
off the commanded one, the open pads had ~2.4 mm per side, and a tilted pad's
corner landed on the cube's top edge.  ``grasp_alignment`` re-reads those
recordings and predicts the pad that hit in each from the hover alone.

So this module plans from the commanded pose but DECIDES from the measured
one:

  1. ``plan_crane_pick`` -- no motion.  Solves the grasp, a vertical ladder of
     1 cm rungs up to the hover, and a via point, with the simulator's own IK
     service (bounded multi-start for the grasp, warm-started rungs above it),
     and checks every segment: the existing whole-arm ``check_arm_path``
     (unchanged), the arm and both pads against the TABLE, both pads' route
     past the object's top edges, and the carried object against the table,
     rails and other objects for the lift and the replacement.
  2. ``execute_crane_pick`` -- PRESENT -> via -> hover, a visible pause, then
     a bounded number of measured corrections at the hover (each one
     re-solved by IK and re-checked before it is commanded), a gated descent
     that re-measures at every rung and stops on the first grip-force reading
     or any object movement, a straddle check on the measured hand before the
     close, a 5 cm lift judged by the OBJECT's measured rise, a hold, the
     replacement, and the return by the checked route.

Every refusal and halt is a valid outcome and is reported with its reason.
Nothing here changes contact physics, force limits, calibration, CLEAR_Z or
the transit rates (``transit.CARRY_HZ`` / ``STEP_HZ`` are used as they are).

Feedback the caller must supply
-------------------------------
``observe()`` returns an ``Observation``: the target's live pose (position and
quaternion) and the right gripper's ``grip_force_n`` from the native state.
That is the whole contact feed this module has.  ``grip_force_n`` sums the
normal force on the two COLLISION PADS against any non-robot geom (object,
table, rail); contacts of the forearm, upper arm or visual shells are not in
it, and the state's ``contacts`` list is empty unless the server records them.
A halt can therefore say "a pad touched something" or "the object moved"; the
absence of either is NOT evidence that nothing touched.  The command panel's
``SimLink`` carries positions only today; giving it the quaternion and the
gripper block is the step that connects this module to the panel worker.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reachy_ai.motion import grasp_alignment as G
from reachy_ai.motion import primitives as P
from reachy_ai.motion.kinematics import (
    _BASE,
    ArmClearanceError,
    R_ARM_JOINTS,
    TargetContact,
    joint_limits,
    joint_path,
    link_frames,
    sample_count,
    within_limits,
)
from reachy_ai.motion.transit import CARRY_HZ, STEP_HZ
from reachy_ai.scene.awareness import SceneModel
from reachy_ai.tasks.pick_place_live import present_joints

log = logging.getLogger("reachy12.tasks.crane_pick_live")

#: The right shoulder in world (kinematics._SHOULDER_Y, torso at z = 1.0).
_SHOULDER_XY = (0.0, -0.19)


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CraneSpec:
    """The approach.  Defaults are the recorded x-face plan, not a new search.

    ``lean_bearing_deg`` is the hand's lean azimuth relative to the bearing
    from the shoulder to the object: 83.27 deg absolute at r2c2 in the
    recorded plan, whose bearing is 23.75 deg.  ``face_axis`` is the object's
    own axis the jaw closes along ("x": pads on the object's +/-x faces); the
    y faces were shown infeasible at this cell (pads staggered ~6 cm by the
    lean, minimax margin -5.1 mm) and are not offered.
    """
    face_axis: str = "x"
    tilt_deg: float = 30.0
    lean_bearing_deg: float = 59.52
    opening_deg: float = -65.0
    close_deg: float = -5.0          # primitives._GRIPPER_CLOSED_DEG
    transit_opening_deg: float = -45.0   # rig_routes.OPEN
    rung_m: float = 0.01
    hover_rungs: int = 9
    via_rungs: int = 18
    lift_rungs: int = 5
    pause_s: float = 10.0
    hold_s: float = 2.0


#: Side-clearance prediction error (m) of the joint-offset model, by how far
#: below the measuring rung the prediction reaches (cm).  Max over every
#: pre-contact rung pair of both recorded crane attempts (34 pairs; Reachy-Lab
#: trial-2026-10-06-crane-aligned-55/data/02-prediction-tolerance.json).
#: Nothing was recorded beyond 7 cm, so a check further away than that uses
#: the 7 cm value AND the planner refuses a plan whose pads would reach the
#: object's top more than 7 cm below a checkpoint (every rung is one).
PREDICTION_TOLERANCE_M = {1: 0.000177, 2: 0.000177, 3: 0.000225,
                          4: 0.000294, 5: 0.000358, 6: 0.000425, 7: 0.000494}


def prediction_tolerance(distance_m: float) -> float:
    k = min(7, max(1, math.ceil(round(distance_m * 100, 6))))
    return PREDICTION_TOLERANCE_M[k]


@dataclass(frozen=True)
class CraneLimits:
    """Correction bounds and detection thresholds, fixed before motion.

    From the recorded attempts (data/01-recorded-attempts-analysis.json):
      * realised jaw-gap centre vs commanded, pre-contact: lateral 2.6-4.6 mm,
        vertical 12.0-12.2 mm low; the lateral shift that would have
        equalised the x-face pads' predicted route clearance was 5.45 mm.
        A correction need beyond the recorded regime means something else is
        wrong, so the caps are those maxima rounded up to the next mm.
      * grip force read exactly 0.0 N in all 2194 pre-contact states and was
        positive at first pad contact; the resting cube moved 0.0000 mm.
    """
    max_lateral_correction_m: float = 0.006
    max_vertical_correction_m: float = 0.013
    hover_corrections: int = 3
    pre_insertion_corrections: int = 1
    contact_force_n: float = 0.0     # halt on any reading above this
    object_moved_m: float = 0.0002   # halt on any movement above this
    #: Lift: the object may trail the hand by at most one rung before it is
    #: called a slip; success is a held rise >= lift - one rung.
    lift_lag_m: float = 0.01
    ik_position_tol_m: float = 0.0005
    max_attempt_s: float = 300.0


# ── Errors and observations ───────────────────────────────────────────────────

class CraneRefused(RuntimeError):
    """The crane pick was refused (before motion, or before the next move).

    ``stage`` names where; ``detail`` carries the numbers.
    """

    def __init__(self, stage: str, reason: str, **detail) -> None:
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Observation:
    position: Tuple[float, float, float]
    quat_wxyz: Tuple[float, float, float, float]
    grip_force_n: float
    grasping: bool
    age_s: float = 0.0


Observe = Callable[[], Observation]
EventLog = Callable[..., None]


# ── Geometry of the approach ──────────────────────────────────────────────────

def grasp_rotation(obj: G.Box, spec: CraneSpec) -> np.ndarray:
    """Target r_wrist2hand rotation: hand tilted ``tilt_deg`` from vertical,
    leaning at the spec's azimuth, jaw closing along the object's face axis."""
    bearing = math.degrees(math.atan2(obj.center[1] - _SHOULDER_XY[1],
                                      obj.center[0] - _SHOULDER_XY[0]))
    t = math.radians(spec.tilt_deg)
    a = math.radians(bearing + spec.lean_bearing_deg)
    z = -np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a),
                   -math.cos(t)])
    k = {"x": 0, "y": 1}[spec.face_axis]
    face = obj.axes[:, k]
    other = obj.axes[:, 1 - k]
    j = np.cross(other, z)           # horizontal projection along the face normal
    j /= np.linalg.norm(j)
    if j @ face < 0:
        j = -j
    y = -j
    return np.column_stack([np.cross(y, z), y, z])


def _rot_err_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = (np.trace(a.T @ b) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _ik_raw(arm, rot: np.ndarray, wrist: np.ndarray, q0) -> Optional[List[float]]:
    M = np.eye(4)
    M[:3, :3] = rot
    M[:3, 3] = np.asarray(wrist) - _BASE
    try:
        return list(arm.inverse_kinematics(M, q0=list(q0)))
    except ValueError:
        return None


def _ik(arm, rot: np.ndarray, wrist: np.ndarray, q0) -> Optional[List[float]]:
    q = _ik_raw(arm, rot, wrist, q0)
    return q if q is not None and within_limits(q) else None


def _inside_limits(q: Sequence[float]) -> List[float]:
    """``q`` with each joint moved inside the planner's travel.  Used ONLY to
    derive an orientation target that the arm can hold; never commanded."""
    return [min(max(float(v), lo), hi) for v, (lo, hi) in zip(q, joint_limits("right"))]


def solve_gap(arm, rot: np.ndarray, gap_target: np.ndarray, gripper_deg: float,
              q0: Sequence[float], tol: float, iters: int = 8,
              relax: bool = False) -> Optional[Tuple[List[float], float]]:
    """IK for the jaw-gap centre (not the wrist) at ``gap_target``.

    The IK service solves for the wrist frame; the gap centre's offset from
    it depends on the wrist roll and the opening, so the wrist target is
    corrected by the remaining gap error and re-solved, warm-started.

    The service minimises position AND orientation error together, so where
    ``rot`` cannot be held at that position (a joint at its stop) it returns a
    compromise that misses the position.  With ``relax`` the orientation
    target then becomes the compromise's own orientation, which lets the
    position converge while the hand's attitude drifts only as far as reach
    requires; the caller judges the drift.  Returns (joints, gap error m).
    """
    h = G.hand_frames(q0, gripper_deg)
    W = np.asarray(gap_target) + (h.wrist - h.gap_mid)
    q = list(q0)
    err = math.inf
    for _ in range(iters):
        q1 = _ik_raw(arm, rot, W, q)
        if q1 is None:
            return None
        if not within_limits(q1):
            if not relax:
                return None
            # The service's bounds are wider than the joints' travel: aim at
            # the attitude the arm can actually hold and solve again.
            q = _inside_limits(q1)
            rot = G.hand_rotation(q)
            err = math.inf
            continue
        q = q1
        e = np.asarray(gap_target) - G.hand_frames(q, gripper_deg).gap_mid
        err = float(np.linalg.norm(e))
        if err < tol:
            break
        W = W + e
        if relax:
            rot = G.hand_rotation(q)
    return (q, err) if err < tol and within_limits(q) else None


#: Bounded start set for the grasp solve (the same 36 the recorded plan's
#: offline seed search used): shoulder pitch x arm yaw x elbow x (roll, yaw).
GRASP_STARTS = [[sp, sr, ay, el, fy, 0.0, 0.0]
                for sp in (-100.0, -70.0, -40.0) for ay in (0.0, 45.0)
                for el in (-60.0, -100.0)
                for (sr, fy) in ((-10.0, -60.0), (-10.0, 0.0), (-10.0, 60.0))]


# ── The plan ──────────────────────────────────────────────────────────────────

@dataclass
class CranePlan:
    object_id: str
    spec: CraneSpec
    limits: CraneLimits
    obj: G.Box                         # the object as planned against
    rot: np.ndarray                    # target hand rotation at the grasp
    gap_targets: List[np.ndarray]      # index k: k rungs above the grasp
    rungs: List[List[float]]           # commanded joints per rung, 0..via_rungs
    present: List[float]
    shift: np.ndarray = field(default_factory=lambda: np.zeros(3))
    insertion_rung: int = 0            # lowest rung with every pad above the top
    jaw_face_limit_deg: float = 0.0    # from the PLANNED pad slack (see jaw_face_limit_deg)
    checks: Dict[str, object] = field(default_factory=dict)

    @property
    def hover(self) -> List[float]:
        return self.rungs[self.spec.hover_rungs]

    @property
    def via(self) -> List[float]:
        return self.rungs[self.spec.via_rungs]

    def descent(self, frm: Optional[int] = None) -> List[List[float]]:
        top = self.spec.hover_rungs if frm is None else frm
        return [self.rungs[k] for k in range(top, -1, -1)]

    def summary(self) -> Dict[str, object]:
        return {"object_id": self.object_id,
                "shift_mm": [round(1000 * float(v), 2) for v in self.shift],
                "insertion_rung": self.insertion_rung,
                "grasp_gap_target": [round(float(v), 4) for v in self.gap_targets[0]],
                "rungs": [[round(v, 3) for v in q] for q in self.rungs],
                "checks": self.checks}


def _solve_ladder(arm, plan_rot: np.ndarray, targets: Sequence[np.ndarray],
                  grip: float, q_grasp: Sequence[float], tol: float,
                  rots: Optional[Sequence[np.ndarray]] = None) -> List[List[float]]:
    """Rungs 1.. above a solved grasp, each warm-started from the one below.

    Each rung's target orientation is the rung below's ACHIEVED orientation
    (or ``rots[k]`` when re-solving an existing ladder), so the hand keeps its
    attitude where it can and drifts only as much as reach requires.
    """
    out = [list(q_grasp)]
    for k in range(1, len(targets)):
        rot = rots[k] if rots is not None else G.hand_rotation(out[-1])
        got = solve_gap(arm, rot, targets[k], grip, out[-1], tol, iters=20,
                        relax=True)
        if got is None:
            raise CraneRefused("plan", f"IK found no pose for rung +{k}",
                               rung=k, target=[float(v) for v in targets[k]])
        out.append(got[0])
    return out


def _segments(plan: CranePlan) -> Dict[str, Tuple[List[List[float]], float, TargetContact, str]]:
    s = plan.spec
    oid = plan.object_id
    r = plan.rungs
    hover, via = s.hover_rungs, s.via_rungs
    desc = [r[k] for k in range(hover, -1, -1)]
    lift = [r[k] for k in range(0, s.lift_rungs + 1)]
    up = [r[k] for k in range(hover, via + 1)]
    return {
        "transit PRESENT->via": ([plan.present, r[via]], s.transit_opening_deg,
                                 TargetContact(oid, hand_contact=False), "carry"),
        "via->hover": (up[::-1], s.transit_opening_deg,
                       TargetContact(oid, hand_contact=True), "carry"),
        "descent hover->grasp": (desc, s.opening_deg,
                                 TargetContact(oid, hand_contact=True), "step"),
        "lift grasp->+lift": (lift, s.close_deg,
                              TargetContact(oid, hand_contact=True, carried=True), "step"),
        "replace +lift->grasp": (lift[::-1], s.close_deg,
                                 TargetContact(oid, hand_contact=True, carried=True), "step"),
        "retract grasp->hover": (desc[::-1], s.opening_deg,
                                 TargetContact(oid, hand_contact=True), "step"),
        "hover->via": (up, s.opening_deg,
                       TargetContact(oid, hand_contact=True), "carry"),
        "return via->PRESENT": ([r[via], plan.present], s.transit_opening_deg,
                                TargetContact(oid, hand_contact=False), "carry"),
    }


def check_crane_plan(planner, scene: SceneModel, plan: CranePlan,
                     offset: Optional[np.ndarray] = None,
                     gripper_open_deg: Optional[float] = None,
                     from_rung: Optional[int] = None) -> Dict[str, object]:
    """Every check the crane pick is held to.  Raises CraneRefused.

    ``offset`` (measured - commanded joints) and ``gripper_open_deg`` (the
    measured opening) make the pad and table checks about the PREDICTED
    realised hand; ``from_rung`` is where the arm is now (the route check
    starts there).  The existing whole-arm check is unchanged and always
    run on the commanded poses.
    """
    s, lim = plan.spec, plan.limits
    off = np.zeros(7) if offset is None else np.asarray(offset, dtype=float)
    g_open = s.opening_deg if gripper_open_deg is None else gripper_open_deg
    rep: Dict[str, object] = {}

    # 1. The existing whole-arm preflight (tube hand, objects + rails).
    for name, (seq, grip, tgt, _rate) in _segments(plan).items():
        try:
            planner.check_arm_path(seq, segment=name, gripper_deg=grip, target=tgt)
        except ArmClearanceError as exc:
            raise CraneRefused("check_arm_path", str(exc), segment=name,
                               link=exc.link, obstacle=exc.obstacle,
                               clearance_m=exc.clearance_m) from exc
    rep["check_arm_path"] = "pass (all segments)"

    # 2. Arm links and pads vs the TABLE (not covered by check_arm_path).
    tab = {}
    for name, (seq, grip, _tgt, _rate) in _segments(plan).items():
        dense = G.densify(seq, 1.0)
        for label, qs in (("commanded", dense),
                          ("predicted", [list(np.asarray(q) + off) for q in dense])):
            g = g_open if grip == s.opening_deg else grip
            tr = G.table_clearance(scene, qs, g)
            tab[f"{name} ({label})"] = {k: round(1000 * v, 1) for k, v in tr.by_link.items()}
            if tr.worst_m < 0.0:
                raise CraneRefused("table", f"{tr.link} {1000 * tr.worst_m:.1f} mm "
                                   f"into the table on '{name}' ({label})",
                                   segment=name, link=tr.link, clearance_m=tr.worst_m,
                                   predicted=(label == "predicted"))
    rep["table_mm"] = tab

    # 3. Both pads' route past the object's top edges, down to grasp depth.
    top = s.hover_rungs if from_rung is None else from_rung
    route = G.route_clearance(plan.descent(top), g_open, plan.obj, offset=off)
    tol = prediction_tolerance(top * s.rung_m)
    rep["pad_route"] = {**route.as_dict(), "tolerance_mm": round(1000 * tol, 3),
                        "from_rung": top}
    if route.min_m < tol:
        side = "thumb" if route.thumb_m < route.finger_m else "finger"
        raise CraneRefused("pad_route", f"{side} pad route clearance "
                           f"{1000 * route.min_m:.2f} mm < {1000 * tol:.3f} mm "
                           f"from rung +{top}", route=route.as_dict())

    # 4. The straddle the descent should end in (predicted), before closing.
    hand0 = G.hand_frames(np.asarray(plan.rungs[0]) + off, g_open)
    if plan.jaw_face_limit_deg <= 0.0:
        # Set once, from the plan's own (commanded) pad slack.
        plan.jaw_face_limit_deg = jaw_face_limit_deg(route.min_m)
    st = G.straddle_check(hand0, plan.obj, scene.table_surface_z, 0.0,
                          plan.jaw_face_limit_deg)
    rep["straddle_predicted"] = {**st.as_dict(),
                                 "jaw_face_limit_deg": round(plan.jaw_face_limit_deg, 2)}
    if not st.ok:
        raise CraneRefused("straddle", f"predicted grasp pose does not straddle: "
                           f"{list(st.failed)}", straddle=st.as_dict())

    # 5. The carried object: lift and replacement vs table, rails, others.
    rel = G.grasp_transform(G.hand_frames(plan.rungs[0], s.opening_deg), plan.obj)
    rest = plan.obj.lowest_z - scene.table_surface_z
    lift = [plan.rungs[k] for k in range(0, s.lift_rungs + 1)]
    cr = G.carried_clearance(scene, plan.object_id, lift, s.close_deg, plan.obj, rel)
    rep["carried"] = {**cr.as_dict(), "resting_table_mm": round(1000 * rest, 2)}
    if cr.others_m < 0.0:
        raise CraneRefused("carried", f"carried object would hit {cr.other_id} "
                           f"({1000 * cr.others_m:.1f} mm)", carried=cr.as_dict())
    if cr.table_m < rest - 1e-4:
        raise CraneRefused("carried", f"carried object would be pushed "
                           f"{1000 * (rest - cr.table_m):.1f} mm into the table",
                           carried=cr.as_dict())
    plan.checks = rep
    return rep


def correctable(exc: CraneRefused) -> bool:
    """A refusal a measured re-centring can address: the pads' route, the
    predicted straddle, or the PREDICTED (not the commanded) hand in the
    table -- all three are the tracking error showing up in a different
    check.  Everything else (whole-arm, carried object, IK) is not."""
    return (exc.stage in ("pad_route", "straddle")
            or (exc.stage == "table" and bool(exc.detail.get("predicted"))))


def jaw_face_limit_deg(slack_m: float) -> float:
    """Jaw-vs-face angle at which rotating a pad about the vertical would by
    itself use up ``slack_m`` at the pad's outer corner (finger pad half-width
    along the face, 14 mm, from the MJCF)."""
    half = float(G.FINGER_PAD[2][0])
    return math.degrees(math.asin(max(0.0, min(1.0, slack_m / half))))


def _insertion_rung(plan: CranePlan) -> int:
    """Lowest rung at which every pad is at least one rung above the object's
    top, on the commanded poses: the last place a lateral correction may be
    made before a pad can meet a top edge."""
    top = plan.obj.highest_z
    for k in range(0, plan.spec.hover_rungs + 1):
        h = G.hand_frames(plan.rungs[k], plan.spec.opening_deg)
        if min(h.thumb.lowest_z, h.finger.lowest_z) - top >= plan.spec.rung_m:
            return k
    raise CraneRefused("plan", "a pad is within one rung of the object's top "
                       "even at the hover")


def plan_crane_pick(planner, scene: SceneModel, object_id: str,
                    pose: Optional[Observation] = None,
                    spec: CraneSpec = CraneSpec(),
                    limits: CraneLimits = CraneLimits(),
                    on_event: Optional[EventLog] = None) -> CranePlan:
    """Plan and check the crane pick.  IK/FK service calls only -- no motion.

    ``pose`` is the object's live pose (the scene model has no live
    orientation).  Raises CraneRefused.
    """
    ev = on_event or (lambda *_a, **_k: None)
    arm = planner._arm
    obj_model = scene.get(object_id)
    if obj_model.kind != "box":
        raise CraneRefused("plan", f"{object_id} is a {obj_model.kind}; the "
                           f"crane pick closes on two flat faces")
    obj = G.object_box(obj_model, *(() if pose is None else
                                     (pose.position, pose.quat_wxyz)))
    rot = grasp_rotation(obj, spec)
    tol = limits.ik_position_tol_m

    # Grasp: bounded multi-start (one IK call per start), then the three
    # highest-elbow candidates are gap-centred; the highest that converges
    # with the orientation held (<= 1 deg) wins.
    g0 = obj.center.copy()
    pre = []
    for start in GRASP_STARTS:
        h = G.hand_frames(start, spec.opening_deg)
        q = _ik(arm, rot, g0 + (h.wrist - h.gap_mid), start)
        if q is not None and _rot_err_deg(G.hand_rotation(q), rot) <= 1.0:
            pre.append(q)
    pre.sort(key=lambda q: -float(link_frames(q)[1][2]))
    found = []
    for q in pre[:3]:
        got = solve_gap(arm, rot, g0, spec.opening_deg, q, tol)
        if got is not None and _rot_err_deg(G.hand_rotation(got[0]), rot) <= 1.0:
            found.append(got[0])
    ev("plan_grasp_starts", tried=len(GRASP_STARTS), converged=len(pre),
       gap_centred=len(found))
    if not found:
        raise CraneRefused("plan", "IK found no grasp pose in the bounded start set")
    q_g = max(found, key=lambda q: float(link_frames(q)[1][2]))

    # Grasp height: equalise "both pad centres below the top" and "lowest pad
    # above the table" -- the two things a too-high and a too-low hand break.
    h = G.hand_frames(q_g, spec.opening_deg)
    below_top = obj.highest_z - max(h.thumb.center[2], h.finger.center[2])
    above_table = min(h.thumb.lowest_z, h.finger.lowest_z) - scene.table_surface_z
    dz = 0.5 * (above_table - below_top)
    # Laterally: equalise the two pads' clearance to the faces.
    pc = G.pad_clearance(h, obj)
    dn = 0.5 * (pc.thumb_side_m - pc.finger_side_m)
    g1 = g0 + dn * np.asarray(pc.face_normal) - np.array([0.0, 0.0, dz])
    got = solve_gap(arm, rot, g1, spec.opening_deg, q_g, tol)
    if got is None:
        raise CraneRefused("plan", "IK lost the grasp pose while centring it")
    q_g = got[0]
    targets = [g1 + np.array([0.0, 0.0, k * spec.rung_m])
               for k in range(spec.via_rungs + 1)]
    rungs = _solve_ladder(arm, rot, targets, spec.opening_deg, q_g, tol)
    plan = CranePlan(object_id, spec, limits, obj, rot, targets, rungs,
                     present_joints())
    plan.insertion_rung = _insertion_rung(plan)
    if plan.insertion_rung > 7:
        raise CraneRefused("plan", "pads would meet the object's top more than "
                           "7 cm below the insertion checkpoint (beyond the "
                           "recorded prediction range)")
    check_crane_plan(planner, scene, plan)
    ev("plan_checked", **plan.summary())
    return plan


def replan(planner, scene: SceneModel, plan: CranePlan, shift: np.ndarray,
           seed_rung: int) -> CranePlan:
    """The same ladder with its gap targets moved by ``shift`` (world, m).

    Re-solved by IK from the existing rungs (each rung keeps its own
    orientation as the target), and NOT checked here -- the caller runs
    ``check_crane_plan`` with the measured offset before commanding it.
    """
    lim = plan.limits
    total = plan.shift + shift
    lat = float(np.linalg.norm(total[:2]))
    if lat > lim.max_lateral_correction_m + 1e-9:
        raise CraneRefused("correction", f"lateral correction {1000 * lat:.2f} mm "
                           f"exceeds the {1000 * lim.max_lateral_correction_m:.0f} mm cap",
                           shift_mm=(1000 * total).round(2).tolist())
    if abs(float(total[2])) > lim.max_vertical_correction_m + 1e-9:
        raise CraneRefused("correction", f"vertical correction {1000 * total[2]:.2f} mm "
                           f"exceeds the {1000 * lim.max_vertical_correction_m:.0f} mm cap",
                           shift_mm=(1000 * total).round(2).tolist())
    arm = planner._arm
    targets = [t + shift for t in plan.gap_targets]
    rots = [G.hand_rotation(q) for q in plan.rungs]
    # Solve outward from the rung the arm is at, so the commanded change there
    # is the smallest one; the rest are warm-started from their neighbours.
    new: Dict[int, List[float]] = {}
    got = solve_gap(arm, rots[seed_rung], targets[seed_rung], plan.spec.opening_deg,
                    plan.rungs[seed_rung], lim.ik_position_tol_m, iters=20, relax=True)
    if got is None:
        raise CraneRefused("correction", f"IK found no corrected pose at rung +{seed_rung}")
    new[seed_rung] = got[0]
    for k in list(range(seed_rung - 1, -1, -1)) + list(range(seed_rung + 1, len(targets))):
        nb = new[k + 1] if k < seed_rung else new[k - 1]
        got = solve_gap(arm, rots[k], targets[k], plan.spec.opening_deg, nb,
                        lim.ik_position_tol_m, iters=20, relax=True)
        if got is None:
            raise CraneRefused("correction", f"IK found no corrected pose at rung +{k}")
        new[k] = got[0]
    out = replace(plan, gap_targets=targets,
                  rungs=[new[k] for k in range(len(targets))], shift=total,
                  checks={})
    out.insertion_rung = _insertion_rung(out)
    return out


# ── Execution ─────────────────────────────────────────────────────────────────

def _read(arm) -> Tuple[List[float], float]:
    return ([float(getattr(arm, n).present_position) for n in R_ARM_JOINTS],
            float(arm.r_gripper.present_position))


class _Halt(Exception):
    def __init__(self, reason: str, **detail):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


#: An observation older than this is not feedback (the native server pushes
#: state at ~50 Hz).
FEEDBACK_STALE_S = 0.5


def execute_crane_pick(robot, planner, scene: SceneModel, plan: CranePlan,
                       observe: Observe, *, on_event: Optional[EventLog] = None,
                       should_abort: Optional[Callable[[], bool]] = None
                       ) -> Dict[str, object]:
    """Fly a checked crane plan from PRESENT and back.  See the module doc.

    Returns an outcome dict.  A refusal or a halt is an outcome, not an
    exception.  The arm must be at PRESENT and turned on.  All motion goes
    through ``motion.primitives``; nothing is commanded that the plan's checks
    (or a re-check made immediately before it) did not cover.
    """
    ev = on_event or (lambda *_a, **_k: None)
    arm = robot.r_arm
    s, lim = plan.spec, plan.limits
    t_start = time.monotonic()
    outcome: Dict[str, object] = {"posture_held": None, "aligned_at_hover": None,
                                  "straddled_before_close": None, "closed": False,
                                  "lifted_and_held": None, "replaced": None,
                                  "halt": None, "refusal": None, "final": None}
    first = observe()
    c0 = np.asarray(first.position, dtype=float)
    if float(np.linalg.norm(c0 - plan.obj.center)) > lim.object_moved_m:
        outcome["refusal"] = "the object is not where the plan was made"
        ev("REFUSED", stage="start", reason=outcome["refusal"],
           planned=plan.obj.center.tolist(), live=c0.tolist())
        return outcome
    ok, joint, off_deg = P.stow_entry_ok(arm)
    if not ok:
        outcome["refusal"] = f"not at PRESENT ({joint} {off_deg:.0f} deg off)"
        ev("REFUSED", stage="start", reason=outcome["refusal"])
        return outcome

    # `above`: the highest ladder rung the arm is at or below -- where a
    # retreat rejoins the checked ladder.
    st = {"monitor": False, "above": s.via_rungs}
    cur = {"plan": plan}

    def P_() -> CranePlan:
        return cur["plan"]

    def monitor() -> Optional[str]:
        if not st["monitor"]:
            return None
        o = observe()
        if o.age_s > FEEDBACK_STALE_S:
            return f"feedback stale ({o.age_s:.2f} s)"
        if o.grip_force_n > lim.contact_force_n:
            return f"grip force {o.grip_force_n:.3f} N (a pad touched something)"
        moved = float(np.linalg.norm(np.asarray(o.position) - c0))
        if moved > lim.object_moved_m:
            return f"object moved {1000 * moved:.2f} mm"
        if time.monotonic() - t_start > lim.max_attempt_s:
            return "attempt time limit"
        if should_abort is not None and should_abort():
            return "aborted by the caller"
        return None

    def stream(a, b, rate_hz: int, max_step_deg: float, check: bool = True) -> None:
        n = max(2, math.ceil(max(abs(float(y) - float(x)) for x, y in zip(a, b))
                             / max_step_deg) + 1)
        for q in joint_path(a, b, n)[1:]:
            if check:
                why = monitor()
                if why:
                    raise _Halt(why)
            P.execute_trajectory(arm, [q], R_ARM_JOINTS, rate_hz=rate_hz)

    def hold(q, seconds: float, check: bool = True) -> None:
        for _ in range(max(1, int(seconds * STEP_HZ))):
            if check:
                why = monitor()
                if why:
                    raise _Halt(why)
            P.execute_trajectory(arm, [q], R_ARM_JOINTS, rate_hz=STEP_HZ)

    def measure(rung: int, label: str):
        plan_ = P_()
        qm, gm = _read(arm)
        cmd = plan_.rungs[rung]
        off = G.joint_offset(qm, cmd)
        o = observe()
        live = G.object_box(scene.get(plan_.object_id), o.position, o.quat_wxyz)
        hm = G.hand_frames(qm, gm)
        hc = G.hand_frames(cmd, gm)
        pc = G.pad_clearance(hm, live)
        ev("MEASURE", label=label, rung=rung,
           q_commanded=[round(v, 3) for v in cmd], q_measured=[round(v, 3) for v in qm],
           joint_offset_deg=[round(float(v), 3) for v in off],
           gripper_measured_deg=round(gm, 2),
           gap_commanded=[round(float(v), 5) for v in hc.gap_mid],
           gap_measured=[round(float(v), 5) for v in hm.gap_mid],
           gap_error_mm=[round(1000 * float(v), 2) for v in hm.gap_mid - hc.gap_mid],
           pads=pc.as_dict(), elbow_z=round(float(link_frames(qm)[1][2]), 4),
           grip_force_n=o.grip_force_n, object=[round(v, 5) for v in o.position],
           object_moved_mm=round(1000 * float(np.linalg.norm(np.asarray(o.position) - c0)), 3))
        return qm, gm, off, live, hm, pc

    def judge(at_rung: int, off, gm, live) -> Dict[str, object]:
        """check_crane_plan on the current plan, from where the arm is."""
        return check_crane_plan(planner, scene, replace(P_(), obj=live), offset=off,
                                gripper_open_deg=gm, from_rung=at_rung)

    def correct(at_rung: int, budget: int, label: str, m) -> Tuple:
        """Judge; if the pads' route or the predicted straddle fails, make up
        to ``budget`` measured corrections, each re-solved and re-checked
        before it is commanded.  Returns the latest measurement."""
        qm, gm, off, live, hm, pc = m
        for i in range(budget + 1):
            try:
                rep = judge(at_rung, off, gm, live)
                ev("ALIGNED", label=label, corrections=i, pad_route=rep["pad_route"],
                   straddle_predicted=rep["straddle_predicted"],
                   shift_mm=[round(1000 * float(v), 2) for v in P_().shift])
                return qm, gm, off, live, hm, pc
            except CraneRefused as exc:
                if not correctable(exc) or i == budget:
                    ev("ALIGN_REFUSED", label=label, corrections=i, stage=exc.stage,
                       reason=exc.reason)
                    raise
                why = exc.reason
            plan_ = P_()
            route = G.route_clearance(plan_.descent(at_rung), gm, live, offset=off)
            end = G.pad_clearance(G.hand_frames(np.asarray(plan_.rungs[0]) + off, gm), live)
            planned_above = float(plan_.gap_targets[0][2] - plan_.shift[2]
                                  - plan_.obj.center[2])
            d = G.centring_shift(end, route, planned_above)
            ev("CORRECTION", label=label, iteration=i + 1, because=why,
               predicted_route=route.as_dict(),
               shift_mm=[round(1000 * float(v), 2) for v in d],
               cumulative_mm=[round(1000 * float(v), 2) for v in plan_.shift + d])
            new = replan(planner, scene, plan_, d, at_rung)
            move = [qm, new.rungs[at_rung]]
            try:
                planner.check_arm_path(move, segment=f"{label} correction {i + 1}",
                                       gripper_deg=gm,
                                       target=TargetContact(plan_.object_id, hand_contact=True))
            except ArmClearanceError as exc2:
                raise CraneRefused("check_arm_path", str(exc2)) from exc2
            tr = G.table_clearance(scene, G.densify(move, 0.25), gm)
            mv = G.route_clearance(move, gm, live)
            if tr.worst_m < 0.0 or mv.min_m <= 0.0:
                raise CraneRefused("correction", "the correcting move itself is not clear",
                                   table_m=tr.worst_m, pads_m=mv.min_m)
            cur["plan"] = new
            ev("CORRECTION_COMMANDED", label=label, iteration=i + 1,
               q_from=[round(v, 3) for v in qm],
               q_to=[round(v, 3) for v in new.rungs[at_rung]])
            stream(qm, new.rungs[at_rung], STEP_HZ, 0.05)
            hold(new.rungs[at_rung], 1.5)
            qm, gm, off, live, hm, pc = measure(at_rung, f"{label} after correction {i + 1}")
        raise AssertionError("unreachable")

    def retreat(opening: float) -> bool:
        """Back up the checked ladder, via, PRESENT -- the documented route.
        The one new line (measured pose -> the rung above) is checked first."""
        st["monitor"] = False
        plan_ = P_()
        qm, gm = _read(arm)
        k = min(st["above"], s.via_rungs)
        link = [qm, plan_.rungs[k]]
        try:
            planner.check_arm_path(link, segment="retreat to the ladder", gripper_deg=gm,
                                   target=TargetContact(plan_.object_id, hand_contact=True))
            tr = G.table_clearance(scene, G.densify(link, 0.25), gm)
            if tr.worst_m < 0.0:
                raise CraneRefused("table", f"retreat line puts {tr.link} in the table")
        except (ArmClearanceError, CraneRefused) as exc:
            ev("RETREAT_REFUSED", reason=str(exc))
            return False
        P.smooth_move(arm, {"r_gripper": opening}, 0.8)
        stream(qm, plan_.rungs[k], STEP_HZ, 0.05, check=False)
        for j in range(k, s.via_rungs):
            slow = j < s.hover_rungs
            stream(plan_.rungs[j], plan_.rungs[j + 1], STEP_HZ if slow else CARRY_HZ,
                   0.05 if slow else 2.0, check=False)
        P.smooth_move(arm, {"r_gripper": s.transit_opening_deg}, 0.5)
        stream(plan_.via, plan_.present, CARRY_HZ, 2.0, check=False)
        hold(plan_.present, 1.0, check=False)
        ev("phase", name="back at PRESENT")
        return True

    # ── approach: PRESENT -> via -> hover ───────────────────────────────────
    ev("phase", name="ATTEMPT_START")
    P.look_at(robot, tuple(plan.obj.center), duration=1.0)
    P.smooth_move(arm, {"r_gripper": s.transit_opening_deg}, 0.5)
    ev("phase", name="transit PRESENT -> via")
    stream(plan.present, plan.via, CARRY_HZ, 2.0, check=False)
    ev("phase", name="via -> hover")
    for k in range(s.via_rungs, s.hover_rungs, -1):
        stream(plan.rungs[k], plan.rungs[k - 1], CARRY_HZ, 2.0, check=False)
    st["above"] = s.hover_rungs
    P.smooth_move(arm, {"r_gripper": s.opening_deg}, 0.8)
    hold(plan.hover, 1.5, check=False)
    st["monitor"] = True

    halt: Optional[_Halt] = None
    try:
        # ── the visible pause at the elevated approach ─────────────────────
        ev("phase", name=f"HOVER pause ({s.pause_s:.0f} s)")
        hold(plan.hover, s.pause_s)
        m = measure(s.hover_rungs, "hover after pause")
        qm, gm, off, live, hm, pc = m
        outcome["posture_held"] = {
            "elbow_z": round(float(link_frames(qm)[1][2]), 4),
            "forearm_min_z": round(float(min(link_frames(qm)[1][2],
                                             link_frames(qm)[2][2])), 4),
            "tilt_from_vertical_deg": round(hm.tilt_from_vertical_deg, 2),
            "jaw_face_angle_deg": round(pc.jaw_face_angle_deg, 2)}

        # ── bounded corrections at the hover ───────────────────────────────
        try:
            m = correct(s.hover_rungs, lim.hover_corrections, "hover", m)
            outcome["aligned_at_hover"] = True
        except CraneRefused as exc:
            outcome["aligned_at_hover"] = False
            outcome["refusal"] = f"{exc.stage}: {exc.reason}"
            raise _Halt(f"refused at the hover -- {exc.stage}: {exc.reason}")

        # ── gated descent: re-measured and re-judged at every rung ─────────
        ev("phase", name="descent")
        left = lim.pre_insertion_corrections
        rung = s.hover_rungs
        while rung > 0:
            st["above"] = rung
            stream(P_().rungs[rung], P_().rungs[rung - 1], STEP_HZ, 0.05)
            rung -= 1
            st["above"] = rung
            hold(P_().rungs[rung], 0.6)
            m = measure(rung, "descent rung")
            if rung == 0:
                break
            try:
                rep = judge(rung, m[2], m[1], m[3])
                ev("CONTINUE", rung=rung, pad_route=rep["pad_route"])
            except CraneRefused as exc:
                may_fix = rung >= P_().insertion_rung and left > 0 and correctable(exc)
                ev("DESCENT_CHECK_FAILED", rung=rung, stage=exc.stage, reason=exc.reason,
                   correction_allowed=may_fix)
                if not may_fix:
                    raise _Halt(f"refused at rung +{rung} -- {exc.stage}: {exc.reason}")
                left -= 1
                try:
                    m = correct(rung, 1, f"pre-insertion rung +{rung}", m)
                except CraneRefused as exc2:
                    raise _Halt(f"refused at rung +{rung} -- {exc2.stage}: {exc2.reason}")

        # ── straddle on the MEASURED hand, then (only then) close ──────────
        hold(P_().rungs[0], 1.0)
        qm, gm, off, live, hm, pc = measure(0, "grasp depth")
        sc = G.straddle_check(hm, live, scene.table_surface_z, 0.0,
                              P_().jaw_face_limit_deg)
        ev("STRADDLE", **sc.as_dict(), jaw_face_limit_deg=round(P_().jaw_face_limit_deg, 2))
        outcome["straddled_before_close"] = sc.ok
        if not sc.ok:
            raise _Halt(f"no straddle at grasp depth: {list(sc.failed)}")
    except _Halt as h:
        halt = h

    if halt is not None:
        st["monitor"] = False
        outcome["halt"] = halt.reason
        qm, gm = _read(arm)
        o = observe()
        ev("HALT", reason=halt.reason, above_rung=st["above"],
           q_measured=[round(v, 3) for v in qm], object=list(o.position),
           quat=list(o.quat_wxyz), grip_force_n=o.grip_force_n)
        if not retreat(s.opening_deg):
            outcome["final"] = "left at the halt pose (retreat refused)"
            ev("ATTEMPT_END", outcome=outcome)
            return outcome
        return _finish(arm, robot, observe, outcome, ev)

    # ── close, lift, hold, replace ────────────────────────────────────────
    plan = P_()
    st["monitor"] = False
    ev("phase", name="close")
    z0 = float(observe().position[2])
    P.smooth_move(arm, {"r_gripper": s.close_deg}, 0.8)
    hold(plan.rungs[0], 1.0, check=False)
    o = observe()
    qm, gm = _read(arm)
    outcome["closed"] = True
    ev("CLOSED", gripper_measured_deg=round(gm, 2), grip_force_n=o.grip_force_n,
       grasping=o.grasping, object=list(o.position))

    ev("phase", name="lift")
    lifted, k = True, 0
    for k in range(1, s.lift_rungs + 1):
        st["above"] = k
        stream(plan.rungs[k - 1], plan.rungs[k], STEP_HZ, 0.05, check=False)
        hold(plan.rungs[k], 0.4, check=False)
        o = observe()
        rise = float(o.position[2]) - z0
        ev("LIFT_RUNG", rung=k, object_rise_mm=round(1000 * rise, 2),
           grip_force_n=o.grip_force_n, grasping=o.grasping, object=list(o.position))
        if rise < k * s.rung_m - lim.lift_lag_m:
            lifted = False
            outcome["halt"] = (f"object did not follow the lift: hand up {k} cm, "
                               f"object up {1000 * rise:.1f} mm")
            ev("HALT", reason=outcome["halt"])
            break
    if lifted:
        ev("phase", name=f"hold ({s.hold_s:.0f} s)")
        o1 = observe()
        hold(plan.rungs[s.lift_rungs], s.hold_s, check=False)
        o2 = observe()
        r1, r2 = float(o1.position[2]) - z0, float(o2.position[2]) - z0
        need = s.lift_rungs * s.rung_m - lim.lift_lag_m
        outcome["lifted_and_held"] = {
            "ok": bool(r1 >= need and r2 >= need), "rise_start_mm": round(1000 * r1, 2),
            "rise_end_mm": round(1000 * r2, 2), "required_mm": round(1000 * need, 1),
            "grasping": o2.grasping, "grip_force_n": o2.grip_force_n,
            "object": list(o2.position), "quat": list(o2.quat_wxyz)}
        ev("HELD", **outcome["lifted_and_held"])
        ev("phase", name="replace")
        for k in range(s.lift_rungs, 0, -1):
            stream(plan.rungs[k], plan.rungs[k - 1], STEP_HZ, 0.05, check=False)
            st["above"] = k - 1
        k = 0
        hold(plan.rungs[0], 0.6, check=False)
    else:
        outcome["lifted_and_held"] = {"ok": False}
    P.smooth_move(arm, {"r_gripper": s.opening_deg}, 0.8)
    hold(plan.rungs[k], 0.8, check=False)
    o = observe()
    outcome["replaced"] = {"object": list(o.position), "quat": list(o.quat_wxyz),
                           "offset_from_start_mm": round(1000 * float(np.linalg.norm(
                               np.asarray(o.position) - c0)), 2)}
    ev("RELEASED", **outcome["replaced"])
    st["above"] = k
    if not retreat(s.opening_deg):
        outcome["final"] = "left at the release pose (retreat refused)"
        ev("ATTEMPT_END", outcome=outcome)
        return outcome
    return _finish(arm, robot, observe, outcome, ev)


def _finish(arm, robot, observe, outcome, ev):
    ok, joint, off_deg = P.stow_entry_ok(arm)
    if ok:
        ev("phase", name="stow_from_side (documented PRESENT -> HOME route)")
        P.stow_from_side(robot, arm)
        outcome["final"] = "HOME"
    else:
        ev("left_at_pose", why=f"not at PRESENT ({joint} {off_deg:.0f} deg off); not stowing")
        outcome["final"] = f"not at PRESENT ({joint} {off_deg:.0f} deg off)"
    o = observe()
    outcome["final_object"] = list(o.position)
    ev("ATTEMPT_END", outcome=outcome)
    return outcome
