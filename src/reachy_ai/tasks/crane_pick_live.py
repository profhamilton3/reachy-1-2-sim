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
absence of either is NOT evidence that nothing touched -- an arm link brushing
the table, for instance, appears in neither, and is only found by replaying the
recorded states offline.  In the panel, the parent's ``SimLink`` (which carries
the quaternion and the gripper block) forwards these to the motion worker.

Replacement and withdrawal (added after the 2026-10-06 attempt)
---------------------------------------------------------------
That attempt lifted the cube, then the thumb pad touched the table after the
release and brushed the replaced cube on the way out.  The first was this
module's own doing: the retreat started its line from the MEASURED pose, and
commanding a pose the arm already sags short of lets it sag again (the hand
dropped ~4 mm).  Every move now starts from the last COMMANDED pose.  The
second came from the cube having been put down 2.9 mm from where it was picked
up, under a ladder planned for the original spot.  So: the lowering stops when
the object's lowest corner reaches the table (support contact, the one contact
intended there) and never goes where the loaded hand's predicted pads would
meet the table; the withdrawal is judged against the object's MEASURED pose
after release, corrected at most once by a bounded sideways shift, or refused;
and the pad-force / object monitor stays on through the withdrawal and the
return.

Tracking compensation (#55: "compensate in the commanded pose or fix the
actuator model?")
----------------------------------------------------------------------
For THIS path the steady-state tracking error is compensated in the commanded
pose, from measurement: the hover correction moves the commanded jaw-gap target
by the error the measured joints show (2026-10-06: ~3.4 mm sideways and
~8.8 mm up for a shoulder pitch settling ~1.1 deg short), re-solved and
re-checked.  What it does NOT address: the actuator model itself (gains and
force ranges are unchanged); the different offset of the LOADED arm while
carrying, which is not corrected, only predicted against the table; and every
other path, including ``pick_place_live``.  Whether the actuator model should
change instead remains open.

Hold slip is not addressed by this control code.  The 2026-10-06 hold crept
~0.75 mm/s with the hand still and no measurable pivot, at about 250 N of pad
normal force summed over both pads (about 125 N per side) on a 1.2 N cube, by
the server's ``grip_force_n`` (the sum over every pad contact point; an earlier
"~380 N" figure was wrong).  An offline step of that recorded state reproduced
the creep, and MuJoCo's no-slip pass removed it.  That pass is now the
simulator default (ADR-0005), so a hold made with it is a different physics
configuration from the earlier ones.  Physical force realism is unvalidated.
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
    joint_path,
    link_frames,
    resolve_within_limits,
    sample_count,
    within_limits,
)
from reachy_ai.motion.kinematics import RESOLVE_MAX_ATTITUDE_DEG, _rotvec
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
    own axis the jaw closes along ("x": pads on the object's +/-x faces).  No
    y-face straddle was found in what was examined (the lean staggers the pads
    ~6 cm vertically; best worst-case margin -5.1 mm), so it is not offered.
    That finding is
    limited to what was examined: the cube at r2c2, opening -65 deg, a
    48-case grid of tilt/lean/jaw-shift and a 12-start optimiser of the worst
    margin, with this rig's joint limits.  It is not a claim about other
    cells, openings or hand models.
    """
    face_axis: str = "x"
    tilt_deg: float = 30.0
    lean_bearing_deg: float = 59.52
    opening_deg: float = -65.0
    #: The wider release (#173): the gripper's travel ends at -68.75 deg, and
    #: this stays 0.75 deg inside it.  Used only when the measured state says
    #: the usual opening leaves a pad under the allowance; offline, the extra
    #: 3 deg took the finger 0.3-0.9 mm further from the cube it had followed.
    wide_opening_deg: float = -68.0
    #: How far the hand backs away along the jaw, on the thumb's side, while
    #: the grip lets go (#174 drag study).  0 for the lift: its put-back
    #: relies on the release drag undoing the ~8 mm the close moves the
    #: object toward the thumb (offline, a 9 mm retreat put 16 of 20 recorded
    #: lifts back more than 5 mm out).  A placement sets it (`crane_place_live`).
    release_retreat_m: float = 0.0
    #: How far the hand backs out along its own axis (up the approach) while
    #: the grip lets go; 0 for the lift.  See `crane_place_live.PlaceSpec`.
    release_backoff_m: float = 0.0
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
#: EMPIRICAL INPUTS FROM TWO RUNS, not guaranteed bounds: they are the largest
#: errors seen so far, used as the go/no-go tolerance, and a later run can
#: exceed them -- which is why every rung is re-measured rather than trusted.
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
    #: A separating retreat (#173, O1) may START with a pad under the
    #: allowance -- the release left it there -- but not deeper into the
    #: object than this.
    separation_start_floor_m: float = 0.0005
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
    #: Why it is stale (`feedback.latest`): None when fresh.
    stale_reason: Optional[str] = None


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


def _gap_resolve(rot: np.ndarray, gap_target: np.ndarray, gripper_deg: float,
                 q_service: Sequence[float], tol: float,
                 rot_weight_m: float) -> Tuple[List[float], float]:
    """An out-of-travel IK answer for the jaw-gap centre, re-solved INSIDE the
    travel (``kinematics.resolve_within_limits``): the gap position is the
    target, the hand rotation a soft term weighted by ``rot_weight_m``."""
    tgt = np.asarray(gap_target, dtype=float)
    Rt = np.asarray(rot, dtype=float)

    def residual(q):
        return np.concatenate([tgt - G.hand_frames(list(q), gripper_deg).gap_mid,
                               rot_weight_m * _rotvec(Rt @ G.hand_rotation(list(q)).T)])
    return resolve_within_limits(residual, q_service, "right", tol_m=tol)


#: Soft-attitude weights (m per rad) for the bounded re-solve.  Strict solves
#: must hold the orientation (judged <= 1 deg by the caller); relaxed ones let
#: it give way as far as the stops require (the plan's checks judge the drift).
_RESOLVE_ROT_WEIGHT_STRICT_M = 0.5
_RESOLVE_ROT_WEIGHT_RELAX_M = 0.05


def solve_gap(arm, rot: np.ndarray, gap_target: np.ndarray, gripper_deg: float,
              q0: Sequence[float], tol: float, iters: int = 8,
              relax: bool = False) -> Optional[Tuple[List[float], float]]:
    """IK for the jaw-gap centre (not the wrist) at ``gap_target``.

    The IK service solves for the wrist frame; the gap centre's offset from
    it depends on the wrist roll and the opening, so the wrist target is
    corrected by the remaining gap error and re-solved, warm-started.

    The service minimises without the joints' stops, so near one it answers
    with a pose slightly past it.  Such an answer is never commanded and never
    clipped: it is re-solved inside the exact travel on the local chain, with
    the gap position as the target and the attitude soft.  Without ``relax``
    the attitude must still be held (the caller judges it, <= 1 deg); with
    ``relax`` it may drift as far as reach and the stops require, and the
    rest of the ladder takes the achieved attitude as its target.  Returns
    (joints, gap error m), or None.
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
            w = _RESOLVE_ROT_WEIGHT_RELAX_M if relax else _RESOLVE_ROT_WEIGHT_STRICT_M
            q2, e2 = _gap_resolve(rot, gap_target, gripper_deg, q1, tol, w)
            # Without relax the attitude is a requirement, not a preference:
            # checked here, because not every caller re-checks it.
            if e2 < tol and within_limits(q2) and (
                    relax or _rot_err_deg(G.hand_rotation(q2), rot) <= RESOLVE_MAX_ATTITUDE_DEG):
                return q2, e2
            return None
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

    # 0. Every commanded pose inside the exact joint travel.  The streamed
    # lines between them are joint-space interpolations, so they stay inside
    # a box of limits whenever their ends do.
    for name, q in [(f"rung +{k}", r) for k, r in enumerate(plan.rungs)] + [("PRESENT", plan.present)]:
        if not within_limits(q):
            raise CraneRefused("joint_limits", f"{name} is outside the joint travel",
                               pose=[round(float(v), 3) for v in q])
    rep["joint_limits"] = "pass (every rung and PRESENT inside the travel)"

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
                    on_event: Optional[EventLog] = None,
                    starts: Optional[Sequence[Sequence[float]]] = None) -> CranePlan:
    """Plan and check the crane pick.  IK/FK service calls only -- no motion.

    ``pose`` is the object's live pose (the scene model has no live
    orientation).  ``starts`` replaces the grasp's IK start set
    (`GRASP_STARTS`); the placement's own attitude brings its own.  Raises
    CraneRefused.
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
    start_set = list(GRASP_STARTS if starts is None else starts)
    for start in start_set:
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
    ev("plan_grasp_starts", tried=len(start_set), converged=len(pre),
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


# ── The way out after the release (#173) ──────────────────────────────────────
#
# Four of the ten Phase 1.5 lifts refused their withdrawal: the release drags
# the cube 5-12 mm (the clamped pads unload into it as the finger opens) and
# it can come to rest against a pad, so the first sample of the way out -- the
# pose the hand is ALREADY in -- is under the allowance, and a correction of
# the climb cannot move it.  Two changes, chosen offline on the ten recorded
# release states (Reachy-Lab issue173-replay):
#
#   * O2, CHECK BEFORE OPENING: open only until the grip lets go, judge the
#     way out for the fully opened hand from the measured state, and open no
#     further if there is none.  The usual opening is tried first, then the
#     wider one; after opening, the measured state is judged again, and the
#     hand opens to the wider angle if a pad still sits under the allowance.
#   * O1, A SEPARATING RETREAT: if the way straight up (or the existing
#     sideways correction) is not clear, the first leg may move along the
#     jaw's closing axis from a start under the allowance, provided the pad
#     under it moves strictly away from the object until it clears it, the
#     other pad stays clear, and no pad starts deeper than
#     `CraneLimits.separation_start_floor_m`.  That leg flies with the monitor
#     off, as the existing correction does; the climb after it is monitored.

@dataclass
class WayOut:
    """A judged way out of the hand after the release.

    ``path`` starts at the last commanded pose.  ``path[1:1 + unmonitored]``
    flies with the monitor off (a correction or separating leg, whose start
    may be in contact with nothing more than the object just released); the
    rest is the climb, monitored.  ``kind`` is ``direct``, ``sideways`` (the
    pre-#173 correction) or ``separating`` (#173, O1).
    """
    kind: str
    path: List[List[float]]
    plan: CranePlan
    unmonitored: int
    report: Dict[str, object]
    shift_mm: Optional[List[float]] = None


def rungs_above(plan: CranePlan, cmd: Sequence[float], above: int) -> List[int]:
    """The ladder rungs above the commanded pose ``cmd``, up to the hover
    (``above`` is the highest rung the hand is at or below)."""
    s = plan.spec
    k = min(max(above, 0), s.hover_rungs)
    z_now = G.hand_frames(cmd, s.opening_deg).gap_mid[2]
    while k < s.hover_rungs and \
            G.hand_frames(plan.rungs[k], s.opening_deg).gap_mid[2] <= z_now + 1e-4:
        k += 1
    return list(range(k, s.hover_rungs + 1))


def _onward(plan: CranePlan) -> List[List[float]]:
    s = plan.spec
    return [plan.rungs[j] for j in range(s.hover_rungs + 1, s.via_rungs + 1)] + \
        [plan.present]


def _pad_series(path, gm: float, live: G.Box, off) -> Tuple[List[float], List[float]]:
    """Thumb and finger clearance to ``live`` at every 0.25 deg sample of the
    commanded ``path``, predicted with the measured offset (the samples of
    `grasp_alignment.route_clearance`)."""
    th, fi = [], []
    for q in G.densify(path, 0.25):
        h = G.hand_frames(np.asarray(q) + off, gm)
        th.append(G.obb_separation(h.thumb, live))
        fi.append(G.obb_separation(h.finger, live))
    return th, fi


def judge_way_out(planner, scene: SceneModel, plan: CranePlan, path, gm: float,
                  live: G.Box, off, *, separating: int = 0,
                  label: str = "") -> Tuple[bool, Dict[str, object]]:
    """Both pads against the measured object and the table, along ``path`` as
    it will be flown (predicted from the measured offset), plus the unchanged
    whole-arm check onward to PRESENT.

    Every sample must clear the allowance, except -- with ``separating`` > 0
    -- the first ``separating`` samples, where a pad may start under it if it
    starts no deeper than the floor and its clearance strictly increases
    until it is clear (O1).
    """
    s, lim = plan.spec, plan.limits
    th, fi = _pad_series(path, gm, live, off)
    climb = (G.hand_frames(path[-1], gm).gap_mid[2]
             - G.hand_frames(path[0], gm).gap_mid[2])
    tol = prediction_tolerance(max(climb, s.rung_m))
    pred = [list(np.asarray(q) + off) for q in G.densify(path, 0.5)]
    tab = G.table_clearance(scene, pred, gm)
    route = G.route_clearance(path, gm, live, offset=off)
    rep: Dict[str, object] = {
        "label": label, "route": route.as_dict(), "tolerance_mm": round(1000 * tol, 3),
        "table_mm": {k: round(1000 * v, 2) for k, v in tab.by_link.items()},
        "start_mm": [round(1000 * th[0], 2), round(1000 * fi[0], 2)],
        "path_rungs": len(path) - 1}
    if separating:
        ok = True
        for series in (th, fi):
            if series[0] < -lim.separation_start_floor_m:
                ok = False
            seg = series[:separating + 1]
            if any(a < tol and b <= a for a, b in zip(seg, seg[1:])):
                ok = False
        ok = ok and min(th[separating:]) >= tol and min(fi[separating:]) >= tol
        rep["separating_samples"] = separating
    else:
        ok = min(th) >= tol and min(fi) >= tol
    ok = ok and tab.worst_m >= 0.0
    if ok:
        try:
            planner.check_arm_path(
                list(path) + _onward(plan), segment=f"withdrawal ({label})",
                gripper_deg=gm,
                target=TargetContact(plan.object_id, hand_contact=True,
                                     center=tuple(live.center)))
        except ArmClearanceError as exc:
            ok = False
            rep["check_arm_path"] = str(exc)
    rep["ok"] = ok
    return ok, rep


def _shifted(planner, plan: CranePlan, cmd, d: np.ndarray):
    """``cmd`` moved by ``d`` (jaw-gap, world) and the ladder moved with it."""
    s, lim = plan.spec, plan.limits
    got = solve_gap(planner._arm, G.hand_rotation(cmd),
                    G.hand_frames(cmd, s.opening_deg).gap_mid + d, s.opening_deg, cmd,
                    lim.ik_position_tol_m, iters=20, relax=True)
    if got is None:
        return None, None
    moved = replace(plan, gap_targets=[t + d for t in plan.gap_targets],
                    rungs=_shift_ladder(planner, plan, d), shift=plan.shift + d,
                    checks={})
    return got[0], moved


#: Every kind of way out, in the order `plan_way_out` tries them.
WAY_OUT_KINDS = ("direct", "sideways", "separating")
#: What the check before opening looks for first, at either opening.
PRE_OPEN_PREFERRED = ("direct", "sideways")


def release_openings(spec: CraneSpec) -> Tuple[float, float]:
    """The openings the check before opening tries, in order: the wider one
    first.  Offline (issue173-replay), every release that opened only to the
    usual angle and later needed the wider one had left the finger resting on
    the cube in between; opening wide at once left it clear."""
    return (spec.wide_opening_deg, spec.opening_deg)


def plan_way_out(planner, scene: SceneModel, plan: CranePlan, cmd, above: int,
                 gm: float, live: G.Box, off, label: str = "",
                 kinds: Sequence[str] = WAY_OUT_KINDS
                 ) -> Tuple[Optional[WayOut], List[Dict[str, object]]]:
    """The first clear way out, in order: straight up the ladder; the bounded
    sideways correction that equalises the route's worst clearances (as
    before #173); the separating retreat (#173, O1).  Returns it (or None)
    and the report of every candidate judged.  ``kinds`` restricts the two
    corrections (the offline comparison of options uses it); the way straight
    up is always tried first."""
    lim = plan.limits
    reports: List[Dict[str, object]] = []
    cmd = list(cmd)

    path = [cmd] + [plan.rungs[j] for j in rungs_above(plan, cmd, above)]
    ok, rep = judge_way_out(planner, scene, plan, path, gm, live, off,
                            label=f"{label} direct")
    reports.append(rep)
    if ok:
        return WayOut("direct", path, plan, 0, rep), reports

    def via(d: np.ndarray, kind: str, separating: bool):
        got, moved = _shifted(planner, plan, cmd, d)
        if got is None:
            reports.append({"label": f"{label} {kind}", "ok": False,
                            "reason": "IK found no shifted pose"})
            return None
        p = [cmd, got] + [moved.rungs[j] for j in rungs_above(moved, cmd, above)]
        n = len(G.densify([cmd, got], 0.25)) - 1 if separating else 0
        ok_, rep_ = judge_way_out(planner, scene, moved, p, gm, live, off,
                                  separating=n, label=f"{label} {kind}")
        rep_["shift_mm"] = (1000 * d).round(2).tolist()
        reports.append(rep_)
        if ok_:
            return WayOut(kind, p, moved, 1, rep_, rep_["shift_mm"])
        return None

    # Sideways, as before #173: equalise the route's worst clearances, and
    # refuse rather than cap a shift that would exceed the bound.
    route = G.route_clearance(path, gm, live, offset=off)
    d = 0.5 * (route.thumb_m - route.finger_m) * np.asarray(route.end.face_normal)
    if "sideways" in kinds and float(np.linalg.norm(d)) <= lim.max_lateral_correction_m \
            and "check_arm_path" not in rep:
        found = via(d, "sideways", separating=False)
        if found is not None:
            return found, reports

    if "separating" not in kinds:
        return None, reports
    # Separating (O1): equalise the START clearances along the jaw's closing
    # axis, bounded, so the pad the release left under the allowance moves
    # away first; the climb follows the ladder shifted the same way.
    start = G.pad_clearance(G.hand_frames(np.asarray(cmd) + off, gm), live)
    d = 0.5 * (start.thumb_m - start.finger_m) * np.asarray(start.face_normal)
    norm = float(np.linalg.norm(d))
    if norm > lim.max_lateral_correction_m:
        d = d * (lim.max_lateral_correction_m / norm)
    found = via(d, "separating", separating=True)
    return found, reports


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

#: In-hand slip bound for the hold: the tolerance to which the pad boxes this
#: measurement is made from are verified against the compiled MJCF
#: (tests/unit/test_crane_grasp_alignment.py, 0.1 mm).  The measurement itself
#: read exactly zero spread over 286 static states in the recorded pause, so
#: this, not sensor noise, is its resolution.
SLIP_RESOLUTION_M = 0.0001


def _in_hand(qm: Sequence[float], gm: float, o: Observation) -> np.ndarray:
    """The object's centre in the thumb frame (m): where it sits in the hand."""
    h = G.hand_frames(qm, gm)
    return h.thumb_rot.T @ (np.asarray(o.position) - h.thumb.center)


def execute_crane_pick(robot, planner, scene: SceneModel, plan: CranePlan,
                       observe: Observe, *, on_event: Optional[EventLog] = None,
                       should_abort: Optional[Callable[[], bool]] = None,
                       stow: bool = True, place=None) -> Dict[str, object]:
    """Fly a checked crane plan from PRESENT and back.  See the module doc.

    ``place`` (a `crane_place_live.PlacePlan` whose ``pick`` is ``plan``)
    turns the lift into a placement (#174): after the hold the object is
    carried to the destination and set down there instead of where it was
    picked up; a halt on the way flies every move since the hold back in
    reverse and sets it down where it was picked up.

    Returns an outcome dict, including ``criteria`` (the success criteria
    fixed before motion, each judged from measurements).  A refusal or a halt
    is an outcome, not an exception.  The arm must be at PRESENT and turned
    on.  All motion goes through ``motion.primitives``; nothing is commanded
    that the plan's checks, or a re-check made immediately before it, did not
    cover.  Every connecting move starts from the LAST COMMANDED pose: the arm
    sits short of its command under gravity, and commanding the measured pose
    would let it sag by that much again (measured 2026-10-06: the thumb pad
    dropped 4.3 mm onto the table that way).

    ``stow`` ends with the documented stow to HOME; without it the arm is left
    at PRESENT for the caller (the panel's abilities end there).
    """
    ev = on_event or (lambda *_a, **_k: None)
    arm = robot.r_arm
    s, lim = plan.spec, plan.limits
    t_start = time.monotonic()
    outcome: Dict[str, object] = {"posture_held": None, "aligned_at_hover": None,
                                  "straddled_before_close": None, "closed": False,
                                  "lifted_and_held": None, "replaced": None,
                                  "withdrawal": None, "halt": None, "refusal": None,
                                  "final": None, "criteria": None}
    first = observe()
    c0 = np.asarray(first.position, dtype=float)
    rest_lowest = (G.object_box(scene.get(plan.object_id), first.position,
                                first.quat_wxyz).lowest_z - scene.table_surface_z)
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

    # `cmd`: the last commanded arm pose -- every move starts from it.
    # `ref`: where the object must stay while the monitor is on.
    # `above`: the highest ladder rung the arm is at or below.
    st = {"monitor": False, "cmd": list(plan.present), "ref": c0.copy(),
          "above": s.via_rungs}
    cur = {"plan": plan}

    def P_() -> CranePlan:
        return cur["plan"]

    def monitor() -> Optional[str]:
        if not st["monitor"]:
            return None
        o = observe()
        if o.age_s > FEEDBACK_STALE_S:
            return (f"feedback stale ({o.age_s:.2f} s"
                    + (f", {o.stale_reason})" if o.stale_reason else ")"))
        if st.get("carrying"):
            # The object is HELD: pad force is the grip, and the object moves
            # with the hand.  What stops a carry is losing the grip.  Where the
            # object sits in the hand is measured at every stop (`in_hand`),
            # not mid-move, where its pose and the joints are read at
            # different moments.
            if o.grip_force_n <= lim.contact_force_n:
                return "the grip force read zero while carrying: the object is not held"
            if st.get("aborting"):
                return None       # the way back is finished whatever is asked
            if time.monotonic() - t_start > lim.max_attempt_s:
                return "attempt time limit"
            if should_abort is not None and should_abort():
                return "aborted by the caller"
            return None
        if o.grip_force_n > lim.contact_force_n:
            return f"grip force {o.grip_force_n:.3f} N (a pad touched something)"
        moved = float(np.linalg.norm(np.asarray(o.position) - st["ref"]))
        if moved > lim.object_moved_m:
            return f"object moved {1000 * moved:.2f} mm"
        if st.get("finishing"):
            # The way home is finished whatever is asked, as the pick arc
            # finishes past its close; pad force and the object still stop it.
            return None
        if time.monotonic() - t_start > lim.max_attempt_s:
            return "attempt time limit"
        if should_abort is not None and should_abort():
            return "aborted by the caller"
        return None

    def stream(b, rate_hz: int, max_step_deg: float, check: bool = True,
               stop_when: Optional[Callable[[], bool]] = None) -> bool:
        """Command the joint line from the last commanded pose to ``b``.
        Returns False if ``stop_when`` ended it early."""
        a = st["cmd"]
        n = max(2, math.ceil(max(abs(float(y) - float(x)) for x, y in zip(a, b))
                             / max_step_deg) + 1)
        for q in joint_path(a, b, n)[1:]:
            if check:
                why = monitor()
                if why:
                    raise _Halt(why)
            P.execute_trajectory(arm, [q], R_ARM_JOINTS, rate_hz=rate_hz)
            st["cmd"] = list(q)
            if stop_when is not None and stop_when():
                return False
        return True

    def hold(seconds: float, check: bool = True) -> None:
        q = st["cmd"]
        for _ in range(max(1, int(seconds * STEP_HZ))):
            if check:
                why = monitor()
                if why:
                    raise _Halt(why)
            P.execute_trajectory(arm, [q], R_ARM_JOINTS, rate_hz=STEP_HZ)

    def measure(rung: Optional[int], label: str):
        plan_ = P_()
        qm, gm = _read(arm)
        cmd = st["cmd"]
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
           thumb_above_table_mm=round(1000 * (hm.thumb.lowest_z - scene.table_surface_z), 2),
           finger_above_table_mm=round(1000 * (hm.finger.lowest_z - scene.table_surface_z), 2),
           grip_force_n=o.grip_force_n, object=[round(v, 5) for v in o.position],
           object_moved_mm=round(1000 * float(np.linalg.norm(np.asarray(o.position) - c0)), 3))
        return qm, gm, off, live, hm, pc

    def judge(at_rung: int, off, gm, live) -> Dict[str, object]:
        """check_crane_plan on the current plan, from where the arm is."""
        return check_crane_plan(planner, scene, replace(P_(), obj=live), offset=off,
                                gripper_open_deg=gm, from_rung=at_rung)

    def correct(at_rung: int, budget: int, label: str, m) -> Tuple:
        """Judge; if the pads' route, the predicted straddle or the predicted
        hand-in-table fails, make up to ``budget`` measured corrections, each
        re-solved and re-checked before it is commanded."""
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
            move = [st["cmd"], new.rungs[at_rung]]
            try:
                planner.check_arm_path(move, segment=f"{label} correction {i + 1}",
                                       gripper_deg=gm,
                                       target=TargetContact(plan_.object_id, hand_contact=True))
            except ArmClearanceError as exc2:
                raise CraneRefused("check_arm_path", str(exc2)) from exc2
            pred = [list(np.asarray(q) + off) for q in G.densify(move, 0.25)]
            tr = G.table_clearance(scene, pred, gm)
            mv = G.route_clearance(move, gm, live, offset=off)
            if tr.worst_m < 0.0 or mv.min_m <= 0.0:
                raise CraneRefused("correction", "the correcting move itself is not clear",
                                   table_m=tr.worst_m, pads_m=mv.min_m)
            cur["plan"] = new
            ev("CORRECTION_COMMANDED", label=label, iteration=i + 1,
               q_from=[round(v, 3) for v in st["cmd"]],
               q_to=[round(v, 3) for v in new.rungs[at_rung]])
            stream(new.rungs[at_rung], STEP_HZ, 0.05)
            hold(1.5)
            qm, gm, off, live, hm, pc = measure(at_rung, f"{label} after correction {i + 1}")
        raise AssertionError("unreachable")

    def measured_state():
        """The hand and the object as they are now, with the joint offset
        from the last commanded pose."""
        plan_ = P_()
        qm, gm = _read(arm)
        off = G.joint_offset(qm, st["cmd"])
        o = observe()
        live = G.object_box(scene.get(plan_.object_id), o.position, o.quat_wxyz)
        return qm, gm, off, o, live

    def way_out(label: str, gripper_deg: Optional[float] = None,
                kinds: Sequence[str] = WAY_OUT_KINDS):
        """`plan_way_out` from the measured state; ``gripper_deg`` judges the
        hand as it WILL be opened (the check before opening) rather than as
        it is."""
        qm, gm, off, o, live = measured_state()
        g = gm if gripper_deg is None else gripper_deg
        found, reports = plan_way_out(planner, scene, P_(), st["cmd"], st["above"],
                                      g, live, off, label=label, kinds=kinds)
        return found, reports, o

    def check_before_opening() -> Optional[float]:
        """O2: with the grip let go, the opening to open to; None when no
        opening has a way out.  An opening the pads clear without a
        separating retreat is preferred, and the wider opening before the
        usual one (`release_openings`)."""
        for kinds in (PRE_OPEN_PREFERRED, WAY_OUT_KINDS):
            for opening in release_openings(s):
                found, reports, _o = way_out(f"before opening to {opening:.0f} deg",
                                             gripper_deg=opening, kinds=kinds)
                ev("WITHDRAWAL_PRECHECK", opening_deg=opening, ok=found is not None,
                   way_out=found.kind if found else None, allowed=list(kinds),
                   candidates=[{k: v for k, v in r.items() if k != "route"}
                               for r in reports])
                if found is not None:
                    return opening
        return None

    def withdraw() -> Dict[str, object]:
        """Judge the way out from where the object now is; open wider if a pad
        is still under the allowance; fly an unmonitored correcting or
        separating leg if one is needed, then climb with the monitor on.
        Returns the withdrawal record; raises _Halt when it refuses or trips."""
        found, reports, o = way_out("after release")
        opened_wider = False
        if found is None or found.kind != "direct":
            qm, gm = _read(arm)
            if gm > s.wide_opening_deg + 0.5:
                # A pad under the allowance with the usual opening: open wider
                # (inside the travel) and judge again before anything moves.
                P.smooth_move(arm, {"r_gripper": s.wide_opening_deg}, 0.3)
                hold(0.5, check=False)
                opened_wider = True
                # The hand is no longer the one judged: judge it again.
                found, reports_w, o = way_out("after opening wider")
                reports += reports_w
        last = reports[-1] if reports else {}
        ev("WITHDRAWAL_JUDGED", label="after release", ok=found is not None,
           way_out=found.kind if found else None, opened_wider=opened_wider,
           route=(found.report if found else reports[0]).get("route"),
           tolerance_mm=(found.report if found else reports[0]).get("tolerance_mm"),
           object=[round(v, 5) for v in o.position], grip_force_n=o.grip_force_n,
           candidates=[{k: v for k, v in r.items() if k != "route"} for r in reports])
        if found is None:
            raise _Halt("withdrawal refused: no clear way out of the replaced object "
                        "(straight up, a bounded sideways shift or a separating retreat)",
                        judged=reports, last=last)
        corrected = None
        if found.unmonitored:
            corrected = {"way_out": found.kind, "shift_mm": found.shift_mm,
                         "route": found.report.get("route"),
                         "start_mm": found.report.get("start_mm")}
            ev("WITHDRAWAL_CORRECTION", **corrected)
            cur["plan"] = found.plan
            for q in found.path[1:1 + found.unmonitored]:
                stream(q, STEP_HZ, 0.05)        # the monitor is off here
            hold(0.5)
        # Climb, the monitor on: a pad force or the object moving stops it.
        st["ref"] = np.asarray(observe().position, dtype=float)
        st["monitor"] = True
        for q in found.path[1 + found.unmonitored:]:
            stream(q, STEP_HZ, 0.05)
        st["above"] = s.hover_rungs
        return {"way_out": found.kind, "opened_wider": opened_wider,
                "judged": [{k: v for k, v in r.items() if k != "route"} for r in reports],
                "corrected": corrected}

    def back_to_present(opening: float) -> None:
        """Hover -> via -> PRESENT by the checked ladder; pad force still watched."""
        plan_ = P_()
        st["finishing"] = True
        for j in range(max(st["above"], s.hover_rungs), s.via_rungs):
            stream(plan_.rungs[j + 1], CARRY_HZ, 2.0)
        P.smooth_move(arm, {"r_gripper": s.transit_opening_deg}, 0.5)
        stream(plan_.present, CARRY_HZ, 2.0)
        st["monitor"] = False
        hold(1.0, check=False)
        ev("phase", name="back at PRESENT")

    def retreat_after_halt() -> bool:
        """After a halt before the close: the measured pose's way back to the
        checked ladder is checked first, then the ladder is climbed."""
        st["monitor"] = False
        plan_ = P_()
        qm, gm = _read(arm)
        k = min(st["above"], s.via_rungs)
        link = [st["cmd"], plan_.rungs[k]]
        off = G.joint_offset(qm, st["cmd"])
        try:
            planner.check_arm_path(link, segment="retreat to the ladder", gripper_deg=gm,
                                   target=TargetContact(plan_.object_id, hand_contact=True))
            pred = [list(np.asarray(q) + off) for q in G.densify(link, 0.25)]
            tr = G.table_clearance(scene, pred, gm)
            if tr.worst_m < 0.0:
                raise CraneRefused("table", f"retreat line puts {tr.link} in the table")
        except (ArmClearanceError, CraneRefused) as exc:
            ev("RETREAT_REFUSED", reason=str(exc))
            return False
        P.smooth_move(arm, {"r_gripper": s.opening_deg}, 0.8)
        stream(plan_.rungs[k], STEP_HZ, 0.05, check=False)
        for j in range(k, s.hover_rungs):
            stream(plan_.rungs[j + 1], STEP_HZ, 0.05, check=False)
        st["above"] = max(k, s.hover_rungs)
        back_to_present(s.opening_deg)
        return True

    def lower_to_support(plan_: CranePlan, k_: int):
        """Lower down ``plan_``'s ladder from rung ``k_`` until the object meets
        the table (its intended support contact), never past where the loaded
        hand's predicted pads would meet it.  Returns (reached, object xyz)."""
        sp = plan_.spec
        qm, gm = _read(arm)
        off = G.joint_offset(qm, st["cmd"])
        lower = [st["cmd"]] + [plan_.rungs[j] for j in range(k_ - 1, -1, -1)] if k_ > 0 \
            else [st["cmd"]]
        dense = G.densify(lower, 0.05)
        # Both pads stay out of the table on the way down, predicted from the
        # loaded hand's measured offset: lower no further than that allows.
        floor = len(dense) - 1
        for i, q in enumerate(dense):
            if G.table_clearance(scene, [list(np.asarray(q) + off)], sp.close_deg).worst_m < 0.0:
                floor = max(i - 1, 0)
                break
        ev("REPLACE_PLAN", setpoints=len(dense), floor_index=floor,
           pads_table_limited=floor < len(dense) - 1)

        def supported() -> bool:
            o_ = observe()
            if o_.age_s > FEEDBACK_STALE_S:
                return False          # no fresh feedback is never support
            b = G.object_box(scene.get(plan_.object_id), o_.position, o_.quat_wxyz)
            return b.lowest_z - scene.table_surface_z <= rest_lowest + lim.object_moved_m

        reached = supported()
        for q in dense[1:floor + 1]:
            if reached:
                break
            P.execute_trajectory(arm, [q], R_ARM_JOINTS, rate_hz=STEP_HZ)
            st["cmd"] = list(q)
            reached = supported()
        hold(0.3, check=False)
        o = observe()
        qm, gm = _read(arm)
        hm = G.hand_frames(qm, gm)
        ev("SUPPORTED", reached=reached, object=list(o.position), quat=list(o.quat_wxyz),
           thumb_above_table_mm=round(1000 * (hm.thumb.lowest_z - scene.table_surface_z), 2),
           finger_above_table_mm=round(1000 * (hm.finger.lowest_z - scene.table_surface_z), 2),
           grip_force_n=o.grip_force_n)
        # Height of the hand above the grasp rung: where `above` sits now.
        z_cmd = G.hand_frames(st["cmd"], sp.close_deg).gap_mid[2]
        st["above"] = max([j for j in range(0, sp.hover_rungs + 1)
                           if G.hand_frames(plan_.rungs[j], sp.close_deg).gap_mid[2]
                           <= z_cmd + 1e-4] or [0])
        return reached, np.asarray(o.position, dtype=float)

    def release_retreat(sp: CraneSpec):
        """The placement's release move (#174): the hand backed off along the
        jaw, on the thumb's side, by ``sp.release_retreat_m``, the ladder moved
        with it.  Checked first (whole arm, table predicted from the measured
        offset); None -- the release of #173 -- when it is off or not clear.
        Returns (joint line to stream in step with the let-go, moved plan)."""
        if sp.release_retreat_m <= 0.0 and sp.release_backoff_m <= 0.0:
            return None
        plan_ = P_()
        qm, gm = _read(arm)
        o = observe()
        hm_ = G.hand_frames(qm, gm)
        d = -sp.release_retreat_m * hm_.jaw - sp.release_backoff_m * hm_.approach
        q_t, moved = _shifted(planner, plan_, st["cmd"], d)
        why = None
        if q_t is None:
            why = "IK found no pose for it"
        else:
            move = [st["cmd"], q_t]
            try:
                planner.check_arm_path(move, segment="release retreat",
                                       gripper_deg=sp.opening_deg,
                                       target=TargetContact(plan_.object_id, hand_contact=True,
                                                            center=tuple(o.position)))
            except ArmClearanceError as exc:
                why = str(exc)
            if why is None:
                off = G.joint_offset(qm, st["cmd"])
                pred = [list(np.asarray(q) + off) for q in G.densify(move, 0.25)]
                for g in (gm, sp.opening_deg):
                    tr = G.table_clearance(scene, pred, g)
                    if tr.worst_m < 0.0:
                        why = f"{tr.link} would reach the table ({1000 * tr.worst_m:.1f} mm)"
                        break
        if why is not None:
            ev("RELEASE_RETREAT", ok=False, reason=why)
            return None
        n = max(2, int(0.7 * int(0.8 * P._INTERP_HZ)) + 1)
        ev("RELEASE_RETREAT", ok=True, retreat_mm=round(1000 * sp.release_retreat_m, 1),
           shift_mm=(1000 * d).round(2).tolist(), q_to=[round(v, 3) for v in q_t])
        return joint_path(st["cmd"], q_t, n)[1:], moved

    def carry_and_lower(pl, trail: List[List[float]]) -> int:
        """Re-plan the destination from the measured hold, climb, carry, and
        lower to `PlaceSpec.check_rung` above the set-down, correcting the
        held object's measured position toward the cell centre at the carry
        height and again there.  Returns the destination rung reached.
        Raises _Halt / CraneRefused (the caller aborts)."""
        from reachy_ai.tasks import crane_place_live as PL

        sp_ = pl.spec
        c = pl.carry_rung
        qm, gm = _read(arm)
        o = observe()
        held = G.object_box(scene.get(pl.object_id), o.position, o.quat_wxyz)
        rel = PL.measured_hold(qm, gm, held)
        cur_pl = PL.replan_from_hold(planner, scene, pl, rel,
                                     offset=G.joint_offset(qm, st["cmd"]))
        ev("CARRY_PLANNED", carry_rung=c, psi_deg=round(cur_pl.psi_deg, 2),
           hold_in_thumb_mm=(1000 * rel[1]).round(2).tolist(),
           predicted_hold_mm=(1000 * pl.rel[1]).round(2).tolist(),
           exits={k_: round(1000 * v["clearance_m"], 1)
                  for k_, v in cur_pl.checks.get("exits", {}).items()})
        ref = _in_hand(qm, gm, o)

        def in_hand(label: str) -> None:
            qm_, gm_ = _read(arm)
            o_ = observe()
            drift = float(np.linalg.norm(_in_hand(qm_, gm_, o_) - ref))
            ev("IN_HAND", label=label, drift_mm=round(1000 * drift, 3),
               grip_force_n=o_.grip_force_n, object=list(o_.position))
            if drift > sp_.in_hand_slip_halt_m:
                raise _Halt(f"the held object moved {1000 * drift:.1f} mm in the hand ({label})")

        def corrections(rung: int, label: str):
            nonlocal cur_pl
            for i in range(sp_.setdown_corrections + 1):
                o_ = observe()
                e = np.array([cur_pl.target[0] - o_.position[0],
                              cur_pl.target[1] - o_.position[1], 0.0])
                err = float(np.linalg.norm(e))
                rec = {"label": label, "rung": rung, "iteration": i,
                       "error_mm": round(1000 * err, 2),
                       "object": [round(v, 5) for v in o_.position]}
                outcome["place"]["corrections"].append(rec)
                ev("SETDOWN_MEASURE", **rec)
                if err <= sp_.setdown_tolerance_m or i == sp_.setdown_corrections:
                    return
                new = replan(planner, scene, P_(), e, rung)       # bounded: CraneRefused
                qm_, gm_ = _read(arm)
                off_ = G.joint_offset(qm_, st["cmd"])
                cand = replace(cur_pl, dest=new)
                PL.check_place(planner, scene, cand, offset=off_, from_rung=rung)
                move = [st["cmd"], new.rungs[rung]]
                try:
                    planner.check_arm_path(move, segment=f"set-down correction ({label})",
                                           gripper_deg=new.spec.close_deg,
                                           target=TargetContact(pl.object_id, hand_contact=True,
                                                                carried=True))
                except ArmClearanceError as exc:
                    raise CraneRefused("check_arm_path", str(exc)) from exc
                cr = G.carried_clearance(scene, pl.object_id, move, new.spec.close_deg,
                                         pl.pick.obj, cur_pl.rel)
                arm_d, _l, _i = G.carried_arm_clearance(move, new.spec.close_deg,
                                                        pl.pick.obj, cur_pl.rel)
                tr = G.table_clearance(scene, [list(np.asarray(q) + off_)
                                               for q in G.densify(move, 0.25)],
                                       new.spec.close_deg)
                if cr.others_m < 0.0 or arm_d < 0.0 or tr.worst_m < 0.0 or \
                        cr.table_m < sp_.carry_table_clearance_m * (rung > sp_.check_rung):
                    raise CraneRefused("correction", "the set-down correction move is not clear",
                                       others_m=cr.others_m, arm_m=arm_d, table_m=tr.worst_m)
                ev("SETDOWN_CORRECTION", label=label, iteration=i + 1,
                   shift_mm=(1000 * e).round(2).tolist(),
                   cumulative_mm=(1000 * new.shift).round(2).tolist())
                cur["plan"] = new
                cur_pl = cand
                stream(new.rungs[rung], STEP_HZ, 0.05)
                trail.append(list(new.rungs[rung]))
                hold(1.0)
                in_hand(f"{label}, after correction {i + 1}")

        ev("phase", name=f"carry to {pl.cell}")
        st["carrying"], st["monitor"] = True, True
        for q in cur_pl.climb()[1:]:
            stream(q, STEP_HZ, 0.1)
            trail.append(list(q))
        st["above"] = c
        stream(cur_pl.dest.rungs[c], STEP_HZ, 0.25)
        trail.append(list(cur_pl.dest.rungs[c]))
        cur["plan"] = cur_pl.dest
        hold(1.0)
        in_hand(f"{pl.cell} +{c}")
        corrections(c, "carry height")
        for j in range(c - 1, sp_.check_rung - 1, -1):
            stream(P_().rungs[j], STEP_HZ, 0.05)
            trail.append(list(P_().rungs[j]))
            st["above"] = j
        hold(0.6)
        in_hand(f"{pl.cell} +{sp_.check_rung}")
        corrections(sp_.check_rung, f"{sp_.check_rung} rungs up")
        st["carrying"], st["monitor"] = False, False
        return sp_.check_rung

    def abort_to_pickup(why: str, trail: List[List[float]]) -> None:
        """Fly every move since the hold back in reverse (each was checked),
        to the pickup ladder's lift rung; the caller sets the object down
        there.  The grip is still watched; nothing else interrupts it."""
        outcome["place"]["aborted"] = why
        ev("ABORT", reason=why, poses_to_retrace=len(trail) - 1)
        ev("phase", name="abort: carry back to where it was picked up")
        st["carrying"], st["monitor"], st["aborting"] = True, True, True
        for q in reversed(trail):
            stream(q, STEP_HZ, 0.1)
        st["carrying"], st["monitor"], st["aborting"] = False, False, False
        cur["plan"] = place.pick
        st["above"] = s.lift_rungs
        # The stop that caused this has been honoured: what is left -- set it
        # down, withdraw, go home -- is finished whatever is asked, as the
        # return already is; pad force and the object still stop it.
        st["finishing"] = True

    # ── approach: PRESENT -> via -> hover ───────────────────────────────────
    ev("phase", name="ATTEMPT_START")
    P.look_at(robot, tuple(plan.obj.center), duration=1.0)
    P.smooth_move(arm, {"r_gripper": s.transit_opening_deg}, 0.5)
    ev("phase", name="transit PRESENT -> via")
    stream(plan.via, CARRY_HZ, 2.0, check=False)
    ev("phase", name="via -> hover")
    for k in range(s.via_rungs, s.hover_rungs, -1):
        stream(plan.rungs[k - 1], CARRY_HZ, 2.0, check=False)
    st["above"] = s.hover_rungs
    P.smooth_move(arm, {"r_gripper": s.opening_deg}, 0.8)
    hold(1.5, check=False)
    st["monitor"] = True

    halt: Optional[_Halt] = None
    try:
        # ── the visible pause at the elevated approach ─────────────────────
        ev("phase", name=f"HOVER pause ({s.pause_s:.0f} s)")
        hold(s.pause_s)
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
            stream(P_().rungs[rung - 1], STEP_HZ, 0.05)
            rung -= 1
            st["above"] = rung
            hold(0.6)
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
        hold(1.0)
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
        if not retreat_after_halt():
            outcome["final"] = "left at the halt pose (retreat refused)"
            return _end(outcome, ev, observe)
        return _finish(arm, robot, observe, outcome, ev, stow)

    # ── close, lift, hold ──────────────────────────────────────────────────
    plan = P_()
    st["monitor"] = False
    ev("phase", name="close")
    z0 = float(observe().position[2])
    P.smooth_move(arm, {"r_gripper": s.close_deg}, 0.8)
    hold(1.0, check=False)
    o = observe()
    qm, gm = _read(arm)
    outcome["closed"] = True
    ev("CLOSED", gripper_measured_deg=round(gm, 2), grip_force_n=o.grip_force_n,
       grasping=o.grasping, object=list(o.position), quat=list(o.quat_wxyz))

    ev("phase", name="lift")
    lifted, k = True, 0
    for k in range(1, s.lift_rungs + 1):
        st["above"] = k
        stream(plan.rungs[k], STEP_HZ, 0.05, check=False)
        hold(0.4, check=False)
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
    hold_rec = None
    if lifted:
        ev("phase", name=f"hold ({s.hold_s:.0f} s)")
        o1 = observe()
        q1, g1 = _read(arm)
        hold(s.hold_s, check=False)
        o2 = observe()
        q2, g2 = _read(arm)
        r1, r2 = float(o1.position[2]) - z0, float(o2.position[2]) - z0
        need = s.lift_rungs * s.rung_m - lim.lift_lag_m
        slip = _in_hand(q2, g2, o2) - _in_hand(q1, g1, o1)
        hand_move = float(np.linalg.norm(G.hand_frames(q2, g2).thumb.center
                                         - G.hand_frames(q1, g1).thumb.center))
        hold_rec = {
            "ok": bool(r1 >= need and r2 >= need), "rise_start_mm": round(1000 * r1, 2),
            "rise_end_mm": round(1000 * r2, 2), "required_mm": round(1000 * need, 1),
            "in_hand_slip_mm": round(1000 * float(np.linalg.norm(slip)), 3),
            "in_hand_slip_vector_mm": (1000 * slip).round(3).tolist(),
            "hand_moved_mm": round(1000 * hand_move, 3),
            "slip_bound_mm": round(1000 * SLIP_RESOLUTION_M, 3),
            "grasping": o2.grasping, "grip_force_n": o2.grip_force_n,
            "object": list(o2.position), "quat": list(o2.quat_wxyz)}
        outcome["lifted_and_held"] = hold_rec
        ev("HELD", **hold_rec)
    else:
        outcome["lifted_and_held"] = {"ok": False}

    def stopped_carrying(reason: str):
        """A halt on the way back to the pickup: the grip was lost.  Nothing
        is opened and nothing more is flown."""
        st["monitor"] = False
        outcome["halt"] = f"stopped while carrying the object back: {reason}"
        ev("HALT", reason=outcome["halt"], stage="abort")
        outcome["final"] = "stopped on the way back to the pickup"
        outcome["criteria"] = _criteria(outcome, arm)
        return _end(outcome, ev, observe)

    # ── placement (#174): carry to the destination and set down there ──────
    # Every commanded pose after the hold goes on `trail`: each was checked
    # before it was commanded, so an abort flies them back in reverse to the
    # pickup ladder and sets the object down where it was picked up.
    trail: List[List[float]] = [list(st["cmd"])]
    at_destination = False
    if place is not None:
        outcome["place"] = {"target_cell": place.target_cell,
                            "target": [round(float(v), 5) for v in place.target],
                            "carry_rung": place.carry_rung, "psi_deg": round(place.psi_deg, 2),
                            "corrections": [], "aborted": None}
        if lifted and hold_rec and hold_rec.get("ok"):
            try:
                k = carry_and_lower(place, trail)
                at_destination = True
            except (_Halt, CraneRefused) as exc:
                why = exc.reason if isinstance(exc, _Halt) else f"{exc.stage}: {exc.reason}"
                try:
                    abort_to_pickup(why, trail)
                except _Halt as h2:
                    return stopped_carrying(h2.reason)
                k = s.lift_rungs
        else:
            outcome["place"]["aborted"] = "not carried: the lift did not hold"
    plan = P_()
    ps = plan.spec

    # ── set-down: lower until the object meets the table, then open ────────
    ev("phase", name=f"set-down at {place.cell}" if at_destination else "replace")
    reached_support, held_at = lower_to_support(plan, k)
    if place is not None and not reached_support and at_destination:
        # Never release in the air: carry it back and set it down where it
        # was picked up, on the ladder it was lifted from.
        try:
            abort_to_pickup("the object did not reach the table at the destination", trail)
        except _Halt as h2:
            return stopped_carrying(h2.reason)
        at_destination = False
        plan, k = P_(), s.lift_rungs
        ps = plan.spec
        ev("phase", name="replace")
        reached_support, held_at = lower_to_support(plan, k)
    if place is not None and not reached_support:
        st["monitor"] = False
        outcome["halt"] = outcome["halt"] or ("the object did not reach the table; holding "
                                              "it rather than release it in the air")
        ev("HALT", reason=outcome["halt"], stage="set-down")
        outcome["final"] = "holding the object above the table (not released)"
        outcome["criteria"] = _criteria(outcome, arm)
        return _end(outcome, ev, observe)
    # ── release (#173, O2): let go, CHECK, and only then open ───────────────
    # Open only until the grip force reads zero: the object is free and the
    # hand has opened no further than it took to free it.  Then judge the way
    # out for the fully opened hand from where the object came to rest, and
    # open no further if neither opening has one -- never open into a refusal.
    # A placement backs the hand away on the thumb's side while it lets go
    # (#174 drag study), when that move checks clear.
    released = lambda: observe().grip_force_n <= lim.contact_force_n  # noqa: E731
    retreat = release_retreat(ps)
    if retreat is not None:
        path, moved = retreat
        let_go, q_last = P.release_grip_retreating(arm, ps.opening_deg, released, path,
                                                   R_ARM_JOINTS, 0.8)
        if q_last is not None:
            st["cmd"] = list(q_last)
        cur["plan"] = moved
    else:
        let_go = P.release_grip(arm, ps.opening_deg, released, 0.8)
    hold(0.5, check=False)
    opening = check_before_opening()
    if opening is not None:
        P.smooth_move(arm, {"r_gripper": opening}, 0.4)
        hold(0.8, check=False)
    o = observe()
    qm, gm = _read(arm)
    hm = G.hand_frames(qm, gm)
    placed = G.object_box(scene.get(plan.object_id), o.position, o.quat_wxyz)
    w, x, y, z = o.quat_wxyz
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
    outcome["replaced"] = {
        "supported_before_open": reached_support, "object": list(o.position),
        # How far the release moved the object (horizontal, held -> released):
        # reported, target <= 5 mm, not a criterion (#173 decision).
        "release_drag_mm": round(1000 * float(np.linalg.norm((np.asarray(o.position)
                                                                - held_at)[:2])), 2),
        "let_go_deg": round(float(let_go), 2), "opening_deg": opening,
        "release_retreat_mm": round(1000 * ps.release_retreat_m, 1) if retreat else 0.0,
        "quat": list(o.quat_wxyz), "tilt_deg": round(tilt, 3),
        "lowest_above_table_mm": round(1000 * (placed.lowest_z - scene.table_surface_z), 3),
        "offset_from_start_mm": round(1000 * float(np.linalg.norm(np.asarray(o.position) - c0)), 2),
        "grip_force_after_open_n": o.grip_force_n,
        "thumb_above_table_mm": round(1000 * (hm.thumb.lowest_z - scene.table_surface_z), 2),
        "finger_above_table_mm": round(1000 * (hm.finger.lowest_z - scene.table_surface_z), 2)}
    if place is not None:
        err = math.hypot(float(o.position[0]) - float(place.target[0]),
                         float(o.position[1]) - float(place.target[1]))
        outcome["place"].update(at_destination=at_destination,
                                error_from_target_mm=round(1000 * err, 2),
                                tilt_deg=round(tilt, 3))
    ev("RELEASED", **outcome["replaced"])

    # ── withdraw: re-judged against where the object now is ────────────────
    ev("phase", name="withdrawal")
    try:
        if opening is None:
            raise _Halt("withdrawal refused before opening: with the grip let go, "
                        "neither opening has a clear way out, so the hand was not "
                        "opened further")
        outcome["withdrawal"] = withdraw()
        outcome["withdrawal"]["clean"] = True
        ev("WITHDRAWN", **outcome["withdrawal"])
        back_to_present(s.opening_deg)
    except _Halt as h:
        st["monitor"] = False
        o = observe()
        qm, gm = _read(arm)
        outcome["halt"] = outcome["halt"] or h.reason
        outcome["withdrawal"] = {"clean": False, "reason": h.reason, **_public(h.detail)}
        ev("HALT", reason=h.reason, stage="withdrawal", object=list(o.position),
           grip_force_n=o.grip_force_n, q_measured=[round(v, 3) for v in qm])
        outcome["final"] = "left beside the object (withdrawal refused or stopped)"
        outcome["criteria"] = _criteria(outcome, arm)
        return _end(outcome, ev, observe)
    outcome["criteria"] = _criteria(outcome, arm)
    return _finish(arm, robot, observe, outcome, ev, stow)


def _public(d):
    """A record without the private (non-JSON) working values."""
    if isinstance(d, dict):
        return {k: _public(v) for k, v in d.items() if not str(k).startswith("_")}
    return d


def _shift_ladder(planner, plan: CranePlan, d: np.ndarray) -> List[List[float]]:
    """The ladder's rungs re-solved with the gap moved by ``d``, keeping each
    rung's own orientation; rungs that cannot be re-solved keep their pose
    (and the caller's route check judges them)."""
    out = []
    prev = None
    for k, q in enumerate(plan.rungs):
        got = solve_gap(planner._arm, G.hand_rotation(q),
                        G.hand_frames(q, plan.spec.opening_deg).gap_mid + d,
                        plan.spec.opening_deg, prev or q, plan.limits.ik_position_tol_m,
                        iters=20, relax=True)
        out.append(got[0] if got is not None else list(q))
        prev = out[-1]
    return out


def _criteria(outcome, arm) -> Dict[str, object]:
    """The success criteria, fixed before motion (see the report), judged."""
    held = outcome.get("lifted_and_held") or {}
    rep = outcome.get("replaced") or {}
    wd = outcome.get("withdrawal") or {}
    at_present = P.stow_entry_ok(arm)[0]
    extra = {}
    pl = outcome.get("place")
    if pl is not None:
        from reachy_ai.tasks.crane_place_live import D6_POSITION_M, D6_TILT_DEG
        extra["placed_on_target"] = bool(
            pl.get("at_destination") and not pl.get("aborted")
            and pl.get("error_from_target_mm") is not None
            and pl["error_from_target_mm"] <= 1000 * D6_POSITION_M
            and pl.get("tilt_deg", 90.0) <= D6_TILT_DEG)
    return {**extra,
        "retained_height": bool(held.get("ok")),
        "slip_within_resolution": (held.get("in_hand_slip_mm") is not None
                                   and held["in_hand_slip_mm"] <= held["slip_bound_mm"]),
        "clean_placement": bool(rep.get("supported_before_open")
                                and rep.get("grip_force_after_open_n", 1.0) == 0.0
                                and rep.get("thumb_above_table_mm", -1.0) >= 0.0
                                and rep.get("finger_above_table_mm", -1.0) >= 0.0),
        "clean_withdrawal": bool(wd.get("clean")),
        "returned_to_present": bool(at_present),
    }


def _end(outcome, ev, observe):
    o = observe()
    outcome["final_object"] = list(o.position)
    ev("ATTEMPT_END", outcome=outcome)
    return outcome


def _finish(arm, robot, observe, outcome, ev, stow: bool = True):
    ok, joint, off_deg = P.stow_entry_ok(arm)
    if ok and stow:
        ev("phase", name="stow_from_side (documented PRESENT -> HOME route)")
        P.stow_from_side(robot, arm)
        outcome["final"] = "HOME"
    elif ok:
        outcome["final"] = "PRESENT"
    else:
        ev("left_at_pose", why=f"not at PRESENT ({joint} {off_deg:.0f} deg off); not stowing")
        outcome["final"] = f"not at PRESENT ({joint} {off_deg:.0f} deg off)"
    return _end(outcome, ev, observe)
