"""Side-on grasp: the hand level, a horizontal slide-in, a vertical lift (#180).

Why this exists
---------------
After #195 the crane (`crane_pick_live`, hand tilted 22.5 deg from vertical)
picks the 6 cm cube from r3c2 only: no top-down attitude straddles it from
any other cell with the palm clear.  The real robot already has another
grasp.  Siva picks and places SIDE-ON (reachy-tabletop-ai CalibrateBoard /
MovePieceLogic, analysed in #185): the hand is level at every taught square,
it comes in from the front, the right or a diagonal, and the pads close
across the object horizontally.

What a side-on plan is
----------------------
It is a `crane_pick_live.CranePlan` with TWO ladders, so the crane's executor
(`execute_crane_pick`) flies it unchanged -- the measured corrections, the
gated slide-in, the straddle check before closing, the lift, the carry, the
set-down, #173's release and way out, the abort path:

  * ``rungs`` -- the APPROACH.  Rung 0 is the grasp; rungs 1..hover go back
    along the approach, level, one rung (1 cm) each, to the STANDOFF; the
    rungs above it go straight up from the standoff to the via.  The arm
    comes down at the standoff, slides in along this line, and after a
    placement withdraws back out along the destination's own.
  * ``lift`` -- the vertical ladder above the grasp, used by the lift, the
    carry and the set-down (`CranePlan.up`).

Planning (`plan_side_pick`, `plan_side_place`)
---------------------------------------------
For each candidate APPROACH DIRECTION -- a cube: its four horizontal face
normals (so the jaw closes on two faces, whatever the cube's yaw); a cylinder:
every 15 deg (any horizontal closing direction is valid) -- and each hand
pitch in `SideSpec.pitches_deg`, the grasp is solved from Siva's taught tip
orientations as IK seeds (`TAUGHT_SEEDS`; attitudes, never positions: the
taught spacing is still open, #185), then the crane's starts.  The jaw-gap
centre is put at the object's centre line, `SideSpec.grasp_height_m` above
the table, centred between the pads, and moved back along the approach
until the palm clears the object's near face by `palm_clearance_m`.

Every crane check is reused, with the hand model that has the palm (#195):
whole-arm (`check_arm_path`), arm/pads/palm against the table (commanded
and predicted), both pads' and the palm's route along the slide-in, the
predicted straddle, the carried object.  Added here: the hand's boxes (pads,
palm, back) against every OTHER object along every hand segment, which makes
a direction blocked by a neighbour fail (`check_side_extras`); the shape and
the opening (`side_shape`, `opening_check`); the cylinder's roll risk
(`cylinder_roll_model`).  The candidate that passes everything with the
largest margin (`plan_margin`) is the plan; every candidate's result is in
the event log.

Planner choice (`plan_lift_choice`, `plan_place_choice`)
-------------------------------------------------------
Both grasps are planned; the one that passes every check with the larger
margin flies, and the record says which.

Nothing here talks to the robot except through the planner's IK/FK service
(planning) and `execute_crane_pick` (motion, `motion.primitives` only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reachy_ai.motion import grasp_alignment as G
from reachy_ai.motion import kinematics as K
from reachy_ai.motion.kinematics import (ArmClearanceError, R_ARM_JOINTS, joint_limits,
                                         link_frames, within_limits)
from reachy_ai.scene.awareness import SceneModel, segment_object_distance
from reachy_ai.tasks import crane_pick_live as CP
from reachy_ai.tasks import crane_place_live as PL
from reachy_ai.tasks.pick_place_live import present_joints

#: Ability route names (`web/panel_abilities.py`, `evaluation.panel_routes`):
#: the panel's lift and place keep theirs; which grasp flew is in the record.
GRASP_SIDE = "side-on"
GRASP_CRANE = "crane"


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SideSpec(CP.CraneSpec):
    """The side-on approach.  Inherits the crane's opening, close, release
    and rung values, which flew at the October mount (#195); the crane's
    tilt / lean / face are not used.  Evidence for every value set here:
    docs/reviews/2026-10-19-side-on-grasp-180/ (README).
    """
    #: The standoff: the slide-in starts 8 rungs (8 cm) back along the
    #: approach.  At the -65 deg opening the hand's leading part (the thumb
    #: pad) runs 29 mm ahead of the jaw-gap centre (hand_geometry.txt), so
    #: from 8 cm a 6 cm cube is at least 2 rungs clear of every pad and the
    #: palm -- the insertion checkpoint (`_insertion_rung`) is 6 at most.
    standoff_rungs: int = 8
    #: The HOVER, where the measured corrections are made, is this many rungs
    #: straight above the standoff.  At the standoff itself the level hand is
    #: at grasp height, its forearm 4 mm and its palm 9 mm above the table as
    #: COMMANDED; the arm sags ~9 mm before the hover correction lifts it, so
    #: the predicted hand there is in the table and the first flights' hover
    #: corrections were refused (flights/lift_v1.jsonl, r3c2).  2 cm up the
    #: sag is corrected first; the descent then comes down the last 2 rungs
    #: and slides in, re-measured at every rung.  ``hover_rungs`` is
    #: standoff + this (the executor's hover index into the approach).
    hover_rise_rungs: int = 2
    hover_rungs: int = 10
    #: The via: up to 15 rungs (15 cm) straight up from the standoff, so the
    #: hand crosses the board up to ~18 cm above the table, over the tallest
    #: object in the scenes (soda_can, 12.2 cm).  Near the robot a level hand
    #: cannot rise that far (the near row stopped at 6-13 cm), so the rise is
    #: as high as the IK reaches, at least `min_via_rise_rungs`; each plan's
    #: own spec carries its `via_rungs`.  The transit to it is checked whole.
    via_rungs: int = 23
    min_via_rise_rungs: int = 3
    #: The vertical ladder above the grasp: up to the highest carry height a
    #: placement uses (`crane_place_live.PlaceSpec.carry_rungs`), or as high
    #: as the IK reaches; at least the lift's own `lift_rungs`.
    lift_ladder_rungs: int = 18
    #: The highest lift rung the plan will use: the lift's own, or a
    #: placement's carry height (set by `plan_side_place`).  A correction
    #: that re-solves the lift ladder keeps at least this many rungs.
    lift_min_rungs: int = 5
    #: The jaw-gap centre's height above the table at the grasp.  With the
    #: hand level, the forearm capsule (radius 30 mm, ending at the wrist,
    #: level with the gap centre) reaches 30 mm below it, the palm and the
    #: hand back 25 mm, the pads 14 mm (hand_geometry.txt); 34 mm leaves the
    #: forearm 4 mm.  Siva's taught tips stand at about the same height
    #: (#185: z -0.3825..-0.3695 against a surface ~0.41 m below the shoulder).
    grasp_height_m: float = 0.034
    #: Hand pitch, approach tilted down from level (deg), tried in this order.
    #: Siva's taught hands are level to within 5.5 deg (taught_seeds.txt:
    #: -5.5..+4.7); 5 deg down raises the wrist and the forearm and turns
    #: r1c3's 1.4 deg joint margin level into 7.4 (reach_level.txt).
    pitches_deg: Tuple[float, ...] = (0.0, 5.0)
    #: Both pad centres at least this far below the object's top, as planned
    #: (the straddle check needs them below it; the 4 cm pool box leaves 6 mm).
    min_pad_below_top_m: float = 0.004
    #: The hand's boxes stay this far from every OTHER object on every hand
    #: segment (commanded).  10 mm: the uncorrected tracking error before the
    #: hover correction at the October mount was 5.3-5.4 mm sideways and
    #: ~9.4 mm low (#195 flights/), and no other object is watched in flight.
    neighbour_clearance_m: float = 0.010
    #: A cylinder may be gripped from any direction; tried every this many deg.
    cylinder_azimuth_step_deg: float = 15.0
    #: The close (close_study.py, search/close_study_*.jsonl).  A level hand's
    #: finger sweeps FORWARD along the gripped face as it closes: from the
    #: -65 deg opening onto a 6 cm cube it meets the face at ~-50 deg, where its
    #: contact edge moves 0.8 mm forward per mm inward, and drags the object
    #: forward while it shoves it across the gap to the thumb; if the drag
    #: carries the contact to the object's near edge, the object is squeezed
    #: out (the first red_cube flight, flights/first.md: 48 mm, nothing held).
    #: Two rules keep the shove short:
    #:
    #: * the opening leaves the object `side_slack_total_m` of free gap in all
    #:   -- 7.8 mm, what the 6 cm cube has at the crane's flown -65 deg (67.8 mm
    #:   free gap, hand_geometry.txt) -- so a 40 mm object is opened to about
    #:   -30 deg, not -65 (at -65 a pool cylinder met the finger at -40 deg,
    #:   was shoved 16 mm and tipped off the table in 4 of 6 fast closes; at
    #:   -30 and -35 every pool box and cylinder close held, drift <= 2.9 mm);
    #: * the object stands `thumb_side_m` from the thumb pad, the finger side
    #:   taking the rest: with 1.0-2.0 mm the 6 cm cube held in 29 of 30 closes
    #:   over +/-3 deg of yaw error and 12-16 mm of depth; with the slack split
    #:   evenly (3.9 mm a side, the crane's rule) one close in eleven, and the
    #:   first flight, squeezed it out.  1.5 mm is twice the largest prediction
    #:   tolerance (0.494 mm) plus 0.5 mm.
    side_slack_total_m: float = 0.0078
    thumb_side_m: float = 0.0015
    #: The close target: this far past the angle at which the finger meets
    #: the object (`closure_deg`), never past the crane's close (-5 deg).  The
    #: crane drives every close to -5 deg; side-on, the finger then over-
    #: travels ~19 deg into the 6 cm cube (12 mm of soft-contact penetration,
    #: ~250 N) and that over-travel is where it drags the cube forward.  With
    #: contact + 15 deg (-35 for the cube, -5 for a 40 mm object) the cube
    #: held in 34 of 36 closes with <= 5.9 mm drift (search/close_study_rule);
    #: contact + 5 left it at ~23 N with one 0.39 mm creep, and a pool
    #: cylinder at contact + 4.5 deg was dropped (close_study_target).
    close_past_contact_deg: float = 15.0
    #: The thumb backs off along the jaw while the grip lets go, in a lift
    #: and a placement alike (the crane's lift keeps 0: its put-back relies
    #: on the release drag undoing its 8 mm close shove).  Without it three of
    #: the first ten side-on lifts left a pad touching the object after the
    #: release (0.02-0.07 N; flights/lift_v1.jsonl).  4 mm, not the crane's
    #: 9 (`crane_place_live.PlaceSpec.release_retreat_m`): the hand moves
    #: toward the finger's side as it backs the thumb off, and side-on the
    #: finger has only the free-gap slack (7.8 mm) on that side, not the
    #: crane's slack plus its 8 mm shove -- 9 mm put the opened finger 1.2 mm
    #: into a placed 6 cm cube (the plan's way-out check, flights/place_v1).
    #: 4 mm is about half the slack.
    release_retreat_m: float = 0.004
    #: Approach directions further than this from the shoulder's bearing to
    #: the object are not tried: none of them reached in reach_level.txt
    #: (reachable spans were within -57..+69 deg of the bearing).
    max_bearing_offset_deg: float = 90.0
    #: Full plans are made for at most this many candidates, best grasp joint
    #: margin first (each costs a ladder solve and every check).
    max_full_plans: int = 4


#: Siva's taught squares as attitude SEEDS (#185; taught_seeds.json): the cell
#: (A = r3 far, C = r1 near, columns agree), the taught approach azimuth (deg,
#: the tip's -z flattened) and one joint solution of the full taught tip pose
#: inside the sim's travel.  Never used as positions.
TAUGHT_SEEDS: Dict[str, Tuple[float, Tuple[float, ...]]] = {
    "r3c1": (54.6, (-30.55, 10.384, 74.072, -33.657, -28.951, -31.018, -0.618)),
    "r3c2": (18.3, (-8.312, 13.463, 23.062, -51.43, 10.972, -24.999, -11.484)),
    "r3c3": (6.6, (-18.316, -20.688, 58.918, -63.78, -29.287, -28.23, -34.824)),
    "r2c1": (44.8, (2.207, 20.113, 37.191, -50.914, 17.402, -29.387, -1.706)),
    "r2c2": (43.9, (-21.365, -18.797, 86.053, -74.212, -24.092, -23.784, -28.841)),
    "r2c3": (2.8, (4.174, -26.422, 51.758, -81.362, -14.787, -39.262, -44.497)),
    "r1c1": (76.3, (-2.999, 0.243, 78.711, -66.715, -1.678, -20.881, -1.699)),
    "r1c2": (82.4, (-10.92, -29.91, 89.481, -86.251, -8.613, -28.241, -2.089)),
    "r1c3": (15.0, (30.865, -33.274, 46.534, -89.365, 4.85, -42.925, -43.251)),
}

#: The hand back (`r_hand_back_col`, #192), (centre, half-extents) in the
#: thumb frame, from native_mujoco/model/reachy_1_2.xml.  The crane's checks
#: use the pads and the palm; a level hand carries the back at the palm's
#: height, beside the board, so it is checked here too.
HAND_BACK = ((0.0, -0.018, -0.003), (0.025, 0.028, 0.019))

#: Contact friction for the roll model.  The pads carry MuJoCo's default 1.0
#: and the pair takes the larger of the two geoms' values, so the contact has
#: at least 1.0; the scene's cylinders declare 0.75 (FWDCenterLabMCC /
#: SivaPool).  The squeeze-out bound uses the lower, the tipping test the higher.
PAD_FRICTION = 1.0
CYLINDER_FRICTION = 0.75


# ── Geometry ──────────────────────────────────────────────────────────────────

def side_rotation(azimuth_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """The hand rotation (wrist frame = SDK tip frame) for a level hand whose
    approach points at ``azimuth_deg`` (world, from +x toward +y), tilted
    ``pitch_deg`` down: the thumb frame's z is minus the approach, its x is
    up (as in every taught pose), so the jaw closes horizontally."""
    a, b = math.radians(azimuth_deg), math.radians(pitch_deg)
    app = np.array([math.cos(a) * math.cos(b), math.sin(a) * math.cos(b), -math.sin(b)])
    side = np.cross(np.array([0.0, 0.0, 1.0]), app)
    side /= np.linalg.norm(side)
    x = np.cross(app, side)
    z = -app
    return np.column_stack([x, np.cross(z, x), z])


def approach_of(rot: np.ndarray) -> np.ndarray:
    """The horizontal unit approach of a hand rotation (minus its z, flattened)."""
    a = -np.asarray(rot)[:, 2].copy()
    a[2] = 0.0
    return a / (np.linalg.norm(a) or 1.0)


def jaw_of(approach: np.ndarray) -> np.ndarray:
    """The horizontal direction across the jaw for a horizontal approach."""
    return np.array([-approach[1], approach[0], 0.0])


def hand_back(hand: G.HandFrames) -> G.Box:
    """The hand-back box (`HAND_BACK`) for ``hand``."""
    R = hand.thumb_rot
    origin = hand.palm.center - R @ np.asarray(G.PALM[1], dtype=float)
    c, h = HAND_BACK
    return G.Box("hand_back", origin + R @ np.asarray(c), R, np.asarray(h))


def free_gap_m(gripper_deg: float) -> float:
    """The free gap between the two pads at an opening (box clearance; what an
    object must pass through), from the MJCF pad boxes."""
    h = G.hand_frames([0.0] * 7, gripper_deg)
    return G.obb_separation(h.thumb, h.finger)


def bearing_deg(xy: Sequence[float]) -> float:
    return PL.bearing_deg(xy)


# ── Shapes ────────────────────────────────────────────────────────────────────

def side_shape(obj) -> str:
    """"cube" or "cylinder" -- or CraneRefused("shape").  The side-on grasp is
    for the objects #180 names: cubes (red_cube and the pool boxes, semantic
    class "cube") and upright cylinders (blue_cylinder and the pool
    cylinders, class "cylinder").  The can and the foam block, and anything
    else, are refused with the reason."""
    if obj.kind == "box" and obj.semantic_class == "cube":
        return "cube"
    if obj.kind == "cylinder" and obj.semantic_class == "cylinder":
        return "cylinder"
    raise CP.CraneRefused(
        "shape", f"{obj.id} is a {obj.semantic_class or 'unclassified'} {obj.kind}; the "
        "side-on grasp is for cubes (red_cube, the pool boxes) and cylinders "
        "(blue_cylinder, the pool cylinders) only", kind=obj.kind,
        semantic_class=obj.semantic_class)


def width_across_m(box: G.Box, jaw: np.ndarray) -> float:
    """The object's extent across the jaw (the box face normal nearest it)."""
    n, _m, i = G.face_axes(box, jaw)
    return 2.0 * float(box.half[i])


def side_opening_deg(width_m: float, spec: SideSpec) -> float:
    """The opening for an object ``width_m`` across the jaw: the free gap
    leaves it `side_slack_total_m`, rounded to 0.5 deg toward wider, and never
    wider than the crane's opening (`CraneSpec.opening_deg`)."""
    g = closure_deg(width_m + spec.side_slack_total_m)
    return max(spec.opening_deg, math.floor(2.0 * g) / 2.0)


def side_close_deg(width_m: float, spec: SideSpec) -> float:
    """The close target for an object ``width_m`` across the jaw:
    `close_past_contact_deg` past `closure_deg`, rounded to 0.5 deg, never
    past the crane's `close_deg`."""
    g = closure_deg(width_m) + spec.close_past_contact_deg
    return min(CP.CraneSpec().close_deg, math.floor(2.0 * g) / 2.0)


def opening_check(obj_id: str, width_m: float, spec: SideSpec) -> Dict[str, float]:
    """The object must fit between the open pads: refused when it is as wide
    as the free gap at the widest planned opening, the crane's (#180: "check
    the opening")."""
    gap = free_gap_m(spec.opening_deg)
    widest = free_gap_m(spec.wide_opening_deg)
    rep = {"width_mm": round(1000 * width_m, 1), "free_gap_mm": round(1000 * gap, 1),
           "free_gap_wide_mm": round(1000 * widest, 1),
           "slack_per_side_mm": round(500 * (gap - width_m), 2)}
    if width_m >= gap:
        raise CP.CraneRefused(
            "opening", f"{obj_id} is {1000 * width_m:.1f} mm across; the open jaw leaves "
            f"{1000 * gap:.1f} mm at {spec.opening_deg:.0f} deg ({1000 * widest:.1f} mm at "
            f"the widest, {spec.wide_opening_deg:.0f} deg)", **rep)
    return rep


def closure_deg(width_m: float) -> float:
    """The gripper angle at which the free gap equals ``width_m`` (bisection
    on `free_gap_m`): where the closing finger first meets a round object."""
    lo, hi = -68.75, 20.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if free_gap_m(mid) > width_m:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cylinder_roll_model(obj, grasp_height_m: float) -> Dict[str, object]:
    """The roll risk of a standing cylinder gripped side-on (#180, Part B).

    * AT CLOSURE: the thumb pad is flat and the finger pad meets the cylinder
      at the closure angle, so the two contact normals make a wedge of that
      angle.  A wedge holds a round object only while it is under twice the
      friction angle; past it the grip squeezes the cylinder out.
    * AT APPROACH AND RELEASE: a push at pad height tips the cylinder rather
      than sliding it when the friction exceeds radius / push height -- and a
      tipped cylinder rolls.  So no pad may touch it on the way in (the pad
      route is checked against its bounding box taken across the jaw, which
      is exact across the jaw), and the release backs the thumb away along
      the jaw before the hand withdraws (`PlaceSpec.release_retreat_m`),
      judged again from where the cylinder then stands (#173).
    """
    r = float(obj.size[0]) / 2.0
    g = closure_deg(2.0 * r)
    squeeze_limit = 2.0 * math.degrees(math.atan(CYLINDER_FRICTION))
    tip_ratio = r / grasp_height_m
    return {"radius_mm": round(1000 * r, 1), "closure_deg": round(g, 2),
            "wedge_deg": round(abs(g), 2), "squeeze_out_limit_deg": round(squeeze_limit, 1),
            "holds": abs(g) < squeeze_limit,
            "push_height_mm": round(1000 * grasp_height_m, 1),
            "radius_over_height": round(tip_ratio, 3), "friction": PAD_FRICTION,
            "a_push_tips_it": PAD_FRICTION > tip_ratio}


def grasp_box(scene: SceneModel, object_id: str, position, quat,
              approach: np.ndarray) -> Tuple[G.Box, Optional[np.ndarray]]:
    """The object as the side-on checks see it, and the cylinder alignment
    (`CranePlan.align`): a cylinder's bounding box is taken across the jaw."""
    obj = scene.get(object_id)
    align = jaw_of(approach) if obj.kind == "cylinder" else None
    return G.object_box(obj, position, quat, align=align), align


def candidate_azimuths(box: G.Box, shape: str, spec: SideSpec,
                       taught_az: Optional[float]) -> List[float]:
    """Approach directions to try, nearest the taught one first.  A cube's
    four horizontal face normals; a cylinder's every `cylinder_azimuth_step_deg`.
    Only those within `max_bearing_offset_deg` of the shoulder's bearing."""
    b = bearing_deg(box.center[:2])
    if shape == "cube":
        out = []
        for i in range(3):
            ax = box.axes[:, i]
            if abs(ax[2]) < 0.7:
                for s in (1.0, -1.0):
                    out.append(math.degrees(math.atan2(s * ax[1], s * ax[0])))
    else:
        step = spec.cylinder_azimuth_step_deg
        out = [-180.0 + step * k for k in range(int(round(360.0 / step)))]
    wrap = lambda d: (d + 180.0) % 360.0 - 180.0  # noqa: E731
    out = [a for a in out if abs(wrap(a - b)) <= spec.max_bearing_offset_deg + 1e-9]
    ref = b if taught_az is None else taught_az
    return sorted(out, key=lambda a: abs(wrap(a - ref)))


def nearest_taught(xy: Sequence[float], scene: SceneModel) -> Optional[str]:
    """The taught square whose sim cell centre is nearest ``xy``."""
    best = None
    for cell in TAUGHT_SEEDS:
        try:
            cx, cy, _z = scene.cell_center(f"cell_{cell}")
        except KeyError:
            continue
        d = math.hypot(cx - xy[0], cy - xy[1])
        if best is None or d < best[0]:
            best = (d, cell)
    return None if best is None else best[1]


def seed_starts(near_cell: Optional[str]) -> List[List[float]]:
    """IK starts: the nearest taught square's joints first, the other taught
    squares, then the crane's start set."""
    order = sorted(TAUGHT_SEEDS, key=lambda c: c != near_cell)
    return [list(TAUGHT_SEEDS[c][1]) for c in order] + [list(q) for q in CP.GRASP_STARTS]


# ── Margins ───────────────────────────────────────────────────────────────────

def joint_margin_mm(q: Sequence[float], gripper_deg: float) -> Tuple[float, str]:
    """How far the jaw-gap centre can move under each joint before that joint
    reaches its stop: the joint's margin (deg) times the gap centre's speed
    under it (mm/deg, finite difference).  Returns (least, joint)."""
    base = G.hand_frames(q, gripper_deg).gap_mid
    best = (math.inf, "")
    for i, (n, v, (a, b)) in enumerate(zip(R_ARM_JOINTS, q, joint_limits("right"))):
        dq = list(q)
        dq[i] = float(dq[i]) + 0.01
        speed = float(np.linalg.norm(G.hand_frames(dq, gripper_deg).gap_mid - base)) / 0.01
        m = min(float(v) - a, b - float(v)) * 1000.0 * speed
        if m < best[0]:
            best = (m, n)
    return best


def _min_nested(v) -> float:
    if isinstance(v, dict):
        return min((_min_nested(x) for x in v.values()), default=math.inf)
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.inf


def plan_margin(pick: CP.CranePlan, place: Optional[PL.PlacePlan] = None,
                scene: Optional[SceneModel] = None) -> Dict[str, object]:
    """One margin for a checked plan, crane or side-on, in millimetres of
    hand travel: the least, over every check that has a threshold, of how
    far the plan stands from that threshold.  Joint stops count as the gap
    centre's travel to the stop (`joint_margin_mm`), so a stop and a
    clearance compare in the same unit.  Returns {"margin_mm", "limit",
    "terms"}; the planner choice picks the larger "margin_mm"."""
    t: Dict[str, float] = {}
    c = pick.checks
    pr = c.get("pad_route") or {}
    tol = float(pr.get("tolerance_mm", 0.0))
    t["pad_route"] = min(pr.get("route_thumb_mm", math.inf),
                         pr.get("route_finger_mm", math.inf)) - tol
    t["palm_route"] = pr.get("route_palm_mm", math.inf) - tol
    sd = c.get("straddle_predicted") or {}
    t["pad_below_top"] = min(sd.get("thumb_center_below_top_mm", math.inf),
                             sd.get("finger_center_below_top_mm", math.inf))
    t["table"] = _min_nested(c.get("table_mm") or {})
    t["carried_lift"] = float((c.get("carried") or {}).get("others_mm", math.inf))
    ex = c.get("side") or {}
    if "neighbours_mm" in ex:
        t["neighbours"] = float(ex["neighbours_mm"])
    poses = [(q, pick.spec.opening_deg) for q in pick.rungs]
    poses += [(pick.up(k), pick.spec.close_deg) for k in range(pick.up_count)]
    if place is not None:
        pc = place.checks
        t["table_place"] = _min_nested(pc.get("table_mm") or {})
        for name, v in (pc.get("held") or {}).items():
            t["held_others"] = min(t.get("held_others", math.inf), float(v.get("others_mm", math.inf)))
            t["held_arm"] = min(t.get("held_arm", math.inf), float(v.get("arm_mm", math.inf)))
        ex_ = [1000.0 * (float(e["clearance_m"]) - _footprint_margin())
               for e in (pc.get("exits") or {}).values() if e.get("open")]
        if ex_:
            t["exit"] = max(ex_)
        if place.hold_source == "predicted" and "pad_route_out" in pc:
            po = pc["pad_route_out"]
            t["pad_route_out"] = min(po.get("route_thumb_mm", math.inf),
                                     po.get("route_finger_mm", math.inf))
        ex2 = pc.get("side") or {}
        if "neighbours_mm" in ex2:
            t["neighbours_place"] = float(ex2["neighbours_mm"])
        d = place.dest
        poses += [(q, d.spec.opening_deg) for q in d.rungs]
        poses += [(d.up(k), d.spec.close_deg) for k in range(place.carry_rung + 1)]
    jm = (math.inf, "")
    for q, g in poses:
        m = joint_margin_mm(q, g)
        if m[0] < jm[0]:
            jm = m
    t["joints"] = jm[0]
    terms = {k: round(v, 2) for k, v in t.items() if math.isfinite(v)}
    lim = min(terms, key=terms.get)
    return {"margin_mm": terms[lim], "limit": lim, "joint": jm[1], "terms": terms}


def _footprint_margin() -> float:
    from reachy_ai.motion import rig_routes as R
    return float(R.FOOTPRINT_MARGIN)


# ── The whole-arm preflight with the box hand ─────────────────────────────────

class BoxHandPlanner:
    """The planner a side-on plan is checked and flown with (#180).

    Everything is the wrapped `CartesianPlanner`'s except `check_arm_path`,
    the whole-arm preflight.  Its arm links are unchanged (the upper arm and
    forearm capsules, every obstacle, the target per `TargetContact`).  Its
    HAND is the MJCF's collision hand -- the two pads, the palm and the hand
    back (#192, #195) as boxes -- instead of the tube (`WHOLE_ARM_HAND_MODEL`):
    the tube is one capsule around the hand's axis with the radius of the
    hand's widest corner over every wrist roll (114 mm at the -65 deg
    opening), which a level hand 34 mm above the table pushes through the
    board.  At r1c1 it read -9 mm into the rail under the board where the
    pads, palm and back stand 84-169 mm clear (docs/reviews/2026-10-19-side-
    on-grasp-180/README).  Same margin (`WHOLE_ARM_MARGIN_M`), same sampling.
    """

    MODEL = "boxes (pads, palm, back)"

    def __init__(self, planner) -> None:
        self._p = planner

    def __getattr__(self, name):
        return getattr(self._p, name)

    def check_arm_path(self, joints_seq, *, segment: str, gripper_deg, target=None) -> None:
        scene = self._p._scene
        if scene is None:
            return
        seq = [list(q)[:7] for q in joints_seq]
        if not seq:
            return
        g = K._GRIPPER_OPEN_LIMIT_DEG if gripper_deg is None else float(gripper_deg)
        skip = {target.object_id} if target is not None else set()
        ids = [o for o in scene.obstacle_ids(include_static=True) if o not in skip]
        boxes = {o: G.object_box(scene.get(o)) for o in ids}
        tgt = None
        if target is not None and scene.get(target.object_id).collides:
            tgt = scene.get(target.object_id)
        worst = [math.inf, "", "", -1]

        def consider(d, oid, link, idx):
            if d < worst[0]:
                worst[:] = [d, oid, link, idx]

        pairs = [(i, q) for i, (u, v) in enumerate(zip(seq, seq[1:]))
                 for q in K.joint_path(u, v, K.sample_count(u, v))] if len(seq) > 1 \
            else [(0, seq[0])]
        for idx, q in pairs:
            caps = [c for c in K.link_capsules(q, "right", g, hand="tube") if c[0] != "hand"]
            for oid, c in scene.clearances(caps, ids=ids).items():
                consider(c.distance, oid, c.link, idx)
            h = G.hand_frames(q, g)
            hand = (h.thumb, h.finger, h.palm, hand_back(h))
            for oid, b in boxes.items():
                # A cheap bound first: every hand box lies within 0.2 m of the wrist.
                lb = float(np.linalg.norm(b.center - h.wrist)) - 0.2 - float(np.linalg.norm(b.half))
                if lb > 0.05:
                    consider(lb, oid, "hand", idx)
                    continue
                for hb in hand:
                    consider(G.obb_separation(hb, b), oid, "hand", idx)
            if tgt is None:
                continue
            if target.carried:
                _s, _e, wrist, R = link_frames(q, "right")
                pad = wrist - R @ K._TOOL
                centre = (float(pad[0]), float(pad[1]), float(pad[2]))
            elif target.center is not None:
                centre = target.center
            else:
                centre = tgt.center
            moved = replace(tgt, center=centre)
            for link, p0, p1, radius in caps:
                d, _at = segment_object_distance(moved, p0, p1, radius)
                consider(d, target.object_id, link, idx)
            if not target.hand_contact:
                tb = G.object_box(moved)
                for hb in hand:
                    consider(G.obb_separation(hb, tb), target.object_id, "hand", idx)
        if worst[0] < K.WHOLE_ARM_MARGIN_M:
            d, oid, link, idx = worst
            raise ArmClearanceError(segment=segment, link=link, obstacle=oid, clearance_m=d,
                                    margin_m=K.WHOLE_ARM_MARGIN_M, model=self.MODEL,
                                    gripper_deg=gripper_deg, pose_index=idx)


def box_hand(planner):
    """``planner`` with the box-hand whole-arm preflight (idempotent)."""
    return planner if isinstance(planner, BoxHandPlanner) else BoxHandPlanner(planner)


def planner_for(plan: CP.CranePlan, planner):
    """The planner a plan is flown with: the box hand for a side-on plan,
    the planner itself (the tube) for the crane."""
    return box_hand(planner) if plan.kind == "side" else planner


# ── Extra checks (neighbours, the hand back, the shape) ───────────────────────

def _others(scene: SceneModel, object_id: str) -> List[G.Box]:
    return [G.object_box(o) for o in scene.objects.values()
            if o.collides and o.id != object_id and "grid-cell" not in o.tags
            and (scene.table is None or o.id != scene.table.id)]


def hand_clearance(scene: SceneModel, object_id: str, seq: Sequence[Sequence[float]],
                   gripper_deg: float, max_step_deg: float = 1.0) -> Tuple[float, str, int]:
    """The hand's boxes (pads, palm, back) against every other colliding
    object along a commanded route: (worst m, object, sample)."""
    boxes = _others(scene, object_id)
    worst = (math.inf, "", -1)
    if not boxes:
        return worst
    for k, q in enumerate(G.densify(seq, max_step_deg)):
        h = G.hand_frames(q, gripper_deg)
        for hb in (h.thumb, h.finger, h.palm, hand_back(h)):
            for o in boxes:
                d = G.obb_separation(hb, o)
                if d < worst[0]:
                    worst = (d, o.name, k)
    return worst


def back_table_clearance(scene: SceneModel, seq: Sequence[Sequence[float]],
                         gripper_deg: float) -> float:
    return min(hand_back(G.hand_frames(q, gripper_deg)).lowest_z - scene.table_surface_z
               for q in G.densify(seq, 1.0))


def _hand_segments(plan: CP.CranePlan) -> Dict[str, Tuple[List[List[float]], float]]:
    """The open-hand segments of a plan: the approach and the way back."""
    s = plan.spec
    r = plan.rungs
    return {"via->hover": ([r[k] for k in range(s.via_rungs, s.hover_rungs - 1, -1)],
                           s.transit_opening_deg),
            "slide-in": ([r[k] for k in range(s.hover_rungs, -1, -1)], s.opening_deg),
            "lift": ([plan.up(k) for k in range(s.lift_rungs + 1)], s.close_deg)}


def check_side_extras(scene: SceneModel, plan: CP.CranePlan, spec: SideSpec,
                      segs: Optional[Dict[str, Tuple[List[List[float]], float]]] = None
                      ) -> Dict[str, object]:
    """The checks a level hand adds to the crane's: neighbours along every
    hand segment, and the hand back above the table.  Raises CraneRefused."""
    rep: Dict[str, object] = {}
    worst = (math.inf, "", "")
    tab = math.inf
    for name, (seq, g) in (segs or _hand_segments(plan)).items():
        d, oid, _k = hand_clearance(scene, plan.object_id, seq, g)
        if d < worst[0]:
            worst = (d, oid, name)
        tab = min(tab, back_table_clearance(scene, seq, g))
    rep["neighbours_mm"] = round(1000 * worst[0], 2) if math.isfinite(worst[0]) else None
    rep["neighbour"] = worst[1] or None
    rep["back_table_mm"] = round(1000 * tab, 2)
    if worst[0] < spec.neighbour_clearance_m:
        raise CP.CraneRefused(
            "neighbours", f"the hand would pass {1000 * worst[0]:.1f} mm from {worst[1]} on "
            f"'{worst[2]}' ({1000 * spec.neighbour_clearance_m:.0f} mm needed)",
            other=worst[1], segment=worst[2], clearance_m=worst[0])
    if tab < 0.0:
        raise CP.CraneRefused("table", f"the hand back would be {-1000 * tab:.1f} mm into "
                              "the table", clearance_m=tab)
    if not math.isfinite(worst[0]):
        rep.pop("neighbours_mm")
    return rep


# ── The plan ──────────────────────────────────────────────────────────────────

#: Null-space refinement of a side-on rung (`refine_margin`): the jaw-gap
#: position is held to 0.3 mm and the hand attitude to 0.5 deg of where the
#: solve put it, while the joints move away from their nearest stops.
_REFINE_POS_M = 0.0003
_REFINE_ROT_DEG = 0.5


def _margins_deg(q: Sequence[float]) -> np.ndarray:
    return np.array([min(float(v) - a, b - float(v))
                     for v, (a, b) in zip(q, joint_limits("right"))])


def refine_margin(q: Sequence[float], rot: np.ndarray, gap_target: np.ndarray,
                  gripper_deg: float, iters: int = 30) -> List[float]:
    """Use the arm's one redundant joint to stand further from the stops.

    The seven-joint arm holds the jaw-gap position and the hand attitude (six
    numbers) with one degree of freedom to spare -- the elbow's swivel -- and
    the ladder solves take whichever pose the IK reaches first, which on a
    level hand runs the wrist onto its stops along the slide-in (#180).  Each
    step here corrects the six-number task by damped least squares and adds a
    move in the task's null space up the gradient of the least joint margin;
    a step is kept only if the gap and the attitude stay within
    `_REFINE_POS_M` / `_REFINE_ROT_DEG` of the target and the least margin
    grows.  Returns the best pose found (``q`` itself if none is better)."""
    def task(qq):
        h = G.hand_frames(qq, gripper_deg)
        return np.concatenate([np.asarray(gap_target) - h.gap_mid,
                               0.1 * CP._rotvec(np.asarray(rot) @ G.hand_rotation(qq).T)])

    def ok(qq):
        e = task(qq)
        return (float(np.linalg.norm(e[:3])) <= _REFINE_POS_M
                and math.degrees(float(np.linalg.norm(e[3:]))) / 0.1 <= _REFINE_ROT_DEG
                and within_limits(qq))

    best = [float(v) for v in q]
    if not ok(best):
        return best
    lo = np.array([a for a, _ in joint_limits("right")])
    hi = np.array([b for _, b in joint_limits("right")])
    cur = np.array(best)
    for _ in range(iters):
        e = task(cur)
        J = np.zeros((6, 7))
        for i in range(7):
            d = cur.copy()
            d[i] += 1e-3
            J[:, i] = -(task(d) - e) / 1e-3
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-8 * np.eye(6))
        m = _margins_deg(cur)
        k = int(np.argmin(m))
        grad = np.zeros(7)
        grad[k] = 1.0 if (cur[k] - lo[k]) < (hi[k] - cur[k]) else -1.0
        null = (np.eye(7) - Jp @ J) @ grad
        if float(np.linalg.norm(null)) < 1e-9:
            break
        moved = False
        for step in (2.0, 1.0, 0.5, 0.25):
            cand = np.clip(cur + Jp @ e + step * null / np.linalg.norm(null), lo, hi)
            if ok(list(cand)) and _margins_deg(cand).min() > m.min() + 1e-3:
                cur, moved = cand, True
                break
        if not moved:
            break
    return [float(v) for v in cur]


def _rise(arm, rot: np.ndarray, start_t: np.ndarray, direction: np.ndarray, q0,
          n_max: int, n_min: int, grip: float, tol: float, what: str
          ) -> Tuple[List[np.ndarray], List[List[float]]]:
    """Rungs from ``q0`` (at ``start_t``) along ``direction``, one rung each,
    warm-started and refined (`refine_margin`), as far as the IK reaches up
    to ``n_max``; refused below ``n_min``."""
    targets, rungs = [], []
    q = list(q0)
    for k in range(1, n_max + 1):
        t = start_t + k * direction
        got = CP.solve_gap(arm, G.hand_rotation(q), t, grip, q, tol, iters=20, relax=True)
        if got is None:
            break
        q = refine_margin(got[0], G.hand_rotation(got[0]), t, grip)
        targets.append(t)
        rungs.append(q)
    if len(rungs) < n_min:
        raise CP.CraneRefused("plan", f"IK found no pose for {what} rung +{len(rungs) + 1}")
    return targets, rungs


def _ladders(arm, rot: np.ndarray, g1: np.ndarray, approach: np.ndarray, q_g,
             spec: SideSpec, tol: float):
    """The approach (level slide-in back to the standoff, then up to the via)
    and the vertical lift ladder, both from the solved grasp ``q_g``.
    Returns (approach targets, approach rungs, lift targets, lift rungs,
    the spec with this plan's own `via_rungs` / `lift_ladder_rungs`)."""
    up = np.array([0.0, 0.0, spec.rung_m])
    back = -spec.rung_m * np.asarray(approach)
    g = spec.opening_deg
    S = spec.standoff_rungs
    q_g = refine_margin(q_g, rot, g1, g)
    a_t, rungs = _rise(arm, rot, g1, back, q_g, S, S, g, tol, "slide-in")
    v_t, v_r = _rise(arm, rot, a_t[-1], up, rungs[-1],
                     spec.via_rungs - S, max(spec.min_via_rise_rungs, spec.hover_rise_rungs + 1),
                     g, tol, "via")
    l_t, lift = _rise(arm, rot, g1, up, q_g, spec.lift_ladder_rungs, spec.lift_rungs, g, tol,
                      "lift")
    s = replace(spec, hover_rungs=S + spec.hover_rise_rungs, via_rungs=S + len(v_r),
                lift_ladder_rungs=len(lift))
    return ([g1] + a_t + v_t, [list(q_g)] + rungs + v_r, [g1] + l_t, [list(q_g)] + lift, s)


def _solve_grasp(arm, rot: np.ndarray, g0: np.ndarray, spec: SideSpec, tol: float,
                 starts: Sequence[Sequence[float]]) -> Optional[List[float]]:
    """The grasp pose for jaw-gap centre ``g0`` at ``rot``: multi-start IK,
    gap-centred, attitude held within 1 deg; the largest joint margin wins."""
    pre = []
    for start in starts:
        h = G.hand_frames(start, spec.opening_deg)
        q = CP._ik(arm, rot, g0 + (h.wrist - h.gap_mid), start)
        if q is not None and CP._rot_err_deg(G.hand_rotation(q), rot) <= 1.0:
            pre.append(q)
    margin = lambda q: min(min(v - a, b - v) for v, (a, b) in  # noqa: E731
                           zip(q, joint_limits("right")))
    pre.sort(key=lambda q: -margin(q))
    for q in pre[:3]:
        got = CP.solve_gap(arm, rot, g0, spec.opening_deg, q, tol)
        if got is not None and CP._rot_err_deg(G.hand_rotation(got[0]), rot) <= 1.0:
            return got[0]
    return None


@dataclass
class Candidate:
    azimuth_deg: float
    pitch_deg: float
    q_grasp: Optional[List[float]] = None
    g1: Optional[np.ndarray] = None
    grasp_margin_deg: float = -math.inf
    refusal: Optional[str] = None
    stage: Optional[str] = None
    plan: Optional[CP.CranePlan] = None
    margin: Optional[Dict[str, object]] = None
    spec: Optional[SideSpec] = None              # with this object's opening

    def record(self) -> Dict[str, object]:
        return {"azimuth_deg": round(self.azimuth_deg, 1), "pitch_deg": self.pitch_deg,
                "grasp_joint_margin_deg": (round(self.grasp_margin_deg, 2)
                                           if math.isfinite(self.grasp_margin_deg) else None),
                "stage": self.stage, "refusal": self.refusal,
                "margin_mm": (self.margin or {}).get("margin_mm"),
                "limit": (self.margin or {}).get("limit")}


def _grasp_candidate(arm, scene: SceneModel, object_id: str, pose, az: float, pitch: float,
                     spec: SideSpec, limits: CP.CraneLimits,
                     starts: Sequence[Sequence[float]]) -> Candidate:
    """Solve and centre the grasp for one approach (no ladders, no checks)."""
    cand = Candidate(az, pitch, spec=spec)
    rot = side_rotation(az, pitch)
    app = approach_of(rot)
    box, _align = grasp_box(scene, object_id, *(pose or (None, None)), approach=app)
    tol = limits.ik_position_tol_m
    top = box.highest_z
    g0 = np.array([box.center[0], box.center[1], scene.table_surface_z + spec.grasp_height_m])
    if top - g0[2] < spec.min_pad_below_top_m:
        cand.stage, cand.refusal = "plan", (
            f"{object_id} is {1000 * (top - scene.table_surface_z):.0f} mm tall; the pads "
            f"at {1000 * spec.grasp_height_m:.0f} mm would not be "
            f"{1000 * spec.min_pad_below_top_m:.0f} mm below its top")
        return cand
    q = _solve_grasp(arm, rot, g0, spec, tol, starts)
    if q is None:
        cand.stage, cand.refusal = "plan", "IK found no level grasp pose for this approach"
        return cand
    # Stand the object `thumb_side_m` off the thumb pad across the jaw, then
    # back off along the approach until the palm clears its near face by
    # `palm_clearance_m`.
    g1 = g0.copy()
    for _ in range(3):
        h = G.hand_frames(q, spec.opening_deg)
        pc = G.pad_clearance(h, box, closing=True)
        dn = pc.thumb_side_m - spec.thumb_side_m
        palm = G.obb_separation(h.palm, box)
        step = dn * np.asarray(pc.face_normal)
        if palm < spec.palm_clearance_m - 1e-5:
            step = step - (spec.palm_clearance_m - palm) * app
        if float(np.linalg.norm(step)) < 2e-5:
            break
        g1 = g1 + step
        got = CP.solve_gap(arm, rot, g1, spec.opening_deg, q, tol)
        if got is None:
            cand.stage, cand.refusal = "plan", "IK lost the grasp pose while centring it"
            return cand
        q = got[0]
    cand.q_grasp, cand.g1 = q, g1
    cand.grasp_margin_deg = min(min(v - a, b - v) for v, (a, b) in
                                zip(q, joint_limits("right")))
    return cand


def _full_plan(planner, scene: SceneModel, object_id: str, pose, cand: Candidate,
               spec: SideSpec, limits: CP.CraneLimits) -> CP.CranePlan:
    arm = planner._arm
    spec = cand.spec or spec
    rot = side_rotation(cand.azimuth_deg, cand.pitch_deg)
    app = approach_of(rot)
    box, align = grasp_box(scene, object_id, *(pose or (None, None)), approach=app)
    a_t, rungs, l_t, lift, spec = _ladders(arm, rot, cand.g1, app, cand.q_grasp, spec,
                                           limits.ik_position_tol_m)
    plan = CP.CranePlan(object_id, spec, limits, box, rot, a_t, rungs, present_joints(),
                        lift=lift, lift_targets=l_t, approach=app, align=align)
    plan.insertion_rung = CP._insertion_rung(plan)
    if plan.insertion_rung > 7:
        raise CP.CraneRefused("plan", "the pads would meet the object more than 7 cm in "
                              "from the insertion checkpoint (beyond the recorded "
                              "prediction range)")
    rep = CP.check_crane_plan(box_hand(planner), scene, plan)
    rep["side"] = check_side_extras(scene, plan, spec)
    rep["side"]["azimuth_deg"] = round(cand.azimuth_deg, 2)
    rep["side"]["pitch_deg"] = cand.pitch_deg
    plan.checks = rep
    return plan


def side_candidates(planner, scene: SceneModel, object_id: str,
                    pose: Optional[CP.Observation] = None, spec: SideSpec = SideSpec(),
                    limits: CP.CraneLimits = CP.CraneLimits(),
                    on_event: Optional[CP.EventLog] = None,
                    azimuths: Optional[Sequence[float]] = None) -> List[Candidate]:
    """Every approach the side-on grasp would try for ``object_id``, each with
    its grasp solved (or its refusal), best grasp joint margin first.  The
    shape and the opening are checked first (CraneRefused)."""
    ev = on_event or (lambda *_a, **_k: None)
    obj = scene.get(object_id)
    shape = side_shape(obj)
    pq = None if pose is None else (pose.position, pose.quat_wxyz)
    box0 = G.object_box(obj, *(pq or (None, None)))
    if shape == "cylinder":
        roll = cylinder_roll_model(obj, spec.grasp_height_m)
        ev("side_roll_model", object_id=object_id, **roll)
        if not roll["holds"]:
            raise CP.CraneRefused("shape", f"{object_id}: the closing wedge "
                                  f"({roll['wedge_deg']} deg) would squeeze it out", **roll)
    near = nearest_taught(box0.center[:2], scene)
    taught_az = TAUGHT_SEEDS[near][0] if near else None
    azs = list(azimuths) if azimuths is not None else candidate_azimuths(box0, shape, spec, taught_az)
    if not azs:
        raise CP.CraneRefused("plan", "no approach direction within reach of the shoulder")
    # The opening, across the jaw of the first approach (a cube's face, a
    # cylinder's diameter: the same for every approach of a square cube).
    b1, _ = grasp_box(scene, object_id, *(pq or (None, None)),
                      approach=approach_of(side_rotation(azs[0])))
    opening = opening_check(object_id, width_across_m(b1, jaw_of(approach_of(side_rotation(azs[0])))),
                            spec)
    ev("side_opening", object_id=object_id, **opening)
    starts = seed_starts(near)
    out = []
    width = opening["width_mm"] / 1000.0
    spec = replace(spec, opening_deg=side_opening_deg(width, spec),
                   close_deg=side_close_deg(width, spec))
    ev("side_opening_chosen", object_id=object_id, opening_deg=round(spec.opening_deg, 2),
       close_deg=spec.close_deg)
    for az in azs:
        for pitch in spec.pitches_deg:
            out.append(_grasp_candidate(planner._arm, scene, object_id, pq, az, pitch, spec,
                                        limits, starts))
    out.sort(key=lambda c: -c.grasp_margin_deg)
    ev("side_candidates", object_id=object_id, taught_square=near, taught_azimuth=taught_az,
       candidates=[c.record() for c in out])
    return out


def plan_side_pick(planner, scene: SceneModel, object_id: str,
                   pose: Optional[CP.Observation] = None, spec: SideSpec = SideSpec(),
                   limits: CP.CraneLimits = CP.CraneLimits(),
                   on_event: Optional[CP.EventLog] = None,
                   azimuths: Optional[Sequence[float]] = None,
                   candidates: Optional[List[Candidate]] = None) -> CP.CranePlan:
    """Plan and check the side-on pick.  IK/FK calls only -- no motion.

    Up to `SideSpec.max_full_plans` approaches with a grasp are planned in
    full and checked; the one with the largest `plan_margin` is returned.
    Raises CraneRefused: the refusal of the best-ranked approach, with every
    approach's in ``detail["candidates"]`` and the event log."""
    ev = on_event or (lambda *_a, **_k: None)
    cands = candidates if candidates is not None else side_candidates(
        planner, scene, object_id, pose, spec, limits, on_event, azimuths)
    pq = None if pose is None else (pose.position, pose.quat_wxyz)
    tried = 0
    for c in cands:
        if c.q_grasp is None or tried >= spec.max_full_plans:
            continue
        tried += 1
        try:
            c.plan = _full_plan(planner, scene, object_id, pq, c, spec, limits)
            c.margin = plan_margin(c.plan)
        except CP.CraneRefused as exc:
            c.stage, c.refusal = exc.stage, exc.reason
        ev("side_pick_candidate", **c.record())
    ok = [c for c in cands if c.plan is not None]
    if not ok:
        first = next((c for c in cands if c.refusal), None)
        raise CP.CraneRefused(first.stage if first else "plan",
                              (first.refusal if first else "no approach") +
                              f" (best of {len(cands)} side-on approaches)",
                              candidates=[c.record() for c in cands])
    best = max(ok, key=lambda c: c.margin["margin_mm"])
    best.plan.checks["margin"] = best.margin
    ev("side_pick_chosen", **best.record())
    return best.plan


def predicted_hold(pick: CP.CranePlan) -> Tuple[np.ndarray, np.ndarray]:
    """Where the object will sit in the level hand once closed: the planned
    grasp moved across the thumb gap (`SideSpec.thumb_side_m`) to the thumb.
    The crane's 8 mm (`crane_place_live.GRASP_SHIFT_M`) is its own close's
    shove across an evenly split gap.  The carry is re-planned from the
    measured hold either way."""
    h = G.hand_frames(pick.rungs[0], pick.spec.opening_deg)
    rot_rel, t_rel = G.grasp_transform(h, pick.obj)
    shift = float(getattr(pick.spec, "thumb_side_m", PL.GRASP_SHIFT_M))
    return rot_rel, t_rel + shift * (h.thumb_rot.T @ -h.closing)


def plan_side_destination(planner, scene: SceneModel, pick: CP.CranePlan,
                          rel: Tuple[np.ndarray, np.ndarray], target_xy: Sequence[float],
                          psi_deg: float, place: PL.PlaceSpec,
                          seeds: Sequence[Sequence[float]] = ()
                          ) -> Tuple[CP.CranePlan, np.ndarray]:
    """The side-on destination: rung 0 holds the object standing on the
    target, the hand turned ``psi_deg`` about the vertical from its grasp
    attitude (still level); the lift ladder rises from it (the set-down comes
    down it) and the approach ladder leaves it level along the turned
    approach (the withdrawal), then rises to the via."""
    # The release is the side-on one (`SideSpec.release_retreat_m`), not the
    # crane placement's.
    s = pick.spec
    lim = replace(pick.limits, max_lateral_correction_m=place.max_setdown_correction_m,
                  max_attempt_s=place.max_attempt_s)
    arm = planner._arm
    Rz = PL._rotz(psi_deg)
    rd = Rz @ G.hand_rotation(pick.rungs[0])
    app = Rz @ np.asarray(pick.approach)
    rot_rel, t_rel = rel
    axes = rd @ rot_rel
    half = pick.obj.half
    target = np.array([float(target_xy[0]), float(target_xy[1]),
                       scene.table_surface_z + float(np.abs(axes[2]) @ half)])
    box = G.Box(pick.object_id, target, axes, half)
    h0 = G.hand_frames(pick.rungs[0], s.opening_deg)
    v = h0.thumb_rot.T @ (h0.gap_mid - h0.thumb.center)
    g0 = target - rd @ t_rel + rd @ v
    tol = lim.ik_position_tol_m
    near = nearest_taught(target[:2], scene)
    q0 = None
    for seed in [pick.rungs[0]] + [list(q) for q in seeds] + seed_starts(near):
        got = CP.solve_gap(arm, rd, g0, s.opening_deg, seed, tol)
        if got is not None and CP._rot_err_deg(G.hand_rotation(got[0]), rd) <= 1.0:
            q0 = got[0]
            break
    if q0 is None:
        raise CP.CraneRefused("destination", "IK found no level pose that sets the object "
                              "down on the target")
    try:
        a_t, rungs, l_t, lift, s = _ladders(arm, rd, g0, app, q0, s, tol)
    except CP.CraneRefused as exc:
        raise CP.CraneRefused("destination", exc.reason) from exc
    align = None if pick.align is None else Rz @ np.asarray(pick.align)
    dest = CP.CranePlan(pick.object_id, s, lim, box, rd, a_t, rungs, list(pick.present),
                        lift=lift, lift_targets=l_t, approach=app, align=align)
    try:
        dest.insertion_rung = CP._insertion_rung(dest)
    except CP.CraneRefused:
        dest.insertion_rung = 0
    return dest, target


def _dest_extras(scene: SceneModel, pp: PL.PlacePlan, spec: SideSpec) -> Dict[str, object]:
    """Neighbours along the carry and the set-down (the hand closed on the
    object) and the destination's withdrawal and the way up from its standoff
    (the hand open); the object itself is the target, not a neighbour."""
    d = pp.dest
    s = d.spec
    segs = {"carry": (pp.climb() + pp.carry(), s.close_deg),
            "set-down": (pp.setdown(), s.close_deg),
            "withdrawal": ([d.rungs[k] for k in range(0, s.hover_rungs + 1)], s.opening_deg),
            "standoff->via": ([d.rungs[k] for k in range(s.hover_rungs, s.via_rungs + 1)],
                              s.transit_opening_deg)}
    return check_side_extras(scene, d, spec, segs)


def plan_side_place(planner, scene: SceneModel, object_id: str, target_cell: str,
                    pose: Optional[CP.Observation] = None, spec: SideSpec = SideSpec(),
                    limits: CP.CraneLimits = CP.CraneLimits(),
                    place: PL.PlaceSpec = PL.PlaceSpec(),
                    on_event: Optional[CP.EventLog] = None,
                    azimuths: Optional[Sequence[float]] = None,
                    max_headings: int = 13) -> PL.PlacePlan:
    """Plan and check a side-on placement.  IK/FK calls only -- no motion.

    The cell checks are `plan_crane_place`'s (a grid cell, in reach, free).
    Then each checked side-on pick (best margin first, up to
    `SideSpec.max_full_plans`) is tried with destination headings nearest
    the bearing change first (`crane_place_live.headings_from`, at most
    ``max_headings``), at each carry height, against every crane placement
    check (`crane_place_live.check_place`) and the side-on extras at the
    destination.  The passing (pick, heading) with the largest `plan_margin`
    wins.  Raises CraneRefused."""
    ev = on_event or (lambda *_a, **_k: None)
    cell_id = target_cell if target_cell.startswith("cell_") else f"cell_{target_cell}"
    try:
        cell = scene.get(cell_id)
    except KeyError:
        raise CP.CraneRefused("destination", f"there is no cell {target_cell} in this scene")
    if "grid-cell" not in cell.tags:
        raise CP.CraneRefused("destination", f"{target_cell} is not a grid cell")
    if "no-right-arm" in cell.tags:
        raise CP.CraneRefused("destination", f"{target_cell} is out of the right arm's reach")
    cx, cy, _tz = scene.cell_center(cell_id)
    tx, ty = PL.aimed((cx, cy), place)
    held = G.object_box(scene.get(object_id), *(() if pose is None else
                                                   (pose.position, pose.quat_wxyz)))
    reach = float(np.linalg.norm(held.half[:2]))
    square = G.Box(object_id, np.array([tx, ty, held.center[2]]), np.eye(3),
                   np.array([reach, reach, held.half[2]]))
    for o in scene.objects.values():
        if not o.collides or o.id == object_id or "grid-cell" in o.tags or \
                (scene.table is not None and o.id == scene.table.id):
            continue
        gap = G.obb_separation(square, G.object_box(o))
        if gap < place.destination_free_m:
            raise CP.CraneRefused("destination", f"{target_cell} is not free: {o.id} is "
                                  f"{1000 * gap:.1f} mm from where {object_id} would stand",
                                  other=o.id, gap_m=gap)
    moved = math.hypot(tx - held.center[0], ty - held.center[1])
    if moved < place.min_transfer_m:
        raise CP.CraneRefused("destination", f"{object_id} is already on {target_cell} "
                              f"({1000 * moved:.0f} mm from its centre)")
    limits = replace(limits, max_attempt_s=place.max_attempt_s)
    cands = side_candidates(planner, scene, object_id, pose, spec, limits, on_event, azimuths)
    pq = None if pose is None else (pose.position, pose.quat_wxyz)
    passing: List[Tuple[Dict[str, object], PL.PlacePlan]] = []
    last: Optional[CP.CraneRefused] = None
    tried = 0
    for c in cands:
        if c.q_grasp is None or tried >= spec.max_full_plans:
            continue
        tried += 1
        try:
            pick = _full_plan(planner, scene, object_id, pq, c, spec, limits)
        except CP.CraneRefused as exc:
            c.stage, c.refusal = exc.stage, exc.reason
            ev("side_pick_candidate", **c.record())
            last = last or exc
            continue
        c.plan = pick
        ev("side_pick_candidate", **c.record())
        psi0 = PL.bearing_deg((tx, ty)) - PL.bearing_deg(pick.obj.center[:2])
        rel = predicted_hold(pick)
        for psi in PL.headings_from(psi0)[:max_headings]:
            try:
                dest, target = plan_side_destination(planner, scene, pick, rel, (tx, ty),
                                                     psi, place)
            except CP.CraneRefused as exc:
                ev("place_heading_refused", grasp=GRASP_SIDE, azimuth_deg=c.azimuth_deg,
                   psi_deg=round(psi, 1), stage=exc.stage, reason=exc.reason)
                last = last or exc
                continue
            done = False
            top = min(pick.up_count, dest.up_count) - 1
            for cr in [c_ for c_ in place.carry_rungs if c_ <= top]:
                pk = replace(pick, spec=replace(pick.spec, lift_min_rungs=cr))
                pp = PL.PlacePlan(object_id, cell_id, target, place, pk, dest, cr, psi, rel,
                                  cell_centre=np.array([cx, cy]))
                try:
                    PL.check_place(box_hand(planner), scene, pp)
                    pp.checks["side"] = _dest_extras(scene, pp, spec)
                except CP.CraneRefused as exc:
                    ev("place_carry_refused", grasp=GRASP_SIDE, azimuth_deg=c.azimuth_deg,
                       carry_rung=cr, psi_deg=round(psi, 1), stage=exc.stage,
                       reason=exc.reason)
                    last = last or exc
                    continue
                m = plan_margin(pick, pp)
                pp.checks["margin"] = m
                ev("place_checked", grasp=GRASP_SIDE, azimuth_deg=round(c.azimuth_deg, 1),
                   pitch_deg=c.pitch_deg, carry_rung=cr, psi_deg=round(psi, 2),
                   margin_mm=m["margin_mm"], limit=m["limit"])
                passing.append((m, pp))
                done = True
                break
            if done:
                break
    if not passing:
        if last is None:
            first = next((c for c in cands if c.refusal), None)
            raise CP.CraneRefused(first.stage if first else "plan",
                                  first.refusal if first else "no side-on approach",
                                  candidates=[c.record() for c in cands])
        raise last
    m, best = max(passing, key=lambda mp: mp[0]["margin_mm"])
    ev("side_place_chosen", azimuth_deg=round(float(math.degrees(math.atan2(
        best.pick.approach[1], best.pick.approach[0]))), 1),
       psi_deg=round(best.psi_deg, 2), carry_rung=best.carry_rung, **m)
    return best


# ── The planner choice ────────────────────────────────────────────────────────

@dataclass
class Choice:
    """Which grasp a request is flown with, and why."""
    used: str
    plans: Dict[str, object] = field(default_factory=dict)      # grasp -> plan
    margins: Dict[str, Optional[Dict[str, object]]] = field(default_factory=dict)
    refusals: Dict[str, Optional[str]] = field(default_factory=dict)

    def record(self) -> Dict[str, object]:
        return {"used": self.used, "margins": self.margins, "refusals": self.refusals}

    @property
    def plan(self):
        return self.plans[self.used]


def _choose(results: Dict[str, Tuple[Optional[object], Optional[Dict[str, object]],
                                    Optional[CP.CraneRefused]]],
            ev: CP.EventLog) -> Choice:
    ok = {k: v for k, v in results.items() if v[0] is not None}
    margins = {k: v[1] for k, v in results.items()}
    refusals = {k: (None if v[2] is None else f"{v[2].stage}: {v[2].reason}")
                for k, v in results.items()}
    if not ok:
        order = [GRASP_SIDE, GRASP_CRANE]
        exc = next(results[k][2] for k in order if k in results and results[k][2] is not None)
        ev("GRASP_CHOSEN", used=None, margins=margins, refusals=refusals)
        raise CP.CraneRefused(exc.stage, "; ".join(f"{k}: {r}" for k, r in refusals.items()
                                                   if r), refusals=refusals)
    used = max(ok, key=lambda k: ok[k][1]["margin_mm"])
    ch = Choice(used, {k: v[0] for k, v in ok.items()}, margins, refusals)
    ev("GRASP_CHOSEN", used=used, margins=margins, refusals=refusals)
    return ch


def plan_lift_choice(planner, scene: SceneModel, object_id: str,
                     pose: Optional[CP.Observation] = None,
                     on_event: Optional[CP.EventLog] = None,
                     grasps: Sequence[str] = (GRASP_SIDE, GRASP_CRANE)) -> Choice:
    """The lift by whichever grasp passes every check with the larger margin."""
    ev = on_event or (lambda *_a, **_k: None)
    res = {}
    if GRASP_SIDE in grasps:
        try:
            p = plan_side_pick(planner, scene, object_id, pose, on_event=on_event)
            res[GRASP_SIDE] = (p, p.checks["margin"], None)
        except CP.CraneRefused as exc:
            res[GRASP_SIDE] = (None, None, exc)
    if GRASP_CRANE in grasps:
        try:
            p = CP.plan_crane_pick(planner, scene, object_id, pose, on_event=on_event)
            res[GRASP_CRANE] = (p, plan_margin(p), None)
        except CP.CraneRefused as exc:
            res[GRASP_CRANE] = (None, None, exc)
    return _choose(res, ev)


def plan_place_choice(planner, scene: SceneModel, object_id: str, target_cell: str,
                      pose: Optional[CP.Observation] = None,
                      on_event: Optional[CP.EventLog] = None,
                      grasps: Sequence[str] = (GRASP_SIDE, GRASP_CRANE)) -> Choice:
    """The placement by whichever grasp passes every check with the larger margin."""
    ev = on_event or (lambda *_a, **_k: None)
    res = {}
    if GRASP_SIDE in grasps:
        try:
            pp = plan_side_place(planner, scene, object_id, target_cell, pose,
                                 on_event=on_event)
            res[GRASP_SIDE] = (pp, pp.checks["margin"], None)
        except CP.CraneRefused as exc:
            res[GRASP_SIDE] = (None, None, exc)
    if GRASP_CRANE in grasps:
        try:
            pp = PL.plan_crane_place(planner, scene, object_id, target_cell, pose,
                                     on_event=on_event)
            res[GRASP_CRANE] = (pp, plan_margin(pp.pick, pp), None)
        except CP.CraneRefused as exc:
            res[GRASP_CRANE] = (None, None, exc)
    return _choose(res, ev)
