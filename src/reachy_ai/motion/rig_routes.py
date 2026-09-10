"""The measured pose set and routes for the FWD Center Lab rig (issue #64).

Extracted from `notebooks/tlh_motion-routine.ipynb`, whose cell sources are the
source of truth for every number here.  The notebook is preserved and is not
imported: importing it would execute its cells, which connect to a robot.

WHY THESE NUMBERS CANNOT BE TIDIED
----------------------------------
The corridor out of the rail pocket is a few millimetres wide in places, and
the numbers were checked at 0.2 degree resolution against the board, all five
rig rails, the pedestal and the robot's own links.  The per-waypoint tolerances
are not a style choice: settling into GRIP_SHUT happens with 79 mm of clearance
all round, while SWING_3 -> HOVER passes the elbow 4.8 mm from the board's
edge.  A single uniform tolerance would be a rewrite, not an extraction.

The comments that explain a number travel WITH the number.  PRESENT in
particular carries the reason it is not the pose it used to be, and without
that reason it looks like a value somebody could improve.

WHICH SCENE
-----------
The pose set is labelled FWDCenterLabMCC.  Endpoint equality does not prove the
path is clear, and the panel runs FWDCenterLabSivaPool.  So a route is not
usable in a scene until it has been flown there and recorded in
`ROUTE_COMPATIBILITY`; `check_route()` refuses the pair otherwise, in the
scene's own name rather than by claiming the arm is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

#: Named postures.  `REST` is the forearm supported on the tabletop; `HOME` is
#: stored in the rail pocket.  They are eleven waypoints apart, and the
#: operator's word for both of them is "rest".
POSTURE_HOME = "home"
POSTURE_REST = "rest"
POSTURE_PRESENT = "present"
#: There is no side-hub posture.  There used to be one at roll -88, invented
#: for `raise_to_side` and reached by a sweep the clearance model puts inside
#: both the board and the outer rail.  PRESENT is the junction now, and PRESENT
#: is a measured pose — see the routes below.

#: Gripper angles.  The sign is inverted from the obvious reading: negative
#: OPENS.  Verified on the physical robot, not inferred from the joint name.
OPEN = -45.0    # ~6.5 cm pad gap
SHUT = 20.0     # ~0.7 cm

R_JOINTS: Tuple[str, ...] = (
    "r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch",
    "r_forearm_yaw", "r_wrist_pitch", "r_wrist_roll", "r_gripper",
)
ARM7: Tuple[str, ...] = R_JOINTS[:7]

#: Tracking tolerance for a guarded waypoint, in degrees.
TRACK_TOL = 6.0

#: The loose tolerance used where the point is the lesson rather than the pose
#: — the joint sweeps, and the wave.  Three weak joints reversing together do
#: not arrive in step, and holding them to TRACK_TOL trips the guard on a move
#: that is behaving exactly as intended.
LESSON_TOL = 90.0

#: Clearance margin, in metres.  Five times the measured degradation between a
#: commanded pose and the realised one, and the same value the joint sweeps
#: use.  It is NOT derived from the worst pad miss: the pad is not the part of
#: the arm that approaches anything.
SAFE_MARGIN = 0.05


def pose(**kw) -> Dict[str, float]:
    """A full seven-joint pose, gripper open unless told otherwise."""
    base = dict.fromkeys(ARM7, 0.0)
    base["r_gripper"] = OPEN
    base.update(kw)
    return base


# ── Verified pose set for FWDCenterLabMCC ────────────────────────────────────
# Checked at 0.2 deg resolution against the board, all five rig rails, the
# pedestal and the robot's own links.  Do not tweak blind — the corridor out of
# the pocket is only a few millimetres wide in places.

HOME = pose()
GRIP_SHUT = pose(r_gripper=SHUT)

# 1. back out of the pocket — extension only, roll stays at 0
BACK = pose(r_gripper=SHUT, r_shoulder_pitch=40.0)

# 2. curl the forearm up tight against the upper arm
CURL = pose(r_gripper=SHUT, r_shoulder_pitch=40.0,
            r_elbow_pitch=-125.0, r_wrist_pitch=45.0)
CURL_HIGH = pose(r_gripper=SHUT, r_shoulder_pitch=70.0,
                 r_elbow_pitch=-120.0, r_wrist_pitch=45.0)
TUCK = pose(r_gripper=SHUT, r_shoulder_pitch=70.0,
            r_elbow_pitch=-120.0, r_wrist_pitch=-45.0)

# 3. swing the folded unit forward and out over the rail
SWING_1 = pose(r_gripper=SHUT, r_shoulder_pitch=37.5, r_shoulder_roll=-32.5,
               r_elbow_pitch=-120.0, r_wrist_pitch=-45.0)
SWING_2 = pose(r_gripper=SHUT, r_shoulder_pitch=20.0, r_shoulder_roll=-35.0,
               r_elbow_pitch=-120.0, r_wrist_pitch=-45.0)
SWING_3 = pose(r_gripper=SHUT, r_shoulder_pitch=-17.5, r_shoulder_roll=-37.5,
               r_elbow_pitch=-120.0, r_wrist_pitch=-45.0)

# 4. unfold onto the board
HOVER = pose(r_gripper=SHUT, r_shoulder_pitch=-40.0, r_shoulder_roll=-10.0,
             r_elbow_pitch=-60.0, r_wrist_pitch=-15.0)
REST_SHUT = pose(r_gripper=SHUT, r_shoulder_pitch=-40.0, r_shoulder_roll=-10.0,
                 r_elbow_pitch=-45.0, r_wrist_pitch=-10.0, r_wrist_roll=30.0)
REST = dict(REST_SHUT, r_gripper=OPEN)

# Raised pose for the joint sweeps and the wave.  Not eyeballed: this is the
# pose that keeps every LINK of the arm clear of every object through every
# sweep.
#
# It used to be (-37.5, -2.0, -80.0), described as "collision-free, so joints
# can be swept to their limits".  That was measured on the gripper pad, which
# sat 21 cm above the board and looked entirely safe.  The pad is not the arm:
# at that pose the ELBOW was at z = 0.778, 3.8 cm above a 0.740 tabletop and
# directly over the near-right grid cell where red_cube stands, with the
# forearm already 5 mm INSIDE the cube before any joint moved.
#
# Lifting the hand does not fix that.  The elbow rides a fixed 0.28 m sphere
# about the shoulder; red_cube's nearest surface is 0.320 m from the shoulder
# and the upper arm's own surface reaches 0.315 m, so the near-right cell lies
# inside the elbow's arc no matter how the wrist is held.  The arm has to be
# pointed where the arc does not sweep: up, and out to the right.
#
#     worst clearance, whole arm, over the whole of routine 2:
#         (-37.5,  -2.0, -80.0)   -1.9 cm   already inside red_cube
#         (-70.0, -25.0, -80.0)  +10.9 cm   vs red_cube, via the upper arm
#                                + 8.9 cm   counting the table and rig as well
PRESENT = pose(r_shoulder_pitch=-70.0, r_shoulder_roll=-25.0,
               r_elbow_pitch=-80.0)


@dataclass(frozen=True)
class Waypoint:
    """One step of a route.

    `tol` is per waypoint because the waypoints are not equally dangerous.
    """

    name: str
    pose: Dict[str, float]
    seconds: float
    tol: float
    #: Which joints must actually arrive.  None means every one but the
    #: gripper, which is right for the pocket corridor — there the wrist angles
    #: are part of the shape that fits.  The presentation route names the gross
    #: joints instead: the wave ends with the forearm yaw at +-60 by design,
    #: and holding the return to that refuses a move that is perfectly safe.
    guard: Optional[Tuple[str, ...]] = None


#: Out of the pocket and onto the board.
PLACE_ROUTE: Tuple[Waypoint, ...] = (
    Waypoint("GRIP_SHUT", GRIP_SHUT, 3.5, 25.0),
    Waypoint("BACK", BACK, 3.0, TRACK_TOL),
    Waypoint("CURL", CURL, 3.0, TRACK_TOL),
    Waypoint("CURL_HIGH", CURL_HIGH, 2.0, TRACK_TOL),
    Waypoint("TUCK", TUCK, 2.0, TRACK_TOL),
    Waypoint("SWING_1", SWING_1, 3.0, TRACK_TOL),
    Waypoint("SWING_2", SWING_2, 2.5, TRACK_TOL),
    Waypoint("SWING_3", SWING_3, 2.5, TRACK_TOL),
    Waypoint("HOVER", HOVER, 2.5, TRACK_TOL),
    Waypoint("REST_SHUT", REST_SHUT, 3.0, TRACK_TOL),
    Waypoint("REST", REST, 3.0, TRACK_TOL),
)

#: The placement route run backwards.  Nothing may cut across it: a direct move
#: from anywhere over the board to HOME drives the upper arm through the
#: board's near edge.
STOW_ROUTE: Tuple[Waypoint, ...] = (
    Waypoint("REST_SHUT", REST_SHUT, 2.0, TRACK_TOL),
    Waypoint("HOVER", HOVER, 2.0, TRACK_TOL),
    Waypoint("SWING_3", SWING_3, 2.5, TRACK_TOL),
    Waypoint("SWING_2", SWING_2, 2.5, TRACK_TOL),
    Waypoint("SWING_1", SWING_1, 2.0, TRACK_TOL),
    Waypoint("TUCK", TUCK, 3.0, TRACK_TOL),
    Waypoint("CURL_HIGH", CURL_HIGH, 2.0, TRACK_TOL),
    Waypoint("CURL", CURL, 2.0, TRACK_TOL),
    Waypoint("BACK", BACK, 3.0, TRACK_TOL),
    Waypoint("GRIP_SHUT", GRIP_SHUT, 3.0, 25.0),
    Waypoint("HOME", HOME, 2.5, 25.0),
)

# ── Getting to the presentation pose (notebook section 4) ───────────────────
#
# THE NOTEBOOK HAS NO SIDE HUB, AND THIS NO LONGER INVENTS ONE.  There used to
# be a `SIDE_HUB` at roll -88 with a two-leg `PRESENT_ROUTE` onto it, none of
# which appears anywhere in the measured route.  The notebook reaches the
# raised pose from REST, in one move (cell 18), and it comes back down by
# flying STOW_ROUTE from PRESENT (cell 34) — whose first waypoint is REST_SHUT.
# Those two moves are what these routes are.
#
# The -88 hub was worse than unmeasured, it was wrong: getting to it swept the
# hand from 0 to -88 of roll at a fixed pitch, which the clearance model puts
# 4.3 to 6.5 cm inside `table_top` — the HAND against the board's near edge —
# for every degree of that sweep, clearing only in the last ten.  The notebook
# names that
# exact manoeuvre as its FAILURE exercise: "Try the sideways version and watch
# it fail ... the arm jams against the outer rail.  That jamming IS the
# feedback."
#
# PRESENT IS THE RAISED SIDE POSE.  It is what "out to the robot's right"
# means here, and unlike the hub it was measured: notebook section 4 records
# +8.9 cm worst whole-arm clearance over the entire routine, against the -1.9
# cm of the pose it replaced.  So there is no separate hub posture any more —
# PRESENT is the junction, and it is a notebook pose.

#: REST -> PRESENT.  Notebook cell 18, one move.  Modelled monotonic the whole
#: way: +4.1 cm a quarter in, +7.2 at the half, +13.8 cm on arrival, with the
#: only negative at the REST end where the forearm is deliberately supported on
#: the board.
#:
#: The notebook flies this with `tol=LESSON_TOL`, which switches the abort off
#: because section 4 is a teaching sweep and stopping it would hide the joint
#: behaviour it exists to show.  That is not a safety judgement about the move,
#: so the guard here is the gross joints at 10 degrees, as every other route
#: in this module uses.
LIFT_TO_PRESENT: Tuple["Waypoint", ...] = ()

#: PRESENT -> REST.  The first move of the notebook's stow (PRESENT ->
#: REST_SHUT, cell 34) followed by the route's own last waypoint, which opens
#: the hand.  Both poses are the measured ones; neither move is new.
LOWER_TO_REST: Tuple["Waypoint", ...] = ()

# ── The wave (notebook 4.6) ──────────────────────────────────────────────────
# Defined relative to PRESENT, so the wave inherits PRESENT's clearance
# argument rather than making its own.
WAVE_A = dict(PRESENT, r_forearm_yaw=-60.0, r_wrist_pitch=25.0,
              r_wrist_roll=-30.0)
WAVE_B = dict(PRESENT, r_forearm_yaw=+60.0, r_wrist_pitch=-25.0,
              r_wrist_roll=+30.0)

#: Three cycles, 1.8 s a swing.  Both measured, not chosen: at 0.9 s the three
#: weak joints reversing together finished tens of degrees short and tripped
#: the tracking guard.  Raising either without re-measuring is how a bounded
#: behaviour stops being bounded.
WAVE_CYCLES = 3
WAVE_SECONDS = 1.8

#: The wave ends where it began.  Returning to tabletop rest or to the pocket
#: is a separate validated transition, not something to append so the arm looks
#: tidy.
WAVE_START = "present"
WAVE_END = "present"

_PRESENT_GUARD = ("r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw",
                  "r_elbow_pitch")

LIFT_TO_PRESENT = (
    Waypoint("PRESENT", PRESENT, 3.0, 10.0, _PRESENT_GUARD),
)

LOWER_TO_REST = (
    Waypoint("REST_SHUT", REST_SHUT, 3.0, TRACK_TOL, _PRESENT_GUARD),
    Waypoint("REST", REST, 3.0, TRACK_TOL, _PRESENT_GUARD),
)

#: HOME -> PRESENT and back, as `primitives.raise_to_side` and
#: `primitives.stow_from_side` fly them: the measured route out of the pocket,
#: then the lift, and the exact reverse coming back.  Named because the posture
#: graph and the compatibility record work in route names.
RAISE_TO_SIDE: Tuple["Waypoint", ...] = PLACE_ROUTE + LIFT_TO_PRESENT
STOW_FROM_SIDE: Tuple["Waypoint", ...] = STOW_ROUTE

# ── Pointing (notebook 4.7) ──────────────────────────────────────────────────
#: Air wanted under the pad.  This number is about POINTING; it is not what
#: keeps the objects on the table — the clearance guard is.  An earlier version
#: added 10 cm of slop on top, which bought nothing (a run with the taller
#: hover still threw the blue cylinder 1.06 m off the board, because the object
#: was hit by the FOREARM in transit and no pad height addresses that) and
#: wrecked the pointing it exists to demonstrate.
POINT_CLEARANCE = 0.06

#: Where the base hover is not enough, the target is LIFTED rather than skipped.
POINT_LIFT_STEP = 0.05
POINT_LIFT_MAX = 0.25

#: How much closer than the destination hover an approach leg may come to the
#: object being reached for.  Self-consistent by construction: the approach may
#: be exactly as close as it has to be and no closer.
POINT_APPROACH_SLACK = 0.02

#: How many hops each approach is broken into.  This is the knob that decides
#: how good an approximation the path clearance is: it models the arm between
#: two poses as a joint-space straight line, which is the assumption that
#: failed when cell_r2c1 reported +5.5 cm and still moved the can 0.189 m — the
#: guard measured the ENDPOINTS, and the can was hit in the middle, where
#: nothing was measured at all.  Six legs puts a measurement every ~4 cm of pad
#: travel.  It never reaches zero error, and pretending otherwise is how the
#: last four versions of this guard were wrong.
POINT_LEGS = 6
POINT_MARGIN = SAFE_MARGIN

#: The joints that decide where the arm sits in the rig.  The wrist angles and
#: the gripper do not move the elbow or forearm through the rails, and on a
#: freshly reset sim they read wherever gravity left them while the motors were
#: off (wrist_roll drifts to ~40 deg, the gripper falls open) — so judging "is
#: the arm home?" on those would report a clean reset as a fault.
GROSS_JOINTS: Tuple[str, ...] = (
    "r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch",
)

#: The joints a route waypoint is JUDGED on — the notebook's `CRITICAL`.
#:
#: The gross four, which put the elbow and forearm where the clearance was
#: measured, plus the WRIST PITCH: that one folds the hand and is part of the
#: shape that fits through the pocket, which is why CURL carries +45.
#:
#: The forearm yaw, the wrist roll and the gripper are NOT in it, and leaving
#: them out is not laziness.  They spin the hand about its own axis and move
#: the elbow nowhere; they are weak (kp=60, 10 Nm) and converge over several
#: waypoints; and a few degrees on them moves the pad by millimetres inside
#: margins of 20 mm or more.  Holding them to TRACK_TOL aborts routes that are
#: in no danger — measured: a stow flown from PRESENT stopped at SWING_2
#: because `r_wrist_roll` was 23 degrees into a 30 degree move, with the whole
#: arm 2.3 cm from anything and the board untouched.
CRITICAL_JOINTS: Tuple[str, ...] = GROSS_JOINTS + ("r_wrist_pitch",)


# ---------------------------------------------------------------------------
# Which routes are validated in which scenes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteValidation:
    """The record of a route having actually been flown in a scene.

    `evidence` is where the run is described — when, against what, and what the
    worst clearance was.  An entry with no evidence is not a validation, and
    `check_route` treats it as one only because somebody wrote it down; keeping
    the field required makes writing one down deliberate.
    """

    route: str
    scene: str
    evidence: str


#: Routes proven in a scene by flying them there.
#:
#: FWDCenterLabMCC is where the pose set was measured, so its entries carry the
#: notebook as their evidence.  FWDCenterLabSivaPool is deliberately ABSENT.
#: The two scenes differ in object population, endpoint equality does not prove
#: the corridor between the endpoints is clear, and the corridor is a few
#: millimetres wide in places.  Adding a row here without a flown run would be
#: the whole safety argument of this module, undone by a one-line edit.
ROUTE_COMPATIBILITY: Tuple[RouteValidation, ...] = (
    RouteValidation(
        "PLACE_ROUTE", "FWDCenterLabMCC",
        "notebooks/tlh_motion-routine.ipynb section 3: flown, board untouched, "
        "waypoint clearances checked at 0.2 deg resolution",
    ),
    RouteValidation(
        "STOW_ROUTE", "FWDCenterLabMCC",
        "notebooks/tlh_motion-routine.ipynb section 5: flown, arm verified at "
        "HOME on the gross joints",
    ),
    RouteValidation(
        "WAVE", "FWDCenterLabMCC",
        "notebooks/tlh_motion-routine.ipynb section 4.6: three cycles at 1.8 s, "
        "scene drift checked afterwards",
    ),
    RouteValidation(
        "LIFT_TO_PRESENT", "FWDCenterLabMCC",
        "notebooks/tlh_motion-routine.ipynb cell 18: REST -> PRESENT in one "
        "move, then the whole of section 4 flown from it with the board "
        "instrumented cell by cell and every object still on its cell",
    ),
    RouteValidation(
        "LOWER_TO_REST", "FWDCenterLabMCC",
        "notebooks/tlh_motion-routine.ipynb cell 34: PRESENT -> REST_SHUT is "
        "the first move of the stow, flown at the end of every section 4 run",
    ),
    RouteValidation(
        "RAISE_TO_SIDE", "FWDCenterLabMCC",
        "PLACE_ROUTE then LIFT_TO_PRESENT, both above: "
        "notebooks/tlh_motion-routine.ipynb flies exactly this pair, in this "
        "order, in sections 3 and 4",
    ),
    RouteValidation(
        "STOW_FROM_SIDE", "FWDCenterLabMCC",
        "STOW_ROUTE, above: the stow in notebooks/tlh_motion-routine.ipynb IS "
        "the return from PRESENT, because its first waypoint is REST_SHUT",
    ),
    # THE SivaPool ROWS THAT WERE HERE WERE DELETED RATHER THAN EDITED.
    #
    # They certified a different implementation.  RAISE_TO_SIDE and
    # STOW_FROM_SIDE used to be four hand-built steps through a roll -88 hub
    # that appears nowhere in the notebook, and their rows rested on "flown six
    # times, board undisturbed" — true, and not evidence about a board that was
    # empty at the time.  Modelled afterwards: the ascent runs 4.3 to 6.5 cm
    # inside `table_top` for the whole of its roll sweep, and the descent's
    # join into CURL puts the hand 6.4 cm inside it.  Both are the gripper
    # through the board's near edge, which is what an operator watched.  A
    # validation row for code that no longer exists is worse than no row.
    # PRESENT_ROUTE and PRESENT_RETURN went with the hub they connected to.
    #
    # The WAVE row for SivaPool went too.  The wave itself was measured there
    # — +13.1, +13.3 and +13.2 cm across three runs — but the row described
    # the approach it was flown through, and that approach was the hub.  The
    # pose the wave happens at has not changed; the way the arm reaches it has.
    #
    # All of these have to be flown in SivaPool and measured before they get a
    # row here, exactly as PLACE_ROUTE and STOW_ROUTE do (#74).
    # POINT is not listed for any scene.  Section 4.7 documents runs that moved
    # objects — a can shifted 0.189 m by an arm reporting positive clearance —
    # so it is not a validated route anywhere, and saying so here is more use
    # than a comment somewhere else saying it is imperfect.
    #
    # FWDCenterLabSivaPool is not listed either, and that is now a MEASURED
    # answer rather than an untested assumption.  See VALIDATION_ATTEMPTS.
)


@dataclass(frozen=True)
class ValidationAttempt:
    """A route flown in a scene and NOT accepted, with the numbers.

    Worth keeping next to the passes.  "Not listed" reads as "nobody has got to
    it yet", which invites someone to add a row on the strength of one clean
    run.  "Flown four times, one pass went 2 mm inside the rail" does not.
    """

    route: str
    scene: str
    when: str
    outcome: str
    detail: str


VALIDATION_ATTEMPTS: Tuple[ValidationAttempt, ...] = (
    ValidationAttempt(
        "RAISE_TO_SIDE", "FWDCenterLabSivaPool", "2026-09-10", "rejected",
        "HOME->PRESENT, flown three times after `raise_to_side` was rebuilt on "
        "the measured route. Complete and arrived at PRESENT every time, 33-40 "
        "s, board undisturbed every run (soda_can on r1c1, foam_block on "
        "r2c3, 0.0000 m). Sampled at 20 Hz through the whole flight, not just "
        "at the waypoints. Worst realised vs the RIG: -0.24, -0.81 and -0.28 "
        "cm, hand vs rig_rail_outer_right, which is SWING_1 — the same "
        "waypoint and the same rail that rejected PLACE_ROUTE below, and the "
        "same verdict follows. Worst overall -4.31, -2.39, -3.75 cm hand vs "
        "table_top, which is REST and is the route's INTENT: the forearm is "
        "deliberately supported on the board there.",
    ),
    ValidationAttempt(
        "STOW_FROM_SIDE", "FWDCenterLabSivaPool", "2026-09-10", "rejected",
        "PRESENT->HOME, flown three times, complete and arrived at HOME every "
        "time, 29-34 s, board undisturbed. Worst realised vs the rig -0.83, "
        "-0.58 and -0.49 cm at the same SWING_1 crossing. Same verdict, same "
        "reason. Note what this is NOT: the four hand-built steps it replaced "
        "held the gripper 4.3-6.5 cm inside the board for 78 degrees of roll "
        "sweep. Grazing a rail by 8 mm and ploughing the board by 65 mm are "
        "not the same finding, and only one of them was ever visible to an "
        "operator watching RViz.",
    ),
    ValidationAttempt(
        "PLACE_ROUTE", "FWDCenterLabSivaPool", "2026-09-10", "rejected",
        "Flown three times, HOME->REST. Twice complete and arrived; the third "
        "stopped at REST_SHUT with r_wrist_roll 9 deg off its 6 deg tolerance "
        "after five re-streamed passes. Board undisturbed every time (all 10 "
        "objects 0.0000 m). Rejected on clearance: worst realised at SWING_1 "
        "vs rig_rail_outer_right was +0.2, +0.1 and +0.2 cm against a planned "
        "+0.6 cm.",
    ),
    ValidationAttempt(
        "STOW_ROUTE", "FWDCenterLabSivaPool", "2026-09-10", "rejected",
        "Flown three times, REST->HOME, complete and arrived every time, board "
        "undisturbed. Rejected on clearance: SWING_1 vs rig_rail_outer_right "
        "measured +0.7, -0.2 and +0.5 cm — the model puts the arm INSIDE the "
        "rail on the second pass. Six SWING_1 samples across both routes: "
        "+0.2, +0.7, +0.1, -0.2, +0.2, +0.5 cm, one of six negative, spread "
        "0.9 cm on a 0.6 cm budget. The corridor completes, and it has no "
        "margin at its tightest waypoint.",
    ),
    ValidationAttempt(
        "POINT", "FWDCenterLabMCC", "2026-08", "rejected",
        "Measured in notebook section 4.7 and not accepted: pad miss up to "
        "21.2 cm across the grid, and cell_r2c1 reported +5.5 cm of clearance "
        "while moving the can 0.189 m. The arm bows off the line the guard "
        "cleared, between the points where anything was measured.",
    ),
)


def attempts_for(route: str, scene: str) -> Tuple[ValidationAttempt, ...]:
    return tuple(a for a in VALIDATION_ATTEMPTS
                 if a.route == route and a.scene == scene)


def validation_for(route: str, scene: str) -> Optional[RouteValidation]:
    for row in ROUTE_COMPATIBILITY:
        if row.route == route and row.scene == scene:
            return row
    return None


def check_route(route: str, scene: str) -> Tuple[bool, str]:
    """May this route be flown in this scene?  (ok, why_not)

    The refusal names the scene rather than the arm.  "I cannot do that in this
    scene" sends someone to the compatibility record; "the arm is unavailable"
    sends them to the robot, which is fine.
    """
    if not route:
        return False, "that action names no route"
    row = validation_for(route, scene)
    if row is not None:
        return True, ""
    tried = attempts_for(route, scene)
    if tried:
        # Say what happened, not just that the row is missing.  "Not listed"
        # invites someone to add a row after one clean run; "flown and
        # rejected, here is the number" does not.  And a route nobody has
        # flown here is a different answer from one that was flown and failed.
        attempt = tried[0]
        if attempt.outcome == "rejected":
            return False, (f"the {route} route was flown in {scene} on "
                           f"{attempt.when} and rejected: {attempt.detail}")
        return False, (f"the {route} route has not been flown in {scene}: "
                       f"{attempt.detail}")
    flown_in = sorted({r.scene for r in ROUTE_COMPATIBILITY if r.route == route})
    if not flown_in:
        # No pass anywhere.  If it was tried somewhere else and failed, that
        # reason travels: a route rejected in the scene it was designed for is
        # not going to be better in one it was not.
        elsewhere = [a for a in VALIDATION_ATTEMPTS
                     if a.route == route and a.outcome == "rejected"]
        if elsewhere:
            return False, (f"the {route} route is not validated in any scene. "
                           f"It was flown in {elsewhere[0].scene} and "
                           f"rejected: {elsewhere[0].detail}")
        return False, (f"the {route} route has not been validated in any "
                       "scene yet, so I will not fly it")
    return False, (f"the {route} route was measured in "
                   f"{', '.join(flown_in)} and has not been flown in {scene}. "
                   "The endpoints matching does not mean the path between them "
                   "is clear, and this corridor is a few millimetres wide in "
                   "places.")


def route_named(name: str) -> Tuple[Waypoint, ...]:
    routes = {"PLACE_ROUTE": PLACE_ROUTE, "STOW_ROUTE": STOW_ROUTE,
              "LIFT_TO_PRESENT": LIFT_TO_PRESENT,
              "LOWER_TO_REST": LOWER_TO_REST,
              "RAISE_TO_SIDE": RAISE_TO_SIDE,
              "STOW_FROM_SIDE": STOW_FROM_SIDE}
    if name not in routes:
        raise KeyError(f"no route called {name!r}")
    return routes[name]


def nearest_waypoint(present: Dict[str, float],
                     route: Tuple[Waypoint, ...] = STOW_ROUTE) -> Tuple[str, float]:
    """The route waypoint the arm is closest to now, and how far off it is.

    Used for recovery, and the reason it is only half an answer is worth
    stating where the function is: `ensure_home()` in the notebook eases onto
    this waypoint and retraces from there, and says plainly that the connecting
    segment is the ONE move on an unverified path.  An arm threaded under the
    front rail is not a routine stow start, and no commanded pose pulls it back
    out — the pad ends up below the tabletop.  So this reports; it does not
    authorise.
    """
    def distance(target: Dict[str, float]) -> float:
        return max(abs(present.get(n, 0.0) - v)
                   for n, v in target.items() if n != "r_gripper")

    best = min(route, key=lambda w: distance(w.pose))
    return best.name, distance(best.pose)


def at_pose(present: Dict[str, float], target: Dict[str, float],
            tol: float = 8.0, joints: Optional[List[str]] = None) -> bool:
    names = list(joints) if joints is not None else [n for n in target
                                                     if n != "r_gripper"]
    return all(abs(present.get(n, 0.0) - target[n]) <= tol for n in names)


# ---------------------------------------------------------------------------
# The posture graph
# ---------------------------------------------------------------------------

#: The named postures a route may start or end at.
POSTURES: Dict[str, Dict[str, float]] = {
    POSTURE_HOME: HOME,
    POSTURE_REST: REST,
    POSTURE_PRESENT: PRESENT,
}

#: Which route takes the arm from one posture to another.
#:
#: A pair that is not here is not a transition.  There is no "just move there"
#: edge, and the absence is the point: a direct move from anywhere over the
#: board to HOME drives the upper arm through the board's near edge, and the
#: side-hub exit that IS named "go home" drives it through the outer rail in at
#: least one scene (#73).  Two motions can both be reasonably called going
#: home; only one of them is measured for the trip you are making.
POSTURE_TRANSITIONS: Dict[Tuple[str, str], str] = {
    (POSTURE_HOME, POSTURE_REST): "PLACE_ROUTE",
    (POSTURE_REST, POSTURE_HOME): "STOW_ROUTE",
    (POSTURE_HOME, POSTURE_PRESENT): "RAISE_TO_SIDE",
    (POSTURE_PRESENT, POSTURE_HOME): "STOW_FROM_SIDE",
    (POSTURE_REST, POSTURE_PRESENT): "LIFT_TO_PRESENT",
    (POSTURE_PRESENT, POSTURE_REST): "LOWER_TO_REST",
    (POSTURE_PRESENT, POSTURE_PRESENT): "WAVE",
}


def posture_of(present: Dict[str, float], tol: float = 8.0) -> Optional[str]:
    """Which named posture the arm is standing at, or None.

    Judged on the gross joints: the wrist angles and the gripper do not decide
    where the arm sits in the rig, and on a freshly reset simulator they read
    wherever gravity left them.
    """
    for name, target in POSTURES.items():
        if at_pose(present, target, tol=tol, joints=list(GROSS_JOINTS)):
            return name
    return None


def transition(frm: str, to: str) -> Optional[str]:
    """The route from one posture to another, or None if there is not one."""
    if frm == to:
        return POSTURE_TRANSITIONS.get((frm, to))
    return POSTURE_TRANSITIONS.get((frm, to))


def path(frm: str, to: str) -> Optional[List[str]]:
    """The routes that get from one posture to another, in order.

    Breadth-first over the measured edges, so "wave" asked of an arm in the
    rail pocket answers RAISE_TO_SIDE rather than refusing.  The arm has to
    leave the pocket before it can do anything, and that is a fact about the
    rig rather than something the operator should have to know and type.

    None means no measured sequence exists.  That is still a real answer: it
    is what stops a plan inventing a way through.
    """
    if frm == to:
        direct = POSTURE_TRANSITIONS.get((frm, to))
        return [direct] if direct else []
    seen = {frm}
    queue: List[Tuple[str, List[str]]] = [(frm, [])]
    while queue:
        here, so_far = queue.pop(0)
        for (a, b), route in POSTURE_TRANSITIONS.items():
            if a != here or b in seen or a == b:
                continue
            steps = so_far + [route]
            if b == to:
                return steps
            seen.add(b)
            queue.append((b, steps))
    return None


def reachable_from(frm: str) -> Tuple[str, ...]:
    return tuple(sorted(to for (f, to) in POSTURE_TRANSITIONS if f == frm))
