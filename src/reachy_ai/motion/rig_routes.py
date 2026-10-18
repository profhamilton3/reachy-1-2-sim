"""The measured pose set and routes for the FWD Center Lab rig (issue #64).

The pose set was extracted from `notebooks/tlh_motion-routine.ipynb`, which is
preserved and not imported: importing it would execute its cells, which
connect to a robot.

THE OCTOBER LAYOUT (#192)
-------------------------
The robot moved.  Since the October layout (owner decision 2026-10-17) the
torso sits at (0.206, 0, 1.15) in the scene frame, and its shoulders are over
the board.  Everything the August mount (torso at (0, 0, 1.0)) needed is gone:
the rail-pocket HOME, the corridor out of it (PLACE_ROUTE, STOW_ROUTE,
RAISE_TO_SIDE, STOW_FROM_SIDE), its poses and its margin rule (#74), and every
route record flown there.  They are in git history and in
docs/reviews/; nothing here replays them (owner decision 2026-10-18).

REST IS HOME.  Siva confirms the arm rests on the table, as in the lab photos,
so REST -- on the board's right margin, outside the grid -- is where the arm is
stored, what "stow", "store" and "put your arm away" mean, and the pose the
simulator starts in.  The posture graph is REST and PRESENT; WAVE and POINT
happen at PRESENT.  See docs/adr/0010-october-layout.md.

The comments that explain a number travel WITH the number.  PRESENT in
particular carries the reason it is not the pose it used to be, and without
that reason it looks like a value somebody could improve.

WHICH SCENE
-----------
Endpoint equality does not prove the path is clear, and the panel runs
FWDCenterLabSivaPool.  So a route is not usable in a scene until it has been
flown there, at the current mount, and recorded in `ROUTE_COMPATIBILITY`;
`check_route()` refuses the pair otherwise, in the scene's own name rather than
by claiming the arm is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

#: Named postures.  `REST` is the hand resting on the board's right margin.
#: There used to be a second, `HOME`, in the rig's rail pocket; in the
#: October layout there is no pocket to be in (#192), and HOME IS REST:
#: `POSTURE_HOME` is kept as a name for callers that mean "stored", and it
#: names the same posture.
POSTURE_REST = "rest"
POSTURE_HOME = POSTURE_REST
POSTURE_PRESENT = "present"

#: Where the robot is mounted, as recorded against each route validation.
#: `check_route` accepts only records made at `CURRENT_MOUNT`: a route flown
#: with the torso somewhere else proves nothing about the arm here.
MOUNT_OCTOBER = "october: torso (0.206, 0, 1.15)"
CURRENT_MOUNT = MOUNT_OCTOBER

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

#: The forearm-footprint guard's own clearance threshold (#82) —
#: `panel_executor._footprint_refusal` refuses a leg whose worst modelled
#: clearance is below this, in metres.
#:
#: NAMED HERE SO THE CHOICE IS VISIBLE, NOT BECAUSE IT HAS BEEN VALIDATED.
#: It is today's actual, previously-implicit behaviour (a bare `d < 0.0` in
#: the guard) written down unchanged — zero margin, not SAFE_MARGIN adopted
#: on purpose.  Whether the guard should use SAFE_MARGIN, this value, or
#: something else is #74's decision: the guard's own probe against
#: ADR-0002's evidence board disagrees with the eighteen flown legs at both
#: 0.0 and SAFE_MARGIN, and reconciling that needs the flight-measurement
#: data neither this constant nor this commit has.
FOOTPRINT_MARGIN = 0.0

#: How far an object may move before the board counts as disturbed, in metres.
#:
#: ONE NUMBER, NOT THREE.  The live executor's post-move check, the episode
#: recorder's success definition and the searched-recipe evaluators all ask the
#: same question — "did this motion move anything?" — and answering it with
#: three separately maintained literals is how they drift apart.  It lives here
#: because it belongs to the rig rather than to any one caller.
#:
#: 2 cm is the panel's own threshold, carried forward rather than re-derived.
#: It is a DISTURBANCE threshold and not a measurement accuracy: the pose
#: estimate is better than this, and a shove is much worse.
OBJECT_DRIFT_TOL = 0.02


def pose(**kw) -> Dict[str, float]:
    """A full seven-joint pose, gripper open unless told otherwise."""
    base = dict.fromkeys(ARM7, 0.0)
    base["r_gripper"] = OPEN
    base.update(kw)
    return base


# ── REST, October layout (#192) ──────────────────────────────────────────────
#
# The hand rests on the board's RIGHT MARGIN, outside the taped grid, as the
# lab photos show the real gripper doing: on its palm and back, fingertips just
# over the board's edge (tip at about (0.34, -0.40) in the scene; the outer
# tape border is at y = -0.2286, the board's edge at -0.349), with the forearm
# along the margin.
#
# FOUND BY LETTING GO OF THE ARM, NOT BY CHOOSING A POSE.  The arm is stored
# here and the simulator starts here with the motors off, exactly like the
# robot, so REST has to be a pose the arm STAYS in when nothing holds it.  An IK
# pose on the margin with the elbow swung out slid off the board's edge in two
# seconds.  This one is a settled pose: dropped motors-off in MuJoCo from many
# starts on the margin, kept only where a second drop moved it by under 2 deg,
# then dropped again for 120 s and kept only where the GRIPPER STAYED SHUT,
# then scored against a soda can, blue cylinder, red cube, pool cylinder, foam
# block and pool box standing on each column-3 cell.  Motors off for 10
# minutes from here: the gross joints drift at most 1.3 deg, the gripper stays
# on its shut stop, and the arm is still.  The forearm yaw (84.5 deg) is what
# holds the hand shut: it turns the hand so gravity closes the finger.  A
# REST whose finger hangs open creeps open over minutes (one did: +18 deg at
# 5 s, -15 at 120 s, the open stop by 11 minutes), and an open hand on the
# margin is the wide hand the footprint guard below assumes away.  The
# shoulder roll does NOT lean on the +20.3 deg stop (docs/adr/0010), so REST
# does not depend on that deviation.
#
# WITH THE HAND'S COLLISION BOXES (#192).  The first October REST rested on
# the finger pad, and the hand's visual shell sank 38 mm into the board,
# because the hand had no collision geometry but the pads.  With the back-of-
# hand and palm boxes the hand rests on them, and that REST no longer stayed
# put (wrist roll fell 35 deg, the gripper fell open), so it was searched
# again; this is the result.
#
# THE GRIPPER IS SHUT AT REST.  The footprint guard sizes the hand to its
# aperture, and an open hand's tube (8-11.5 cm radius) on a 10.8 cm margin
# reaches into column 3 whatever the arm does.  Shut, it is 3.9-6.7 cm.
#
#     the footprint guard's own number (`footprint.route_footprint_clearances`,
#     every leg of LOWER_TO_REST and LIFT_TO_PRESENT), one object on the cell:
#         r1c3 +1.8 cm   r2c3 +12.0 cm   r3c3 +20.8 cm   (worst of six objects)
#     the August REST and its routes, at the August mount, same objects:
#         r1c3 +2.3 cm   r2c3 -5.7 cm   r3c3 -9.4 cm    (every r2c3/r3c3 refused;
#                                                      pool_box_3 on r2c3: -3.7)
#
# Numbers and the search: docs/reviews/2026-10-17-october-layout-192/.
REST = pose(r_gripper=SHUT, r_shoulder_pitch=10.5, r_shoulder_roll=15.0,
            r_arm_yaw=-55.0, r_elbow_pitch=-81.0, r_forearm_yaw=84.5,
            r_wrist_pitch=-4.5, r_wrist_roll=-17.5)
#: REST already has the hand shut.  The name is kept for callers that mean
#: "rest, closed", and it is the same pose.
REST_SHUT = REST

#: REST's arm shape swung OUT to the side, at PRESENT's roll and level
#: pitch: the hand clear of the board's right edge.  The lift and the lowering
#: pass through it, so the hand comes onto the margin from outside the board
#: rather than across column 3.  The straight joint-space line from PRESENT to
#: REST crossed a soda can on r1c3 by -2.2 cm; via this pose the worst is the
#: +3.3 cm of REST itself.
REST_OUT = dict(REST, r_shoulder_pitch=0.0, r_shoulder_roll=-25.0)

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

#: PRESENT with the hand shut: the gripper closes up here before the arm
#: descends to REST, and opens up here after the lift, so the legs over the
#: board are judged with the narrow hand (see REST).
PRESENT_SHUT = dict(PRESENT, r_gripper=SHUT)


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
    #: gripper.  The routes here name the gross joints instead: the wave ends
    #: with the forearm yaw at +-60 by design, and holding the return to that
    #: refuses a move that is perfectly safe.
    guard: Optional[Tuple[str, ...]] = None


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

#: The wave ends where it began.  Returning to REST is a separate validated
#: transition, not something to append so the arm looks tidy.
WAVE_START = "present"
WAVE_END = "present"

# ── Getting to the presentation pose, and back ──────────────────────────────
#
# PRESENT IS THE RAISED SIDE POSE and the junction of the posture graph: the
# wave and pointing start and end there, and REST is one route away.  There is
# no side hub; one at roll -88, reached by a sweep the clearance model put
# inside both the board and the outer rail, was removed before #192.

_PRESENT_GUARD = ("r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw",
                  "r_elbow_pitch")

#: The tolerance is 8, not 10, and it matches `posture_of`'s on purpose.
#:
#: At 10 the route could ARRIVE and still not be RECOGNISED: flown three
#: times, one run finished between 8 and 12 degrees off PRESENT, reported
#: success, and the wave that followed refused with "the arm is not at the
#: presentation pose" — which is true, confusing, and avoidable.  A route that
#: says it got there and a posture check that says it did not are the same
#: question asked twice with different answers.
#:
#: REST -> PRESENT, October layout (#192).  Out to the side first, so the hand
#: leaves the margin away from column 3, then up; the hand opens only once it
#: is raised.  Flown offline in MuJoCo with tall objects on column 3: nothing
#: moved, worst MJCF distance 3.6 cm (docs/reviews/2026-10-17-october-layout-192).
LIFT_TO_PRESENT = (
    Waypoint("REST_OUT", REST_OUT, 3.0, TRACK_TOL, _PRESENT_GUARD),
    Waypoint("PRESENT_SHUT", PRESENT_SHUT, 3.0, 8.0, _PRESENT_GUARD),
    Waypoint("PRESENT", PRESENT, 1.5, 8.0, _PRESENT_GUARD),
)

#: PRESENT -> REST, October layout: the reverse.  The hand shuts while raised,
#: swings out, and comes onto the margin from outside the board.
LOWER_TO_REST = (
    Waypoint("PRESENT_SHUT", PRESENT_SHUT, 1.5, 8.0, _PRESENT_GUARD),
    Waypoint("REST_OUT", REST_OUT, 3.0, TRACK_TOL, _PRESENT_GUARD),
    Waypoint("REST", REST, 3.0, TRACK_TOL, _PRESENT_GUARD),
)

# ── Pointing (notebook 4.7) ──────────────────────────────────────────────────
#: Air wanted under the pad.  This number is about POINTING; it is not what
#: keeps the objects on the table — the clearance guard is.  An earlier version
#: added 10 cm of slop on top, which bought nothing (a run with the taller
#: hover still threw the blue cylinder 1.06 m off the board, because the object
#: was hit by the FOREARM in transit and no pad height addresses that) and
#: wrecked the pointing it exists to demonstrate.
POINT_CLEARANCE = 0.06

#: The floor under the hover, and it is a floor rather than a target.
#:
#: The notebook derives the hover from the tallest thing on the board plus
#: POINT_CLEARANCE, then takes the larger of that and this.  On an empty board
#: the first term drops out entirely and this is the whole answer — which is
#: the simple case worth getting right first, because every occupied case is
#: this plus the height of what is standing there.
#:
#: Derived from the scene's own geometry, not guessed.  Object heights in
#: FWDCenterLabSivaPool, and the hover each one implies when it is the only
#: thing on the board:
#:
#:     empty board                    -            12.0 cm   (this floor)
#:     pool_box_1..3               4.0 cm          12.0 cm   (floor binds)
#:     foam_block                  5.0 cm          12.0 cm   (floor binds)
#:     red_cube                    6.0 cm          12.0 cm   (floor binds)
#:     pool_cyl_1..3               8.0 cm          14.0 cm
#:     blue_cylinder              10.0 cm          16.0 cm
#:     soda_can                   12.2 cm          18.2 cm
#:
#: Which is what the notebook means by "17 cm over the table is barely 6 cm
#: over a can".
POINT_HOVER_FLOOR = 0.12

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

#: The joints that decide where the arm sits.  The wrist angles and the gripper
#: do not move the elbow or forearm, and on a freshly reset sim they read
#: wherever gravity left them while the motors were off -- so judging "is the
#: arm at rest?" on those would report a clean reset as a fault.
GROSS_JOINTS: Tuple[str, ...] = (
    "r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch",
)

#: The joints a route waypoint is JUDGED on — the notebook's `CRITICAL`.
#:
#: The gross four, which put the elbow and forearm where the clearance was
#: measured, plus the WRIST PITCH: that one folds the hand.
#:
#: The forearm yaw, the wrist roll and the gripper are NOT in it, and leaving
#: them out is not laziness.  They spin the hand about its own axis and move
#: the elbow nowhere; they are weak (kp=60, 10 Nm) and converge over several
#: waypoints; and a few degrees on them moves the pad by millimetres inside
#: margins of 20 mm or more.  Holding them to TRACK_TOL aborts routes that are
#: in no danger -- measured, in August: a stow flown from PRESENT stopped
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
    #: Where the robot was mounted when this was flown (#192).  Required: a
    #: record that does not say where the robot stood cannot be held to the
    #: mount it is checked against.
    mount: str = ""

    def __post_init__(self) -> None:
        if not self.mount:
            raise ValueError(f"the {self.route} record in {self.scene} names no mount")


#: Routes proven in a scene by flying them there.
#:
#: EVERY ROW IS AT THE OCTOBER MOUNT, torso at (0.206, 0, 1.15) (#192).  The
#: August rows -- the rail-pocket corridor and the routes flown from it -- went
#: with the August mount (owner decision 2026-10-18); their evidence is in git
#: history and docs/reviews/.  The evidence here is OFFLINE physics: the
#: simulator's own MuJoCo model and contact settings, the arm streamed with the
#: motors on, without the server, SDK or bridge
#: (docs/reviews/2026-10-17-october-layout-192/).  FWDCenterLabSivaPool and
#: FWDCenterLabMCC share the board, grid and rig.
#:
#: Adding a row here without a flown run would be the whole safety argument of
#: this module, undone by a one-line edit.
ROUTE_COMPATIBILITY: Tuple[RouteValidation, ...] = (
    RouteValidation(
        "LIFT_TO_PRESENT", "FWDCenterLabSivaPool",
        "2026-10-18 (#192, offline physics, fly_rest.py and fly_wave_point.py): "
        "REST -> REST_OUT -> PRESENT_SHUT -> PRESENT, from the REST the hand "
        "collision boxes settle at, flown with soda_can / blue_cylinder / "
        "red_cube or blue_cylinder / soda_can / foam_block standing on r1c3 / "
        "r2c3 / r3c3, and as the approach of every WAVE and POINT flight "
        "below (12 flights). Arrived within 1.0 deg on the gross joints; no "
        "object moved (0.0 mm); worst MJCF distance to any of them 1.8 cm "
        "(blue_cylinder on r1c3). Only robot contacts: the forearm and the "
        "hand's palm and back on table_top, leaving REST.",
        mount=MOUNT_OCTOBER,
    ),
    RouteValidation(
        "LOWER_TO_REST", "FWDCenterLabSivaPool",
        "2026-10-18 (#192, offline physics): PRESENT -> PRESENT_SHUT -> "
        "REST_OUT -> REST, the same layouts as LIFT_TO_PRESENT and the return "
        "of every WAVE and POINT flight (12 flights). Arrived within 0.3 deg; "
        "no object moved (0.0 mm); worst MJCF distance 1.9 cm (blue_cylinder "
        "on r1c3). Only robot contact: the forearm coming to rest on "
        "table_top.",
        mount=MOUNT_OCTOBER,
    ),
    RouteValidation(
        "WAVE", "FWDCenterLabSivaPool",
        "2026-10-18 (#192, offline physics, fly_wave_point.py): REST -> "
        "LIFT_TO_PRESENT -> WAVE (3 cycles, 1.8 s a swing, rig_motion.wave "
        "unchanged) -> LOWER_TO_REST, in three layouts: soda_can / "
        "blue_cylinder / red_cube on r1c3 / r2c3 / r3c3; blue_cylinder / "
        "soda_can / foam_block on the same cells; soda_can on r2c2 with "
        "blue_cylinder on r1c1 and pool_cyl_1 on r3c1. All three cycles every "
        "time, back at PRESENT within 1.0 deg; no object moved (0.0 mm); no "
        "robot contact during the wave; worst MJCF distance during it 3.6 cm "
        "(soda_can on r1c3).",
        mount=MOUNT_OCTOBER,
    ),
    RouteValidation(
        "POINT", "FWDCenterLabSivaPool",
        "2026-10-18 (#192, offline physics, fly_wave_point.py): the panel's "
        "own `_point_at_target` (hover height, point_at's lift search and "
        "escort, the return to PRESENT), flown from REST via LIFT_TO_PRESENT "
        "and back by LOWER_TO_REST. AN EMPTY BOARD: all nine cells, miss 0.4 "
        "to 1.4 cm, no lift, nothing touched. ONE OBJECT AT A TIME on the "
        "centre cell (pool_box_1, foam_block, red_cube, pool_cyl_1, "
        "blue_cylinder, soda_can), every cell and then the object itself: "
        "57 of 60 cells pointed (miss 0.3 to 1.5 cm, lifts of 5-10 cm over "
        "r1c1, r1c2 and r2c2); cell_r1c2 refused three times by the guard "
        "(foam_block: 4.8 cm on leg 5 of 6 under the 5 cm margin; "
        "blue_cylinder and soda_can before any motion). The object itself: "
        "4 of 6 (the boxes and blue_cylinder, miss 0.6 cm); pool_cyl_1 and "
        "soda_can were stopped by the escort's realised check on leg 6, the "
        "tube hand reading 0.1 and 0.3 cm inside the object while the MJCF "
        "hand was 3.0 to 3.1 cm clear, then flown back to PRESENT (an earlier "
        "flight from the previous REST stopped blue_cylinder the same way, at "
        "0.6 cm). NOTHING MOVED in any of the 69 "
        "pointings (0.0 mm), and the arm touched nothing but table_top at "
        "REST. TWO OR MORE OBJECTS IS NOT COVERED, and `_ability_available` "
        "still enforces that. August, for comparison: 8 of 9 empty-board "
        "cells (r3c1 out of reach), miss 2.2 to 8.8 cm.",
        mount=MOUNT_OCTOBER,
    ),
    RouteValidation(
        "LIFT_TO_PRESENT", "FWDCenterLabMCC",
        "2026-10-18 (#192, offline physics): as the FWDCenterLabSivaPool row; "
        "MCC is the same board, grid and rig.",
        mount=MOUNT_OCTOBER,
    ),
    RouteValidation(
        "LOWER_TO_REST", "FWDCenterLabMCC",
        "2026-10-18 (#192, offline physics): as the FWDCenterLabSivaPool row; "
        "MCC is the same board, grid and rig.",
        mount=MOUNT_OCTOBER,
    ),
    RouteValidation(
        "WAVE", "FWDCenterLabMCC",
        "2026-10-18 (#192, offline physics): as the FWDCenterLabSivaPool row; "
        "MCC is the same board, grid and rig.",
        mount=MOUNT_OCTOBER,
    ),
    RouteValidation(
        "POINT", "FWDCenterLabMCC",
        "2026-10-18 (#192, offline physics): as the FWDCenterLabSivaPool row; "
        "MCC is the same board, grid and rig.",
        mount=MOUNT_OCTOBER,
    ),
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


#: Empty at the October mount: every rejection on record was an August flight
#: (#192).  Kept so the next rejection is written down with its numbers.
VALIDATION_ATTEMPTS: Tuple[ValidationAttempt, ...] = ()


def attempts_for(route: str, scene: str) -> Tuple[ValidationAttempt, ...]:
    return tuple(a for a in VALIDATION_ATTEMPTS
                 if a.route == route and a.scene == scene)


def validation_for(route: str, scene: str,
                   mount: Optional[str] = None) -> Optional[RouteValidation]:
    """The record of `route` flown in `scene` at `mount` (default: the current
    mount), or None."""
    mount = CURRENT_MOUNT if mount is None else mount
    for row in ROUTE_COMPATIBILITY:
        if row.route == route and row.scene == scene and row.mount == mount:
            return row
    return None


def check_route(route: str, scene: str) -> Tuple[bool, str]:
    """May this route be flown in this scene?  (ok, why_not)

    The refusal names the scene rather than the arm.  "I cannot do that in this
    scene" sends someone to the compatibility record; "the arm is unavailable"
    sends them to the robot, which is fine.

    The record must be at `CURRENT_MOUNT`: one flown with the robot mounted
    somewhere else is refused, and says so.
    """
    if not route:
        return False, "that action names no route"
    row = validation_for(route, scene)
    if row is None and any(r.route == route and r.scene == scene
                           for r in ROUTE_COMPATIBILITY):
        # Flown here, but with the robot mounted somewhere else (#192).  Said
        # in those words: "not validated" would send someone looking for a
        # missing run, and the run exists; it is the robot that moved.
        return False, (f"the {route} route was flown in {scene} with the "
                       f"robot mounted elsewhere, and it has not been re-flown "
                       f"since the robot moved ({CURRENT_MOUNT})")
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
    flown_in = sorted({r.scene for r in ROUTE_COMPATIBILITY
                       if r.route == route and r.mount == CURRENT_MOUNT})
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
                   "is clear.")


def route_named(name: str) -> Tuple[Waypoint, ...]:
    routes = {"LIFT_TO_PRESENT": LIFT_TO_PRESENT,
              "LOWER_TO_REST": LOWER_TO_REST}
    if name not in routes:
        raise KeyError(f"no route called {name!r}")
    return routes[name]


def nearest_waypoint(present: Dict[str, float],
                     route: Tuple[Waypoint, ...] = LIFT_TO_PRESENT + LOWER_TO_REST,
                     ) -> Tuple[str, float]:
    """The route waypoint the arm is closest to now, and how far off it is.

    For the recovery message, and only half an answer: moving onto the
    nearest waypoint is the ONE move on an unverified path.  So this reports;
    it does not authorise.
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

#: The named postures a route may start or end at.  No separate HOME: in the
#: October layout HOME IS REST (#192).
POSTURES: Dict[str, Dict[str, float]] = {
    POSTURE_REST: REST,
    POSTURE_PRESENT: PRESENT,
}

#: Which route takes the arm from one posture to another.
#:
#: A pair that is not here is not a transition.  There is no "just move there"
#: edge, and the absence is the point: two motions can both be reasonably
#: called going to rest; only one of them is measured for the trip you are
#: making.
POSTURE_TRANSITIONS: Dict[Tuple[str, str], str] = {
    (POSTURE_REST, POSTURE_PRESENT): "LIFT_TO_PRESENT",
    (POSTURE_PRESENT, POSTURE_REST): "LOWER_TO_REST",
    (POSTURE_PRESENT, POSTURE_PRESENT): "WAVE",
}


#: The legs of each route #82 checks as a swept capsule path against LIVE
#: object poses before flying, because nothing below `primitives` has a scene
#: and cannot know an object drifted into it.  In the October layout (#192)
#: that is the whole of LOWER_TO_REST and LIFT_TO_PRESENT, every waypoint,
#: because the whole of each is over or beside the board; both legs are
#: checked whichever way the arm is going, from wherever it starts.
#:
#: WAVE and POINT are absent on purpose: neither lands the arm on REST or
#: departs from it.  The wave happens at PRESENT, and pointing is guarded by
#: its own escort (`motion.escort`), against the live poses, leg by leg.
FOOTPRINT_LEGS: Dict[str, Tuple[Dict[str, float], ...]] = {
    "LOWER_TO_REST": (PRESENT, PRESENT_SHUT, REST_OUT, REST),
    "LIFT_TO_PRESENT": (REST, REST_OUT, PRESENT_SHUT, PRESENT),
}


def posture_of(present: Dict[str, float], tol: float = 8.0) -> Optional[str]:
    """Which named posture the arm is standing at, or None.

    Judged on the gross joints: the wrist angles and the gripper do not decide
    where the arm sits, and on a freshly reset simulator they read wherever
    gravity left them.
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

    Breadth-first over the measured edges, so "wave" asked of an arm at REST
    answers LIFT_TO_PRESENT rather than refusing.  The arm has to be raised
    before it can wave, and that is a fact about the rig rather than something
    the operator should have to know and type.

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
