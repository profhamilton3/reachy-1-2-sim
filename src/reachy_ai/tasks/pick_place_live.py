"""Scene-aware pick-and-place arc against a LIVE simulator (issue #51).

Migration note
--------------
`_run_segment` and `pick_and_place` were moved here verbatim from
`scripts/demo_pick_place.py`, with one change: the place site is now a
parameter instead of a lookup in that script's `_PLACE_XY` table.  The demo
imports them from here and passes its own table, so its behaviour is
unchanged; the command panel's executor passes a grid cell's world xy.

The extraction is the point.  This arc is the only pick-and-place motion in the
repository that has been driven against physics and tuned for it — the slow
over-table transit rate, the descend-straight-down approach, the re-abduction
on the way out.  A second copy inside the panel would drift from it, and the
one that drifted would be the one moving the arm.

NOT the offline episode runner
------------------------------
`native_mujoco/episode_runner.py` owns its own `SimulationCore` and resets it
at the start of every run.  Driving it from the panel would advance a separate
world and report that world's result as the live one, which is precisely the
false claim the panel must never make.  This module instead commands the same
world the page displays, through the v1 SDK the compatibility core already
bridges to the native server.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence, Tuple

from reachy_ai.motion import primitives as P
from reachy_ai.motion.kinematics import R_ARM_JOINTS

log = logging.getLogger("reachy12.tasks.pick_place_live")

XY = Tuple[float, float]

_STEP_HZ = 25
# Over-table transit segments (swing-in, lift, carry, swing-out) stream at a
# SLOWER rate so the physics arm has time to actually track each setpoint at the
# commanded clear height instead of lagging and sagging below it.  ~8 Hz gives
# the arm ~125 ms per setpoint — plenty for the stiffened actuators to converge,
# so the hand holds the high arc tightly over the table.
_CARRY_HZ = 8

# The arm arcs through this height (m) when travelling between pick and place.
# Chosen 0.26 m above the table surface (0.74) so the whole hand AND forearm are
# visibly clear of the tabletop and any objects sitting on it (the pad-only
# collision model doesn't see the forearm, so we keep the transit high).
CLEAR_Z = 1.00


def run_segment(arm, planner, start, end, seed, steps, attacher=None,
                object_id=None, rate_hz=_STEP_HZ):
    """Plan a collision-checked Cartesian segment and execute it.  Returns the
    final joint solution (seed for the next segment).  ``rate_hz`` sets the
    streaming rate — pass a slower rate for over-table transits so the physics
    arm tracks the commanded height instead of sagging."""
    traj, cart = planner.plan_segment(start, end, steps, seed)

    on_step = None
    if attacher is not None and object_id is not None:
        def on_step(i, q, _c=cart):
            attacher.follow(object_id, _c[i])

    P.execute_trajectory(arm, traj, R_ARM_JOINTS, rate_hz=rate_hz, on_step=on_step)
    return traj[-1]


def pick_and_place(robot, planner, scene, attacher, object_id, seed, side_pad,
                   place_xy: XY,
                   should_abort: Optional[Callable[[], bool]] = None,
                   on_phase: Optional[Callable[[str], None]] = None):
    """Full pick-and-place cycle for one object, approaching over the table's
    right side.  Starts and ends at the SIDE_HIGH hub (``side_pad``).  Returns
    the ending seed.

    ``place_xy`` is the world (x, y) to place at.  ``should_abort``, when given,
    is checked between segments: a cancel is honoured at a phase boundary rather
    than mid-trajectory, so the arm is never left part-way through a planned
    Cartesian move with the gripper in an undefined state.  ``on_phase`` reports
    the phase about to run.

    Motion arc — the arm is at the SIDE_HIGH hub (up and out to the robot's
    right, clear of the table) at entry and exit.  Every segment is a
    collision-checked Cartesian move; the hand only ever enters the table
    footprint at or above CLEAR_Z, then descends straight down onto the object:

      SIDE_HIGH → [over table @ CLEAR_Z] → above pick
                → [down] → hover → grasp → close gripper
                → [up]   → above pick
                → [over table @ CLEAR_Z] → above place
                → [down] → place → open gripper
                → [up]   → above place
                → [over table @ CLEAR_Z] → SIDE_HIGH
    """
    arm = robot.r_arm

    hover = scene.hover_point(object_id)   # object top + 0.05 m
    grasp = scene.grasp_point(object_id)   # object centre (gripper pads land here)
    place = scene.rest_point(place_xy, object_id)

    # High-clearance waypoints directly above the pick and place sites.  The arm
    # crosses the table only at CLEAR_Z.
    above_pick = (hover[0], hover[1], CLEAR_Z)
    above_place = (place_xy[0], place_xy[1], CLEAR_Z)

    def phase(name: str) -> bool:
        """Announce a phase, and report whether the caller wants to stop.

        Checked only here, between segments.  Aborting inside a streamed
        trajectory would stop the arm at an arbitrary point of a planned move,
        and with the gripper possibly half closed on an object.
        """
        if on_phase is not None:
            on_phase(name)
        return bool(should_abort and should_abort())

    # 1. Head looks toward the object.
    P.look_at(robot, grasp, duration=1.0)

    # 2. Swing IN from the side, over the table, to directly above the object.
    #    Slow rate so the arm holds the high arc (no sag) while crossing over.
    if phase("swing in over table"):
        return seed
    log.info("── %s: swing in over table to above pick", object_id)
    seed = run_segment(arm, planner, side_pad, above_pick, seed, 40,
                       rate_hz=_CARRY_HZ)

    # 3. Descend straight down: above → hover → grasp.
    if phase("descend to hover"):
        return seed
    log.info("── %s: descend to hover", object_id)
    seed = run_segment(arm, planner, above_pick, hover, seed, 25)
    if phase("descend to grasp"):
        return seed
    log.info("── %s: descend to grasp", object_id)
    seed = run_segment(arm, planner, hover, grasp, seed, 15)

    # 4. Close gripper.  Past this point an abort still finishes the arc: an
    #    object released mid-carry drops onto whatever is below it.
    phase("close gripper")
    log.info("── %s: close gripper", object_id)
    P.close_gripper(arm)
    if attacher is not None:
        attacher.follow(object_id, grasp)

    # 5. Lift straight up to clear height above the pick site (slow, so the arm
    #    actually reaches the clear height before carrying).
    phase("lift to clear height")
    log.info("── %s: lift to clear height", object_id)
    seed = run_segment(arm, planner, grasp, above_pick, seed, 30, attacher,
                       object_id, rate_hz=_CARRY_HZ)

    # 6. Head tracks toward the place site while the arm swings across.
    P.look_at(robot, (place_xy[0], place_xy[1], scene.table_surface_z),
              duration=0.8)

    # 7. Carry over the table at clear height to above the place site — slow rate
    #    keeps the hand pinned to the high arc the whole way across.
    phase("carry over table")
    log.info("── %s: carry over table at clear height", object_id)
    seed = run_segment(arm, planner, above_pick, above_place, seed, 40, attacher,
                       object_id, rate_hz=_CARRY_HZ)

    # 8. Descend straight down onto the place position.
    phase("descend to place")
    log.info("── %s: descend to place", object_id)
    seed = run_segment(arm, planner, above_place, place, seed, 30, attacher,
                       object_id)

    # 9. Open gripper; release marker.
    phase("release")
    log.info("── %s: open gripper (release)", object_id)
    P.open_gripper(arm)
    if attacher is not None:
        attacher.release(object_id, place)

    # 10. Lift straight up to clear height, then return to the side hub by
    #     RE-ABDUCTING the shoulder (roll-first, closed-loop).  A Cartesian
    #     swing-out streams all joints together, which stalls the shoulder-roll
    #     under physics, so use the sequenced raise instead.
    phase("retract")
    log.info("── %s: retract to clear height", object_id)
    seed = run_segment(arm, planner, place, above_place, seed, 25,
                       rate_hz=_CARRY_HZ)
    log.info("── %s: re-abduct back out to the side hub", object_id)
    P.raise_to_side(arm)
    return [P.SIDE_HIGH[n] for n in R_ARM_JOINTS]


def side_hub(planner) -> Tuple[Sequence[float], XY]:
    """The SIDE_HIGH transit hub as (joint seed, pad world point)."""
    seed = [P.SIDE_HIGH[n] for n in R_ARM_JOINTS]
    return seed, planner.fk_world(seed)
