"""Transit constants of the pick/place arc, moved -- NOT re-derived (#56).

They used to live in ``tasks/pick_place_live.py`` beside a comment saying the
pad-only collision model did not see the forearm.  Since #56 the arc is
preflighted against the WHOLE arm, so that rationale is gone; these values are
unchanged and have NOT been re-measured.  Moving them does not complete #56's
"move the compensations" item.

CLEAR_Z
    The height (m) the hand arcs through between pick and place: 0.26 m above
    the table surface (0.74).  It is the transit height every arc inherits.
    The preflight now sees the forearm at this height, but lowering it is a
    behaviour change that needs a simulator measurement nobody has authorised.

CARRY_HZ
    Streaming rate (Hz) of the over-table transit segments (swing-in, lift,
    carry, retract, return).  A TRACKING COMPENSATION: ~8 Hz gives the physics
    arm ~125 ms per setpoint, enough for the stiffened actuators to converge
    so the hand holds the high arc instead of lagging and sagging.  It is
    about physics tracking (#55 / E1 territory), not about planner blindness.

STEP_HZ
    Streaming rate (Hz) of the descents onto and off the object.
"""

CLEAR_Z = 1.00
CARRY_HZ = 8
STEP_HZ = 25
