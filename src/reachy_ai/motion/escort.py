"""Fly a move in short legs, re-measuring clearance from the pose actually reached.

WHY THIS EXISTS.

The clearance guard in ``kinematics.CartesianPlanner`` is a *plan-time* check.
It takes the pose the arm is about to be commanded to, builds the three link
capsules, and measures them against the objects on the board.  Everything about
that is right except its subject: it describes an arm that tracks perfectly, and
this one does not.

The gap was measured, not assumed.  Planning each grid cell, flying it, then
recomputing clearance from the joint angles actually reached gave a worst
degradation of 0.9 cm — small enough that a 5 cm margin looked ample.  Then
``cell_r2c1`` reported +5.5 cm of air to the soda can and moved it 0.189 m.

Both numbers are true, and the contradiction is the whole point:

  * the 0.9 cm was measured at the ENDPOINTS of each move;
  * the can was hit somewhere in the MIDDLE of one.

``joint_path`` models a ``goto`` as a straight line in joint space, and each
joint does run monotonically from its start to its goal — but not in step.  The
shoulder (kp 300) arrives well before the wrists (kp 60), so the pose at t=0.5
is nowhere near the midpoint of that line, and the arm bows out of the corridor
the guard cleared.  Both endpoints check out; the belly of the move does not.
No amount of margin fixes this, because the deviation is not bounded by anything
the planner can see before the move.

WHAT CLOSING THE LOOP MEANS HERE.

Cut the move into short legs.  Before each leg, check it the old way — that part
was never wrong, only incomplete.  After each leg, READ THE ARM, rebuild the
capsules from the angles it actually reached, and measure the clearance that
really happened.  If that has eaten into the margin, stop, and retreat to the
last pose whose realised clearance was good.

The legs are what make the after-the-fact measurement worth anything.  Between
two realised poses this still assumes a joint-space straight line, exactly the
assumption that failed above — but over a sixth of a move the joints have had
much less room to get out of step, so the assumption is much closer to true.
Shortening the legs shrinks the residual; it never reaches zero.

WHAT IT STILL CANNOT DO.  It detects a bad leg AFTER flying it.  A leg that goes
from ample clearance to contact in one hop is caught too late to prevent the
contact — the guarantee is that the *rest* of the move does not happen, not that
nothing is ever touched.  ``scene_drift`` remains the evidence; this is the
mechanism that keeps one disturbance from becoming five.

Pure logic plus two injected callables, so it is unit-testable on the host with
no simulator and no SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

__all__ = ["Leg", "EscortResult", "escort"]

Joints = Sequence[float]


def _lerp(a: Joints, b: Joints, t: float) -> List[float]:
    return [x + t * (y - x) for x, y in zip(list(a)[:7], list(b)[:7])]


def _drift(a: Joints, b: Joints) -> float:
    return max((abs(x - y) for x, y in zip(list(a)[:7], list(b)[:7])), default=0.0)


def _dist(c: Any) -> float:
    """Clearance distance, with ``None`` (nothing to check) reading as clear."""
    return float("inf") if c is None else c.distance


def required_of(object_id: str, margin: float,
                margins: Optional[Mapping[str, float]]) -> float:
    """How much air ``object_id`` in particular has to be given."""
    if margins is None:
        return margin
    return margins.get(object_id, margin)


def binding(clearances: Mapping[str, Any], margin: float,
            margins: Optional[Mapping[str, float]] = None):
    """The object with the least slack against ITS OWN margin.

    Not the nearest object — the nearest one may be the one being deliberately
    approached, which is allowed to be near.  Slack is what decides, so an
    object 1 cm away with a 0.5 cm margin is fine while one 4 cm away with a
    5 cm margin is not.  Returns (Clearance, required) or (None, None).
    """
    if not clearances:
        return None, None
    oid = min(clearances,
              key=lambda o: clearances[o].distance - required_of(o, margin, margins))
    return clearances[oid], required_of(oid, margin, margins)


@dataclass(frozen=True)
class Leg:
    """One sub-move, as planned and as flown."""

    index: int              # 1-based
    fraction: float         # fraction of the whole commanded move this leg ends at
    commanded: List[float]  # the seven angles this leg was told to reach
    reached: List[float]    # the seven angles it actually reached
    planned: Any            # binding Clearance predicted over the leg, or None
    realised: Any           # binding Clearance measured over the leg, or None
    flown: bool             # False when the plan check refused it
    planned_all: Dict[str, Any] = field(default_factory=dict)
    realised_all: Dict[str, Any] = field(default_factory=dict)

    @property
    def drift(self) -> float:
        """Worst per-joint gap between the commanded pose and the reached one."""
        return _drift(self.commanded, self.reached)

    @property
    def degraded(self) -> float:
        """How much worse the realised clearance was than the planned one (m).

        Positive means the world was tighter than the model promised — the
        quantity the margin is supposed to cover.  Zero when there was nothing
        to measure against, which is not the same as "nothing went wrong" and is
        why a planner with no scene reports no degradation rather than a number.

        Compared PER OBJECT, not between the two binding Clearances: which
        object binds can change between the plan and the flight — that is the
        cell_r2c1 failure exactly, planned against red_cube and flown into the
        soda can — and subtracting one object's distance from another's would
        turn the most interesting case into a meaningless number.
        """
        if not self.planned_all or not self.realised_all:
            if self.planned is None or self.realised is None:
                return 0.0
            return self.planned.distance - self.realised.distance
        shared = [self.planned_all[o].distance - self.realised_all[o].distance
                  for o in self.planned_all if o in self.realised_all]
        return max(shared) if shared else 0.0


@dataclass(frozen=True)
class EscortResult:
    """Outcome of an escorted move."""

    completed: bool
    fraction: float          # how far along the commanded move the arm was taken
    legs: List[Leg] = field(default_factory=list)
    stopped_by: str = ""     # "" | "plan" | "realised" | "unreachable"
    reason: str = ""
    backed_off: bool = False
    start: List[float] = field(default_factory=list)
    reached: List[float] = field(default_factory=list)

    @property
    def worst_planned(self) -> Any:
        cs = [l.planned for l in self.legs if l.planned is not None]
        return min(cs, key=lambda c: c.distance) if cs else None

    @property
    def worst_realised(self) -> Any:
        cs = [l.realised for l in self.legs if l.realised is not None]
        return min(cs, key=lambda c: c.distance) if cs else None

    @property
    def worst_degraded(self) -> float:
        """Largest planned-minus-realised gap over the legs actually flown.

        This is the number the whole module exists to expose.  Watch it across
        runs: if it stays near zero the plan-time guard was telling the truth
        and the legs are only costing time; when it jumps, the guard was about
        to be wrong and the loop is what caught it.
        """
        flown = [l.degraded for l in self.legs if l.flown]
        return max(flown) if flown else 0.0

    def __str__(self) -> str:
        if self.completed:
            return (f"arrived in {len(self.legs)} legs; worst realised "
                    f"{self.worst_realised}, degraded "
                    f"{self.worst_degraded * 100:+.1f} cm")
        return (f"STOPPED at {self.fraction * 100:.0f}% ({self.stopped_by}) — "
                f"{self.reason}" + ("; backed off" if self.backed_off else ""))


def escort(
    planner,
    q_to: Joints,
    send: Callable[[List[float], float], None],
    read: Callable[[], List[float]],
    *,
    margin: float,
    abort_margin: Optional[float] = None,
    margins: Optional[Mapping[str, float]] = None,
    abort_margins: Optional[Mapping[str, float]] = None,
    legs: int = 6,
    duration: float = 2.0,
    min_leg_secs: float = 0.4,
    gripper_deg: Optional[float] = None,
    refresh: Optional[Callable[[], Any]] = None,
    back_off: bool = True,
    steps: int = 7,
    on_leg: Optional[Callable[[Leg], None]] = None,
    **clearance_kw,
) -> EscortResult:
    """Fly ``q_to`` in ``legs`` guarded steps, checking the arm after each one.

    ``send(joints, secs)`` commands the seven arm joints and blocks until the
    move is done; ``read()`` returns the seven angles the arm is at now.  Both
    are injected so this is testable without a robot, and so the caller keeps
    control of gripper commands, settle passes and tracking tolerances.

    ``margin`` authorises the next leg (the plan-time check, unchanged).
    ``abort_margin`` is the floor the *realised* clearance must stay above, and
    defaults to half of ``margin``: the margin is a budget, and spending half of
    it on error the model did not predict is the signal to stop.  With the
    measured endpoint degradation at 0.9 cm and a 5 cm margin, a realised 2.5 cm
    means something is happening that is roughly three times worse than anything
    seen in calibration.  That is a stop, not a tighter margin.

    ``margins`` overrides ``margin`` for named objects, and it is what makes
    approaching one of them possible at all.  A single margin forces a choice
    between two wrong answers: hold the target to 5 cm and no approach is ever
    allowed, since hovering 6 cm over a can puts the hand about 1 cm from it; or
    drop the target from the obstacle set, which is what this module used to do
    and means NOTHING guards the object you are reaching for.  Measured cost of
    the second: blue_cylinder hovered to 1.6 cm with 6.1 cm reported, moved
    0.123 m, and the loop saw nothing wrong because it was not looking.
    ``margins={"soda_can": 0.005}`` keeps the can in the set and asks only that
    the arm not pass through it, while the rest of the board keeps the full
    margin.  ``abort_margins`` does the same for the realised floor; an object
    with a margin but no abort margin gets half of its own, not half of the
    global one.

    ``refresh`` is called before each measurement.  Pass it: the objects move,
    and a guard reading their loaded positions is checking a board that no
    longer exists.  Not passing it is how you get a clean guard over wreckage.

    ``gripper_deg`` sizes the hand capsule to its aperture — travel shut.  An
    open hand is a 7.5 cm tube about the wrist axis against a shut one's 5.2 cm,
    and omitting this makes the guard assume the wider one.
    """
    if legs < 1:
        raise ValueError("legs must be at least 1")
    if abort_margin is None:
        abort_margin = margin / 2.0
    # An object given its own plan margin gets half of THAT as its realised
    # floor, not half of the global one — otherwise naming a can at 0.5 cm would
    # silently hold it to a 2.5 cm floor and refuse every approach anyway.
    floors: Dict[str, float] = {o: m / 2.0 for o, m in (margins or {}).items()}
    floors.update(abort_margins or {})

    def measure_path(a: Joints, b: Joints):
        return planner.path_clearances(a, b, steps=steps,
                                       gripper_deg=gripper_deg, **clearance_kw)

    q_start = list(read())[:7]
    flown: List[Leg] = []
    here = q_start
    safe = q_start          # last pose whose realised clearance was acceptable
    safe_fraction = 0.0

    for i in range(1, legs + 1):
        fraction = i / legs
        q_cmd = _lerp(q_start, q_to, fraction)

        if refresh is not None:
            refresh()
        planned_all = measure_path(here, q_cmd)
        planned, planned_req = binding(planned_all, margin, margins)

        if planned is not None and planned.distance < planned_req:
            leg = Leg(i, fraction, q_cmd, list(here), planned, None, False,
                      planned_all, {})
            flown.append(leg)
            if on_leg is not None:
                on_leg(leg)
            # Usually the arm is fine where it stands and only the road ahead is
            # blocked — every leg so far ended above the floor, or the loop
            # would already have stopped.  Usually, not always: refresh() may
            # have just told us an object moved, and the pose that was safe when
            # the arm arrived is not safe any more.  Check before leaving it.
            backed = False
            standing, standing_req = binding(
                planner.clearances(here, gripper_deg, **clearance_kw)
                if planner.scene is not None else {}, margin, floors)
            if (back_off and standing is not None
                    and standing.distance < standing_req and safe != here):
                send(safe, max(duration / legs, min_leg_secs))
                here = list(read())[:7]
                backed = True
            return EscortResult(
                completed=False, fraction=safe_fraction, legs=flown,
                stopped_by="plan",
                reason=f"{planned} on leg {i} of {legs}, under the "
                       f"{planned_req * 100:.1f} cm margin it is held to",
                backed_off=backed, start=q_start, reached=list(here),
            )

        send(q_cmd, max(duration / legs, min_leg_secs))
        previous, here = here, list(read())[:7]

        if refresh is not None:
            refresh()
        realised_all = measure_path(previous, here)
        realised, realised_req = binding(realised_all, abort_margin, floors)

        leg = Leg(i, fraction, q_cmd, list(here), planned, realised, True,
                  planned_all, realised_all)
        flown.append(leg)
        if on_leg is not None:
            on_leg(leg)

        if realised is not None and realised.distance < realised_req:
            backed = False
            if back_off and safe != here:
                # Deliberately unguarded, and deliberately not re-planned: this
                # retraces a leg the arm has just flown, from the far end back
                # to a pose whose realised clearance was measured and good.  A
                # guarded retreat can refuse, and a refused retreat leaves the
                # arm parked exactly where it should not be.
                send(safe, max(duration / legs, min_leg_secs))
                here = list(read())[:7]
                backed = True
            return EscortResult(
                completed=False, fraction=safe_fraction, legs=flown,
                stopped_by="realised",
                reason=f"{realised} once leg {i} of {legs} had been flown "
                       f"(planned {planned}), under the "
                       f"{realised_req * 100:.1f} cm floor",
                backed_off=backed, start=q_start, reached=list(here),
            )

        safe, safe_fraction = here, fraction

    return EscortResult(
        completed=True, fraction=1.0, legs=flown, stopped_by="", reason="",
        backed_off=False, start=q_start, reached=list(here),
    )
