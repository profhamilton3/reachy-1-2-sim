"""Goal assignment, C0 segmentation, and the affected-segment finder
(plan §7.1, §2.3).

Assigns each commanded right-arm setpoint to the goto/pass goal it belongs
to, using the route's own waypoint poses from ``rig_routes`` (never copied),
and locates the PLACE_ROUTE "affected segment": from the last setpoint of
the HOVER goto, through the HOVER hold, the HOVER -> REST_SHUT goto and any
re-stream passes, up to the REST_SHUT waypoint check.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from reachy_ai.motion.rig_routes import R_JOINTS, Waypoint  # noqa: E402

#: T1 (review §3.2/M1): every pose this module sees must already be radians
#: -- ``_units.route_rad``'s job, done once by the caller, never here. This
#: is the "assert this where practical" the assignment asks for: it catches
#: a raw ``rig_routes.Waypoint`` (degrees) reaching this module before it
#: can silently mis-segment every command.
_PLAUSIBLE_RAD_BOUND = 2.0 * 3.141592653589793


def _assert_rad8(pose: Dict[str, float], where: str) -> None:
    for j, v in pose.items():
        assert abs(v) < _PLAUSIBLE_RAD_BOUND, (
            f"{where}: {j}={v} is not a plausible radian value -- "
            "pass a route_rad()-converted pose, never a degree one")


#: Tight on purpose: a minimum-jerk clean flight is numerically exact, so
#: this only needs to absorb float roundoff. A looser bound risks two
#: closely-spaced waypoints' segments overlapping and mis-assigning real,
#: still-in-progress samples from one goto to its predecessor (or successor).
ON_SEGMENT_TOL_DEG = 1e-4
GOAL_LOOKAHEAD = 4

SKIPPED_WAYPOINT = "skipped_waypoint"
EXTRA_WAYPOINT = "extra_waypoint"


def _on_segment(start8: Dict[str, float], goal8: Dict[str, float],
                 value8: Dict[str, float], tol_deg: float) -> bool:
    """Right-arm joints AND the gripper (``R_JOINTS``): a waypoint transition
    that moves only the gripper (REST_SHUT -> REST) is still a real goal
    change and must be detectable, exactly like an arm-only move."""
    tol = np.radians(tol_deg)
    for j in R_JOINTS:
        a, b, v = start8.get(j, 0.0), goal8.get(j, 0.0), value8.get(j, 0.0)
        lo, hi = (a, b) if a <= b else (b, a)
        if not (lo - tol <= v <= hi + tol):
            return False
    return True


def _pose8(wp: Waypoint) -> Dict[str, float]:
    return wp.pose


def _exact_equal8(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return all(a.get(j, 0.0) == b.get(j, 0.0) for j in R_JOINTS)


def ulp32(x: float) -> float:
    """One float32 ULP of ``x`` (a non-negative magnitude, float64), via
    ``np.spacing`` at float32 precision -- the exact quantisation step of
    a ``float32(position_rad)`` value on the wire. Used by R-over (below)
    and by ``pathcheck``'s N2/N3/C1'/C4' (owner rulings, 2026-09-25).
    ``np.spacing`` is sign-aware (``np.spacing(-0.7) < 0``, the step
    towards ``-inf``) -- ``abs()`` here, always, so callers can add this
    to either side of a bound without re-deriving the sign themselves."""
    return abs(float(np.spacing(np.float32(x))))


def _on_segment_strict(
    start8: Dict[str, float], goal8: Dict[str, float], value8: Dict[str, float],
    prev8: Dict[str, float], tol_deg: float,
) -> bool:
    """R-carry + R-const + R-over (coordinator ruling, 2026-09-25 stage-A
    review, §3 R-carry; owner rulings R-over): ``prev8`` is the
    IMMEDIATELY PRECEDING command's own value (or ``start_pose8`` for the
    leg's first command) -- the same reference every call site at row
    ``i`` uses, regardless of which candidate goto is being tested.

    1. A joint whose value here is BIT-EXACT to ``prev8[j]`` is a carry
       (R-carry §1): accepted unconditionally, for both moving and
       constant joints -- a carry carries no positional information
       about which goto it belongs to, so it must never fail (or
       establish) a segment-membership test.
    2. A CONSTANT joint (nominal ``start8[j] == goal8[j]`` -- report §4
       C2's own definition) that is NOT a carry must equal the nominal
       value EXACTLY (R-const/N1) -- never a bootstrapped reference from
       whatever the goto's own first accepted sample happened to be
       (that bootstrap is exactly the G-a defect: a leading carry from
       the PREVIOUS goto poisoned the reference, rejecting the later,
       correct, exact samples).
    3. A MOVING joint that is not a carry uses the tol box on its START
       side (unchanged), but on its GOAL side the box is tightened to 1
       float32 ULP of the goal (R-over): a setpoint beyond the goal, in
       the direction of travel, by more than that never belongs to this
       goto -- it is goto k+1's own (or later's), never admitted here
       merely because it is inside the old, far more permissive
       ``ON_SEGMENT_TOL_DEG`` box."""
    tol = np.radians(tol_deg)
    for j in R_JOINTS:
        a, b, v = start8.get(j, 0.0), goal8.get(j, 0.0), value8.get(j, 0.0)
        if prev8.get(j, 0.0) == v:
            continue  # R-carry: a bit-exact carry, accepted regardless
        if a == b:
            if v != a:
                return False  # R-const/N1: a non-carry constant sample must equal S exactly
            continue
        if b >= a:
            lo_bound, hi_bound = a - tol, b + ulp32(b)
        else:
            lo_bound, hi_bound = b - ulp32(b), a + tol
        if not (lo_bound <= v <= hi_bound):
            return False
    return True


def carry_mask(
    targets8: Sequence[Dict[str, float]], start_pose8: Dict[str, float], *,
    withdraw: Optional[Sequence[Dict[str, bool]]] = None,
) -> List[Dict[str, bool]]:
    """Per command, per right-arm joint: ``True`` iff that joint's value
    here is bit-exact to the IMMEDIATELY PRECEDING command's value for
    that joint (``start_pose8`` for the leg's own first command) --
    R-carry ruling §1's definition. The single source of truth: every
    carry-aware caller (``assign_goals``'s own R-const/R-over admission,
    via ``_on_segment_strict``, and ``pathcheck``'s N1/N3/C1'/C2) uses
    this exact rule, so they can never disagree on what counts as a
    carry.

    ``withdraw`` (D-2, R-carry' provenance; additive, default ``None``
    leaves every caller's existing behaviour unchanged): per command, per
    joint, ``True`` forces that position's own carry exemption OFF
    regardless of bit-equality -- the provenance recheck's own way of
    saying "this repeat's chain origin was a genuine echo, so it no
    longer counts as a clean carry here." Never used by
    ``assign_goals`` itself (goal assignment is not changed by D-2)."""
    out: List[Dict[str, bool]] = []
    prev = start_pose8
    for i, tgt in enumerate(targets8):
        row = {j: (prev.get(j, 0.0) == tgt.get(j, 0.0)) for j in R_JOINTS}
        if withdraw is not None:
            for j in R_JOINTS:
                if row[j] and withdraw[i].get(j, False):
                    row[j] = False
        out.append(row)
        prev = tgt
    return out


@dataclass
class GoalAssignment:
    goal_index: List[Optional[int]]     # per command in `targets8`; None = indeterminate
    ok: bool
    violation_index: Optional[int] = None
    violation_kind: Optional[str] = None


def assign_goals(
    route: Sequence[Waypoint], start_pose8: Dict[str, float],
    targets8: Sequence[Dict[str, float]], *, tol_deg: float = ON_SEGMENT_TOL_DEG,
) -> GoalAssignment:
    """Assigns each of ``targets8`` (right-arm pose dicts, one per commanded
    setpoint) the index into ``route`` of the goto/pass it belongs to.
    Consecutive setpoints toward the same goal (including re-stream passes,
    which command ``wp.pose`` exactly) get the same index. A target found
    only on an EARLIER goal's segment (a regression) is flagged
    ``extra_waypoint``; jumping more than one goal ahead is flagged
    ``skipped_waypoint``. A target on no known segment is left
    indeterminate (``None``) rather than guessed at -- expected on an
    echo-corrupted (A-like) leg."""
    _assert_rad8(start_pose8, "assign_goals: start_pose8")
    for wp in route:
        _assert_rad8(_pose8(wp), f"assign_goals: route[{wp.name}]")

    n = len(targets8)
    result: List[Optional[int]] = [None] * n
    cur = 0
    seg_start = dict(start_pose8)
    violation_index: Optional[int] = None
    violation_kind: Optional[str] = None

    for i in range(n):
        tgt = targets8[i]
        #: R-carry: the reference for THIS row's own carry test is fixed
        #: by its position in the file, never by which candidate goto is
        #: being tried -- computed once per row, passed unchanged to
        #: every `_on_segment_strict` call below.
        prev8 = targets8[i - 1] if i > 0 else start_pose8

        if cur < len(route) and _on_segment_strict(seg_start, _pose8(route[cur]), tgt, prev8, tol_deg):
            # R-tie (coordinator ruling, 2026-09-25, §1): a setpoint that
            # bit-equals cur's own goal AND DIFFERS from the immediately
            # preceding command belongs to cur -- it continues cur's
            # approach. Under correct reporting, goto cur+1 starts from
            # the PRECEDING command's value (report §4), so this setpoint
            # cannot itself be cur+1's anchor. Only a setpoint that equals
            # BOTH cur's goal AND the preceding command (a CARRY of an
            # already-exact value) is ambiguous between "cur, still held"
            # and "cur+1's own anchor" -- the lookahead tie-break applies
            # to that carry case only.
            prev_tgt = targets8[i - 1] if i > 0 else None
            exact_to_goal = _exact_equal8(tgt, _pose8(route[cur]))
            carry_of_prev = prev_tgt is not None and _exact_equal8(tgt, prev_tgt)
            if (cur + 1 < len(route) and i + 1 < n
                    and exact_to_goal and carry_of_prev):
                nxt_start = _pose8(route[cur])
                nxt_goal = _pose8(route[cur + 1])
                nxt = targets8[i + 1]
                if (_on_segment(nxt_start, nxt_goal, nxt, tol_deg)
                        and not _exact_equal8(nxt, nxt_start)
                        and not _exact_equal8(nxt, nxt_goal)):
                    cur = cur + 1
                    seg_start = nxt_start
                    result[i] = cur
                    continue
            result[i] = cur
            continue

        advanced = False
        for look in range(1, min(GOAL_LOOKAHEAD, len(route) - cur) + 1):
            idx = cur + look
            if idx >= len(route):
                break
            candidate_start = _pose8(route[idx - 1])
            if _on_segment_strict(candidate_start, _pose8(route[idx]), tgt, prev8, tol_deg):
                if look > 1 and violation_index is None:
                    violation_index, violation_kind = i, SKIPPED_WAYPOINT
                cur = idx
                seg_start = candidate_start
                result[i] = cur
                advanced = True
                break
        if advanced:
            continue

        for idx in range(0, cur):
            prior_start = start_pose8 if idx == 0 else _pose8(route[idx - 1])
            if _on_segment_strict(prior_start, _pose8(route[idx]), tgt, prev8, tol_deg):
                if violation_index is None:
                    violation_index, violation_kind = i, EXTRA_WAYPOINT
                result[i] = idx
                advanced = True
                break
        if not advanced:
            result[i] = None  # indeterminate

    return GoalAssignment(result, violation_index is None, violation_index, violation_kind)


def c0_goal_sequence(assignment: GoalAssignment, route: Sequence[Waypoint]) -> List[str]:
    """The de-duplicated sequence of waypoint names actually visited
    (consecutive repeats collapsed -- re-stream passes are not separate
    visits), for comparison against ``[wp.name for wp in route]``."""
    seq: List[str] = []
    last: Optional[int] = None
    for idx in assignment.goal_index:
        if idx is None or idx == last:
            continue
        seq.append(route[idx].name)
        last = idx
    return seq


# ---------------------------------------------------------------------------
# Affected segment (plan §7.1): PLACE_ROUTE's HOVER -> REST_SHUT span
# ---------------------------------------------------------------------------

@dataclass
class AffectedSegment:
    start_command_index: int   # last setpoint of the HOVER goto
    end_command_index: int     # last setpoint checked against REST_SHUT
    hover_goal_index: int
    rest_shut_goal_index: int
    indeterminate: bool = False


def find_affected_segment(
    assignment: GoalAssignment, route: Sequence[Waypoint],
    hover_name: str = "HOVER", rest_shut_name: str = "REST_SHUT",
) -> Optional[AffectedSegment]:
    """The affected segment runs from the LAST setpoint assigned to
    ``hover_name`` (the end of the HOVER goto -- the hold that follows is
    part of the segment too, but it has no commands of its own to bound)
    through every setpoint assigned to ``rest_shut_name`` (its goto and any
    re-stream passes). Returns ``None`` if either waypoint's name is not in
    ``route``, or the segment cannot be bounded because the goal
    assignment never determinately reaches ``rest_shut_name``
    (``AffectedSegment.indeterminate=True`` in that case, still returned
    with whatever bound was found, so callers can report "inconclusive
    baseline" for an A cycle vs. STOP for a B cycle per plan §7.1)."""
    try:
        hover_idx = next(i for i, wp in enumerate(route) if wp.name == hover_name)
        rest_idx = next(i for i, wp in enumerate(route) if wp.name == rest_shut_name)
    except StopIteration:
        return None

    hover_positions = [i for i, g in enumerate(assignment.goal_index) if g == hover_idx]
    rest_positions = [i for i, g in enumerate(assignment.goal_index) if g == rest_idx]
    if not hover_positions:
        return None
    start = hover_positions[-1]
    if not rest_positions or rest_positions[-1] <= start:
        return AffectedSegment(start, start, hover_idx, rest_idx, indeterminate=True)
    end = rest_positions[-1]
    return AffectedSegment(start, end, hover_idx, rest_idx, indeterminate=False)
