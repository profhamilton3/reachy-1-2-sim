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
    n = len(targets8)
    result: List[Optional[int]] = [None] * n
    cur = 0
    seg_start = dict(start_pose8)
    violation_index: Optional[int] = None
    violation_kind: Optional[str] = None

    for i in range(n):
        tgt = targets8[i]
        if cur < len(route) and _on_segment(seg_start, _pose8(route[cur]), tgt, tol_deg):
            # Tie-break the goto boundary: a value that bit-exactly equals
            # cur's own goal is ambiguous between "cur, still held" and
            # "cur+1's own tau=0 anchor" (a real goto starts from the
            # cached goal -- see pathcheck.py's C1). If the NEXT command
            # has already moved on into cur+1's segment, THIS one was
            # cur+1's anchor, not one more cur sample.
            if (cur + 1 < len(route) and i + 1 < n
                    and _exact_equal8(tgt, _pose8(route[cur]))):
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
            if _on_segment(candidate_start, _pose8(route[idx]), tgt, tol_deg):
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
            if _on_segment(prior_start, _pose8(route[idx]), tgt, tol_deg):
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
