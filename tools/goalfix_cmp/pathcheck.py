"""Commanded-path checks C0-C8 (report §4; plan §7.2), against the route's
own waypoint data from ``rig_routes`` -- tolerances and gripper limits are
imported, never copied.

Operates on one leg's worth of ``joint_command`` rows at a time (a full
21-vector target per row, in ``evidence.py``'s ``_JOINT_ORDER``), plus the
``segments.assign_goals`` result for the same rows. All timing (C2's skew,
C3, C4, C6) is simulation time, from each command's own bracket.
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

from reachy_ai.motion.rig_routes import R_JOINTS, ARM7, Waypoint  # noqa: E402
from reachy_ai.motion.kinematics import (  # noqa: E402
    _GRIPPER_OPEN_LIMIT_DEG, _GRIPPER_SHUT_LIMIT_DEG,
)

from tools.goalfix_cmp import segments as seg  # noqa: E402
from tools.goalfix_cmp._minjerk import implied_tau, pose_at  # noqa: E402

GRIPPER_LO_RAD, GRIPPER_HI_RAD = sorted(
    (np.radians(_GRIPPER_OPEN_LIMIT_DEG), np.radians(_GRIPPER_SHUT_LIMIT_DEG)))
C1_TOL_RAD = 1e-6
C2_SKEW_TOL_S = 0.030
C2_ILL_CONDITIONED_DEG = 0.05
C4_LEAD_S = 0.010  # "T - 10 ms"

ALL_CHECKS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")


@dataclass
class CheckResult:
    check_id: str
    passed: bool
    first_violation_index: Optional[int] = None
    detail: str = ""


def _goal8(wp: Waypoint) -> Dict[str, float]:
    return wp.pose


def _idx_for_goal(assignment: seg.GoalAssignment, k: int) -> List[int]:
    return [i for i, g in enumerate(assignment.goal_index) if g == k]


def _exact(a: float, b: float) -> bool:
    return a == b


def _near_end(value: float, a: float, b: float, tol_deg: float = C2_ILL_CONDITIONED_DEG) -> bool:
    return (abs(np.degrees(value - a)) <= tol_deg or abs(np.degrees(value - b)) <= tol_deg)


# ---------------------------------------------------------------------------
# C0
# ---------------------------------------------------------------------------

def check_c0(assignment: seg.GoalAssignment, route: Sequence[Waypoint]) -> CheckResult:
    if not assignment.ok:
        return CheckResult("C0", False, assignment.violation_index,
                            f"{assignment.violation_kind} at index {assignment.violation_index}")
    got = seg.c0_goal_sequence(assignment, route)
    want = [wp.name for wp in route]
    if got != want:
        return CheckResult("C0", False, None, f"goal sequence {got} != route {want}")
    return CheckResult("C0", True)


# ---------------------------------------------------------------------------
# C1: start continuity
# ---------------------------------------------------------------------------

_UNSET = object()


def check_c1(targets8: Sequence[Dict[str, float]], start_pose8: Dict[str, float],
             assignment: seg.GoalAssignment) -> CheckResult:
    prev: Dict[str, float] = dict(start_pose8)
    last_goal = _UNSET  # sentinel: the very first command is always a boundary,
    # even if goal assignment could not place it (still checked against
    # start_pose8, per C1's own "route start" case).
    for i, tgt in enumerate(targets8):
        g = assignment.goal_index[i]
        if g != last_goal:  # first setpoint of a new goto (or the very first command)
            for j in R_JOINTS:
                if abs(tgt.get(j, 0.0) - prev.get(j, 0.0)) > C1_TOL_RAD:
                    return CheckResult("C1", False, i,
                                        f"joint {j} jumps {prev.get(j,0.0)} -> {tgt.get(j,0.0)} "
                                        "at a goto's first setpoint")
            last_goal = g
        prev = tgt
    return CheckResult("C1", True)


# ---------------------------------------------------------------------------
# C2: on-segment, with skew; C3: monotone tau
# ---------------------------------------------------------------------------

def check_c2_c3(
    targets8: Sequence[Dict[str, float]], t_hi_s: Sequence[float],
    start_pose8: Dict[str, float], route: Sequence[Waypoint],
    assignment: seg.GoalAssignment,
) -> Tuple[CheckResult, CheckResult]:
    """C2 keeps the tau-agreement check, WITH the report's own 0.05deg
    near-end exemption (inverting m^-1 is ill-conditioned there). C3
    (coordinator ruling, 2026-09-25 stage-2a rulings addendum, A2) is
    computed independently, on RAW VALUES, with NO near-end exemption:
    report §4 C3 ("each joint's implied tau_j never decreases") states no
    exemption at all -- the 0.05deg exemption is stated only for C2. Since
    the minimum-jerk profile is monotone in tau, "tau never decreases" is
    exactly "the value never moves back away from the goal", which is
    checkable directly with no inversion, near the ends too, and with NO
    tolerance (float32 rounding of a monotone float64 sequence is itself
    non-decreasing)."""
    seg_start = dict(start_pose8)
    last_goal: Optional[int] = None
    last_tau: Dict[str, float] = {}
    last_val: Dict[str, float] = {}
    c2_fail: Optional[CheckResult] = None
    c3_fail: Optional[CheckResult] = None

    for i, tgt in enumerate(targets8):
        if c2_fail is not None and c3_fail is not None:
            break
        g = assignment.goal_index[i]
        if g is None:
            continue
        if g != last_goal:
            seg_start = start_pose8 if g == 0 else _goal8(route[g - 1])
            last_tau = {}
            last_val = {}
        goal8 = _goal8(route[g])

        taus = {}
        constant_violation = False
        for j in ARM7:
            a, b, v = seg_start.get(j, 0.0), goal8.get(j, 0.0), tgt.get(j, 0.0)
            if a == b:
                if v != a:
                    constant_violation = True
                continue
            # C3 (A2): raw-value monotonicity toward the goal, every
            # sample, no near-end exemption, no tolerance.
            if c3_fail is None and j in last_val:
                moved_away = (v < last_val[j]) if b > a else (v > last_val[j])
                if moved_away:
                    c3_fail = CheckResult(
                        "C3", False, i, f"joint {j} moved away from the goal (raw value)")
            last_val[j] = v
            if _near_end(v, a, b):
                continue
            taus[j] = implied_tau(a, b, v)

        if constant_violation and c2_fail is None:
            c2_fail = CheckResult("C2", False, i, "a constant joint moved")

        if taus:
            spread_s = (max(taus.values()) - min(taus.values())) * _leg_seconds(route, g)
            if spread_s > C2_SKEW_TOL_S + 1e-6 and c2_fail is None:
                c2_fail = CheckResult(
                    "C2", False, i, f"per-joint tau spread {spread_s*1000:.1f} ms > 30 ms")
            last_tau.update(taus)

        last_goal = g

    return (c2_fail or CheckResult("C2", True)), (c3_fail or CheckResult("C3", True))


def _leg_seconds(route: Sequence[Waypoint], goal_index: int) -> float:
    return route[goal_index].seconds


# ---------------------------------------------------------------------------
# C4: end on goal
# ---------------------------------------------------------------------------

def check_c4(targets8: Sequence[Dict[str, float]], start_pose8: Dict[str, float],
             route: Sequence[Waypoint], assignment: seg.GoalAssignment) -> CheckResult:
    for k, wp in enumerate(route):
        idxs = _idx_for_goal(assignment, k)
        if not idxs:
            continue
        last_i = idxs[-1]
        seg_start = start_pose8 if k == 0 else _goal8(route[k - 1])
        goal8 = _goal8(wp)
        T = wp.seconds
        tau_ref = max(0.0, (T - C4_LEAD_S) / T) if T > 0 else 1.0
        for j in ARM7:
            a, b = seg_start.get(j, 0.0), goal8.get(j, 0.0)
            if a == b:
                continue
            residual = abs(pose_at(a, b, tau_ref) - b)
            v = targets8[last_i].get(j, 0.0)
            if abs(v - b) > residual + 1e-9:
                return CheckResult("C4", False, last_i,
                                    f"{wp.name}: joint {j} ends {abs(v-b)} rad off goal "
                                    f"(residual budget {residual})")
    return CheckResult("C4", True)


# ---------------------------------------------------------------------------
# C5: re-stream passes hold exactly
# ---------------------------------------------------------------------------

def _is_exact_goal(target8: Dict[str, float], goal8: Dict[str, float]) -> bool:
    return all(target8.get(j, 0.0) == goal8.get(j, 0.0) for j in R_JOINTS)


def check_c5(targets8: Sequence[Dict[str, float]], route: Sequence[Waypoint],
             assignment: seg.GoalAssignment) -> CheckResult:
    for k, wp in enumerate(route):
        idxs = _idx_for_goal(assignment, k)
        if not idxs:
            continue
        goal8 = _goal8(wp)
        first_exact = next((i for i in idxs if _is_exact_goal(targets8[i], goal8)), None)
        if first_exact is None:
            continue
        for i in idxs:
            if i >= first_exact and not _is_exact_goal(targets8[i], goal8):
                return CheckResult("C5", False, i,
                                    f"{wp.name}: setpoint after arrival is not exactly wp.pose")
    return CheckResult("C5", True)


# ---------------------------------------------------------------------------
# C6: holds contain carries only
# ---------------------------------------------------------------------------

def check_c6(targets8: Sequence[Dict[str, float]], assignment: seg.GoalAssignment) -> CheckResult:
    """A hold has no commands of its own -- it is the GAP between the last
    command assigned to goal k and the first assigned to goal k+1. So a
    command that is indeterminate (``segments.py`` could not place it) AND
    sits exactly in that gap (between two CONSECUTIVE, different goals) is,
    by elimination, a real command issued during what should have been
    silence -- it must at least be a carry. An indeterminate command
    sandwiched between two commands of the SAME goal is instead a
    same-goto artefact (e.g. an echo-corrupted sample mid-trajectory) and
    is left to C0/C2 to judge, never double-counted here."""
    n = len(targets8)
    for i in range(1, n - 1):
        if assignment.goal_index[i] is not None:
            continue
        prev_g, next_g = assignment.goal_index[i - 1], assignment.goal_index[i + 1]
        if prev_g is None or next_g is None or next_g != prev_g + 1:
            continue
        if targets8[i] != targets8[i - 1]:
            return CheckResult("C6", False, i, "non-carry command found inside a hold")
    return CheckResult("C6", True)


# ---------------------------------------------------------------------------
# C7: gripper sequencing
# ---------------------------------------------------------------------------

def check_c7(targets8: Sequence[Dict[str, float]], start_pose8: Dict[str, float],
             route: Sequence[Waypoint], assignment: seg.GoalAssignment) -> CheckResult:
    for i, tgt in enumerate(targets8):
        g_val = tgt.get("r_gripper", 0.0)
        if not (GRIPPER_LO_RAD - 1e-6 <= g_val <= GRIPPER_HI_RAD + 1e-6):
            return CheckResult("C7", False, i, f"gripper {g_val} rad outside commanded range")

    for k, wp in enumerate(route):
        idxs = _idx_for_goal(assignment, k)
        if not idxs:
            continue
        seg_start = start_pose8 if k == 0 else _goal8(route[k - 1])
        goal8 = _goal8(wp)
        gripper_changes = seg_start.get("r_gripper", 0.0) != goal8.get("r_gripper", 0.0)
        arm_changes = any(seg_start.get(j, 0.0) != goal8.get(j, 0.0) for j in ARM7)

        if not gripper_changes:
            g0 = targets8[idxs[0]].get("r_gripper", 0.0)
            for i in idxs:
                if targets8[i].get("r_gripper", 0.0) != g0:
                    return CheckResult("C7", False, i,
                                        f"{wp.name}: gripper moved outside a gripper-changing waypoint")
        if gripper_changes and not arm_changes:
            a0 = {j: targets8[idxs[0]].get(j, 0.0) for j in ARM7}
            for i in idxs:
                for j in ARM7:
                    if targets8[i].get(j, 0.0) != a0[j]:
                        return CheckResult("C7", False, i,
                                            f"{wp.name}: arm joint {j} moved during a gripper-only waypoint")
    return CheckResult("C7", True)


# ---------------------------------------------------------------------------
# C8: other 13 joints carry only
# ---------------------------------------------------------------------------

_OTHER_JOINT_SLICE = slice(8, 21)


def check_c8(targets21: np.ndarray, seed_keyframe21: Optional[Sequence[float]] = None) -> Tuple[CheckResult, Optional[float]]:
    """Returns (result, first_command_seed_deviation). The first command's
    own non-right-arm seed is reported descriptively (``max|target -
    keyframe|``), never as a C8 failure -- C8 only applies from the second
    command onward."""
    seed_dev = None
    if len(targets21) and seed_keyframe21 is not None:
        seed_dev = float(np.max(np.abs(
            np.asarray(targets21[0][_OTHER_JOINT_SLICE]) - np.asarray(seed_keyframe21)[_OTHER_JOINT_SLICE])))
    for i in range(1, len(targets21)):
        prev, cur = targets21[i - 1][_OTHER_JOINT_SLICE], targets21[i][_OTHER_JOINT_SLICE]
        if not np.array_equal(prev, cur):
            return CheckResult("C8", False, i, "a non-right-arm joint changed"), seed_dev
    return CheckResult("C8", True), seed_dev


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all(
    targets21: np.ndarray, t_hi_s: Sequence[float], start_pose8: Dict[str, float],
    route: Sequence[Waypoint], guard: Sequence[str],
    seed_keyframe21: Optional[Sequence[float]] = None,
    assignment: Optional[seg.GoalAssignment] = None,
) -> Dict[str, CheckResult]:
    """``assignment`` is normally computed fresh (C0's own job), but a
    caller checking one specific downstream check in isolation (e.g. a
    single-violation fixture for C4/C5/C6/C7) may pass in the assignment
    computed from the CLEAN trajectory the fixture perturbed -- otherwise
    a large enough injected violation can itself push the sample out of
    ``assign_goals``'s tolerance and turn it indeterminate, silently
    hiding the very violation being tested rather than exercising the
    check's own logic."""
    # T1 (review §3.2/M1): ``route``/``start_pose8`` must already be radians
    # (``_units.route_rad``'s job) -- caught here, not silently mis-checked.
    seg._assert_rad8(start_pose8, "pathcheck.run_all: start_pose8")
    for wp in route:
        seg._assert_rad8(_goal8(wp), f"pathcheck.run_all: route[{wp.name}]")
    targets8 = [{name: float(row[i]) for i, name in enumerate(R_JOINTS)} for row in targets21]
    if assignment is None:
        assignment = seg.assign_goals(route, start_pose8, targets8)

    c0 = check_c0(assignment, route)
    c1 = check_c1(targets8, start_pose8, assignment)
    c2, c3 = check_c2_c3(targets8, t_hi_s, start_pose8, route, assignment)
    c4 = check_c4(targets8, start_pose8, route, assignment)
    c5 = check_c5(targets8, route, assignment)
    c6 = check_c6(targets8, assignment)
    c7 = check_c7(targets8, start_pose8, route, assignment)
    c8, seed_dev = check_c8(targets21, seed_keyframe21)

    return {
        "C0": c0, "C1": c1, "C2": c2, "C3": c3, "C4": c4,
        "C5": c5, "C6": c6, "C7": c7, "C8": c8,
        "_seed_deviation_rad": seed_dev,
        "_assignment": assignment,
    }


def results_to_json(results: Dict[str, "CheckResult"]) -> Dict[str, object]:
    out = {}
    for cid in ALL_CHECKS:
        r = results[cid]
        out[cid] = {"passed": r.passed, "first_violation_index": r.first_violation_index,
                    "detail": r.detail}
    out["seed_deviation_rad"] = results.get("_seed_deviation_rad")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json as _json

    from tools.goalfix_cmp import evidence as ev
    from tools.goalfix_cmp._io import IntegrityError, RC_INCONCLUSIVE, RC_OK, RC_STOP, write_result

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-dir", required=True)
    p.add_argument("--states", default="states.jsonl")
    p.add_argument("--commands", default="commands.jsonl")
    p.add_argument("--sha256sums", default="SHA256SUMS")
    p.add_argument("--route", required=True, choices=["PLACE_ROUTE", "LIFT_TO_PRESENT"])
    p.add_argument("--arm", choices=["A", "B"], required=True)
    p.add_argument("--command-start-index", type=int, required=True)
    p.add_argument("--command-end-index", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    try:
        evd = ev.verify_and_load(args.evidence_dir, args.states, args.commands, args.sha256sums)
    except (ev.EvidenceError, IntegrityError) as exc:
        return write_result(args.out, RC_INCONCLUSIVE, {"ok": False, "reason": str(exc)})

    from reachy_ai.motion import rig_routes as R
    from tools.goalfix_cmp._units import pose8_rad, route_rad

    # T1 (review §3.2/M1): convert once, at this one boundary -- route_rad's
    # whole point. `start_pose8` is still the route's nominal start (HOME/
    # REST), not the leg's actual `turn_on` goal (M5); that wiring is T4's
    # end-to-end `cycle` CLI, which this single-leg CLI is not.
    route = route_rad(getattr(R, args.route))
    guard = R.CRITICAL_JOINTS if args.route == "PLACE_ROUTE" else R._PRESENT_GUARD
    start_pose8 = pose8_rad(R.HOME if args.route == "PLACE_ROUTE" else R.REST)

    jc_idx = np.nonzero(evd.commands.joint_command_mask())[0]
    jc_idx = jc_idx[(jc_idx >= args.command_start_index) & (jc_idx <= args.command_end_index)]

    # T2: an unplaceable command inside this leg's range is evidence
    # incomplete, never silently skipped or gated as if it were a carry.
    try:
        ev.check_no_unplaceable_in_range(evd, jc_idx.tolist())
    except ev.EvidenceError as exc:
        return write_result(args.out, RC_INCONCLUSIVE, {"ok": False, "reason": str(exc)})

    targets21 = evd.commands.target_rad[jc_idx]
    t_hi_s = [evd.brackets[i].t_hi for i in jc_idx]

    results = run_all(targets21, t_hi_s, start_pose8, route, guard)
    payload = results_to_json(results)

    if args.arm == "B":
        rc = RC_STOP if not all(results[c].passed for c in ALL_CHECKS) else RC_OK
    else:
        rc = RC_OK  # A is reported only; C1/C2/C5 are expected to fail
    payload["ok"] = rc == RC_OK
    payload["arm"] = args.arm
    return write_result(args.out, rc, payload)


def main() -> int:
    return _cli()


if __name__ == "__main__":
    raise SystemExit(main())
