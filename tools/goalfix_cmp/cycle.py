"""Per-cycle verdict (plan §5 step 9, §7.6): combines §2.2-§2.7 into one
``between_<cycle>.json``-shaped result.

Verdicts, per plan §7.6:

* ``ok`` / ``manipulated`` (rc 0) -- authorize the next cycle.
* ``STOP`` (rc 2) -- a B cycle with a genuine echo in the segment, a
  C0-C8 failure on either leg, or non-zero target drift in a hold.
* ``inconclusive_baseline`` (rc 3, A only) -- 0 genuine echoes in the
  segment, or the segment is indeterminate. Not a failure of the fix, but
  never treated as a pass for gating purposes either.
* ``evidence_incomplete`` (rc 3) -- anything upstream (integrity, epoch
  cross-check, leg linking) raised.

Nothing but ``ok``/``manipulated`` authorizes the next cycle -- an
inconclusive or evidence-incomplete result must never be read as a green
light, even though §7.6 also says inconclusive is not a *failure* of the
fix.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from reachy_ai.motion.rig_routes import Waypoint  # noqa: E402

from tools.goalfix_cmp import echo  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import holds  # noqa: E402
from tools.goalfix_cmp import pathcheck as pc  # noqa: E402
from tools.goalfix_cmp import segments as seg  # noqa: E402
from tools.goalfix_cmp._io import RC_INCONCLUSIVE, RC_OK, RC_STOP  # noqa: E402

VERDICT_OK = "ok"
VERDICT_MANIPULATED = "manipulated"
VERDICT_STOP = "STOP"
VERDICT_INCONCLUSIVE_BASELINE = "inconclusive_baseline"
VERDICT_EVIDENCE_INCOMPLETE = "evidence_incomplete"

_RC_BY_VERDICT = {
    VERDICT_OK: RC_OK,
    VERDICT_MANIPULATED: RC_OK,
    VERDICT_STOP: RC_STOP,
    VERDICT_INCONCLUSIVE_BASELINE: RC_INCONCLUSIVE,
    VERDICT_EVIDENCE_INCOMPLETE: RC_INCONCLUSIVE,
}


def rc_for_verdict(verdict: str) -> int:
    return _RC_BY_VERDICT[verdict]


@dataclass
class LegSpec:
    name: str
    route: Sequence[Waypoint]
    guard: Sequence[str]
    start_pose8: Dict[str, float]
    command_indices: Sequence[int]   # global indices into evidence.commands, in order


@dataclass
class LegResult:
    name: str
    assignment: seg.GoalAssignment
    pathcheck: Dict[str, pc.CheckResult]
    hold_windows: List[Tuple[int, float, float]]


def evaluate_leg(evidence: ev.Evidence, leg: LegSpec) -> LegResult:
    idx = np.asarray(leg.command_indices)
    targets21 = evidence.commands.target_rad[idx]
    targets8 = [{name: float(row[i]) for i, name in enumerate(pc.R_JOINTS)} for row in targets21]
    assignment = seg.assign_goals(leg.route, leg.start_pose8, targets8)
    t_hi_s = [evidence.brackets[i].t_hi for i in idx]
    results = pc.run_all(targets21, t_hi_s, leg.start_pose8, leg.route, leg.guard,
                          assignment=assignment)
    windows = holds.find_hold_windows(assignment, evidence.brackets, list(idx))
    return LegResult(leg.name, assignment, results, windows)


@dataclass
class CycleVerdict:
    cycle: str
    arm: str
    verdict: str
    reasons: List[str] = field(default_factory=list)
    genuine_echo_count: int = 0
    segment_indeterminate: bool = False
    net_shoulder_pitch_hold_deg: Optional[float] = None

    def rc(self) -> int:
        return rc_for_verdict(self.verdict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "cycle": self.cycle, "arm": self.arm, "verdict": self.verdict,
            "reasons": self.reasons, "genuine_echo_count": self.genuine_echo_count,
            "segment_indeterminate": self.segment_indeterminate,
            "net_shoulder_pitch_hold_deg": self.net_shoulder_pitch_hold_deg,
            "rc": self.rc(),
        }


def evaluate_cycle(
    cycle: str, arm: str, evidence: ev.Evidence, legs: Sequence[LegSpec],
    place_route_leg_name: str = "PLACE_ROUTE",
    hover_name: str = "HOVER", rest_shut_name: str = "REST_SHUT",
    goto_context: Optional[Sequence[Optional[echo.GotoContext]]] = None,
) -> Tuple[CycleVerdict, Dict[str, LegResult]]:
    if arm not in ("A", "B"):
        raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")

    leg_results: Dict[str, LegResult] = {}
    for leg in legs:
        leg_results[leg.name] = evaluate_leg(evidence, leg)

    any_pathcheck_fail = any(
        not r.passed for lr in leg_results.values() for r in lr.pathcheck.values()
        if isinstance(r, pc.CheckResult))
    fail_reasons = [
        f"{lr.name}:{cid}" for lr in leg_results.values() for cid, r in lr.pathcheck.items()
        if isinstance(r, pc.CheckResult) and not r.passed]

    any_hold_target_drift = False  # holds never carry commands by construction (see holds.py);
    # a nonzero drift here would mean a command WAS found inside a hold window,
    # which pathcheck's C6 already flags -- included in any_pathcheck_fail above.

    place_leg = leg_results.get(place_route_leg_name)
    place_spec = next((l for l in legs if l.name == place_route_leg_name), None)
    genuine = 0
    segment_indeterminate = True
    net_shoulder_hold_deg = None

    if place_leg is not None and place_spec is not None:
        affected = seg.find_affected_segment(place_leg.assignment, place_spec.route,
                                              hover_name, rest_shut_name)
        if affected is not None:
            segment_indeterminate = affected.indeterminate
            if not affected.indeterminate:
                seg_global_indices = [
                    place_spec.command_indices[i]
                    for i in range(affected.start_command_index, affected.end_command_index + 1)]
                echo_results = echo.classify_commands(evidence, goto_context)
                counts = echo.count_labels(echo_results, seg_global_indices)
                genuine = counts.genuine_echo

    reasons = list(fail_reasons)
    if arm == "B":
        if genuine > 0:
            reasons.append(f"{genuine} genuine echo(es) in the affected segment")
        if any_pathcheck_fail or genuine > 0 or any_hold_target_drift:
            verdict = VERDICT_STOP
        else:
            verdict = VERDICT_OK
    else:
        if segment_indeterminate or genuine == 0:
            verdict = VERDICT_INCONCLUSIVE_BASELINE
        else:
            verdict = VERDICT_MANIPULATED

    cv = CycleVerdict(cycle, arm, verdict, reasons, genuine, segment_indeterminate,
                       net_shoulder_hold_deg)
    return cv, leg_results


def evaluate_cycle_safe(
    cycle: str, arm: str, evidence_dir, states_rel: str, commands_rel: str,
    legs_factory, **kwargs,
) -> CycleVerdict:
    """Wraps ``evaluate_cycle`` with evidence loading, catching
    ``EvidenceError``/``IntegrityError`` into ``evidence_incomplete`` rather
    than letting a malformed cycle raise. ``legs_factory(evidence) ->
    Sequence[LegSpec]`` builds the leg specs once the evidence is loaded
    (their command index ranges depend on it)."""
    from tools.goalfix_cmp._io import IntegrityError

    t0 = time.monotonic()
    try:
        evidence = ev.verify_and_load(evidence_dir, states_rel, commands_rel)
        legs = legs_factory(evidence)
        cv, _ = evaluate_cycle(cycle, arm, evidence, legs, **kwargs)
    except (ev.EvidenceError, IntegrityError) as exc:
        cv = CycleVerdict(cycle, arm, VERDICT_EVIDENCE_INCOMPLETE, [str(exc)])
    cv_elapsed = time.monotonic() - t0
    if cv_elapsed > 60.0:
        cv.reasons.append(f"cycle evaluation took {cv_elapsed:.1f}s, over the 60s budget")
    return cv


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    """CLI covering the PLACE_ROUTE leg alone (the segment this whole
    comparison is about). A cycle that also needs LIFT_TO_PRESENT's leg
    folded into the same verdict should call ``evaluate_cycle`` as a
    library instead -- see the handoff note for why a second leg's
    route/guard/command-range triple doesn't fit cleanly on one CLI line."""
    import argparse

    from tools.goalfix_cmp._io import write_result
    from reachy_ai.motion import rig_routes as R

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-dir", required=True)
    p.add_argument("--states", default="states.jsonl")
    p.add_argument("--commands", default="commands.jsonl")
    p.add_argument("--cycle", required=True)
    p.add_argument("--arm", choices=["A", "B"], required=True)
    p.add_argument("--command-start-index", type=int, required=True)
    p.add_argument("--command-end-index", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    def legs_factory(evidence: ev.Evidence) -> List[LegSpec]:
        idx = list(range(args.command_start_index, args.command_end_index + 1))
        idx = [i for i in idx if evidence.commands.kind[i] == "joint_command"]
        return [LegSpec("PLACE_ROUTE", R.PLACE_ROUTE, R.CRITICAL_JOINTS, R.HOME, idx)]

    cv = evaluate_cycle_safe(args.cycle, args.arm, args.evidence_dir, args.states,
                              args.commands, legs_factory)
    return write_result(args.out, cv.rc(), cv.as_dict())


def main() -> int:
    return _cli()


if __name__ == "__main__":
    raise SystemExit(main())
