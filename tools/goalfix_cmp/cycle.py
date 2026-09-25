"""End-to-end per-cycle verdict (plan §5 step 9, §7.6; assignment T4):
combines §2.2-§2.7 into one ``between_<cycle>.json``-shaped result.

Verdicts, per plan §7.6:

* ``ok`` / ``manipulated`` (rc 0) -- authorize the next cycle.
* ``STOP`` (rc 2) -- a B cycle with a genuine echo in the segment, a
  C0-C8 failure on either leg, non-zero target drift in a hold, a failed
  provenance/compliance/start-variant gate, or an indeterminate/missing
  affected segment.
* ``inconclusive_baseline`` (rc 3, A only) -- 0 genuine echoes in the
  segment, or the segment is indeterminate. Not a failure of the fix, but
  never treated as a pass for gating purposes either.
* ``evidence_incomplete`` (rc 3) -- anything upstream (integrity, epoch
  cross-check, leg linking, an unplaceable command) raised.

Nothing but ``ok``/``manipulated`` authorizes the next cycle.

Both legs (setup ``PLACE_ROUTE`` + flight ``LIFT_TO_PRESENT``) are derived
from the linker sidecars and evaluated in one verdict (T4 item 1-3); each
leg's start is its own ``turn_on`` goal, read from evidence, never
hard-coded to ``R.HOME``/``R.REST`` (M5). ``goto_context`` is built from
the live goal assignment and path coincidence runs in this CLI (item 4).
Hold drift is measured from evidence, never asserted (item 5). Provenance,
compliance and the start-variant gate are called and gated on (item 6).
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
for _p in (_REPO / "src", _REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.goalfix_cmp import clearance as cl  # noqa: E402
from tools.goalfix_cmp import echo  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import holds  # noqa: E402
from tools.goalfix_cmp import initial  # noqa: E402
from tools.goalfix_cmp import pathcheck as pc  # noqa: E402
from tools.goalfix_cmp import provenance as pv  # noqa: E402
from tools.goalfix_cmp import segments as seg  # noqa: E402
from tools.goalfix_cmp._io import RC_INCONCLUSIVE, RC_OK, RC_STOP  # noqa: E402
from tools.goalfix_cmp._units import RadWaypoint, route_rad  # noqa: E402

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


class CycleInputError(ev.EvidenceError):
    """Anything about the leg/sidecar wiring itself (as opposed to the raw
    states.jsonl/commands.jsonl integrity ``EvidenceError`` already
    covers) that makes a cycle un-evaluable. Subclasses EvidenceError so
    every existing ``except ev.EvidenceError`` catch site keeps working."""


# ---------------------------------------------------------------------------
# Leg resolution (T4 items 1-2): sidecar -> Leg -> commands, by bracket
# ---------------------------------------------------------------------------

@dataclass
class LegSpec:
    name: str                          # "setup" | "flight"
    route_rad: Sequence                # RadWaypoint sequence (or any .name/.pose/.seconds mock)
    guard: Sequence[str]
    command_indices: List[int]         # global indices into evidence.commands, in order
    start_pose8: Dict[str, float]      # the turn_on goal, radians, from evidence
    #: T4/T7: the state in force at this leg's own turn_on. Optional --
    #: derived from evidence (ev.leg_turn_on_state_index) when omitted, so
    #: a caller building a LegSpec by hand (a fixture, not resolve_leg)
    #: need not compute it itself.
    turn_on_state_index: Optional[int] = None


def resolve_leg(
    evidence: ev.Evidence, sidecar_path, leg_label: str,
    route_deg: Sequence, guard: Sequence[str],
) -> LegSpec:
    """Loads ``sidecar_path`` (a linker ``<log>.link.json``), locates the
    leg's aligned-state span (``evidence.leg_from_sidecar``), maps it to
    command indices BY BRACKET (``evidence.commands_in_leg`` -- never an
    operator-supplied index, per T4 item 2), and reads the leg's start
    pose as the goal in force at its own ``turn_on`` (T4 item 3 / M5;
    never ``R.HOME``/``R.REST``)."""
    import json
    p = Path(sidecar_path)
    if not p.is_file():
        raise CycleInputError(f"{leg_label}: missing sidecar {p}")
    try:
        sidecar = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise CycleInputError(f"{leg_label}: malformed sidecar {p}: {exc}") from exc

    leg = ev.leg_from_sidecar(leg_label, sidecar, evidence.states)
    idx = ev.commands_in_leg(evidence, leg)
    if not idx:
        raise CycleInputError(f"{leg_label}: no commands found for this leg's aligned-state span")
    turn_on_idx = ev.leg_turn_on_state_index(evidence, idx)
    if turn_on_idx is None or turn_on_idx < 0:
        raise CycleInputError(f"{leg_label}: cannot locate the state in force at turn_on")

    start_pose8 = {
        name: float(evidence.states.position_rad[turn_on_idx, k])
        for k, name in enumerate(pc.R_JOINTS)
    }
    return LegSpec(leg_label, route_rad(route_deg), guard, idx, start_pose8, turn_on_idx)


# ---------------------------------------------------------------------------
# goto_context (T4 item 4): built from the live goal assignment
# ---------------------------------------------------------------------------

def build_goto_context(
    route_rad_: Sequence[RadWaypoint], start_pose8: Dict[str, float],
    assignment: seg.GoalAssignment,
) -> List[Optional[echo.GotoContext]]:
    out: List[Optional[echo.GotoContext]] = [None] * len(assignment.goal_index)
    seg_start = dict(start_pose8)
    last_goal: Optional[int] = None
    for i, g in enumerate(assignment.goal_index):
        if g is None:
            continue
        if g != last_goal:
            seg_start = start_pose8 if g == 0 else route_rad_[g - 1].pose
            last_goal = g
        goal8 = route_rad_[g].pose
        out[i] = echo.GotoContext(start8=dict(seg_start), goal8=dict(goal8),
                                   seconds=route_rad_[g].seconds)
    return out


# ---------------------------------------------------------------------------
# Per-leg evaluation
# ---------------------------------------------------------------------------

@dataclass
class LegResult:
    name: str
    assignment: seg.GoalAssignment
    pathcheck: Dict[str, pc.CheckResult]
    hold_stats: List[holds.HoldStats]
    goto_context: List[Optional[echo.GotoContext]]
    truncated_final_line_inside_leg: bool


def evaluate_leg(evidence: ev.Evidence, leg: LegSpec) -> LegResult:
    idx = np.asarray(leg.command_indices)
    epoch = int(evidence.commands.epoch[idx[0]])
    targets21 = evidence.commands.target_rad[idx]
    targets8 = [{name: float(row[i]) for i, name in enumerate(pc.R_JOINTS)} for row in targets21]
    assignment = seg.assign_goals(leg.route_rad, leg.start_pose8, targets8)
    t_hi_s = [evidence.brackets[i].t_hi for i in idx]
    results = pc.run_all(targets21, t_hi_s, leg.start_pose8, leg.route_rad, leg.guard,
                          assignment=assignment)

    windows = holds.find_hold_windows(assignment, evidence.brackets, list(idx))
    hold_stats: List[holds.HoldStats] = []
    for w in windows:
        goal_index = w[0]
        target8 = leg.route_rad[goal_index].pose
        hs = holds.hold_window_stats(w, epoch, target8, evidence, list(idx))
        if hs is not None:
            hold_stats.append(hs)

    ctx = build_goto_context(leg.route_rad, leg.start_pose8, assignment)

    # T10.2: a truncated final states.jsonl line is flagged; rc 3 if it
    # would fall inside THIS leg -- detectable only when this leg's own
    # command range reaches the very end of the (truncation-shortened)
    # recording, since the dropped row's own content is unknown.
    truncated_inside = (
        evidence.commands.truncated_final_line
        and len(idx) > 0 and int(idx.max()) == len(evidence.commands) - 1)

    return LegResult(leg.name, assignment, results, hold_stats, ctx, truncated_inside)


# ---------------------------------------------------------------------------
# §7.3-7.5 metrics
# ---------------------------------------------------------------------------

@dataclass
class CycleMetrics:
    leg_start_shoulder_pitch_error_deg: Optional[float] = None
    delta_cmd_max_cm: Optional[float] = None
    delta_cmd_segment_max_cm: Optional[float] = None
    wrist_ball_delta_cm: Optional[float] = None
    post_arrival_rise_deg: Optional[float] = None
    net_shoulder_pitch_hold_deg: Optional[float] = None
    c5_c6_pass: bool = True
    open_questions: List[str] = field(default_factory=list)


def compute_place_route_metrics(
    evidence: ev.Evidence, place_leg: LegSpec, place_result: LegResult,
    affected, scene,
) -> CycleMetrics:
    """§7.3-7.5, computed from evidence where report §5 defines the
    quantity precisely enough; ``null`` (via ``open_questions``) where it
    does not -- never invented (assignment §2 T4 item 8)."""
    m = CycleMetrics()
    c5_ok = place_result.pathcheck["C5"].passed
    c6_ok = place_result.pathcheck["C6"].passed
    m.c5_c6_pass = c5_ok and c6_ok

    idx = np.asarray(place_leg.command_indices)
    r_shoulder_k = pc.R_JOINTS.index("r_shoulder_pitch")

    # Net shoulder-pitch TARGET movement over the HOVER hold (report §5:
    # "+0.59/+2.00/+4.80 deg (-39.69 -> -37.70 median)" -- end-of-hold
    # target minus start-of-hold target). Descriptive only; never a
    # verdict (plan §7.1/§7.6).
    hover_hold = next(
        (hs for hs in place_result.hold_stats
         if place_leg.route_rad[hs.window.goal_index].name == "HOVER"), None)
    if hover_hold is not None:
        if hover_hold.window_command_indices:
            first_t = evidence.commands.target_rad[hover_hold.window_command_indices[0], r_shoulder_k]
            last_t = evidence.commands.target_rad[hover_hold.window_command_indices[-1], r_shoulder_k]
            m.net_shoulder_pitch_hold_deg = float(np.degrees(last_t - first_t))
        else:
            m.net_shoulder_pitch_hold_deg = 0.0  # no commands in the hold -- target never moved

    # Leg-start shoulder-pitch error: |realised position at the start of
    # the affected segment (the first setpoint of the HOVER->REST_SHUT
    # goto) - HOVER's own shoulder-pitch target|, per report §2's "the
    # REST_SHUT leg starts from a target N deg from HOVER" framing.
    if affected is not None and not affected.indeterminate:
        seg_start_cmd_global = place_leg.command_indices[affected.start_command_index]
        seg_start_state_idx = evidence.brackets[seg_start_cmd_global].hi_state_index
        hover_target_rad = place_leg.route_rad[affected.hover_goal_index].pose["r_shoulder_pitch"]
        if seg_start_state_idx is not None:
            realised = float(evidence.states.position_rad[seg_start_state_idx, 0])  # r_shoulder_pitch, ARM7[0]
            m.leg_start_shoulder_pitch_error_deg = float(
                np.degrees(abs(realised - hover_target_rad)))

        # Post-arrival rise: arm's rise from its closest approach to HOVER
        # (within the hold window) to the segment's own start -- report §5:
        # "arm rise from closest approach to leg start, +0.98/+1.77/+3.25
        # deg". Needs the hold window's own realised trace; only available
        # when the HOVER hold was found above.
        if hover_hold is not None and seg_start_state_idx is not None:
            lo, hi = hover_hold.window.first_state_index, hover_hold.window.last_state_index
            trace = evidence.states.position_rad[lo:hi + 1, 0]
            if len(trace):
                closest_idx_local = int(np.argmin(np.abs(trace - hover_target_rad)))
                closest_val = float(trace[closest_idx_local])
                realised_at_seg_start = float(evidence.states.position_rad[seg_start_state_idx, 0])
                m.post_arrival_rise_deg = float(np.degrees(realised_at_seg_start - closest_val))
    else:
        m.open_questions.append(
            "leg_start_shoulder_pitch_error_deg/post_arrival_rise_deg: affected segment "
            "is indeterminate or missing")

    # Delta-cmd/Delta-trk (§7.3): needs a SceneModel for the cycle's board
    # -- left null with a flagged question when the caller has none (V1
    # has no board/scene wired at all; a live comparison run would pass
    # one in).
    if scene is None:
        m.open_questions.append(
            "delta_cmd_max_cm/delta_cmd_segment_max_cm/wrist_ball_delta_cm: no SceneModel "
            "was supplied for this run (plan §7.3 needs the cycle's board scene)")
    else:
        commanded = cl.commanded_samples(evidence.commands.target_rad[idx, :8])
        realised, _ = cl.realised_samples_from_states(evidence.states.position_rad[
            [evidence.brackets[i].hi_state_index for i in idx if
             evidence.brackets[i].hi_state_index is not None], :8])
        deltas = cl.compute_deltas(
            "PLACE_ROUTE", None, 200, scene, commanded, realised)
        if deltas:
            m.delta_cmd_max_cm = max(d.delta_cmd_cm for d in deltas)
            if affected is not None and not affected.indeterminate:
                seg_local = range(affected.start_command_index, affected.end_command_index + 1)
                seg_commanded = [commanded[i] for i in seg_local if i < len(commanded)]
                if seg_commanded:
                    seg_deltas = cl.compute_deltas(
                        "PLACE_ROUTE", None, 200, scene, seg_commanded,
                        realised[:len(seg_commanded)])
                    if seg_deltas:
                        m.delta_cmd_segment_max_cm = max(d.delta_cmd_cm for d in seg_deltas)

    return m


# ---------------------------------------------------------------------------
# Cycle-level orchestration
# ---------------------------------------------------------------------------

@dataclass
class CycleVerdict:
    cycle: str
    arm: str
    verdict: str
    reasons: List[str] = field(default_factory=list)
    genuine_echo_count: int = 0
    segment_indeterminate: bool = False
    net_shoulder_pitch_hold_deg: Optional[float] = None
    metrics: Optional[CycleMetrics] = None
    checks: Dict[str, object] = field(default_factory=dict)
    unplaceable_command_indices: List[int] = field(default_factory=list)
    tools_sha256: str = ""
    validation_only: bool = False

    def rc(self) -> int:
        if self.validation_only:
            return RC_INCONCLUSIVE
        return rc_for_verdict(self.verdict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "cycle": self.cycle, "arm": self.arm, "verdict": self.verdict,
            "reasons": self.reasons, "genuine_echo_count": self.genuine_echo_count,
            "segment_indeterminate": self.segment_indeterminate,
            "net_shoulder_pitch_hold_deg": self.net_shoulder_pitch_hold_deg,
            "metrics": (vars(self.metrics) if self.metrics else None),
            "checks": self.checks,
            "unplaceable_command_indices": self.unplaceable_command_indices,
            "tools_sha256": self.tools_sha256,
            "validation_only": self.validation_only,
            "rc": self.rc(),
        }


def _package_sha256() -> str:
    """sha256 over every ``tools/goalfix_cmp/*.py`` source file, sorted --
    what T5's ``summary`` checks each ``between_*.json`` against, so a
    cycle evaluated by a stale/different tooling checkout is caught."""
    import hashlib
    h = hashlib.sha256()
    for p in sorted(_HERE.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def evaluate_cycle(
    cycle: str, arm: str, evidence: ev.Evidence, legs: Sequence[LegSpec], *,
    scene=None,
    provenance_gate: Optional[pv.GateResult] = None,
    compliance_gate: Optional[Tuple[bool, str]] = None,
    start_variant_gate_result: Optional[Tuple[bool, Optional[str], Optional[dict]]] = None,
    guard_note: str = "",
    place_route_leg_name: str = "setup",
    hover_name: str = "HOVER", rest_shut_name: str = "REST_SHUT",
    skip_gates: bool = False,
) -> Tuple[CycleVerdict, Dict[str, LegResult]]:
    """``skip_gates=True`` (V1's ``--validation-mode``, and callers that
    only want the commanded-path/echo verdict in isolation) omits the
    provenance/compliance/start_variant gates from the verdict entirely --
    never treated as "missing" -- rather than requiring every caller to
    thread three inputs it may have no way to supply."""
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

    any_hold_target_drift = any(
        any(v != 0.0 for v in hs.target_drift.values())
        for lr in leg_results.values() for hs in lr.hold_stats)
    drift_reasons = [
        f"{lr.name}: hold at goal {hs.window.goal_index} drifted {hs.target_drift}"
        for lr in leg_results.values() for hs in lr.hold_stats
        if any(v != 0.0 for v in hs.target_drift.values())]

    truncated_inside_leg = [lr.name for lr in leg_results.values()
                             if lr.truncated_final_line_inside_leg]

    # T4 item 4: goto_context, built per leg, merged into one full-length
    # array so echo.classify_commands sees path coincidence for every leg.
    full_goto_context: List[Optional[echo.GotoContext]] = [None] * len(evidence.commands)
    leg_turn_on_map: Dict[int, int] = {}
    for leg in legs:
        lr = leg_results[leg.name]
        for local_i, global_i in enumerate(leg.command_indices):
            full_goto_context[global_i] = lr.goto_context[local_i]
        turn_on_idx = leg.turn_on_state_index
        if turn_on_idx is None:
            turn_on_idx = ev.leg_turn_on_state_index(evidence, leg.command_indices)
        if turn_on_idx is not None:
            leg_turn_on_map[leg.command_indices[0]] = turn_on_idx

    echo_results = echo.classify_commands(
        evidence, full_goto_context, leg_turn_on_state_index=leg_turn_on_map)

    place_leg = next((l for l in legs if l.name == place_route_leg_name), None)
    place_lr = leg_results.get(place_route_leg_name)
    genuine = 0
    segment_indeterminate = True
    affected = None

    if place_leg is not None and place_lr is not None:
        affected = seg.find_affected_segment(place_lr.assignment, place_leg.route_rad,
                                              hover_name, rest_shut_name)
        if affected is not None:
            segment_indeterminate = affected.indeterminate
            if not affected.indeterminate:
                seg_global_indices = [
                    place_leg.command_indices[i]
                    for i in range(affected.start_command_index, affected.end_command_index + 1)]
                counts = echo.count_labels(echo_results, seg_global_indices)
                genuine = counts.genuine_echo

    metrics = None
    if place_leg is not None and place_lr is not None:
        metrics = compute_place_route_metrics(evidence, place_leg, place_lr, affected, scene)

    reasons = list(fail_reasons) + drift_reasons
    for name in truncated_inside_leg:
        reasons.append(f"{name}: commands.jsonl's final line was truncated inside this leg")

    checks: Dict[str, object] = {}
    gate_failed = False
    gate_incomplete = False
    if skip_gates:
        checks["gates_skipped"] = True
    elif provenance_gate is not None:
        checks["provenance"] = {"ok": provenance_gate.ok, "reason": provenance_gate.reason}
        if not provenance_gate.ok:
            gate_failed = True
            reasons.append(f"provenance: {provenance_gate.reason}")
    else:
        gate_incomplete = True
    if skip_gates:
        pass
    elif compliance_gate is not None:
        ok, detail = compliance_gate
        checks["compliance"] = {"ok": ok, "detail": detail}
        if not ok:
            gate_failed = True
            reasons.append(f"compliance: {detail}")
    else:
        gate_incomplete = True
    if skip_gates:
        pass
    elif start_variant_gate_result is not None:
        ok, reason, _doc = start_variant_gate_result
        checks["start_variant"] = {"ok": ok, "reason": reason}
        if not ok:
            gate_incomplete = True
            reasons.append(f"start_variant: {reason}")
    else:
        gate_incomplete = True
    if guard_note:
        checks["guard_note"] = guard_note

    verdict: str
    if truncated_inside_leg:
        verdict = VERDICT_EVIDENCE_INCOMPLETE
    elif arm == "B":
        if genuine > 0:
            reasons.append(f"{genuine} genuine echo(es) in the affected segment")
        if segment_indeterminate:
            reasons.append("affected segment is indeterminate or missing (B)")
        if any_pathcheck_fail or genuine > 0 or any_hold_target_drift or segment_indeterminate or gate_failed:
            verdict = VERDICT_STOP
        elif gate_incomplete:
            verdict = VERDICT_EVIDENCE_INCOMPLETE
        else:
            verdict = VERDICT_OK
    else:
        if gate_failed:
            verdict = VERDICT_STOP
        elif segment_indeterminate or genuine == 0:
            verdict = VERDICT_INCONCLUSIVE_BASELINE
        else:
            verdict = VERDICT_MANIPULATED

    cv = CycleVerdict(
        cycle, arm, verdict, reasons, genuine, segment_indeterminate,
        metrics.net_shoulder_pitch_hold_deg if metrics else None, metrics, checks,
        list(evidence.unplaceable_command_indices), _package_sha256())
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json as _json

    from tools.goalfix_cmp._io import IntegrityError, write_result
    from reachy_ai.motion import rig_routes as R

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ev-dir", required=True)
    p.add_argument("--states", default="states.jsonl")
    p.add_argument("--commands", default="commands.jsonl")
    p.add_argument("--sha256sums", default="SHA256SUMS")
    p.add_argument("--control-dir", required=True)
    p.add_argument("--cycle", required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--arm", choices=["A", "B"], required=True)
    p.add_argument("--arm-map")
    p.add_argument("--expected-host-sha")
    p.add_argument("--setup-sidecar", help="override the guessed <control-dir>/<cycle>-setup.link.json")
    p.add_argument("--flight-sidecar", help="override the guessed <control-dir>/<cycle>-flight.link.json")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--validation-mode", action="store_true",
        help="V1: skip the provenance/compliance/start_variant gates (which do not exist "
             "for this session) and ALWAYS exit 3 -- refused unless --ev-dir is outside "
             "~/b4-goalfix-cmp-*")
    args = p.parse_args(argv)

    if args.validation_mode:
        ev_dir_resolved = str(Path(args.ev_dir).expanduser().resolve())
        home = str(Path.home())
        forbidden = str(Path(home) / "b4-goalfix-cmp-")
        if ev_dir_resolved.startswith(forbidden):
            return write_result(args.out, RC_INCONCLUSIVE, {
                "ok": False,
                "reason": "--validation-mode is refused when --ev-dir is under ~/b4-goalfix-cmp-*"})

    try:
        evd = ev.verify_and_load(args.ev_dir, args.states, args.commands, args.sha256sums)

        control_dir = Path(args.control_dir)
        setup_sidecar = args.setup_sidecar or str(control_dir / f"{args.cycle}-setup.link.json")
        flight_sidecar = args.flight_sidecar or str(control_dir / f"{args.cycle}-flight.link.json")
        setup_leg = resolve_leg(evd, setup_sidecar, "setup", R.PLACE_ROUTE, R.CRITICAL_JOINTS)
        flight_leg = resolve_leg(evd, flight_sidecar, "flight", R.LIFT_TO_PRESENT, R._PRESENT_GUARD)
        overlap = set(setup_leg.command_indices) & set(flight_leg.command_indices)
        if overlap:
            raise CycleInputError(f"setup and flight legs overlap at command indices {sorted(overlap)}")

        provenance_gate = compliance_gate = start_variant_result = None
        guard_note = (
            "report §4 states which guard set applies to which route (CRITICAL_JOINTS for "
            "PLACE_ROUTE, _PRESENT_GUARD for LIFT_TO_PRESENT) but never specifies how a guard "
            "enters C0-C8 mechanically; per the T4 instruction, no use was invented. guard is "
            "recorded here for the record and does not gate any check.")

        if not args.validation_mode:
            versions_path = control_dir / f"versions_{args.cycle}.json"
            if not versions_path.is_file():
                provenance_gate = pv.GateResult(False, f"missing {versions_path}")
            else:
                doc = _json.loads(versions_path.read_text())
                arm_map_doc = _json.loads(Path(args.arm_map).read_text()) if args.arm_map else []
                arm_map = {e["rep"]: pv.ArmMapEntry(**e) for e in arm_map_doc}
                observed = pv.ObservedCycle(
                    running_image_id=doc.get("running_image_id", ""),
                    opt_hashes=doc.get("opt_hashes", {}),
                    supervisor_start_times=doc.get("supervisor_start_times", {}),
                    recreate_timestamp=doc.get("recreate_timestamp", 0.0),
                    host_git_sha=doc.get("host_native_kernel_sha", ""),
                    host_tree_dirty=bool(doc.get("host_tree_dirty", True)))
                provenance_gate = pv.check_cycle(
                    args.rep, arm_map, observed,
                    expected_host_sha=args.expected_host_sha or "",
                    required_supervisor_programs=list(doc.get("supervisor_start_times", {})))

            compliance_path = control_dir / f"prep_{args.cycle}.json"
            if compliance_path.is_file():
                prep = _json.loads(compliance_path.read_text())
                expected = prep.get("expected_compliant21")
                actual = prep.get("actual_compliant21")
                if expected is not None and actual is not None:
                    compliance_gate = initial.check_compliance(actual, expected)
                else:
                    compliance_gate = (False, f"{compliance_path}: missing compliance vectors")
            else:
                compliance_gate = (False, f"missing {compliance_path}")

            start_variant_result = initial.start_variant_gate(control_dir, args.cycle)

        cv, _ = evaluate_cycle(
            args.cycle, args.arm, evd, [setup_leg, flight_leg],
            provenance_gate=provenance_gate, compliance_gate=compliance_gate,
            start_variant_gate_result=start_variant_result, guard_note=guard_note,
            skip_gates=args.validation_mode)
        if args.validation_mode:
            cv.validation_only = True
    except (ev.EvidenceError, IntegrityError, initial.StartVariantGateUnavailable) as exc:
        return write_result(args.out, RC_INCONCLUSIVE, {"ok": False, "reason": str(exc)})

    return write_result(args.out, cv.rc(), cv.as_dict())


def main() -> int:
    return _cli()


if __name__ == "__main__":
    raise SystemExit(main())
