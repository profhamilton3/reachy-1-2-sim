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
    #: Q-hold ruling (2026-09-25 stage-1 rulings, §2): the sidecar's own
    #: aligned span (``ev.Leg.t_lo``/``t_hi``, sim_time_s of the leg's
    #: FIRST/LAST aligned state) -- the outer edge of the lead-in and
    #: parked-tail windows. ``None`` for a caller (a fixture, not
    #: resolve_leg) that has no sidecar of its own; those two windows are
    #: then skipped, never assumed empty.
    sidecar_t_lo: Optional[float] = None
    sidecar_t_hi: Optional[float] = None


def load_verified_scene(sidecar_path):
    """CB1 (merge verdict, 2026-09-25 stage-repairs assignment §4): loads
    the ``SceneModel`` named by the setup sidecar's own ``scene.path``
    (as ``scripts/link_e1_flight.py``'s ``build_base_sidecar`` writes it),
    after RE-hashing that same ``extends`` chain -- via
    ``scripts/e1_identity.py``'s own ``_extends_chain``/``_sha256_file``
    (imported, never reimplemented) -- and confirming it still matches
    the sidecar's own recorded ``scene.chain_sha256``. A missing
    ``scene`` block, a missing scene file, or ANY chain hash mismatch is
    ``CycleInputError`` (rc 3), never a silently-loaded, possibly-stale
    scene."""
    import json as _json

    import e1_identity  # noqa: E402  (scripts/ is on sys.path, see the module header)

    p = Path(sidecar_path)
    sidecar = _json.loads(p.read_text())
    scene_block = sidecar.get("scene") or {}
    scene_path_str = scene_block.get("path")
    if not scene_path_str:
        raise CycleInputError(f"sidecar {p} has no scene.path")
    scene_path = Path(scene_path_str)
    if not scene_path.is_file():
        raise CycleInputError(f"scene file {scene_path} (from sidecar {p}) does not exist")

    recorded_chain_sha256 = scene_block.get("chain_sha256") or {}
    current_chain = e1_identity._extends_chain(scene_path)
    current_chain_sha256 = {str(cp): e1_identity._sha256_file(cp) for cp in current_chain}
    if current_chain_sha256 != recorded_chain_sha256:
        raise CycleInputError(
            f"scene chain hash mismatch for {scene_path}: sidecar recorded "
            f"{recorded_chain_sha256}, current files hash to {current_chain_sha256}")

    from reachy_ai.scene.awareness import SceneModel
    return SceneModel.from_yaml(str(scene_path))


def resolve_leg(
    evidence: ev.Evidence, sidecar_path, leg_label: str,
    route_deg: Sequence, guard: Sequence[str], *,
    expected_route_name: Optional[str] = None,
    expected_server_run_dir: Optional[str] = None,
) -> LegSpec:
    """Loads ``sidecar_path`` (a linker ``<log>.link.json``), locates the
    leg's aligned-state span (``evidence.leg_from_sidecar``), maps it to
    command indices BY BRACKET (``evidence.commands_in_leg`` -- never an
    operator-supplied index, per T4 item 2), and reads the leg's start
    pose as the goal in force at its own ``turn_on`` (T4 item 3 / M5;
    never ``R.HOME``/``R.REST``).

    MB2b (merge verdict, 2026-09-25 stage-repairs assignment §3; CB3):
    ``expected_route_name``/``expected_server_run_dir`` -- when given
    (never by default, for backward compatibility with callers that have
    no cycle manifest at all) -- verify the sidecar's OWN ``log`` field
    names the right route (``scripts/link_e1_flight.py``'s real sidecars
    are named ``route_clearance_<ROUTE>_<ts>.link.json``, never the
    guessed ``<cycle>-{setup,flight}.link.json``) and that its
    ``server_run_dir`` matches the run this leg's evidence actually
    came from."""
    import json
    p = Path(sidecar_path)
    if not p.is_file():
        raise CycleInputError(f"{leg_label}: missing sidecar {p}")
    try:
        sidecar = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise CycleInputError(f"{leg_label}: malformed sidecar {p}: {exc}") from exc

    if expected_route_name is not None:
        log_name = sidecar.get("log", "")
        if expected_route_name not in log_name:
            raise CycleInputError(
                f"{leg_label}: sidecar's log {log_name!r} does not name route "
                f"{expected_route_name!r}")
    if expected_server_run_dir is not None:
        sc_run_dir = sidecar.get("server_run_dir")
        if sc_run_dir is not None and sc_run_dir != expected_server_run_dir:
            raise CycleInputError(
                f"{leg_label}: sidecar's server_run_dir {sc_run_dir!r} != "
                f"this cycle's own run {expected_server_run_dir!r}")

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
    return LegSpec(leg_label, route_rad(route_deg), guard, idx, start_pose8, turn_on_idx,
                   sidecar_t_lo=leg.t_lo, sidecar_t_hi=leg.t_hi)


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
    duplicate_state_indices_inside: List[int] = field(default_factory=list)
    #: A1: a command precedes the turn_on match in the leg's own span
    #: (or no in-span command matches it at all).
    lead_in_violation: bool = False
    lead_in_detail: str = ""
    #: A1: the route's own final waypoint was never determinately
    #: reached, so the parked-tail window could not be bounded at all
    #: (reported, never gated -- there is nothing to check).
    parked_tail_indeterminate: bool = False
    #: CB7 (readiness review, 2026-09-25 stage-repairs assignment §4):
    #: plan §7.2's "also reported per waypoint: re-stream pass counts,
    #: arrival errors[,] and the r_wrist_pitch shortfall (descriptive)"
    #: -- one entry per route waypoint, REPORT-ONLY (never read by any
    #: verdict/rc computation in this module).
    per_waypoint: List[Dict[str, object]] = field(default_factory=list)


def _per_waypoint_report(
    targets8: Sequence[Dict[str, float]], route: Sequence, assignment: seg.GoalAssignment,
) -> List[Dict[str, object]]:
    """CB7: for each route waypoint, the re-stream pass count (plan
    §7.2/C5's own "first exact match, then every later setpoint at that
    goal" -- counted from the first exact arrival onward, inclusive) and
    the arrival error per ARM7 joint in degrees (C4's own `v - b`, the
    LAST setpoint assigned to this waypoint minus its goal, signed --
    never reduced against C4's residual budget, since this is descriptive,
    not a check). `r_wrist_pitch`'s own entry in `arrival_error_deg` IS
    the "wrist-pitch shortfall" the review names alongside it -- the
    review lists them together as one reported group, not two separate
    formulas, and no other definition of "shortfall" appears anywhere in
    the plan or report. `None` (never a fabricated 0) for a waypoint no
    command was ever assigned to."""
    report: List[Dict[str, object]] = []
    for k, wp in enumerate(route):
        idxs = pc._idx_for_goal(assignment, k)
        if not idxs:
            report.append({"name": wp.name, "restream_pass_count": 0, "arrival_error_deg": None})
            continue
        goal8 = pc._goal8(wp)
        first_exact = next(
            (i for i in idxs if pc._is_exact_goal(targets8[i], goal8)), None)
        restream_pass_count = (
            sum(1 for i in idxs if i >= first_exact) if first_exact is not None else 0)
        last_i = idxs[-1]
        arrival_error_deg = {
            j: float(np.degrees(targets8[last_i].get(j, 0.0) - goal8.get(j, 0.0)))
            for j in pc.ARM7}
        report.append({
            "name": wp.name, "restream_pass_count": restream_pass_count,
            "arrival_error_deg": arrival_error_deg})
    return report


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

    # Coordinator ruling (2026-09-25 stage-2a rulings addendum, A6): C6
    # now uses the SAME settle-hold window rule Q-hold implemented for
    # the cycle-level drift gate (§2), replacing the old structural
    # single-sandwich check (which only ever inspected an INDETERMINATE
    # command sandwiched between two consecutive different goals -- never
    # a stray that R-tie/R-const now correctly assign to a real goto).
    # Overridden here (evaluate_leg is the shipped call site that
    # assembles the final per-leg C0-C8 table every caller, including the
    # CLI, actually sees) rather than in pathcheck.check_c6 itself, so
    # check_c6's existing library-level unit tests (which pass bare
    # targets8/assignment, with no evidence/brackets) keep working
    # unchanged.
    c6_bad_index: Optional[int] = None
    c6_detail = ""
    for hs in hold_stats:
        if hs.window.goal_index < 0:
            continue  # lead-in/parked-tail are gated separately, not part of C6's own scope
        if any(v != 0.0 for v in hs.target_drift.values()):
            c6_bad_index = hs.window_command_indices[0] if hs.window_command_indices else None
            c6_detail = f"non-carry command found inside the hold at goal {hs.window.goal_index}"
            break
    results["C6"] = pc.CheckResult("C6", c6_bad_index is None, c6_bad_index, c6_detail)

    # A1 (coordinator ruling, 2026-09-25 stage-2a rulings addendum): the
    # lead-in and parked-tail windows must be bounded by a command found
    # BY ITS OWN PROPERTY (the turn_on present-match / the final
    # waypoint's own assignment), never by "the leg's first/last
    # command" -- a stray occupying that position would otherwise BECOME
    # the boundary and vanish from the window by construction (the same
    # defect Q-hold closed for the settle-hold windows).
    lead_in_violation = False
    lead_in_detail = ""
    if leg.sidecar_t_lo is not None:
        turn_on_match = holds.find_turn_on_command(evidence, list(idx))
        if turn_on_match is None:
            lead_in_violation = True
            lead_in_detail = "no in-span command matches the turn_on present-position property"
        else:
            lead_in_hi = float(evidence.states.sim_time_s[turn_on_match.state_index])
            try:
                es = holds.edge_window_stats(
                    goal_index=holds.LEAD_IN_GOAL_INDEX, t_lo_s=leg.sidecar_t_lo, t_hi_s=lead_in_hi,
                    epoch=epoch, last_k_target8=leg.start_pose8, evidence=evidence,
                    all_command_indices=list(idx))
            except ValueError as exc:
                raise CycleInputError(f"{leg.name}: lead-in window: {exc}") from exc
            if es is not None:
                hold_stats.append(es)
            if turn_on_match.violation:
                lead_in_violation = True
                lead_in_detail = turn_on_match.detail

    parked_tail_indeterminate = False
    if leg.sidecar_t_hi is not None:
        final_idx = holds.find_final_waypoint_last_command(assignment, list(idx), len(leg.route_rad))
        if final_idx is None:
            parked_tail_indeterminate = True
        else:
            parked_lo = evidence.brackets[final_idx].t_hi
            if parked_lo is not None:
                last_k_target8 = {name: float(evidence.commands.target_rad[final_idx, k])
                                   for k, name in enumerate(pc.R_JOINTS)}
                try:
                    es = holds.edge_window_stats(
                        goal_index=holds.PARKED_TAIL_GOAL_INDEX, t_lo_s=parked_lo,
                        t_hi_s=leg.sidecar_t_hi, epoch=epoch, last_k_target8=last_k_target8,
                        evidence=evidence, all_command_indices=list(idx))
                except ValueError as exc:
                    raise CycleInputError(f"{leg.name}: parked-tail window: {exc}") from exc
                if es is not None:
                    hold_stats.append(es)

    ctx = build_goto_context(leg.route_rad, leg.start_pose8, assignment)

    # T10.2: a truncated final states.jsonl line is flagged; rc 3 if it
    # would fall inside THIS leg -- detectable only when this leg's own
    # command range reaches the very end of the (truncation-shortened)
    # recording, since the dropped row's own content is unknown.
    truncated_inside = (
        evidence.commands.truncated_final_line
        and len(idx) > 0 and int(idx.max()) == len(evidence.commands) - 1)

    # T10.4: a duplicate sim_step state (the same sim_step seen twice,
    # e.g. a pause) INSIDE this leg's own span (bracketed by its own
    # commands' hi_state_index) is evidence incomplete; outside any leg
    # it is reported only (evidence.states.duplicate_indices, surfaced
    # unconditionally in the cycle payload).
    hi_indices = [evidence.brackets[i].hi_state_index for i in idx
                  if evidence.brackets[i].hi_state_index is not None]
    duplicates_inside: List[int] = []
    if hi_indices:
        lo, hi = min(hi_indices), max(hi_indices)
        duplicates_inside = [d for d in evidence.states.duplicate_indices if lo <= d <= hi]

    return LegResult(leg.name, assignment, results, hold_stats, ctx, truncated_inside,
                      duplicates_inside, lead_in_violation=lead_in_violation,
                      lead_in_detail=lead_in_detail,
                      parked_tail_indeterminate=parked_tail_indeterminate,
                      per_waypoint=_per_waypoint_report(targets8, leg.route_rad, assignment))


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

    # Leg-start shoulder-pitch error (CB2, merge verdict, 2026-09-25
    # stage-repairs assignment §4): |commanded TARGET at the segment's
    # first setpoint - HOVER's own shoulder-pitch target|, from
    # commands.target_rad -- report line 91: "the tolerance report
    # measured a leg-start SETPOINT 0.98-4.87deg from HOVER; here the
    # TARGET at leg start is ..."; line 101: "the REST_SHUT leg starts
    # from that moved TARGET, not from HOVER" (the "commanded start").
    # states.position_rad is the REALISED position, a different
    # quantity (post_arrival_rise_deg, below, report line 72: "arm
    # rise") -- using it here would report 0.0 as "refutes" a lag that
    # is entirely in the realised trace and never in the commanded
    # target at all.
    if affected is not None and not affected.indeterminate:
        seg_start_cmd_global = place_leg.command_indices[affected.start_command_index]
        seg_start_state_idx = evidence.brackets[seg_start_cmd_global].hi_state_index
        hover_target_rad = place_leg.route_rad[affected.hover_goal_index].pose["r_shoulder_pitch"]
        commanded = float(evidence.commands.target_rad[seg_start_cmd_global, r_shoulder_k])
        m.leg_start_shoulder_pitch_error_deg = float(
            np.degrees(abs(commanded - hover_target_rad)))

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
                # CB1 (merge verdict, 2026-09-25 stage-repairs assignment
                # §4): realised must be taken from the SAME seg_local
                # positions as seg_commanded (the affected segment's own
                # span within the leg), not realised[:len(seg_commanded)]
                # -- which silently took samples from the LEG's own
                # start regardless of where the segment actually begins.
                seg_realised = [realised[i] for i in seg_local if i < len(realised)]
                if seg_commanded and seg_realised:
                    seg_deltas = cl.compute_deltas(
                        "PLACE_ROUTE", None, 200, scene, seg_commanded, seg_realised)
                    if seg_deltas:
                        m.delta_cmd_segment_max_cm = max(d.delta_cmd_cm for d in seg_deltas)

        # CB1 (merge verdict, 2026-09-25 stage-repairs assignment §4):
        # wrist_ball_delta_cm is intentionally left null even with a
        # scene supplied. The plan (§7.5: "B's wrist_ball Δ falls by
        # ... against the median of the manipulated A cycles") and the
        # review ("compute wrist_ball_delta_cm with the vendored
        # indep_wrist_ball") name two DIFFERENT, both-plausible
        # readings: (a) delta_trk_cm for the link=="wrist_ball" entries
        # already present in `deltas`/`seg_deltas` above (hand="shells"
        # only -- "tube" has no wrist_ball capsule), or (b) a
        # cross-check-style delta computed directly via
        # indep_wrist_ball/cross_check_wrist_ball against one particular
        # scene object at one particular instant. Neither the plan nor
        # the report says which, against which object id, or at which
        # instant (a whole-segment worst-case? a single waypoint?) --
        # per the "no invented definitions" rule, this is flagged, not
        # guessed.
        m.open_questions.append(
            "wrist_ball_delta_cm: report §5/plan §7.3-7.5 and the review name two "
            "different readings (delta_trk_cm for the existing link==\"wrist_ball\" "
            "compute_deltas entries, vs. a cross-check computed directly via "
            "indep_wrist_ball/cross_check_wrist_ball) and do not say which, against "
            "which scene object, or at which instant -- left null rather than guessed")

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
    duplicate_state_indices: List[int] = field(default_factory=list)
    tools_sha256: str = ""
    validation_only: bool = False
    #: CB7 (readiness review, 2026-09-25 stage-repairs assignment §4):
    #: plan §7.1's control run ("the same test is run against states
    #: shifted -20s of simulation time... per cycle, the segment metric
    #: is the number of genuine echoes and their fraction of new
    #: targets"), scoped to the SAME affected segment as
    #: genuine_echo_count -- currently computed only inside the separate
    #: echo CLI, never reported alongside the cycle's own genuine count.
    #: REPORT-ONLY: never read by rc()/verdict.
    control_genuine_echo_count: Optional[int] = None
    control_new_target_count: Optional[int] = None
    #: CB7: plan §7.2's "also reported per waypoint: re-stream pass
    #: counts, arrival errors, and the r_wrist_pitch shortfall
    #: (descriptive)" -- keyed by leg name. REPORT-ONLY.
    per_waypoint: Dict[str, List[Dict[str, object]]] = field(default_factory=dict)

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
            "duplicate_state_indices": self.duplicate_state_indices,
            "tools_sha256": self.tools_sha256,
            "validation_only": self.validation_only,
            "control_genuine_echo_count": self.control_genuine_echo_count,
            "control_new_target_count": self.control_new_target_count,
            "per_waypoint": self.per_waypoint,
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

    def _hold_violates(hs: holds.HoldStats) -> bool:
        # Q-hold ruling (§2): the lead-in window's content rule is "any
        # joint_command inside it is a violation" -- stricter than the
        # settle-hold/parked-tail "non-carry (drift != 0) is a
        # violation" rule, since nothing should be commanded at all
        # before turn_on.
        if hs.window.goal_index == holds.LEAD_IN_GOAL_INDEX:
            return bool(hs.window_command_indices)
        return any(v != 0.0 for v in hs.target_drift.values())

    any_hold_target_drift = any(
        _hold_violates(hs) for lr in leg_results.values() for hs in lr.hold_stats)
    drift_reasons = [
        f"{lr.name}: hold at goal {hs.window.goal_index} drifted {hs.target_drift}"
        for lr in leg_results.values() for hs in lr.hold_stats if _hold_violates(hs)]

    truncated_inside_leg = [lr.name for lr in leg_results.values()
                             if lr.truncated_final_line_inside_leg]
    duplicate_inside_leg = {lr.name: lr.duplicate_state_indices_inside
                             for lr in leg_results.values() if lr.duplicate_state_indices_inside}

    # A1: a stray command ahead of the real turn_on match (or no match
    # at all) is a lead-in violation.
    any_lead_in_violation = any(lr.lead_in_violation for lr in leg_results.values())
    lead_in_reasons = [f"{lr.name}: lead-in violation: {lr.lead_in_detail}"
                        for lr in leg_results.values() if lr.lead_in_violation]

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
    control_genuine_echo_count: Optional[int] = None
    control_new_target_count: Optional[int] = None

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

                # CB7 (readiness review, 2026-09-25 stage-repairs
                # assignment §4): plan §7.1's control run, scoped to the
                # SAME affected segment -- REPORT-ONLY, never read below.
                control_results = echo.classify_commands(
                    evidence, full_goto_context, shift_s=-echo.CONTROL_SHIFT_S,
                    leg_turn_on_state_index=leg_turn_on_map)
                control_counts = echo.count_labels(control_results, seg_global_indices)
                control_genuine_echo_count = control_counts.genuine_echo
                control_new_target_count = (
                    sum(control_counts.as_dict().values()) - control_counts.carry)

    metrics = None
    if place_leg is not None and place_lr is not None:
        metrics = compute_place_route_metrics(evidence, place_leg, place_lr, affected, scene)

    reasons = list(fail_reasons) + drift_reasons + lead_in_reasons
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

    if duplicate_inside_leg:
        for name, dups in duplicate_inside_leg.items():
            reasons.append(f"{name}: duplicate sim_step state(s) inside this leg: {dups}")

    verdict: str
    if truncated_inside_leg or duplicate_inside_leg:
        verdict = VERDICT_EVIDENCE_INCOMPLETE
    elif arm == "B":
        if genuine > 0:
            reasons.append(f"{genuine} genuine echo(es) in the affected segment")
        if segment_indeterminate:
            reasons.append("affected segment is indeterminate or missing (B)")
        if (any_pathcheck_fail or genuine > 0 or any_hold_target_drift or any_lead_in_violation
                or segment_indeterminate or gate_failed):
            verdict = VERDICT_STOP
        elif gate_incomplete:
            verdict = VERDICT_EVIDENCE_INCOMPLETE
        else:
            verdict = VERDICT_OK
    else:
        # MB1 (merge verdict, 2026-09-25 stage-repairs assignment §3):
        # a missing or failed gate is non-progressing for BOTH arms.
        # The old code checked only gate_failed here, so a missing
        # start_variant (or provenance/compliance never supplied) on an
        # A cycle with genuine echoes silently authorized as
        # `manipulated` (P1/P1b) instead of reporting the incomplete
        # evidence.
        if gate_failed:
            verdict = VERDICT_STOP
        elif gate_incomplete:
            verdict = VERDICT_EVIDENCE_INCOMPLETE
        elif segment_indeterminate or genuine == 0:
            verdict = VERDICT_INCONCLUSIVE_BASELINE
        else:
            verdict = VERDICT_MANIPULATED

    cv = CycleVerdict(
        cycle, arm, verdict, reasons, genuine, segment_indeterminate,
        metrics.net_shoulder_pitch_hold_deg if metrics else None, metrics, checks,
        unplaceable_command_indices=list(evidence.unplaceable_command_indices),
        duplicate_state_indices=list(evidence.states.duplicate_indices),
        tools_sha256=_package_sha256(),
        control_genuine_echo_count=control_genuine_echo_count,
        control_new_target_count=control_new_target_count,
        per_waypoint={lr.name: lr.per_waypoint for lr in leg_results.values()})
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


def _check_no_unplaceable_in_legs(evidence: ev.Evidence, legs: Sequence[LegSpec]) -> None:
    """MB5: for each leg, checks every raw ``joint_command`` global index
    from the leg's own first PLACEABLE command up to (but not including)
    the next leg's own first command, the next ``reset`` row, or the end
    of ``commands.jsonl`` -- whichever comes first. Raises
    ``ev.EvidenceError`` (mapped to rc 3 by the caller) naming every
    unplaceable index found in that range."""
    starts = sorted(leg.command_indices[0] for leg in legs if leg.command_indices)
    for leg in legs:
        if not leg.command_indices:
            continue
        start = leg.command_indices[0]
        later_starts = [s for s in starts if s > start]
        end = min(later_starts) if later_starts else len(evidence.commands)
        for i in range(start, end):
            if evidence.commands.kind[i] == "reset":
                end = i
                break
        ev.check_no_unplaceable_in_range(evidence, list(range(start, end)))


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
    # MB2 (merge verdict, 2026-09-25 stage-repairs assignment §3): these
    # are OPERATOR inputs, never derived from the document under test.
    # Not `required=True` at the argparse level -- --validation-mode
    # legitimately has none of this (the gates do not exist for that
    # session) -- enforced below, only when not in validation mode.
    p.add_argument("--required-supervisor-programs", nargs="+")
    p.add_argument("--expected-bridge-sha-a")
    p.add_argument("--expected-bridge-sha-b")
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

        # MB2b (merge verdict, 2026-09-25 stage-repairs assignment §3;
        # CB3): the CONTROL DIR carries a cycle manifest
        # (cycle_<id>.json: rep, cycle, and the setup/flight sidecar
        # FILENAMES as the linker actually wrote them -- never the
        # guessed <cycle>-{setup,flight}.link.json pattern, which does
        # not match link_e1_flight.py's real naming,
        # route_clearance_<ROUTE>_<ts>.link.json). Opt-in: a caller with
        # no manifest (every existing synthetic fixture; --validation-
        # mode's own V1 sessions, which have no linker manifest either)
        # keeps the old guessed-name/--setup-sidecar/--flight-sidecar
        # behaviour, unchanged.
        manifest_path = control_dir / f"cycle_{args.cycle}.json"
        manifest = None
        if manifest_path.is_file():
            try:
                manifest = _json.loads(manifest_path.read_text())
            except _json.JSONDecodeError as exc:
                raise CycleInputError(f"malformed cycle manifest {manifest_path}: {exc}") from exc
            if manifest.get("rep") != args.rep:
                raise CycleInputError(
                    f"cycle manifest rep {manifest.get('rep')!r} != --rep {args.rep!r}")
            if manifest.get("cycle") != args.cycle:
                raise CycleInputError(
                    f"cycle manifest cycle {manifest.get('cycle')!r} != --cycle {args.cycle!r}")

        run_dir = str(Path(args.ev_dir).resolve())
        if manifest is not None:
            setup_sidecar = args.setup_sidecar or str(control_dir / manifest["setup_sidecar"])
            flight_sidecar = args.flight_sidecar or str(control_dir / manifest["flight_sidecar"])
            setup_leg = resolve_leg(evd, setup_sidecar, "setup", R.PLACE_ROUTE, R.CRITICAL_JOINTS,
                                     expected_route_name="PLACE_ROUTE", expected_server_run_dir=run_dir)
            flight_leg = resolve_leg(evd, flight_sidecar, "flight", R.LIFT_TO_PRESENT, R._PRESENT_GUARD,
                                      expected_route_name="LIFT_TO_PRESENT", expected_server_run_dir=run_dir)
        else:
            setup_sidecar = args.setup_sidecar or str(control_dir / f"{args.cycle}-setup.link.json")
            flight_sidecar = args.flight_sidecar or str(control_dir / f"{args.cycle}-flight.link.json")
            setup_leg = resolve_leg(evd, setup_sidecar, "setup", R.PLACE_ROUTE, R.CRITICAL_JOINTS)
            flight_leg = resolve_leg(evd, flight_sidecar, "flight", R.LIFT_TO_PRESENT, R._PRESENT_GUARD)
        overlap = set(setup_leg.command_indices) & set(flight_leg.command_indices)
        if overlap:
            raise CycleInputError(f"setup and flight legs overlap at command indices {sorted(overlap)}")

        if manifest is not None:
            # "both legs lie in ONE epoch" -- MB2b's own wording. The
            # rep <-> epoch correspondence itself has no source in the
            # plan, report, or linker code found during this stage
            # (grepped scripts/e1_stage1/*.py and link_e1_flight.py for
            # any "rep"-to-"epoch" mapping; none exists) -- per the
            # instruction not to invent a definition, that half of this
            # item is NOT gated here and is an open question in the
            # handoff, not silently assumed. The "one epoch" part alone
            # (which IS fully determined by the evidence itself) is
            # checked.
            setup_epoch = int(evd.commands.epoch[setup_leg.command_indices[0]])
            flight_epoch = int(evd.commands.epoch[flight_leg.command_indices[0]])
            if setup_epoch != flight_epoch:
                raise CycleInputError(
                    f"setup leg (epoch {setup_epoch}) and flight leg (epoch {flight_epoch}) "
                    "do not lie in the same epoch")

            # "no leg or epoch is claimed by two cycles" -- a claims
            # ledger shared across cycle CLI invocations against the
            # SAME control_dir (each invocation only sees its own
            # cycle otherwise).
            claims_path = control_dir / "_epoch_claims.json"
            claims = _json.loads(claims_path.read_text()) if claims_path.is_file() else {}
            claim_key = str(setup_epoch)
            existing = claims.get(claim_key)
            if existing is not None and existing != args.cycle:
                raise CycleInputError(
                    f"epoch {setup_epoch} is already claimed by cycle {existing!r}, "
                    f"not {args.cycle!r}")
            claims[claim_key] = args.cycle
            claims_path.write_text(_json.dumps(claims))

        # MB5 (merge verdict, 2026-09-25 stage-repairs assignment §3):
        # commands_in_leg (resolve_leg's own building block) SKIPS
        # unplaceable commands when assembling leg.command_indices --
        # so an unplaceable joint_command trailing a leg (P7: the
        # flight leg's own last 10 commands) was never checked against
        # ANY range at all, and the CLI never called
        # check_no_unplaceable_in_range (only pathcheck/echo's OWN CLIs
        # did). Checked here from each leg's own first PLACEABLE
        # command's GLOBAL index to the next leg's own start, or the
        # next reset row, or the end of commands.jsonl -- "from the
        # leg's first command to the next leg or reset in that epoch".
        _check_no_unplaceable_in_legs(evd, [setup_leg, flight_leg])

        # CB1 (merge verdict, 2026-09-25 stage-repairs assignment §4):
        # the setup sidecar's own "scene" block (present -- real
        # link_e1_flight.py sidecars carry it; absent -- V1's own
        # harness never wired one, and every synthetic fixture in this
        # test suite) decides whether a scene is used at all. A scene
        # block that IS present but names a missing file or a chain hash
        # that no longer matches is CycleInputError (rc 3) -- never
        # silently dropped to "no scene". No block at all is scene=None,
        # unchanged from before (compute_place_route_metrics's own
        # open_questions branch).
        scene = None
        setup_sidecar_doc = _json.loads(Path(setup_sidecar).read_text())
        if setup_sidecar_doc.get("scene"):
            scene = load_verified_scene(setup_sidecar)

        provenance_gate = compliance_gate = start_variant_result = None
        guard_note = (
            "report §4 states which guard set applies to which route (CRITICAL_JOINTS for "
            "PLACE_ROUTE, _PRESENT_GUARD for LIFT_TO_PRESENT) but never specifies how a guard "
            "enters C0-C8 mechanically; per the T4 instruction, no use was invented. guard is "
            "recorded here for the record and does not gate any check.")

        if not args.validation_mode:
            # MB2: every one of these is an OPERATOR input, required and
            # never derived from the document under test. A missing
            # operator input is the caller's own configuration error --
            # CycleInputError (rc 3), not a "failed" gate (which would
            # imply the DATA disagreed with a correctly-supplied check).
            if not args.expected_host_sha:
                raise CycleInputError("--expected-host-sha is required and must be non-empty")
            if not args.required_supervisor_programs:
                raise CycleInputError("--required-supervisor-programs is required and must be non-empty")
            if not args.arm_map:
                raise CycleInputError("--arm-map is required")
            if not args.expected_bridge_sha_a or not args.expected_bridge_sha_b:
                raise CycleInputError(
                    "--expected-bridge-sha-a and --expected-bridge-sha-b are both required")

            arm_map_doc = _json.loads(Path(args.arm_map).read_text())
            arm_map_entries = [pv.ArmMapEntry(**e) for e in arm_map_doc]
            arm_map = {e.rep: e for e in arm_map_entries}
            map_check = pv.validate_arm_map(
                arm_map_entries,
                expected_bridge_sha={"A": args.expected_bridge_sha_a, "B": args.expected_bridge_sha_b})
            entry = arm_map.get(args.rep)
            if not map_check.ok:
                provenance_gate = pv.GateResult(False, f"arm_map: {map_check.reason}")
            elif entry is None:
                provenance_gate = pv.GateResult(False, f"rep {args.rep} not in arm_map")
            elif entry.arm != args.arm:
                # MB2 P5/P5b: the arm is DERIVED from arm_map[rep]; a
                # --arm that contradicts it is rejected, never silently
                # evaluated under the operator's own (wrong) label.
                provenance_gate = pv.GateResult(
                    False, f"--arm {args.arm!r} contradicts arm_map[{args.rep}].arm {entry.arm!r}")
            else:
                versions_path = control_dir / f"versions_{args.cycle}.json"
                if not versions_path.is_file():
                    provenance_gate = pv.GateResult(False, f"missing {versions_path}")
                elif "recreate_timestamp" not in _json.loads(versions_path.read_text()):
                    provenance_gate = pv.GateResult(
                        False, f"{versions_path}: missing recreate_timestamp")
                else:
                    doc = _json.loads(versions_path.read_text())
                    # MB2 P5c: versions.bridge_arm/bridge_sha must equal
                    # the arm_map entry -- never read before, so a
                    # contradicting document passed silently.
                    if doc.get("bridge_arm") != entry.arm or doc.get("bridge_sha") != entry.bridge_sha:
                        provenance_gate = pv.GateResult(
                            False,
                            f"versions.bridge_arm/bridge_sha "
                            f"({doc.get('bridge_arm')!r}/{doc.get('bridge_sha')!r}) != "
                            f"arm_map[{args.rep}] ({entry.arm!r}/{entry.bridge_sha!r})")
                    else:
                        observed = pv.ObservedCycle(
                            running_image_id=doc.get("running_image_id", ""),
                            opt_hashes=doc.get("opt_hashes", {}),
                            supervisor_start_times=doc.get("supervisor_start_times", {}),
                            recreate_timestamp=doc.get("recreate_timestamp", 0.0),
                            host_git_sha=doc.get("host_native_kernel_sha", ""),
                            host_tree_dirty=bool(doc.get("host_tree_dirty", True)))
                        provenance_gate = pv.check_cycle(
                            args.rep, arm_map, observed,
                            expected_host_sha=args.expected_host_sha,
                            required_supervisor_programs=args.required_supervisor_programs)

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
            args.cycle, args.arm, evd, [setup_leg, flight_leg], scene=scene,
            provenance_gate=provenance_gate, compliance_gate=compliance_gate,
            start_variant_gate_result=start_variant_result, guard_note=guard_note,
            skip_gates=args.validation_mode)
        if args.validation_mode:
            cv.validation_only = True
    except (ev.EvidenceError, IntegrityError, initial.StartVariantGateUnavailable) as exc:
        return write_result(args.out, RC_INCONCLUSIVE, {"ok": False, "reason": str(exc)})
    except Exception as exc:
        # MB6 (merge verdict, 2026-09-25 stage-repairs assignment §3):
        # a catch-all around the whole CLI body -- invariant 1 ("a
        # Python exception must never escape a CLI") held only for the
        # specific, anticipated exceptions above. A malformed arm_map/
        # versions document (JSONDecodeError, a missing dict key ->
        # KeyError/TypeError, probes P6a-c) escaped as an untranslated
        # traceback with no JSON written at all. The specific catches
        # above are kept for their own messages; this is the fallback,
        # never the first choice.
        return write_result(args.out, RC_INCONCLUSIVE,
                             {"ok": False, "reason": f"{type(exc).__name__}: {exc}"})

    return write_result(args.out, cv.rc(), cv.as_dict())


def main() -> int:
    return _cli()


if __name__ == "__main__":
    raise SystemExit(main())
