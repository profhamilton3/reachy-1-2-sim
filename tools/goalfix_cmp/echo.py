"""Echo classification (plan §7.1), extending the verified
``classify()`` from
IITG-Reachy-Project/outputs/analysis-2026-09-23-goal-feedback-verification/verify_goal_feedback.py
(ported with an equivalence test in
tests/unit/test_goalfix_cmp_equivalence.py) with:

* simulation-time lookback, keyed per reset epoch, instead of wall time;
* the ``[t_lo, t_hi]`` bracket's own uncertainty -- a match found only at
  the ``t_hi`` state (produced by/after the command's own application, so
  it can never be a valid echo SOURCE) is ``timing-ambiguous``, not a
  genuine echo;
* two "exact match, but legitimate" subclasses carved out of what the
  original script counted as an echo unconditionally: **start coincidence**
  (the first command of an epoch repeating the pose already in force) and
  **path coincidence** (the value the minimum-jerk trajectory itself would
  produce, at an implied tau that agrees with the command's other moving
  joints within 30 ms of simulation time -- C2);
* a control run, the same classifier against states shifted -20 s of
  simulation time within the same epoch.

Only the right-arm 8 joints are classified (``R_JOINTS`` / ``ARM7`` plus the
gripper), matching plan §7.1's "new right-arm target" scope.
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

from reachy_ai.motion.rig_routes import R_JOINTS  # noqa: E402

from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp._minjerk import implied_tau, pose_at  # noqa: E402

LOOKBACK_S = 0.5
CONTROL_SHIFT_S = 20.0
C2_SKEW_TOL_S = 0.030
ILL_CONDITIONED_DEG = 0.05
ULP_FLOAT32_FACTOR = 1  # matches within `n` float32 ULPs; the plan asks for 1

CARRY = "carry"
FRESH = "fresh"
GENUINE_ECHO = "genuine_echo"
START_COINCIDENCE = "start_coincidence"
PATH_COINCIDENCE = "path_coincidence"
TIMING_AMBIGUOUS = "timing_ambiguous"

EXACT_MATCH_CLASSES = frozenset(
    {GENUINE_ECHO, START_COINCIDENCE, PATH_COINCIDENCE, TIMING_AMBIGUOUS})


@dataclass(frozen=True)
class GotoContext:
    """What ``echo.py`` needs from ``segments.py`` to test path coincidence
    for one command: the current goto's start/goal pose (right-arm joint
    name -> radians) and its commanded duration."""
    start8: Dict[str, float]
    goal8: Dict[str, float]
    seconds: float


@dataclass
class JointResult:
    label: str
    age_s: Optional[float] = None          # genuine_echo / start / path / ambiguous
    source_state_index: Optional[int] = None
    #: H5/B5: True only for a PATH_COINCIDENCE with no other moving,
    #: non-near-end, non-carry joint in the same command to compare tau
    #: agreement against -- the rule is vacuous in that case (on-path
    #: alone decides it), reported here rather than silently folded in.
    tau_agreement_vacuous: bool = False


@dataclass
class CommandResult:
    command_index: int
    joints: Dict[str, JointResult] = field(default_factory=dict)  # keyed by R_JOINTS name


def _f32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32).astype(np.float64)


def _epoch_state_slice(states: ev.States, epoch: int):
    idx = np.nonzero(states.epoch == epoch)[0]
    return idx, states.sim_time_s[idx], _f32(states.right_arm_position_rad()[idx])


def classify_commands(
    evidence: ev.Evidence,
    goto_context: Optional[Sequence[Optional[GotoContext]]] = None,
    shift_s: float = 0.0,
    leg_turn_on_state_index: Optional[Dict[int, int]] = None,
) -> List[Optional[CommandResult]]:
    """One ``CommandResult`` per ``joint_command`` row in ``evidence.commands``
    (``None`` for ``reset`` rows). ``shift_s`` (e.g. ``-CONTROL_SHIFT_S``)
    reruns the same classifier against states shifted in simulation time,
    for the control rate; a shifted window reaching before its epoch's own
    start is reported as unavailable (``None`` joint results, not a false
    "fresh").

    ``leg_turn_on_state_index`` (T7/plan §7.1/review §3.3): maps each LEG's
    own first command's GLOBAL index to the GLOBAL state index of the
    present-position reading in force at that leg's own ``turn_on`` --
    "the goal set at turn_on" the plan's rule refers to. There is one such
    pair per leg, and a cycle's second leg (``LIFT_TO_PRESENT``, in the
    same reset epoch as its setup leg) has one just as much as the first.
    An exact match on a leg's own first command is START_COINCIDENCE only
    when its source is THIS specific state -- not any state the general
    0.5s lookback happens to turn up, which a fixture can otherwise plant
    to look like a genuine echo or a path coincidence on the very first
    command of a file (see the equivalence/boundary/path-coincidence
    tests, none of which involve a real turn_on at all).

    Omitted, this falls back to "the epoch's absolute first command,
    checked against the epoch's own first state" (the pre-T7 behaviour)
    for backward compatibility with callers that only ever classify a
    single-leg epoch. T4's end-to-end `cycle` CLI is what supplies every
    leg's own pair for a real multi-leg cycle.
    """
    states = evidence.states
    commands = evidence.commands
    goto_context = goto_context or [None] * len(commands)
    out: List[Optional[CommandResult]] = [None] * len(commands)

    epoch_cache: Dict[int, tuple] = {}
    prev_target: Dict[int, np.ndarray] = {}  # per-epoch previous command's 8-vector

    for i in range(len(commands)):
        if commands.kind[i] != "joint_command":
            continue
        epoch = int(commands.epoch[i])
        bracket = evidence.brackets[i]
        result = CommandResult(i)
        out[i] = result
        target8 = commands.target_rad[i, :8]

        if epoch not in epoch_cache:
            epoch_cache[epoch] = _epoch_state_slice(states, epoch)
        e_state_idx, e_sim_time, e_pos32 = epoch_cache[epoch]
        epoch_start_s = float(e_sim_time[0]) if len(e_sim_time) else 0.0

        prev = prev_target.get(epoch)
        ctx = goto_context[i] if i < len(goto_context) else None

        # H5/B5 (owner-approved plan §7.1; coordinator Stage B
        # authorization): the tau-agreement comparison this command's own
        # path-coincidence subclassification needs -- one implied tau per
        # OTHER moving, non-near-end, non-carry joint of THIS command,
        # computed once here (not per candidate joint) so `_subclassify`
        # can compare against every one of them.
        moving_taus: Dict[str, float] = {}
        if ctx is not None:
            for jn, jname in enumerate(R_JOINTS):
                if prev is not None and target8[jn] == prev[jn]:
                    continue  # R-carry: not a setpoint of this goto
                if jname not in ctx.start8 or jname not in ctx.goal8:
                    continue
                ja, jb = ctx.start8[jname], ctx.goal8[jname]
                if ja == jb:
                    continue
                jv = target8[jn]
                if _is_near_end(jv, ja, jb):
                    continue
                moving_taus[jname] = implied_tau(ja, jb, jv)

        if bracket.unplaceable:
            for j, name in enumerate(R_JOINTS):
                is_carry = prev is not None and target8[j] == prev[j]
                result.joints[name] = JointResult(CARRY if is_carry else FRESH)
            prev_target[epoch] = target8
            continue

        t_hi = bracket.t_hi + shift_s
        lo_bound = t_hi - LOOKBACK_S
        unavailable = shift_s != 0.0 and lo_bound < epoch_start_s - 1e-9
        a = int(np.searchsorted(e_sim_time, lo_bound))
        # Unambiguous window: every state strictly before the (possibly
        # shifted) reference instant `t_hi`. The state AT `t_hi` itself is
        # excluded here and checked separately (`hi_match`, below) because
        # -- for the unshifted, real bracket -- that state was produced
        # by/after the command's own application and can never be a valid
        # echo SOURCE (report §4's "matching against the t_hi state" note).
        hi_idx = int(np.searchsorted(e_sim_time, t_hi))
        b_unambig = hi_idx

        for j, name in enumerate(R_JOINTS):
            value = target8[j]
            is_carry = prev is not None and value == prev[j]
            if is_carry:
                result.joints[name] = JointResult(CARRY)
                continue

            if unavailable:
                result.joints[name] = JointResult("unavailable")
                continue

            win_pos = e_pos32[a:b_unambig, j] if b_unambig > a else e_pos32[0:0, j]
            win_time = e_sim_time[a:b_unambig]
            hits = np.nonzero(win_pos == value)[0]

            hi_match = (0 <= hi_idx < len(e_sim_time)
                        and e_pos32[hi_idx, j] == value)

            if len(hits) == 0 and not hi_match:
                result.joints[name] = JointResult(FRESH)
                continue

            if len(hits) == 0 and hi_match:
                result.joints[name] = JointResult(
                    TIMING_AMBIGUOUS, age_s=float(t_hi - e_sim_time[hi_idx]),
                    source_state_index=int(e_state_idx[hi_idx]))
                continue

            src_local = int(hits[-1])  # most recent match in the unambiguous window
            src_age = float(t_hi - win_time[src_local])
            src_global = int(e_state_idx[a + src_local])

            if leg_turn_on_state_index is None:
                turn_on_state_idx = int(e_state_idx[0]) if prev is None else None
            else:
                turn_on_state_idx = leg_turn_on_state_index.get(i)
            label, vacuous = _subclassify(
                name=name, value=value, epoch=epoch, command_index=i,
                turn_on_state_index=turn_on_state_idx, ctx=ctx,
                src_global_index=src_global, moving_taus=moving_taus)
            result.joints[name] = JointResult(label, age_s=src_age, source_state_index=src_global,
                                               tau_agreement_vacuous=vacuous)

        prev_target[epoch] = target8

    return out


def _is_near_end(value: float, a: float, b: float, tol_deg: float = ILL_CONDITIONED_DEG) -> bool:
    return (abs(np.degrees(value - a)) <= tol_deg or abs(np.degrees(value - b)) <= tol_deg)


def _subclassify(*, name: str, value: float, epoch: int, command_index: int,
                  turn_on_state_index: Optional[int], ctx: Optional[GotoContext],
                  src_global_index: int,
                  moving_taus: Optional[Dict[str, float]] = None) -> str:
    """An exact match's subclass, per plan §7.1's table.

    T7 (review §3.3): the plan's own rule is "the first setpoint after
    turn_on equals the goal set at turn_on" -- an exact match on a leg's
    own first command, whose source is specifically the present-position
    reading in force AT that leg's turn_on (turn_on itself sends no
    joint_command; the match is against the readback that was already
    streaming continuously right up to and through it), never any other
    state the lookback happens to turn up. The pre-T7 rule pinned that
    reference to the EPOCH's absolute first state unconditionally, which a
    leg starting after any settle -- let alone a cycle's SECOND leg --
    can never satisfy; E8 is exactly this.

    H5/B5 (V1-a; owner-approved plan §7.1; coordinator Stage B
    authorization): the non-near-end branch used to accept any value that
    round-trips through ``implied_tau``/``pose_at`` for SOME tau in
    [start, goal] -- true for every value in range, so it never actually
    tested plan §7.1's own rule ("at an implied tau that agrees with the
    command's other moving joints within 30 ms (C2)"). Now: on-path AND
    that tau agrees, within 30 ms of SIMULATION TIME (using this goto's
    own ``seconds``), with EVERY OTHER entry in ``moving_taus`` (every
    other moving, non-near-end, non-carry joint of this SAME command). No
    such other joint -> the rule is vacuous: still path coincidence (the
    on-path test alone), reported via ``JointResult.tau_agreement_vacuous``.
    The near-end exemption (Q-echo) is unchanged."""
    if turn_on_state_index is not None and src_global_index == turn_on_state_index:
        return START_COINCIDENCE, False
    if ctx is not None and name in ctx.start8 and name in ctx.goal8:
        a, b = ctx.start8[name], ctx.goal8[name]
        if a != b:
            if _is_near_end(value, a, b):
                # Q-echo (coordinator ruling, 2026-09-25 stage-1 rulings,
                # §3): plan §7.1 defines path coincidence as the
                # minimum-jerk setpoint on THIS goto's own path, "at an
                # implied tau that agrees ... (C2)" -- and C2 exempts a
                # near-end joint from that tau-agreement test, it does
                # not disqualify the joint from being on-path. A match
                # near an end that lies within [start, goal] of the
                # CURRENT goto is a path coincidence (a direction
                # reversal's early minimum-jerk samples pass back through
                # values the lagging plant held moments earlier -- V1's
                # F7, cycle 2 setup, r_shoulder_roll). Off-path near an
                # end (never observed on real B data, but not excluded by
                # construction) is left to genuine_echo -- the exemption
                # is from the tau check, never a blanket "near an end is
                # safe" rule.
                lo, hi = (a, b) if a <= b else (b, a)
                if lo <= value <= hi:
                    return PATH_COINCIDENCE, False
            else:
                tau = implied_tau(a, b, value)
                predicted = pose_at(a, b, tau)
                on_path = _within_float32_ulps(predicted, value, ULP_FLOAT32_FACTOR)
                if on_path:
                    others = {jn: jt for jn, jt in (moving_taus or {}).items() if jn != name}
                    if not others:
                        return PATH_COINCIDENCE, True  # vacuous: no other joint to compare
                    seconds = ctx.seconds
                    agrees = all(
                        abs(tau - other_tau) * seconds <= C2_SKEW_TOL_S + 1e-9
                        for other_tau in others.values())
                    if agrees:
                        return PATH_COINCIDENCE, False
    return GENUINE_ECHO, False


def _within_float32_ulps(a: float, b: float, n: int) -> bool:
    fa, fb = np.float32(a), np.float32(b)
    if fa == fb:
        return True
    ia = fa.view(np.int32)
    ib = fb.view(np.int32)
    return abs(int(ia) - int(ib)) <= n


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class Counts:
    carry: int = 0
    fresh: int = 0
    genuine_echo: int = 0
    start_coincidence: int = 0
    path_coincidence: int = 0
    timing_ambiguous: int = 0
    unavailable: int = 0
    #: H5/B5: of `path_coincidence`, how many had no OTHER moving,
    #: non-near-end, non-carry joint in the same command to check tau
    #: agreement against (the rule was vacuous -- on-path alone decided
    #: it). Counted and reported, never silently folded into the total.
    path_coincidence_vacuous: int = 0

    def add(self, label: str) -> None:
        setattr(self, label, getattr(self, label) + 1)

    def as_dict(self) -> Dict[str, int]:
        return dict(carry=self.carry, fresh=self.fresh, genuine_echo=self.genuine_echo,
                    start_coincidence=self.start_coincidence,
                    path_coincidence=self.path_coincidence,
                    timing_ambiguous=self.timing_ambiguous,
                    unavailable=self.unavailable,
                    path_coincidence_vacuous=self.path_coincidence_vacuous)


def count_labels(results: Sequence[Optional[CommandResult]],
                  command_indices: Optional[Sequence[int]] = None) -> Counts:
    counts = Counts()
    indices = command_indices if command_indices is not None else range(len(results))
    for i in indices:
        r = results[i]
        if r is None:
            continue
        for jr in r.joints.values():
            counts.add(jr.label)
            if jr.label == PATH_COINCIDENCE and jr.tau_agreement_vacuous:
                counts.path_coincidence_vacuous += 1
    return counts


def echo_age_distribution_s(results: Sequence[Optional[CommandResult]]) -> List[float]:
    ages = []
    for r in results:
        if r is None:
            continue
        for jr in r.joints.values():
            if jr.label == GENUINE_ECHO and jr.age_s is not None:
                ages.append(jr.age_s)
    return ages


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from tools.goalfix_cmp._io import (
        IntegrityError, RC_INCONCLUSIVE, RC_OK, RC_STOP, write_result,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-dir", required=True)
    p.add_argument("--states", default="states.jsonl")
    p.add_argument("--commands", default="commands.jsonl")
    p.add_argument("--sha256sums", default="SHA256SUMS")
    p.add_argument("--arm", choices=["A", "B"], required=True)
    p.add_argument("--segment-start-index", type=int, default=None,
                    help="first joint_command index of the affected segment "
                         "(segments.py); omit to classify the whole leg")
    p.add_argument("--segment-end-index", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    try:
        evd = ev.verify_and_load(args.evidence_dir, args.states, args.commands, args.sha256sums)
    except (ev.EvidenceError, IntegrityError) as exc:
        return write_result(args.out, RC_INCONCLUSIVE, {"ok": False, "reason": str(exc)})

    seg_idx = None
    if args.segment_start_index is not None and args.segment_end_index is not None:
        seg_idx = list(range(args.segment_start_index, args.segment_end_index + 1))

    # T2: an unplaceable command in the scoped range (the segment, or the
    # whole file if no segment was given) is evidence incomplete -- never
    # silently classified as carry/fresh for gating purposes.
    gate_range = seg_idx if seg_idx is not None else list(
        np.nonzero(evd.commands.joint_command_mask())[0])
    try:
        ev.check_no_unplaceable_in_range(evd, gate_range)
    except ev.EvidenceError as exc:
        return write_result(args.out, RC_INCONCLUSIVE, {"ok": False, "reason": str(exc)})

    results = classify_commands(evd)
    control = classify_commands(evd, shift_s=-CONTROL_SHIFT_S)

    counts = count_labels(results, seg_idx)
    control_counts = count_labels(control, seg_idx)
    scoped_results = results if seg_idx is None else [
        results[i] for i in seg_idx if 0 <= i < len(results)]

    genuine = counts.genuine_echo
    indeterminate_segment = seg_idx is not None and any(
        r is not None and any(jr.label == "unavailable" for jr in r.joints.values())
        for r in scoped_results)

    if args.arm == "B":
        rc = RC_STOP if genuine > 0 else RC_OK
    else:  # arm == "A"
        rc = RC_INCONCLUSIVE if (genuine == 0 or indeterminate_segment) else RC_OK

    payload = {
        "ok": rc == RC_OK,
        "arm": args.arm,
        "counts": counts.as_dict(),
        "control_counts": control_counts.as_dict(),
        "echo_age_s_median": (
            float(np.median(echo_age_distribution_s(scoped_results)))
            if echo_age_distribution_s(scoped_results) else None),
    }
    return write_result(args.out, rc, payload)


def main() -> int:
    return _cli()


if __name__ == "__main__":
    raise SystemExit(main())
