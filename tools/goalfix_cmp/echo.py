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
) -> List[Optional[CommandResult]]:
    """One ``CommandResult`` per ``joint_command`` row in ``evidence.commands``
    (``None`` for ``reset`` rows). ``shift_s`` (e.g. ``-CONTROL_SHIFT_S``)
    reruns the same classifier against states shifted in simulation time,
    for the control rate; a shifted window reaching before its epoch's own
    start is reported as unavailable (``None`` joint results, not a false
    "fresh")."""
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

            label = _subclassify(
                name=name, value=value, epoch=epoch, command_index=i,
                is_first_command_of_epoch=(prev is None), ctx=ctx,
                epoch_start_state_index=int(e_state_idx[0]),
                src_global_index=src_global)
            result.joints[name] = JointResult(label, age_s=src_age, source_state_index=src_global)

        prev_target[epoch] = target8

    return out


def _subclassify(*, name: str, value: float, epoch: int, command_index: int,
                  is_first_command_of_epoch: bool, ctx: Optional[GotoContext],
                  epoch_start_state_index: int, src_global_index: int) -> str:
    """An exact match's subclass, per plan §7.1's table."""
    if is_first_command_of_epoch and src_global_index == epoch_start_state_index:
        return START_COINCIDENCE
    if ctx is not None and name in ctx.start8 and name in ctx.goal8:
        a, b = ctx.start8[name], ctx.goal8[name]
        if a != b:
            # Ill-conditioned within 0.05 deg of either end -- excluded from
            # the tau-agreement check (report §4, C2); a match that close
            # to an endpoint is left to the genuine-echo bucket rather than
            # guessed at.
            near_end = (abs(np.degrees(value - a)) <= ILL_CONDITIONED_DEG
                        or abs(np.degrees(value - b)) <= ILL_CONDITIONED_DEG)
            if not near_end:
                tau = implied_tau(a, b, value)
                predicted = pose_at(a, b, tau)
                if _within_float32_ulps(predicted, value, ULP_FLOAT32_FACTOR):
                    return PATH_COINCIDENCE
    return GENUINE_ECHO


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

    def add(self, label: str) -> None:
        setattr(self, label, getattr(self, label) + 1)

    def as_dict(self) -> Dict[str, int]:
        return dict(carry=self.carry, fresh=self.fresh, genuine_echo=self.genuine_echo,
                    start_coincidence=self.start_coincidence,
                    path_coincidence=self.path_coincidence,
                    timing_ambiguous=self.timing_ambiguous,
                    unavailable=self.unavailable)


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
