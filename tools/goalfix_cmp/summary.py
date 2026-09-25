"""Cycle aggregation: the checkpoint and the between-cycle summary (plan §5
step 9, §7.5-§7.7, D5).

§7.5's outcomes are evaluated only on VALID cycles (never a STOPped or
evidence-incomplete one), with A's median taken only from cycles this
package's own §7.6 rule calls ``manipulated`` -- an inconclusive-baseline A
cycle contributes no data point, it does not count as a zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from tools.goalfix_cmp import cycle as cyc
from tools.goalfix_cmp._io import RC_INCONCLUSIVE, RC_OK, RC_STOP

OUTCOME_SUPPORTS = "supports_echo_to_offset"
OUTCOME_REFUTES = "refutes"
OUTCOME_OTHERWISE_INCONCLUSIVE = "otherwise_inconclusive"
OUTCOME_INCONCLUSIVE_COMPARISON = "inconclusive_comparison"
OUTCOME_STOPPED = "stopped"

#: §7.5's own reference values.
B_LEG_START_ERROR_MAX_DEG = 1.0
B_DELTA_CMD_MAX_CM = 0.05
B_WRIST_BALL_DROP_MIN_CM = 0.5
REFUTE_LEG_START_ERROR_DEG = 2.0
REFUTE_POST_ARRIVAL_RISE_DEG = 1.5


@dataclass
class CycleMetrics:
    """Per-cycle numbers §7.5 needs, beyond the pass/fail verdict itself.
    Optional: a cycle whose clearance/hold analysis was not run still gets
    a verdict, just not a §7.5 contribution (reported, not guessed at)."""
    leg_start_shoulder_pitch_error_deg: Optional[float] = None
    delta_cmd_max_cm: Optional[float] = None
    wrist_ball_delta_cm: Optional[float] = None
    post_arrival_rise_deg: Optional[float] = None
    c5_c6_pass: bool = True


@dataclass
class CheckpointResult:
    n_cycles: int
    any_stop: bool
    a_inconclusive_count: int
    hold_idle: bool
    reason: str = ""


def checkpoint_after(verdicts: Sequence[cyc.CycleVerdict], n: int = 4) -> CheckpointResult:
    first_n = list(verdicts[:n])
    any_stop = any(v.verdict == cyc.VERDICT_STOP for v in first_n)
    a_first_n = [v for v in first_n if v.arm == "A"]
    a_inconclusive = sum(1 for v in a_first_n if v.verdict == cyc.VERDICT_INCONCLUSIVE_BASELINE)
    # "If both A cycles so far are inconclusive, hold the server idle and
    # ask the owner" (plan §11 D5 / checkpoint text).
    hold_idle = len(a_first_n) >= 2 and a_inconclusive == len(a_first_n)
    reason = ("both A cycles so far are inconclusive baseline" if hold_idle else
              ("a B cycle STOPped" if any_stop else ""))
    return CheckpointResult(len(first_n), any_stop, a_inconclusive, hold_idle, reason)


@dataclass
class SummaryResult:
    n_cycles: int
    outcome: str
    detail: str
    b_median_wrist_ball_cm: Optional[float] = None
    a_manipulated_median_wrist_ball_cm: Optional[float] = None

    def rc(self) -> int:
        if self.outcome == OUTCOME_STOPPED:
            return RC_STOP
        if self.outcome == OUTCOME_INCONCLUSIVE_COMPARISON:
            return RC_INCONCLUSIVE
        return RC_OK

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_cycles": self.n_cycles, "outcome": self.outcome, "detail": self.detail,
            "b_median_wrist_ball_cm": self.b_median_wrist_ball_cm,
            "a_manipulated_median_wrist_ball_cm": self.a_manipulated_median_wrist_ball_cm,
            "rc": self.rc(),
        }


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def aggregate(
    verdicts: Sequence[cyc.CycleVerdict], metrics: Dict[str, CycleMetrics],
    *, a_target_count: int = 6, a_manipulated_min: int = 4,
) -> SummaryResult:
    """``metrics``: {cycle: CycleMetrics}, aligned by ``CycleVerdict.cycle``.
    Per plan §7.6: a B STOP ends evaluation immediately (incomplete and
    unbalanced, no §7.5 verdict). Otherwise, fewer than
    ``a_manipulated_min`` of ``a_target_count`` A cycles being manipulated
    makes the relative (A-vs-B) part of §7.5 "inconclusive comparison";
    B's absolute criteria are still reported."""
    stop = next((v for v in verdicts if v.verdict == cyc.VERDICT_STOP), None)
    if stop is not None:
        return SummaryResult(len(verdicts), OUTCOME_STOPPED,
                              f"B cycle {stop.cycle} STOPped: {'; '.join(stop.reasons)}")

    a_cycles = [v for v in verdicts if v.arm == "A"]
    b_cycles = [v for v in verdicts if v.arm == "B"]
    manipulated_a = [v for v in a_cycles if v.verdict == cyc.VERDICT_MANIPULATED]

    a_wrist_ball = [metrics[v.cycle].wrist_ball_delta_cm for v in manipulated_a
                    if v.cycle in metrics and metrics[v.cycle].wrist_ball_delta_cm is not None]
    b_wrist_ball = [metrics[v.cycle].wrist_ball_delta_cm for v in b_cycles
                    if v.cycle in metrics and metrics[v.cycle].wrist_ball_delta_cm is not None]
    b_leg_start_err = [metrics[v.cycle].leg_start_shoulder_pitch_error_deg for v in b_cycles
                       if v.cycle in metrics
                       and metrics[v.cycle].leg_start_shoulder_pitch_error_deg is not None]
    b_delta_cmd = [metrics[v.cycle].delta_cmd_max_cm for v in b_cycles
                   if v.cycle in metrics and metrics[v.cycle].delta_cmd_max_cm is not None]

    a_median = _median(a_wrist_ball)
    b_median = _median(b_wrist_ball)

    inconclusive_comparison = len(a_cycles) >= a_target_count and len(manipulated_a) < a_manipulated_min

    b_supports = (
        bool(b_leg_start_err) and all(e <= B_LEG_START_ERROR_MAX_DEG for e in b_leg_start_err)
        and bool(b_delta_cmd) and all(d <= B_DELTA_CMD_MAX_CM for d in b_delta_cmd)
        and (a_median is not None and b_median is not None
             and (a_median - b_median) >= B_WRIST_BALL_DROP_MIN_CM))

    b_refutes = (
        (b_leg_start_err and max(b_leg_start_err) >= REFUTE_LEG_START_ERROR_DEG)
        or any(m.post_arrival_rise_deg is not None
               and m.post_arrival_rise_deg >= REFUTE_POST_ARRIVAL_RISE_DEG
               and m.c5_c6_pass for m in (metrics.get(v.cycle) for v in b_cycles) if m))

    if inconclusive_comparison:
        outcome, detail = (OUTCOME_INCONCLUSIVE_COMPARISON,
                            f"only {len(manipulated_a)}/{len(a_cycles)} A cycles manipulated "
                            f"(need >= {a_manipulated_min}/{a_target_count})")
    elif b_refutes:
        outcome, detail = OUTCOME_REFUTES, "a B cycle shows a large leg-start error or post-arrival rise"
    elif b_supports:
        outcome, detail = OUTCOME_SUPPORTS, "B meets all three §7.5 absolute/relative criteria"
    else:
        outcome, detail = OUTCOME_OTHERWISE_INCONCLUSIVE, "neither the support nor the refute criteria are met"

    return SummaryResult(len(verdicts), outcome, detail, b_median, a_median)


# ---------------------------------------------------------------------------
# §7.7 tripwires
# ---------------------------------------------------------------------------

TRIPWIRE_PATTERNS: Dict[str, str] = {
    "reset_ack_timeout": "reset ack timed out",
    "unexpected_reset_ack": "unexpected reset_ack",
    "reset_sh_mismatch": "ack mismatch",
    "control_held_refusal": "control_held",
    "lease_acquisition": "lease acquisition",
    "pause_message": "pause",
}

#: B is expected 0 on every tripwire; A is expected 0 except
#: "unexpected_reset_ack" (pre-existing, not a stop -- plan §7.7 note).
B_EXPECTED_ZERO = tuple(TRIPWIRE_PATTERNS)
A_EXPECTED_ZERO = tuple(k for k in TRIPWIRE_PATTERNS if k != "unexpected_reset_ack")


def count_tripwires(log_text: str) -> Dict[str, int]:
    lower = log_text.lower()
    return {name: lower.count(pattern.lower()) for name, pattern in TRIPWIRE_PATTERNS.items()}


def check_tripwires(counts: Dict[str, int], arm: str) -> List[str]:
    expected_zero = B_EXPECTED_ZERO if arm == "B" else A_EXPECTED_ZERO
    return [name for name in expected_zero if counts.get(name, 0) != 0]
