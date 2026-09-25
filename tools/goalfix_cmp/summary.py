"""Cycle aggregation: the checkpoint and the between-cycle summary (plan §5
step 9, §7.5-§7.7, D5; assignment T5, T9).

§7.5's outcomes are evaluated only on VALID cycles (never a STOPped or
evidence-incomplete one), with A's median taken only from cycles this
package's own §7.6 rule calls ``manipulated`` -- an inconclusive-baseline A
cycle contributes no data point, it does not count as a zero.

T5 (review §3.7/M5): a B cycle that is ``evidence_incomplete`` or has any
missing/null §7.5 metric now makes ``aggregate`` rc 3, never ``supports``;
an incomplete session (fewer than ``a_target_count`` cycles per arm, with
no STOP) is reported ``incomplete``, never silently evaluated as if it
were the full session; "inconclusive comparison" is gated purely on
``len(manipulated_a) < a_manipulated_min`` -- the old code additionally
required at least ``a_target_count`` A cycles to already EXIST, which let
an incomplete session report ``supports``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from tools.goalfix_cmp import cycle as cyc
from tools.goalfix_cmp._io import RC_INCONCLUSIVE, RC_OK, RC_STOP

OUTCOME_SUPPORTS = "supports_echo_to_offset"
OUTCOME_REFUTES = "refutes"
OUTCOME_OTHERWISE_INCONCLUSIVE = "otherwise_inconclusive"
OUTCOME_INCONCLUSIVE_COMPARISON = "inconclusive_comparison"
OUTCOME_STOPPED = "stopped"
OUTCOME_INCOMPLETE = "incomplete"

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
    #: T5: a missing cycle, an evidence_incomplete cycle, or (set by the
    #: CLI, which alone can compare tools_sha256) a hash mismatch. The old
    #: checkpoint ignored evidence_incomplete entirely.
    any_incomplete: bool = False
    missing_cycles: int = 0

    def rc(self) -> int:
        if self.any_incomplete:
            return RC_INCONCLUSIVE
        if self.any_stop:
            return RC_STOP
        if self.hold_idle:
            return RC_INCONCLUSIVE
        return RC_OK


def checkpoint_after(
    verdicts: Sequence[cyc.CycleVerdict], n: int = 4, *, hash_mismatch: bool = False,
) -> CheckpointResult:
    first_n = list(verdicts[:n])
    missing = max(0, n - len(first_n))
    any_stop = any(v.verdict == cyc.VERDICT_STOP for v in first_n)
    any_evidence_incomplete = any(v.verdict == cyc.VERDICT_EVIDENCE_INCOMPLETE for v in first_n)
    any_incomplete = missing > 0 or any_evidence_incomplete or hash_mismatch
    a_first_n = [v for v in first_n if v.arm == "A"]
    a_inconclusive = sum(1 for v in a_first_n if v.verdict == cyc.VERDICT_INCONCLUSIVE_BASELINE)
    # "If both A cycles so far are inconclusive, hold the server idle and
    # ask the owner" (plan §11 D5 / checkpoint text).
    hold_idle = len(a_first_n) >= 2 and a_inconclusive == len(a_first_n)
    reason = (f"missing {missing} of {n} cycles" if missing else
              ("hash mismatch against the running tools package" if hash_mismatch else
               ("an evidence_incomplete cycle" if any_evidence_incomplete else
                ("both A cycles so far are inconclusive baseline" if hold_idle else
                 ("a B cycle STOPped" if any_stop else "")))))
    return CheckpointResult(len(first_n), any_stop, a_inconclusive, hold_idle, reason,
                             any_incomplete, missing)


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
        if self.outcome in (OUTCOME_INCONCLUSIVE_COMPARISON, OUTCOME_INCOMPLETE):
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


#: The §7.5 metrics EVERY valid B cycle must carry -- missing/null on any
#: one of these is rc 3, never silently skipped (T5/review §3.7/M5).
_REQUIRED_B_METRICS: tuple = (
    "leg_start_shoulder_pitch_error_deg", "delta_cmd_max_cm", "wrist_ball_delta_cm")


def aggregate(
    verdicts: Sequence[cyc.CycleVerdict], metrics: Dict[str, CycleMetrics],
    *, a_target_count: int = 6, a_manipulated_min: int = 4,
) -> SummaryResult:
    """``metrics``: {cycle: CycleMetrics}, aligned by ``CycleVerdict.cycle``.
    Per plan §7.6: a B STOP ends evaluation immediately (incomplete and
    unbalanced, no §7.5 verdict). An ``evidence_incomplete`` cycle, a
    missing §7.5 metric on any valid B cycle, or a session short of
    ``a_target_count`` cycles per arm all give ``incomplete`` -- also no
    §7.5 verdict, and never ``supports`` (T5). Otherwise, fewer than
    ``a_manipulated_min`` A cycles being manipulated (regardless of how
    many A cycles exist) makes the relative (A-vs-B) part of §7.5
    "inconclusive comparison"; B's absolute criteria are still reported."""
    stop = next((v for v in verdicts if v.verdict == cyc.VERDICT_STOP), None)
    if stop is not None:
        return SummaryResult(len(verdicts), OUTCOME_STOPPED,
                              f"B cycle {stop.cycle} STOPped: {'; '.join(stop.reasons)}")

    incomplete_cycles = [v for v in verdicts if v.verdict == cyc.VERDICT_EVIDENCE_INCOMPLETE]
    if incomplete_cycles:
        names = ", ".join(v.cycle for v in incomplete_cycles)
        return SummaryResult(len(verdicts), OUTCOME_INCOMPLETE,
                              f"evidence_incomplete cycle(s): {names}")

    a_cycles = [v for v in verdicts if v.arm == "A"]
    b_cycles = [v for v in verdicts if v.arm == "B"]
    if len(a_cycles) < a_target_count or len(b_cycles) < a_target_count:
        return SummaryResult(
            len(verdicts), OUTCOME_INCOMPLETE,
            f"only {len(a_cycles)} A / {len(b_cycles)} B cycles (need {a_target_count} each)")

    for v in b_cycles:
        m = metrics.get(v.cycle)
        missing = (list(_REQUIRED_B_METRICS) if m is None else
                   [f for f in _REQUIRED_B_METRICS if getattr(m, f) is None])
        if missing:
            return SummaryResult(
                len(verdicts), OUTCOME_INCOMPLETE,
                f"B cycle {v.cycle} is missing §7.5 metric(s): {missing}")

    manipulated_a = [v for v in a_cycles if v.verdict == cyc.VERDICT_MANIPULATED]

    a_wrist_ball = [metrics[v.cycle].wrist_ball_delta_cm for v in manipulated_a
                    if v.cycle in metrics and metrics[v.cycle].wrist_ball_delta_cm is not None]
    b_wrist_ball = [metrics[v.cycle].wrist_ball_delta_cm for v in b_cycles]
    b_leg_start_err = [metrics[v.cycle].leg_start_shoulder_pitch_error_deg for v in b_cycles]
    b_delta_cmd = [metrics[v.cycle].delta_cmd_max_cm for v in b_cycles]

    a_median = _median(a_wrist_ball)
    b_median = _median(b_wrist_ball)

    inconclusive_comparison = len(manipulated_a) < a_manipulated_min

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
# §7.7 tripwires (T9/review §3.7)
# ---------------------------------------------------------------------------

#: Every pattern here is anchored to a real, cited source line -- never a
#: bare word search. The old "lease acquisition" pattern matched nothing
#: the native server ever logs (it logs "Execution lease granted to …",
#: server.py:1059); "pause" was a bare substring that happened to match
#: unrelated log noise. Both are fixed below. "control_held_refusal" and
#: "pause_message" are kept as NAMED counters (§7.7's own table groups
#: them with lease_acquisition under one row) but their patterns are
#: `None`: nothing in server.py, mujoco_remote_backend.py or
#: reset_watcher.py ever writes either concept to a log line (`code=
#: "control_held"`, server.py:1037, is sent over the wire, never logged;
#: the "pause" websocket message type, server.py:1091-1092, is likewise
#: never logged) -- flagged in the handoff rather than matched against an
#: invented string.
TRIPWIRE_SOURCES: Dict[str, str] = {
    # "Reset ack timed out …" (mujoco_remote_backend.py:333, the bridge)
    # and "… reset ack timed out or was refused …"
    # (reset_watcher.py:154, the watcher) -- one case-insensitive pattern
    # matches both real strings.
    "reset_ack_timeout": "bridge_or_watcher",
    # "Unexpected reset_ack id=%s (expected %s)" (mujoco_remote_backend.py:468).
    "unexpected_reset_ack": "bridge",
    # reset.sh's own STOP prefix (scripts/e1_stage1/reset.sh:50,58,71:
    # `echo "STOP: reset $GEN not verified: …" | tee -a "$P/control/stop"`)
    # -- covers "ack mismatch" (reset_verify.py's own reason string) AND
    # every OTHER STOP reset.sh can emit, per the plan §7.7 row's own
    # wording ("ack mismatch or other STOP"); the old pattern matched only
    # the first.
    "reset_sh_mismatch": "reset_sh",
    # "Execution lease granted to %r (mover %r, %.0fs)" (server.py:1059).
    "lease_acquisition": "native",
    "control_held_refusal": None,   # ungroundable -- see module docstring above
    "pause_message": None,          # ungroundable -- see module docstring above
}

_TRIPWIRE_PATTERN_TEXT: Dict[str, str] = {
    "reset_ack_timeout": "reset ack timed out",
    "unexpected_reset_ack": "unexpected reset_ack",
    "reset_sh_mismatch": "stop: reset",
    "lease_acquisition": "execution lease granted to",
}

#: B is expected 0 on every tripwire; A is expected 0 except
#: "unexpected_reset_ack" (pre-existing, not a stop -- plan §7.7 note).
B_EXPECTED_ZERO = tuple(TRIPWIRE_SOURCES)
A_EXPECTED_ZERO = tuple(k for k in TRIPWIRE_SOURCES if k != "unexpected_reset_ack")

#: Test-drift guard (T9: "add a test that greps each source line") reads
#: these; kept as data, not just prose, so the test can iterate them.
TRIPWIRE_SOURCE_LINES: Dict[str, tuple] = {
    "reset_ack_timeout": (
        ("mujoco_remote_backend.py", 333, "Reset ack timed out after %.1f s"),
        ("reset_watcher.py", 154, "reset ack timed out or was refused")),
    "unexpected_reset_ack": (
        ("mujoco_remote_backend.py", 468, "Unexpected reset_ack id=%s"),),
    "reset_sh_mismatch": (
        ("scripts/e1_stage1/reset.sh", 50, "STOP: reset $GEN not verified"),),
    "lease_acquisition": (
        ("native_mujoco/server.py", 1059, "Execution lease granted to %r"),),
}


def count_tripwires(
    *, bridge_log: Optional[str] = None, watcher_log: Optional[str] = None,
    reset_sh_log: Optional[str] = None, native_log: Optional[str] = None,
) -> Dict[str, Optional[int]]:
    """One count per ``TRIPWIRE_SOURCES`` key, or ``None`` when the source
    log its pattern needs was not supplied (T9: "a missing log file" is
    evidence incomplete, never assumed 0) or when the pattern itself is
    ungroundable (``control_held_refusal``/``pause_message``)."""
    logs = {"bridge": bridge_log, "watcher": watcher_log,
            "reset_sh": reset_sh_log, "native": native_log}
    out: Dict[str, Optional[int]] = {}
    for name, source in TRIPWIRE_SOURCES.items():
        pattern = _TRIPWIRE_PATTERN_TEXT.get(name)
        if pattern is None:
            out[name] = None
            continue
        if source == "bridge_or_watcher":
            texts = [t for t in (bridge_log, watcher_log) if t is not None]
        else:
            texts = [logs[source]] if logs.get(source) is not None else []
        if not texts:
            out[name] = None
            continue
        out[name] = sum(t.lower().count(pattern) for t in texts)
    return out


@dataclass
class TripwireCheck:
    violated: List[str] = field(default_factory=list)     # nonzero, required-zero
    incomplete: List[str] = field(default_factory=list)   # None (missing log / ungroundable)


def check_tripwires(counts: Dict[str, Optional[int]], arm: str) -> TripwireCheck:
    expected_zero = B_EXPECTED_ZERO if arm == "B" else A_EXPECTED_ZERO
    violated, incomplete = [], []
    for name in expected_zero:
        c = counts.get(name)
        if c is None:
            incomplete.append(name)
        elif c != 0:
            violated.append(name)
    return TripwireCheck(violated, incomplete)


# ---------------------------------------------------------------------------
# CLI (T5): reads only between_*.json files, checked against the running
# tools package before anything else is trusted.
# ---------------------------------------------------------------------------

class SummaryCliError(RuntimeError):
    """Anything about the between_*.json files themselves (missing,
    malformed, or hashed against a different tools package) that makes a
    checkpoint/final call un-evaluable. Callers map this to rc=3."""


def _cycle_rep_key(cycle: str) -> int:
    m = re.search(r"r(\d+)$", cycle)
    return int(m.group(1)) if m else 0


def load_between_files(control_dir) -> "tuple[list, dict, bool]":
    """Every ``between_*.json`` in ``control_dir``, ordered by the cycle
    id's own trailing ``r<rep>``. Returns ``(verdicts, metrics,
    hash_mismatch)`` -- ``hash_mismatch`` is True if ANY file's
    ``tools_sha256`` disagrees with the running ``tools/goalfix_cmp``
    package (T5: "checking each file's tools_sha256 against the running
    package"); that file's own verdict/metrics are excluded rather than
    trusted."""
    from pathlib import Path as _Path
    running_sha = cyc._package_sha256()
    files = sorted(_Path(control_dir).glob("between_*.json"),
                    key=lambda p: _cycle_rep_key(p.stem[len("between_"):]))
    verdicts: List[cyc.CycleVerdict] = []
    metrics: Dict[str, CycleMetrics] = {}
    hash_mismatch = False
    for f in files:
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError as exc:
            raise SummaryCliError(f"{f.name}: malformed JSON: {exc}") from exc
        if doc.get("tools_sha256") != running_sha:
            hash_mismatch = True
            continue
        v = cyc.CycleVerdict(
            doc["cycle"], doc["arm"], doc["verdict"], doc.get("reasons", []),
            doc.get("genuine_echo_count", 0), doc.get("segment_indeterminate", False))
        verdicts.append(v)
        m = doc.get("metrics")
        if m:
            metrics[doc["cycle"]] = CycleMetrics(
                leg_start_shoulder_pitch_error_deg=m.get("leg_start_shoulder_pitch_error_deg"),
                delta_cmd_max_cm=m.get("delta_cmd_max_cm"),
                wrist_ball_delta_cm=m.get("wrist_ball_delta_cm"),
                post_arrival_rise_deg=m.get("post_arrival_rise_deg"),
                c5_c6_pass=m.get("c5_c6_pass", True))
    return verdicts, metrics, hash_mismatch


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from tools.goalfix_cmp._io import write_result

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    cp = sub.add_parser("checkpoint")
    cp.add_argument("--control-dir", required=True)
    cp.add_argument("--n", type=int, default=4)
    cp.add_argument("--out", required=True)

    fn = sub.add_parser("final")
    fn.add_argument("--control-dir", required=True)
    fn.add_argument("--arm-map")
    fn.add_argument("--a-target-count", type=int, default=6)
    fn.add_argument("--a-manipulated-min", type=int, default=4)
    fn.add_argument("--out", required=True)

    args = p.parse_args(argv)

    try:
        verdicts, metrics, hash_mismatch = load_between_files(args.control_dir)
        if args.cmd == "checkpoint":
            r = checkpoint_after(verdicts, args.n, hash_mismatch=hash_mismatch)
            payload = {
                "n_cycles": r.n_cycles, "any_stop": r.any_stop,
                "a_inconclusive_count": r.a_inconclusive_count, "hold_idle": r.hold_idle,
                "reason": r.reason, "any_incomplete": r.any_incomplete,
                "missing_cycles": r.missing_cycles,
            }
            return write_result(args.out, r.rc(), payload)
        else:
            if hash_mismatch:
                r = SummaryResult(len(verdicts), OUTCOME_INCOMPLETE,
                                   "one or more between_*.json files were hashed against a "
                                   "different tools/goalfix_cmp package")
            else:
                r = aggregate(verdicts, metrics, a_target_count=args.a_target_count,
                               a_manipulated_min=args.a_manipulated_min)
            return write_result(args.out, r.rc(), r.as_dict())
    except (SummaryCliError, OSError, KeyError) as exc:
        from tools.goalfix_cmp._io import RC_INCONCLUSIVE
        return write_result(args.out, RC_INCONCLUSIVE, {"ok": False, "reason": str(exc)})


def main() -> int:
    return _cli()


if __name__ == "__main__":
    raise SystemExit(main())
