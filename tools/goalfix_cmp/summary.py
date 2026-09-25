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
#: unrelated log noise. Both are fixed below.
#:
#: MB7 (merge verdict, 2026-09-25 stage-repairs assignment §3; owner
#: decision (a), readiness review §4 Q2): "control_held_refusal" WAS
#: marked ungroundable ("no log call anywhere") -- that was wrong for the
#: bridge's own connection. Native sends `Error(code="control_held")`
#: without logging it (server.py:1037), but the BRIDGE logs every server
#: error it receives on its own connection: `log.warning("Server error:
#: [%s] %s", code, message)` (mujoco_remote_backend.py:486-488). So a
#: refusal sent to the bridge's own connection IS in the bridge log as
#: "Server error: [control_held]" -- groundable via `bridge_log`, same as
#: the others. "pause_message" stays ungroundable via log TEXT (the
#: "pause" websocket message, server.py:1091-1092, is never logged) --
#: its own, separate, states.jsonl-based counter is `PAUSE_TRIPWIRE`
#: below, per the owner's decision.
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
    # "Server error: [%s] %s" % ("control_held", …) -- the BRIDGE's own
    # log of a native refusal sent to ITS connection
    # (mujoco_remote_backend.py:486-488). Refusals to any OTHER client
    # connection are logged nowhere -- a blind spot, reported by
    # `MB7_BLIND_SPOTS`, never assumed 0.
    "control_held_refusal": "bridge",
    "pause_message": None,          # ungroundable via log text -- see PAUSE_TRIPWIRE below
}

_TRIPWIRE_PATTERN_TEXT: Dict[str, str] = {
    "reset_ack_timeout": "reset ack timed out",
    "unexpected_reset_ack": "unexpected reset_ack",
    "reset_sh_mismatch": "stop: reset",
    "lease_acquisition": "execution lease granted to",
    "control_held_refusal": "server error: [control_held]",
}

#: MB7's own {count, observable_scope, blind_spots} shape (assignment §3):
#: the three counters the owner decision (a) accepts, wired into
#: checkpoint/final. Blind spots are ALWAYS listed, never folded into a
#: 0 count.
MB7_OBSERVABLE_SCOPE: Dict[str, str] = {
    "lease_acquisition": "native log: 'Execution lease granted to %r' (server.py:1059)",
    "control_held_refusal": ("bridge log: 'Server error: [control_held]' "
                              "(mujoco_remote_backend.py:486-488)"),
    "pause": ("states.jsonl: rows with paused==true (server.py:863), plus "
              "in-leg duplicate sim_step (a state pushed while paused)"),
}
MB7_BLIND_SPOTS: Dict[str, str] = {
    "lease_acquisition": "a refused acquire ('lease already held') is sent only in "
                          "control_ack and is never logged",
    "control_held_refusal": "a refusal sent to any OTHER client connection (not the "
                             "bridge's own) is logged nowhere",
    "pause": "a pause that lands on a step between state pushes records no states at "
             "all, leaving only a wall_time_ns gap -- its absence proves nothing",
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
    "control_held_refusal": (
        ("mujoco_remote_backend.py", 486, "Server error: [%s] %s"),),
}


# ---------------------------------------------------------------------------
# MB7's "pause" counter: states.jsonl-based, never log text (owner
# decision (a), readiness review §4 Q2).
# ---------------------------------------------------------------------------

def count_pause_tripwire(states_paths: Optional[Sequence[str]] = None) -> Optional[int]:
    """Rows with ``paused == true`` (server.py:863's own field, protocol.py
    ``State.paused``), plus every duplicate ``sim_step`` within the SAME
    epoch of each ``states.jsonl`` -- a state pushed while the sim is not
    advancing shares its predecessor's ``sim_step`` exactly. ``None`` only
    when no path was supplied at all (a missing log, per T9); the KNOWN
    blind spot (a pause landing between two state pushes, per
    ``MB7_BLIND_SPOTS["pause"]``) is never folded into a 0 count -- it is
    simply unmeasurable, and is reported separately, never gated."""
    if not states_paths:
        return None
    from pathlib import Path as _Path
    count = 0
    for p in states_paths:
        rows = [json.loads(line) for line in _Path(p).read_text().splitlines() if line.strip()]
        count += sum(1 for r in rows if r.get("paused") is True)
        epoch = 0
        seen_steps = {0: set()}
        prev_step = None
        for r in rows:
            step = r.get("sim_step")
            if prev_step is not None and step is not None and step < prev_step:
                epoch += 1
                seen_steps[epoch] = set()
            if step in seen_steps[epoch]:
                count += 1
            else:
                seen_steps[epoch].add(step)
            prev_step = step
    return count


def mb7_tripwire_report(
    *, bridge_log: Optional[str] = None, native_log: Optional[str] = None,
    states_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """MB7's own ``{count, observable_scope, blind_spots}`` shape, for
    exactly the three counters the owner decision (a) accepts. A missing
    required log/path gives ``count=None`` (rc 3, per the enforcement
    rule below) -- the documented blind spot is reported regardless,
    every time, never 0 and never omitted."""
    counts_by_text = count_tripwires(bridge_log=bridge_log, native_log=native_log)
    pause_count = count_pause_tripwire(states_paths)
    raw = {
        "lease_acquisition": counts_by_text["lease_acquisition"],
        "control_held_refusal": counts_by_text["control_held_refusal"],
        "pause": pause_count,
    }
    return {
        name: {"count": raw[name], "observable_scope": MB7_OBSERVABLE_SCOPE[name],
               "blind_spots": [MB7_BLIND_SPOTS[name]]}
        for name in ("lease_acquisition", "control_held_refusal", "pause")
    }


@dataclass
class MB7Enforcement:
    #: names with count > 0 -- always a violation (STOP), for both arms:
    #: none of these three counters has an "except" carve-out (unlike
    #: the OLD unexpected_reset_ack).
    violated: List[str] = field(default_factory=list)
    #: names whose required log/path was not supplied at all.
    missing_log: List[str] = field(default_factory=list)


def check_mb7_tripwires(report: Dict[str, Dict[str, object]]) -> MB7Enforcement:
    violated, missing_log = [], []
    for name, entry in report.items():
        c = entry.get("count")
        if c is None:
            missing_log.append(name)
        elif c != 0:
            violated.append(name)
    return MB7Enforcement(violated, missing_log)


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


# ---------------------------------------------------------------------------
# F1 (2026-09-25 pr144-f1-f6 assignment): every log argument the CLI takes
# is a PATH, never text. `count_tripwires`/`mb7_tripwire_report` above stay
# text-in (library helpers legitimately called with text directly by their
# own unit tests, per the assignment's own instruction) -- this is the ONE
# seam the CLI itself must go through, so every CLI-level caller reads the
# file's actual contents rather than (as at 0722476) handing the path
# STRING itself to a pattern count. Kept as a small, separately named
# function so a mutation that reverts to "pass the path" or "treat a
# missing file as empty" has one obvious shipped call site to target.
# ---------------------------------------------------------------------------

def read_log_text(path: str) -> str:
    """The full text of the log file at ``path``. Raises ``SummaryCliError``
    (mapped to rc 3 by every caller), naming ``path``, when it does not
    exist, is not a regular file, or cannot be read/decoded. An EXISTING,
    EMPTY file is valid evidence of no matching lines and returns ``""``
    (count 0), never an error."""
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.is_file():
        raise SummaryCliError(f"log file not found or not a regular file: {path}")
    try:
        return p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SummaryCliError(f"cannot read log file {path}: {exc}") from exc


def _cycle_rep_key(cycle: str) -> int:
    m = re.search(r"r(\d+)$", cycle)
    return int(m.group(1)) if m else 0


def load_between_files(
    control_dir, arm_map: Dict[int, object], *, required_reps: Optional[Sequence[int]] = None,
) -> "tuple[list, dict, bool]":
    """MB3 (merge verdict, 2026-09-25 stage-repairs assignment §3):
    cycles are KEYED BY REP FROM ``arm_map`` (an already-validated
    ``{rep: ArmMapEntry}`` -- ``validate_arm_map``'s own frozen
    ``ARM_MAP_ORDER``, "ABBABAABABBA", is the caller's job to enforce
    before this is called), never by file count or filename sort order.
    Rejects (raises ``SummaryCliError``, mapped to rc 3 by the caller):
    a ``validation_only`` file (never authorizing), an ``rc`` inconsistent
    with its own ``verdict``, a cycle whose rep is not in ``arm_map``, a
    duplicate rep, an arm that differs from ``arm_map[rep].arm``, and
    malformed JSON. ``required_reps`` (given by the checkpoint's own
    ``--n``), if given, additionally requires every rep in it to be
    present -- probe P10's "r1 missing, r2..r5 present" bug: the old code
    took bare ``verdicts[:n]`` positionally, so a SHORTFALL was the only
    detectable shape of "missing"; a missing rep in the MIDDLE with
    enough total files was invisible.

    Returns ``(verdicts, metrics, hash_mismatch)`` -- ``hash_mismatch``
    is True if ANY file's ``tools_sha256`` disagrees with the running
    ``tools/goalfix_cmp`` package; that file's own verdict/metrics are
    excluded rather than trusted, same as before."""
    from pathlib import Path as _Path
    running_sha = cyc._package_sha256()
    files = sorted(_Path(control_dir).glob("between_*.json"))
    by_rep: Dict[int, cyc.CycleVerdict] = {}
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
        if doc.get("validation_only"):
            raise SummaryCliError(f"{f.name}: validation_only results never authorize")
        verdict = doc.get("verdict")
        rc = doc.get("rc")
        if verdict is None or rc != cyc.rc_for_verdict(verdict):
            raise SummaryCliError(
                f"{f.name}: rc {rc!r} is inconsistent with verdict {verdict!r}")
        cycle_id = doc.get("cycle")
        rep = _cycle_rep_key(cycle_id) if cycle_id else 0
        if rep not in arm_map:
            raise SummaryCliError(f"{f.name}: cycle {cycle_id!r} has no rep {rep} in the arm map")
        if rep in by_rep:
            raise SummaryCliError(
                f"{f.name}: duplicate rep {rep} (cycle {cycle_id!r} claims a rep "
                f"already claimed by {by_rep[rep].cycle!r})")
        entry_arm = getattr(arm_map[rep], "arm", None)
        if doc.get("arm") != entry_arm:
            raise SummaryCliError(
                f"{f.name}: arm {doc.get('arm')!r} != arm_map[{rep}].arm {entry_arm!r}")
        v = cyc.CycleVerdict(
            cycle_id, doc["arm"], verdict, doc.get("reasons", []),
            doc.get("genuine_echo_count", 0), doc.get("segment_indeterminate", False))
        by_rep[rep] = v
        m = doc.get("metrics")
        if m:
            metrics[cycle_id] = CycleMetrics(
                leg_start_shoulder_pitch_error_deg=m.get("leg_start_shoulder_pitch_error_deg"),
                delta_cmd_max_cm=m.get("delta_cmd_max_cm"),
                wrist_ball_delta_cm=m.get("wrist_ball_delta_cm"),
                post_arrival_rise_deg=m.get("post_arrival_rise_deg"),
                c5_c6_pass=m.get("c5_c6_pass", True))

    if required_reps is not None:
        missing = sorted(set(required_reps) - set(by_rep))
        if missing:
            raise SummaryCliError(f"missing required rep(s) {missing}")

    verdicts = [by_rep[r] for r in sorted(by_rep)]
    return verdicts, metrics, hash_mismatch


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from tools.goalfix_cmp import provenance as pv
    from tools.goalfix_cmp._io import RC_INCONCLUSIVE, RC_STOP, write_result

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    # MB3 (merge verdict, 2026-09-25 stage-repairs assignment §3): BOTH
    # subcommands now require a validated arm map -- cycles are keyed
    # by rep from it, never by file count or filename sort.
    # F2 (merge verdict review-2026-09-25-pr144-0722476-merge-verdict.md
    # §2; 2026-09-25 pr144-f1-f6 assignment): tripwires are now MANDATORY,
    # never opt-in -- at 0722476, omitting every one of --bridge-log/
    # --native-log/--states (as test_no_log_flags_is_unaffected pinned)
    # skipped tripwire evaluation entirely and authorized rc 0 on evidence
    # that was never actually checked for a reset-ack timeout, a
    # control_held refusal, or a pause. Not `required=True` at the
    # argparse level: argparse's own missing-required-arg exit is status
    # 2, which would read as a STOP to a caller that only looks at rc,
    # not as the "evidence incomplete" it actually is -- so a missing
    # input is instead detected in the CLI body and returned as rc 3 with
    # a JSON reason (see below).
    cp = sub.add_parser("checkpoint")
    cp.add_argument("--control-dir", required=True)
    cp.add_argument("--arm-map", required=True)
    cp.add_argument("--expected-bridge-sha-a", required=True)
    cp.add_argument("--expected-bridge-sha-b", required=True)
    cp.add_argument("--n", type=int, default=4)
    cp.add_argument("--bridge-log")
    cp.add_argument("--native-log")
    cp.add_argument("--states", nargs="+")
    cp.add_argument("--out", required=True)

    fn = sub.add_parser("final")
    fn.add_argument("--control-dir", required=True)
    fn.add_argument("--arm-map", required=True)
    fn.add_argument("--expected-bridge-sha-a", required=True)
    fn.add_argument("--expected-bridge-sha-b", required=True)
    fn.add_argument("--a-target-count", type=int, default=6)
    fn.add_argument("--a-manipulated-min", type=int, default=4)
    fn.add_argument("--bridge-log")
    fn.add_argument("--native-log")
    fn.add_argument("--states", nargs="+")
    fn.add_argument("--out", required=True)

    args = p.parse_args(argv)

    try:
        arm_map_doc = json.loads(open(args.arm_map).read())
        arm_map_entries = [pv.ArmMapEntry(**e) for e in arm_map_doc]
        map_check = pv.validate_arm_map(
            arm_map_entries,
            expected_bridge_sha={"A": args.expected_bridge_sha_a, "B": args.expected_bridge_sha_b})
        if not map_check.ok:
            raise SummaryCliError(f"arm_map: {map_check.reason}")
        arm_map = {e.rep: e for e in arm_map_entries}

        # F2: tripwires are mandatory for both checkpoint and final -- a
        # missing input (any one of the three) is rc 3 in code, never
        # argparse's own status 2 and never a silent rc 0.
        if args.bridge_log is None or args.native_log is None or args.states is None:
            raise SummaryCliError(
                "checkpoint/final require the full tripwire input set: "
                "--bridge-log, --native-log and --states are all mandatory "
                "(F2 -- tripwires are never opt-in)")
        # F1 (2026-09-25 pr144-f1-f6 assignment): `args.bridge_log`/
        # `args.native_log` are PATHS (argparse strings naming files on
        # disk, per the --bridge-log/--native-log help text) -- at 0722476
        # these were handed STRAIGHT to `mb7_tripwire_report` (which counts
        # patterns in whatever text it is given), so the CLI counted
        # patterns in the PATH STRING itself, never the file's contents.
        # `read_log_text` is the one seam that turns a path into text; a
        # missing/unreadable/undecodable file raises `SummaryCliError`
        # (rc 3, via the except clause below), naming the path -- never
        # silently treated as empty.
        bridge_text = read_log_text(args.bridge_log)
        native_text = read_log_text(args.native_log)
        tripwire_report = mb7_tripwire_report(
            bridge_log=bridge_text, native_log=native_text, states_paths=args.states)
        tripwire_check = check_mb7_tripwires(tripwire_report)
        if tripwire_check.missing_log:
            raise SummaryCliError(
                f"tripwire counter(s) missing their required log/path: "
                f"{tripwire_check.missing_log}")

        if args.cmd == "checkpoint":
            verdicts, metrics, hash_mismatch = load_between_files(
                args.control_dir, arm_map, required_reps=range(1, args.n + 1))
            r = checkpoint_after(verdicts, args.n, hash_mismatch=hash_mismatch)
            payload = {
                "n_cycles": r.n_cycles, "any_stop": r.any_stop,
                "a_inconclusive_count": r.a_inconclusive_count, "hold_idle": r.hold_idle,
                "reason": r.reason, "any_incomplete": r.any_incomplete,
                "missing_cycles": r.missing_cycles,
            }
            rc = r.rc()
            if tripwire_report is not None:
                payload["tripwires"] = tripwire_report
                if tripwire_check.violated and rc == RC_OK:
                    rc = RC_STOP
                    payload["reason"] = payload["reason"] or f"tripwire(s) violated: {tripwire_check.violated}"
            return write_result(args.out, rc, payload)
        else:
            verdicts, metrics, hash_mismatch = load_between_files(args.control_dir, arm_map)
            if hash_mismatch:
                r = SummaryResult(len(verdicts), OUTCOME_INCOMPLETE,
                                   "one or more between_*.json files were hashed against a "
                                   "different tools/goalfix_cmp package")
            else:
                r = aggregate(verdicts, metrics, a_target_count=args.a_target_count,
                               a_manipulated_min=args.a_manipulated_min)
            out_payload = r.as_dict()
            rc = r.rc()
            if tripwire_report is not None:
                out_payload["tripwires"] = tripwire_report
                if tripwire_check.violated and rc == RC_OK:
                    rc = RC_STOP
            return write_result(args.out, rc, out_payload)
    except (SummaryCliError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return write_result(args.out, RC_INCONCLUSIVE, {"ok": False, "reason": str(exc)})
    except Exception as exc:
        # MB6-style catch-all, kept here too for the same reason.
        return write_result(args.out, RC_INCONCLUSIVE,
                             {"ok": False, "reason": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    return _cli()


if __name__ == "__main__":
    raise SystemExit(main())
