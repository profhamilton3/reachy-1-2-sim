"""The fixed 7-run harness validation batch (written and dry-run only; NOT executed).

OWNER SCOPE (fixed in advance): exactly 7 Stage A slice regenerations, 4 cycles
each in the order A B B A = 14 clean-B cycles = 28 clean-B leg-evaluations.  A
labelled cycles are recorded but are not the object of the validation.  This is
NOT a repeat-until-green loop: all 7 runs are executed, in order, with no
replacement and no extension.  Stop early ONLY on an infrastructure or capture
failure (harness exception or non-zero harness exit, capture cross-check
failure, missing artifact, ``reset_verify`` failure, free disk below the floor).
C0-C8 outcomes, verdicts and C2 spreads NEVER stop, retry, replace or extend
the batch.  Neither this batch nor decision D1 is authorized until the owner
says so.

Entry point: ``tests/fixtures/goalfix_cmp/run_validation_batch.sh`` (clone,
pin verification, environment; then ``validation_batch.py run``).

Sub-commands (all stdlib-only at import; the SDK/numpy are imported lazily and
only by ``harness``/``run`` -- ``preflight`` and ``--dry-run`` import neither):

* ``preflight``  validates arguments, pins, tree/blob equalities and the output
  path, prints the exact per-run plan, and exits.  This is what ``--dry-run``
  runs.
* ``run``        the batch controller (verifies the clone, runs the runs).
* ``harness``    one run's harness process (Stage A slice + capture shim +
  capture cross-check), spawned by ``run`` as a subprocess so that stdout,
  stderr and the true exit status are recorded and a hang can be timed out.

NATIVESTUB COMPLIANCE LIMITATION (stated in every SUMMARY header)
-----------------------------------------------------------------
``NativeStub(compliant=...)`` sets ONE flag for all 21 joints
(``tests/integration/native_stub.py:33-43``); after ``turn_on("r_arm")`` only the
right arm is stiff.  The slice's ``prep_<cycle>.json`` still carries the
pre-revision-4 fixture vector (antennas stiff, neck compliant;
``stage_a_slice.py:719-726``, unchanged by owner ruling).  The full-gate
compliance check therefore FAILS on every cycle (and would also fail against the
revision-4 vector, on the neck and antennas).  The summary reports full-gate
output verbatim, labels that gate failure a documented harness limitation,
never reports a passing native-compliance result and never describes
validation-mode (path-only) results as full-gate success.

CAPTURE TIMING (see capture.py): capture-on rates are not capture-off rates.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
for _p in (_REPO, _REPO / "src", _REPO / "native_mujoco", _REPO / "scripts",
           _REPO / "tests" / "integration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Fixed pins and parameters
# ---------------------------------------------------------------------------

M_SHA = "155fc1549812146a8579080c72560c2c1875262c"    # candidate host pin (origin/main)
A_SHA = "8c0dad2d62dde791c8912fa723b8c2b7f190e1e8"    # baseline bridge revision (label only)
B_SHA = "67730a1ecf646544825fb60c00129cf46de6d307"    # fixed bridge revision
N_RUNS = 7
ARM_LABELS = ("A", "B", "B", "A")
CYCLE_NAMES = tuple(f"S2-B4-c-r{i}" for i in range(1, 5))      # stage_a_slice's defaults
OPT_FILES = ("fake_reachy_server.py", "mujoco_remote_backend.py", "reset_watcher.py")
KEYFRAME_REL = "native_mujoco/model/reachy_1_2.xml"
KEYFRAME_NAME = "home"
FORBIDDEN_OUT_PREFIXES = ("/tmp", "/private/tmp")
MIN_FREE_BYTES = 5 * 2 ** 30            # stated floor: 5 GiB free before each run
HARNESS_TIMEOUT_S = 3600
CLI_TIMEOUT_S = 1800
FIXED_SIM_STEP = 5000                    # plan §5 P5
EXPECTED_HOST_LABEL_NOTE = "host sha == reviewed branch head P"
C2_TOL_S = 0.030                         # pathcheck.C2_SKEW_TOL_S (asserted at analysis time)

DRIVER_REL = "tests/fixtures/goalfix_cmp/validation_batch.py"

NATIVESTUB_LIMITATION = (
    "NativeStub(compliant=...) sets one flag for all 21 joints (tests/integration/native_stub.py:33-43); "
    "after turn_on('r_arm') only the right arm is stiff. The slice's prep_<cycle>.json still carries the "
    "pre-revision-4 fixture vector (antennas stiff, neck compliant; stage_a_slice.py:719-726, unchanged by "
    "owner ruling). The full-gate compliance check therefore FAILS on every cycle (it would also fail "
    "against the revision-4 vector, on the neck and antennas). That gate failure is a documented HARNESS "
    "LIMITATION, not evidence about the bridge; no passing native-compliance result is reported anywhere "
    "in this summary, and validation-mode (path-only) results are NOT full-gate success.")

STATS_NOTE = (
    "Statistics (descriptive). Counts are reported first. Where an upper bound is given it is the one-sided "
    "95% bound 1 - 0.05^(1/n) for zero observed events. Per leg-evaluation (planned n = 28, about 10.1%) it "
    "ASSUMES INDEPENDENT LEGS, which is NOT established: legs share a run, a process and a harness state. "
    "Per run (planned n = 7, about 34.8%) it assumes independent runs. Clean results cannot reconstruct the "
    "lost 84.3 ms or 0722476 events, and say nothing about native-stack reliability or neck/compliance "
    "behaviour. Capture-on rates are not capture-off rates (the shim adds work between per-joint submits).")


class Refusal(Exception):
    """A refusal to start (bad argument, pin, tree or path). Never a result."""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def one_sided_upper_bound(n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) upper bound on a per-trial rate after ``n`` trials with
    zero events: ``1 - alpha**(1/n)``.  Independence of the trials is an ASSUMPTION."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 1.0 - alpha ** (1.0 / n)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_out_path(path: str, forbidden: Sequence[str] = FORBIDDEN_OUT_PREFIXES) -> str:
    """The absolute ``--out`` path, or ``Refusal``: must be absolute after ``~``
    expansion, outside ``/tmp`` (also ``/private/tmp``), and must not exist."""
    if not path:
        raise Refusal("--out is empty")
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        raise Refusal(f"--out must be absolute: {path!r}")
    parent = os.path.realpath(os.path.dirname(p) or "/")
    real = os.path.join(parent, os.path.basename(p))
    for pre in forbidden:
        pre_real = os.path.realpath(pre)
        for cand in (real, os.path.normpath(p)):
            for base in (pre, pre_real):
                if cand == base or cand.startswith(base.rstrip("/") + "/"):
                    raise Refusal(f"--out must be outside {pre}: {path!r}")
    if os.path.lexists(p):
        raise Refusal(f"--out already exists: {path!r}")
    if not os.path.isdir(os.path.dirname(p)):
        raise Refusal(f"the parent directory of --out does not exist: {path!r}")
    return p


def check_runs(runs: Any) -> int:
    try:
        n = int(runs)
    except (TypeError, ValueError):
        raise Refusal(f"--runs must be the integer {N_RUNS}, got {runs!r}")
    if n != N_RUNS:
        raise Refusal(f"--runs must be exactly {N_RUNS} (fixed in advance), got {n}")
    return n


def check_pin_format(pin: str) -> str:
    if len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise Refusal(f"--pin must be a full 40-hex-digit SHA, got {pin!r}")
    return pin


def _git(repo, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise Refusal(f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip()}")
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _git_show_bytes(repo, rev_path: str) -> bytes:
    proc = subprocess.run(["git", "-C", str(repo), "show", rev_path], capture_output=True)
    if proc.returncode != 0:
        raise Refusal(f"git show {rev_path} failed in {repo}: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def source_pin_facts(repo, pin: str) -> Dict[str, Any]:
    """Pin equalities evaluated against a git repo (the source repo before the clone, the
    clone after).  Raises ``Refusal`` on any mismatch."""
    for label, rev in (("P", pin), ("M", M_SHA), ("B", B_SHA)):
        if not _git(repo, "cat-file", "-t", f"{rev}^{{commit}}", check=False):
            raise Refusal(f"commit {label}={rev} is not present in {repo}")
    p_tools = _git(repo, "rev-parse", f"{pin}:tools/goalfix_cmp")
    m_tools = _git(repo, "rev-parse", f"{M_SHA}:tools/goalfix_cmp")
    if p_tools != m_tools:
        raise Refusal(f"tools/goalfix_cmp tree differs: P has {p_tools}, M has {m_tools}")
    blobs = {}
    for f in OPT_FILES:
        sp = sha256_bytes(_git_show_bytes(repo, f"{pin}:{f}"))
        sb = sha256_bytes(_git_show_bytes(repo, f"{B_SHA}:{f}"))
        if sp != sb:
            raise Refusal(f"{f} at P differs from B's blob (sha256 {sp} vs {sb})")
        blobs[f] = sp
    kf_p = sha256_bytes(_git_show_bytes(repo, f"{pin}:{KEYFRAME_REL}"))
    kf_m = sha256_bytes(_git_show_bytes(repo, f"{M_SHA}:{KEYFRAME_REL}"))
    if kf_p != kf_m:
        raise Refusal(f"{KEYFRAME_REL} at P differs from M's ({kf_p} vs {kf_m})")
    if not _git(repo, "cat-file", "-t", f"{pin}:{DRIVER_REL}", check=False):
        raise Refusal(f"{DRIVER_REL} is not present at P")
    return {"P": pin, "M": M_SHA, "B": B_SHA, "A": A_SHA,
            "P_tools_goalfix_cmp_tree": p_tools, "M_tools_goalfix_cmp_tree": m_tools,
            "opt_file_sha256": blobs, "keyframe_rel": KEYFRAME_REL, "keyframe_sha256": kf_m}


def preflight(pin: str, out: str, repo: str, runs: Any, venv: str) -> Dict[str, Any]:
    check_pin_format(pin)
    check_runs(runs)
    out_abs = check_out_path(out)
    py = Path(os.path.expanduser(venv)) / "bin" / "python"
    if not (py.exists() and os.access(str(py), os.X_OK)):
        raise Refusal(f"--venv has no executable bin/python: {venv!r}")
    repo = os.path.expanduser(repo)
    if not (Path(repo) / ".git").exists():
        raise Refusal(f"source repo {repo!r} is not a git checkout")
    facts = source_pin_facts(repo, pin)
    if not _git(repo, "for-each-ref", "--contains", pin, "--count=1", "--format=%(refname)"):
        raise Refusal("P is not reachable from any ref of the source repo; "
                      "`git clone --local` would not carry it")
    facts.update({"out": out_abs, "venv": os.path.expanduser(venv), "runs": N_RUNS,
                  "source_repo": repo})
    return facts


def plan_lines(pin: str, out: str) -> List[str]:
    lines = [
        "VALIDATION BATCH PLAN (fixed scope; nothing below has been executed by --dry-run)",
        f"  pin P (reviewed branch head): {pin}",
        f"  M (host pin, tools tree must equal): {M_SHA}",
        f"  B (bridge blobs must equal): {B_SHA}     A (label only): {A_SHA}",
        f"  out: {out}",
        f"  source: git clone --local --no-checkout <repo> {out}/src ; git checkout --detach {pin}",
        f"  runs: exactly {N_RUNS}, in order, no replacement, no extension; stop early ONLY on "
        "infrastructure/capture failure (harness exception or non-zero exit, capture cross-check failure, "
        f"missing artifact, reset_verify failure, free disk < {MIN_FREE_BYTES // 2 ** 30} GiB)",
        f"  cycles per run: {len(ARM_LABELS)} in order {' '.join(ARM_LABELS)} "
        f"({', '.join(CYCLE_NAMES)}); B legs per run = {2 * ARM_LABELS.count('B')}; "
        f"total B leg-evaluations = {N_RUNS * 2 * ARM_LABELS.count('B')}",
    ]
    for i in range(1, N_RUNS + 1):
        rd = f"{out}/run-{i:02d}"
        lines += [
            f"  run-{i:02d}:",
            f"    1. harness (subprocess): stage_a_slice.run_stage_a_session + emit_stage_a_evidence "
            f"-> {rd}/ev, {rd}/control with the capture shim active; stdout/stderr -> {rd}/harness.stdout|stderr",
            f"    2. capture: {rd}/capture.jsonl, {rd}/crosscheck.json, {rd}/classification.json",
            f"    3. cycle CLI per cycle, full gates (--expected-host-sha {pin}) and --validation-mode; "
            f"summary checkpoint --n 4 -> {rd}/cli/",
            f"    4. analysis (report-only): shipped C0-C8 per leg, per-command C2 tau spread/implied tau, "
            f"join of every command > 30 ms to its capture class -> {rd}/analysis.json, c2_per_command.jsonl",
            f"    5. neck/seed extractor per cycle (keyframe {KEYFRAME_REL} '{KEYFRAME_NAME}') -> {rd}/neck/",
            f"    6. {rd}/run_manifest.json and {rd}/SHA256SUMS over every file in the run directory",
        ]
    lines += [
        f"  then: {out}/SUMMARY.md, {out}/summary.json, {out}/environment.json, root {out}/SHA256SUMS",
        "  NativeStub compliance limitation: " + NATIVESTUB_LIMITATION,
    ]
    return lines


# ---------------------------------------------------------------------------
# Stop-condition logic (pure)
# ---------------------------------------------------------------------------

def classify_failure(facts: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """The ONLY thing that can stop the batch: an infrastructure or capture failure.
    Returns ``{"kind", "detail"}`` or ``None``.  Any other key in ``facts`` (a C2
    failure, a verdict, a spread, an event count) is deliberately never consulted."""
    if facts.get("disk_free_ok") is False:
        return {"kind": "disk_low", "detail": f"free disk below {MIN_FREE_BYTES} bytes"}
    if facts.get("harness_exception"):
        return {"kind": "harness_exception", "detail": str(facts.get("harness_exception"))}
    if facts.get("harness_timed_out"):
        return {"kind": "harness_timeout", "detail": f"harness exceeded {HARNESS_TIMEOUT_S} s"}
    hx = facts.get("harness_exit_status")
    if hx not in (0, None) and not (hx == 3 and facts.get("capture_ok") is False):
        return {"kind": "harness_exit", "detail": f"harness exit status {hx}"}
    if facts.get("capture_ok") is not True:
        return {"kind": "capture_crosscheck",
                "detail": str(facts.get("capture_detail") or "capture cross-check did not pass")}
    bad = [rc for rc in (facts.get("reset_verify_rcs") or []) if rc != 0]
    if bad:
        return {"kind": "reset_verify", "detail": f"reset_verify rc {bad}"}
    missing = list(facts.get("missing_artifacts") or [])
    if missing:
        return {"kind": "missing_artifact", "detail": ", ".join(missing)}
    return None


# ---------------------------------------------------------------------------
# C2 diagnostic (report-only; uses the shipped helpers)
# ---------------------------------------------------------------------------

def c2_records(targets8, start_pose8, route, assignment, withdraw=None) -> Dict[str, Any]:
    """Per-command tau figures for every command with a goal assignment, mirroring
    ``pathcheck.check_c2_c3``'s C2 loop with the shipped helpers
    (``segments.carry_mask``, ``pathcheck.implied_tau``, ``_near_end``, ``_goal8``,
    ``_leg_seconds``).  ``first_fail`` is the first local index where C2 (a constant
    joint moved, or a tau spread > 30 ms + 1e-6 s) would fail."""
    from tools.goalfix_cmp import pathcheck as pc
    from tools.goalfix_cmp import segments as seg

    carries = seg.carry_mask(targets8, start_pose8, withdraw=withdraw)
    seg_start = dict(start_pose8)
    last_goal = None
    recs: List[dict] = []
    first_fail: Optional[int] = None
    first_reason = ""
    for i, tgt in enumerate(targets8):
        g = assignment.goal_index[i]
        if g is None:
            continue
        if g != last_goal:
            seg_start = start_pose8 if g == 0 else pc._goal8(route[g - 1])
        goal8 = pc._goal8(route[g])
        taus: Dict[str, float] = {}
        const_violation = False
        for j in pc.ARM7:
            a, b, v = seg_start.get(j, 0.0), goal8.get(j, 0.0), tgt.get(j, 0.0)
            is_carry = carries[i][j]
            if a == b:
                if not is_carry and v != a:
                    const_violation = True
                continue
            if is_carry:
                continue
            if pc._near_end(v, a, b):
                continue
            taus[j] = pc.implied_tau(a, b, v)
        spread_s = None
        if taus:
            spread_s = (max(taus.values()) - min(taus.values())) * pc._leg_seconds(route, g)
        fail = const_violation or (spread_s is not None and spread_s > pc.C2_SKEW_TOL_S + 1e-6)
        if fail and first_fail is None:
            first_fail = i
            first_reason = ("a constant joint moved" if const_violation
                            else f"per-joint tau spread {spread_s * 1000:.1f} ms > 30 ms")
        recs.append({"local_index": i, "goal": g, "n_tau": len(taus), "taus": taus,
                     "spread_ms": None if spread_s is None else spread_s * 1000.0,
                     "const_violation": const_violation})
        last_goal = g
    return {"records": recs, "first_fail": first_fail, "first_reason": first_reason}


def c2_diagnostic(targets8, start_pose8, route, assignment, withdraw_rows) -> Dict[str, Any]:
    """Runs ``c2_records`` plain and (only if anything is withdrawn on this leg, exactly as
    ``cycle.evaluate_cycle`` does) with the R-carry' withdrawals.  The diagnostic C2 is the
    plain result if it fails, else the withdrawn result -- the shipped first-pass-then-recheck
    order."""
    plain = c2_records(targets8, start_pose8, route, assignment, None)
    any_w = bool(withdraw_rows) and any(any(r.values()) for r in withdraw_rows)
    wd = c2_records(targets8, start_pose8, route, assignment, withdraw_rows) if any_w else None
    if plain["first_fail"] is not None:
        passed, idx, detail = False, plain["first_fail"], plain["first_reason"]
    elif wd is not None and wd["first_fail"] is not None:
        passed, idx, detail = False, wd["first_fail"], "R-carry′: " + wd["first_reason"]
    else:
        passed, idx, detail = True, None, ""
    return {"passed": passed, "first_fail": idx, "detail": detail, "plain": plain, "withdrawn": wd}


def join_commands_to_capture(over_ms_globals: Sequence[int], kinds: Sequence[str],
                             joined: Dict[Tuple[int, int], int], classification: Dict[str, Any],
                             ) -> List[Dict[str, Any]]:
    """Joins each command (global index into commands.jsonl rows) to its capture class.
    ``joined`` maps (gen, seq) -> joint_command row ordinal (from ``cross_check``)."""
    ord_of_global: Dict[int, int] = {}
    n = -1
    for gi, k in enumerate(kinds):
        if k == "joint_command":
            n += 1
            ord_of_global[gi] = n
    key_of_ord = {o: k for k, o in joined.items()}
    by_key = {(b["gen"], b["seq"]): b for b in classification["builds"]}
    gaps = {(g["gen"], g["batch_id"]): g["gap_ns"] for g in classification["batch_gaps_ns"]}
    out = []
    for gi in over_ms_globals:
        o = ord_of_global.get(gi)
        key = key_of_ord.get(o) if o is not None else None
        b = by_key.get(key) if key is not None else None
        if b is None:
            out.append({"global_index": gi, "class": "unjoined", "gen": None, "seq": None,
                        "newest_source_batch_gap_ns": None, "source_span_ns": None})
            continue
        newest = max(b["source_batches"]) if b["source_batches"] else None
        out.append({"global_index": gi, "class": b["class"], "gen": key[0], "seq": key[1],
                    "source_batches": b["source_batches"], "source_span_ns": b["source_span_ns"],
                    "newest_source_batch_gap_ns": gaps.get((key[0], newest)) if newest is not None else None})
    return out


# ---------------------------------------------------------------------------
# Summary (pure)
# ---------------------------------------------------------------------------

CLASSES = ("mixed_batch", "split_involved_not_mixed", "neither", "unjoined")
CHECKS_REPORTED = ("C0", "C2", "C5", "C7")


def _b_leg_outcomes(rec: dict) -> Dict[str, Any]:
    """C0/C2/C5/C7 outcomes, max C2 spread and >30 ms class counts over this run's B legs."""
    legs = 0
    fails = {c: 0 for c in CHECKS_REPORTED}
    max_spread = None
    classes = {c: 0 for c in CLASSES}
    unverified = 0
    for cyc in rec.get("cycles") or []:
        if cyc.get("arm") != "B":
            continue
        for leg_name, leg in (cyc.get("legs") or {}).items():
            legs += 1
            for c in CHECKS_REPORTED:
                r = leg.get(c)
                if r is not None and r.get("passed") is False:
                    fails[c] += 1
            c2 = (cyc.get("c2") or {}).get(leg_name) or {}
            if c2.get("reproduced") is False:
                unverified += 1
            sp = c2.get("max_spread_ms")
            if sp is not None and (max_spread is None or sp > max_spread):
                max_spread = sp
            for k, v in (c2.get("classes") or {}).items():
                classes[k] = classes.get(k, 0) + v
    return {"b_leg_evaluations": legs, "fails": fails, "max_c2_spread_ms": max_spread,
            "classes_gt30": classes, "c2_diag_unreproduced_legs": unverified}


def annotate_full_gate(fg: Optional[dict]) -> Dict[str, Any]:
    """Full-gate output verbatim, plus the fixed annotation.  A passing native-compliance
    result is NEVER reported: an unexpected pass is withheld and flagged for inspection."""
    if not fg:
        return {"rc": None, "verdict": None, "compliance_gate": "no output",
                "annotation": "full-gate run produced no output"}
    comp = fg.get("compliance")
    if comp is None:
        shown, ann = "not reported by the CLI", "compliance check not present in the output"
    elif comp.get("ok") is False:
        shown = f"FAILED: {comp.get('detail', '')}"
        ann = "documented harness limitation (NativeStub one-flag compliance); not evidence about the bridge"
    else:
        shown = "WITHHELD (unexpected pass; raw output must be inspected)"
        ann = ("UNEXPECTED: the compliance gate did not fail; this is NOT reported as a native-compliance "
               "pass (NativeStub cannot model per-joint compliance)")
    return {"rc": fg.get("rc"), "verdict": fg.get("verdict"), "compliance_gate": shown,
            "annotation": ann}


def build_summary(records: Sequence[dict], meta: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Compact summary over per-run records (any of ``ok``/``failed``/``not_run``).  A failed
    run or a capture failure gives a PARTIAL summary; nothing is replaced or extended."""
    runs_out = []
    totals = {"b_leg_evaluations": 0, "fails": {c: 0 for c in CHECKS_REPORTED},
              "classes_gt30": {c: 0 for c in CLASSES}, "split_batches": 0, "mixed_commands": 0,
              "split_involved_not_mixed_commands": 0, "runs_ok": 0, "runs_failed": 0,
              "runs_not_run": 0, "runs_with_any_event": 0, "c2_diag_unreproduced_legs": 0}
    for rec in records:
        status = rec.get("status")
        cap = rec.get("capture") or {}
        out = _b_leg_outcomes(rec)
        cycles = []
        for cy in rec.get("cycles") or []:
            cycles.append({
                "cycle": cy.get("name"), "arm": cy.get("arm"),
                "full_gate": annotate_full_gate(cy.get("full_gate")),
                "validation_mode_path_only": {
                    "verdict": (cy.get("validation_mode") or {}).get("verdict"),
                    "rc": (cy.get("validation_mode") or {}).get("rc"),
                    "note": "path-only (--validation-mode); NOT full-gate success"},
            })
        entry = {
            "run": rec.get("run"), "status": status, "failure": rec.get("failure"),
            "harness_exit_status": (rec.get("harness") or {}).get("exit_status"),
            "capture_ok": cap.get("ok"),
            "cycles": cycles,
            "b_legs": out,
            "capture_counts": {
                "split_batches": cap.get("n_split_batches"), "mixed_commands": cap.get("n_mixed"),
                "split_involved_not_mixed": cap.get("n_split_involved_not_mixed"),
                "batches": cap.get("n_batches"), "builds": cap.get("n_builds")},
        }
        runs_out.append(entry)
        if status == "ok":
            totals["runs_ok"] += 1
        elif status == "failed":
            totals["runs_failed"] += 1
        else:
            totals["runs_not_run"] += 1
        if status in ("ok", "failed"):
            totals["b_leg_evaluations"] += out["b_leg_evaluations"]
            for c in CHECKS_REPORTED:
                totals["fails"][c] += out["fails"][c]
            for c in CLASSES:
                totals["classes_gt30"][c] += out["classes_gt30"].get(c, 0)
            totals["split_batches"] += cap.get("n_split_batches") or 0
            totals["mixed_commands"] += cap.get("n_mixed") or 0
            totals["split_involved_not_mixed_commands"] += cap.get("n_split_involved_not_mixed") or 0
            totals["c2_diag_unreproduced_legs"] += out["c2_diag_unreproduced_legs"]
            if any(out["fails"].values()):        # event = a C0/C2/C5/C7 failure on a B leg
                totals["runs_with_any_event"] += 1

    partial = totals["runs_failed"] > 0 or totals["runs_not_run"] > 0 or len(records) < N_RUNS
    n_legs = totals["b_leg_evaluations"]
    n_runs_done = totals["runs_ok"] + totals["runs_failed"]
    bounds = {}
    for c in CHECKS_REPORTED:
        if n_legs > 0 and totals["fails"][c] == 0:
            bounds[c] = {"events": 0, "n": n_legs, "one_sided_95pct_upper_bound": one_sided_upper_bound(n_legs),
                         "assumption": "independent legs (NOT established)"}
        else:
            bounds[c] = {"events": totals["fails"][c], "n": n_legs, "one_sided_95pct_upper_bound": None,
                         "assumption": "not given: events observed or no evaluation"}
    if n_runs_done > 0 and totals["runs_with_any_event"] == 0:
        bounds["per_run_any_event"] = {
            "events": 0, "n": n_runs_done, "one_sided_95pct_upper_bound": one_sided_upper_bound(n_runs_done),
            "assumption": "independent runs"}
    else:
        bounds["per_run_any_event"] = {
            "events": totals["runs_with_any_event"], "n": n_runs_done, "one_sided_95pct_upper_bound": None,
            "assumption": "not given: events observed or no run"}

    summary = {
        "scope": {"planned_runs": N_RUNS, "planned_b_leg_evaluations": N_RUNS * 2 * ARM_LABELS.count("B"),
                  "partial": partial, "d1_authorized": False,
                  "batch_authorized_note": "prepared for owner-authorized execution only"},
        "meta": meta, "runs": runs_out, "totals": totals, "bounds": bounds,
        "nativestub_compliance_limitation": NATIVESTUB_LIMITATION, "statistics_note": STATS_NOTE,
    }

    md: List[str] = ["# Validation batch summary", ""]
    if partial:
        md += ["**PARTIAL: the batch did not complete all 7 runs. Nothing was replaced or extended.**", ""]
    md += ["## Header", "",
           f"- Pin P: `{meta.get('pin')}`; M `{M_SHA}`; bridge blobs equal B `{B_SHA}`.",
           f"- Scope: exactly {N_RUNS} Stage A slice runs (A B B A) = 28 planned clean-B leg-evaluations. "
           "A-labelled cycles are recorded, not the object of the validation.",
           "- D1 and the simulator comparison are NOT authorized by this batch.",
           f"- **NativeStub compliance limitation.** {NATIVESTUB_LIMITATION}",
           f"- {STATS_NOTE}", "", "## Per run", "",
           "| run | status | harness exit | capture ok | B-leg C0 / C2 / C5 / C7 fails | max C2 spread (ms) | "
           ">30 ms: mixed / split-involved / neither / unjoined | split batches | mixed commands |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in runs_out:
        b = r["b_legs"]
        cl = b["classes_gt30"]
        md.append(
            f"| {r['run']} | {r['status']} | {r['harness_exit_status']} | {r['capture_ok']} | "
            f"{b['fails']['C0']} / {b['fails']['C2']} / {b['fails']['C5']} / {b['fails']['C7']} | "
            f"{'-' if b['max_c2_spread_ms'] is None else round(b['max_c2_spread_ms'], 3)} | "
            f"{cl['mixed_batch']} / {cl['split_involved_not_mixed']} / {cl['neither']} / {cl['unjoined']} | "
            f"{r['capture_counts']['split_batches']} | {r['capture_counts']['mixed_commands']} |")
        if r["failure"]:
            md.append(f"|  | failure: {r['failure']['kind']}: {r['failure']['detail']} | | | | | | | |")
    md += ["", "## Per cycle (full gates verbatim; validation mode is path-only)", "",
           "| run | cycle | arm | full-gate rc | full-gate verdict | full-gate compliance check | "
           "validation-mode verdict (path-only) |", "|---|---|---|---|---|---|---|"]
    for r in runs_out:
        for c in r["cycles"]:
            fg = c["full_gate"]
            md.append(f"| {r['run']} | {c['cycle']} | {c['arm']} | {fg['rc']} | {fg['verdict']} | "
                      f"{fg['compliance_gate']} ({fg['annotation']}) | "
                      f"{c['validation_mode_path_only']['verdict']} |")
    t = totals
    md += ["", "## Totals over B leg-evaluations (descriptive event counts)", "",
           f"- runs ok / failed / not run: {t['runs_ok']} / {t['runs_failed']} / {t['runs_not_run']}",
           f"- B leg-evaluations executed: {t['b_leg_evaluations']} (planned 28)",
           f"- C0 / C2 / C5 / C7 failing legs: {t['fails']['C0']} / {t['fails']['C2']} / "
           f"{t['fails']['C5']} / {t['fails']['C7']}",
           f"- commands > 30 ms by capture class: {t['classes_gt30']}",
           f"- split batches (all cycles, capture-on): {t['split_batches']}; mixed-batch commands: "
           f"{t['mixed_commands']}; split-involved-not-mixed commands: {t['split_involved_not_mixed_commands']}",
           f"- legs where the C2 diagnostic did NOT reproduce the shipped C2 (per-command figures unverified): "
           f"{t['c2_diag_unreproduced_legs']}", "", "## Upper bounds (only for zero observed events)", ""]
    for k, v in bounds.items():
        if v["one_sided_95pct_upper_bound"] is None:
            md.append(f"- {k}: {v['events']} event(s) in n = {v['n']}; bound not given ({v['assumption']}).")
        else:
            md.append(f"- {k}: 0 events in n = {v['n']}; one-sided 95% upper bound "
                      f"{100 * v['one_sided_95pct_upper_bound']:.1f}% per trial; assumption: {v['assumption']}.")
    md.append("")
    return summary, "\n".join(md)


# ---------------------------------------------------------------------------
# Verification of the clone and the imports
# ---------------------------------------------------------------------------

def verify_source(src: str, pin: str) -> Dict[str, Any]:
    src_p = Path(src)
    if _git(src, "status", "--porcelain", check=False) != "":
        raise Refusal(f"clone {src} has a dirty tree")
    head = _git(src, "rev-parse", "HEAD")
    if head != pin:
        raise Refusal(f"clone HEAD {head} != pin {pin}")
    facts = source_pin_facts(src, pin)
    for f in OPT_FILES:
        if sha256_file(src_p / f) != facts["opt_file_sha256"][f]:
            raise Refusal(f"working-tree {f} differs from its committed blob")
    facts["clone_head"] = head
    facts["worktree_clean"] = True
    return facts


def assert_imports_from(src: str) -> Dict[str, str]:
    """The modules whose provenance matters must come from the clone, never the primary checkout."""
    import importlib
    root = Path(src).resolve()
    seen = {}
    for name in ("mujoco_remote_backend", "fake_reachy_server", "tools.goalfix_cmp",
                 "tools.goalfix_cmp_report.neck_seed", "reachy_ai"):
        mod = importlib.import_module(name)
        f = Path(getattr(mod, "__file__", None) or list(mod.__path__)[0]).resolve()
        if root not in f.parents:
            raise Refusal(f"module {name} imported from {f}, not from {root}")
        seen[name] = str(f)
    return seen


# ---------------------------------------------------------------------------
# Child: one harness run with capture
# ---------------------------------------------------------------------------

def _load_jsonl(path) -> List[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def cmd_harness(args) -> int:
    run_dir = Path(args.run_dir)
    ev_dir, control_dir = run_dir / "ev", run_dir / "control"
    result: Dict[str, Any] = {"started_utc": utc_now(), "exception": None}
    exc_text = None
    stage = None
    evidence = None
    try:
        assert_imports_from(args.src)
        import capture as capmod
        import stage_a_slice as sas
        if (sas.BRIDGE_SHA["A"], sas.BRIDGE_SHA["B"]) != (A_SHA, B_SHA):
            raise Refusal("stage_a_slice.BRIDGE_SHA differs from this driver's A/B labels")
        with capmod.Capture() as c:
            stage = sas.run_stage_a_session(ev_dir, control_dir)
            try:
                evidence = sas.emit_stage_a_evidence(stage, ev_dir)
            finally:
                stage.session.close()
        records = c.records()
        c.write_jsonl(run_dir / "capture.jsonl")
        rows = _load_jsonl(evidence.run_dir / "commands.jsonl")
        overrides: Dict[int, Dict[int, float]] = {}
        for cy in stage.cycles:
            for e in cy.injected_echoes:
                overrides.setdefault(e.stub_command_index, {})[e.joint_index] = e.value_rad
        res = capmod.cross_check(records, rows, allowed_overrides=overrides)
        cl = capmod.classify(records)
        (run_dir / "crosscheck.json").write_text(json.dumps(res.as_dict(), indent=2, sort_keys=True))
        (run_dir / "classification.json").write_text(json.dumps(cl, indent=2, sort_keys=True))
        result.update({
            "ev_dir": str(ev_dir), "control_dir": str(control_dir), "run_dir": str(evidence.run_dir),
            "arm_map": str(evidence.arm_map_path), "native_log": str(evidence.native_log_path),
            "joined": [[g, s, o] for (g, s), o in sorted(res.joined.items())],
            "cross_check_ok": res.ok,
            "capture": {"n_batches": cl["n_batches"], "n_builds": cl["n_builds"],
                        "n_mixed": cl["n_mixed"], "n_split_batches": cl["n_split_batches"],
                        "n_split_involved_not_mixed": cl["n_split_involved_not_mixed"],
                        "n_unattributed_submits": cl["n_unattributed_submits"]},
            "cycles": [{
                "name": cy.name, "rep": cy.rep, "arm": cy.arm, "epoch": cy.reset_gen,
                "reset_verify_rc": cy.reset_verify_rc, "n_injected_echoes": len(cy.injected_echoes),
            } for cy in stage.cycles],
        })
        code = 0 if res.ok else 3
    except BaseException as exc:                      # noqa: BLE001 -- recorded, never swallowed
        exc_text = traceback.format_exc()
        result["exception"] = f"{type(exc).__name__}: {exc}"
        (run_dir / "harness_exception.txt").write_text(exc_text)
        code = 1
    result["finished_utc"] = utc_now()
    (run_dir / "harness_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return code


# ---------------------------------------------------------------------------
# Parent: per-run processing
# ---------------------------------------------------------------------------

def _run_proc(argv: Sequence[str], cwd: str, env: Dict[str, str], stdout: Path, stderr: Path,
              timeout: int) -> Dict[str, Any]:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    timed_out = False
    with open(stdout, "wb") as fo, open(stderr, "wb") as fe:
        try:
            proc = subprocess.run(list(argv), cwd=cwd, env=env, stdout=fo, stderr=fe, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = None, True
    return {"argv": list(argv), "rc": rc, "timed_out": timed_out, "seconds": time.monotonic() - t0,
            "stdout": stdout.name, "stderr": stderr.name}


def _read_json(path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _write_sha256sums(directory: Path, name: str = "SHA256SUMS",
                      exclude_top: Sequence[str] = ()) -> None:
    lines = []
    for p in sorted(directory.rglob("*")):
        if not p.is_file() or p.name == name and p.parent == directory:
            continue
        rel = p.relative_to(directory)
        if rel.parts and rel.parts[0] in exclude_top:
            continue
        lines.append(f"{sha256_file(p)}  ./{rel.as_posix()}")
    (directory / name).write_text("\n".join(lines) + ("\n" if lines else ""))


def analyze_run(run_dir: Path, hres: dict, capture_records: List[dict]) -> Dict[str, Any]:
    """Report-only analysis (step 4): shipped C0-C8 per leg, per-command C2 figures, and the join
    of every command > 30 ms to its capture class.  Any disagreement with the shipped C2 is
    reported and that leg's per-command figures are marked unverified (never a stop)."""
    import numpy as np
    from reachy_ai.motion import rig_routes as R
    from tools.goalfix_cmp import cycle as cyc
    from tools.goalfix_cmp import echo
    from tools.goalfix_cmp import evidence as ev
    from tools.goalfix_cmp import pathcheck as pc
    from tools.goalfix_cmp_report import neck_seed as ns
    import capture as capmod

    assert abs(pc.C2_SKEW_TOL_S - C2_TOL_S) < 1e-12
    evd = ev.verify_and_load(hres["run_dir"], "states.jsonl", "commands.jsonl", "derived-SHA256SUMS")
    classification = capmod.classify(capture_records)
    joined = {(g, s): o for g, s, o in hres["joined"]}
    control = Path(hres["control_dir"])
    out_cycles: List[dict] = []
    per_cmd_lines: List[str] = []
    for cy in hres["cycles"]:
        name, arm = cy["name"], cy["arm"]
        try:
            manifest = json.loads((control / f"cycle_{name}.json").read_text())
            legs = []
            for label, key, route_name in ns._LEGS:
                spec, leg, why = ns._locate_leg(evd, control, manifest, label, key, route_name,
                                                str(Path(hres["run_dir"]).resolve()))
                if spec is None:
                    raise ev.EvidenceError(f"could not locate {label} leg: {why}")
                legs.append(spec)
            cv, leg_results = cyc.evaluate_cycle(name, arm, evd, legs, skip_gates=True)
        except (ev.EvidenceError, OSError, ValueError) as exc:
            # A data/evidence problem is recorded for this cycle and is NOT a stop condition.
            out_cycles.append({"name": name, "arm": arm, "error": f"{type(exc).__name__}: {exc}"})
            continue

        # Re-derive the R-carry' withdrawal rows exactly as cycle.evaluate_cycle does.
        full_ctx = [None] * len(evd.commands)
        turn_on_map: Dict[int, int] = {}
        for spec in legs:
            lr = leg_results[spec.name]
            for local_i, gi in enumerate(spec.command_indices):
                full_ctx[gi] = lr.goto_context[local_i]
            t_idx = spec.turn_on_state_index
            if t_idx is None:
                t_idx = ev.leg_turn_on_state_index(evd, spec.command_indices)
            if t_idx is not None:
                turn_on_map[spec.command_indices[0]] = t_idx
        echo_results = echo.classify_commands(evd, full_ctx, leg_turn_on_state_index=turn_on_map)

        leg_out: Dict[str, dict] = {}
        c2_out: Dict[str, dict] = {}
        for spec in legs:
            lr = leg_results[spec.name]
            leg_out[spec.name] = {
                cid: {"passed": lr.pathcheck[cid].passed,
                      "index": lr.pathcheck[cid].first_violation_index,
                      "detail": lr.pathcheck[cid].detail}
                for cid in pc.ALL_CHECKS}
            idx = list(spec.command_indices)
            withdraw_rows = []
            for gi in idx:
                cr = echo_results[gi]
                withdraw_rows.append({j: (cr is not None and cr.joints[j].label == echo.ECHO_CARRY)
                                      for j in pc.R_JOINTS})
            t21 = evd.commands.target_rad[np.asarray(idx)]
            t8 = [{j: float(row[k]) for k, j in enumerate(pc.R_JOINTS)} for row in t21]
            diag = c2_diagnostic(t8, spec.start_pose8, spec.route_rad, lr.assignment, withdraw_rows)
            shipped = lr.pathcheck["C2"]
            reproduced = (diag["passed"] == shipped.passed
                          and (diag["first_fail"] == shipped.first_violation_index))
            over: List[int] = []
            max_spread = None
            for mode in ("plain", "withdrawn"):
                d = diag[mode]
                if d is None:
                    continue
                for r in d["records"]:
                    if r["n_tau"] >= 2:
                        gi = idx[r["local_index"]]
                        per_cmd_lines.append(json.dumps({
                            "cycle": name, "arm": arm, "leg": spec.name, "mode": mode,
                            "global_index": gi, "local_index": r["local_index"], "goal": r["goal"],
                            "n_tau": r["n_tau"], "taus": r["taus"], "spread_ms": r["spread_ms"],
                            "verified": reproduced}, sort_keys=True))
                        if max_spread is None or r["spread_ms"] > max_spread:
                            max_spread = r["spread_ms"]
                        if r["spread_ms"] > (pc.C2_SKEW_TOL_S + 1e-6) * 1000.0 and gi not in over:
                            over.append(gi)
            joins = join_commands_to_capture(over, evd.commands.kind, joined, classification)
            classes = {c: 0 for c in CLASSES}
            for j in joins:
                classes[j["class"]] += 1
            c2_out[spec.name] = {
                "reproduced": reproduced, "diag_passed": diag["passed"], "diag_index": diag["first_fail"],
                "shipped_passed": shipped.passed, "shipped_index": shipped.first_violation_index,
                "max_spread_ms": max_spread,
                "n_multi_tau_commands": sum(1 for r in diag["plain"]["records"] if r["n_tau"] >= 2),
                "n_gt_30ms": len(over), "classes": classes, "joins": joins,
                "figures_status": "verified" if reproduced else "UNVERIFIED (diagnostic disagrees with shipped C2)"}
        out_cycles.append({"name": name, "arm": arm, "legs": leg_out, "c2": c2_out,
                           "verdict_skip_gates": cv.verdict, "reasons_skip_gates": cv.reasons})
    (run_dir / "c2_per_command.jsonl").write_text("\n".join(per_cmd_lines) + ("\n" if per_cmd_lines else ""))
    doc = {"cycles": out_cycles, "report_only": True,
           "class_definitions": "capture.classify (mixed_batch / split_involved_not_mixed / neither)"}
    (run_dir / "analysis.json").write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    return doc


def _cli_env(src: str) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [f"{src}/src", f"{src}/native_mujoco", f"{src}/scripts", f"{src}/tests/integration"])
    env["REQUIRE_REACHY_SDK"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def process_run(idx: int, out: Path, src: str, pin: str, keyframe_sha256: str) -> Dict[str, Any]:
    rd = out / f"run-{idx:02d}"
    rd.mkdir()
    (rd / "cli").mkdir()
    env = _cli_env(src)
    rec: Dict[str, Any] = {"run": rd.name, "index": idx, "status": "ok", "failure": None,
                           "started_utc": utc_now(), "cycles": []}
    facts: Dict[str, Any] = {"disk_free_ok": True, "missing_artifacts": []}

    # Steps 1-2: harness + capture (child process).
    h = _run_proc([sys.executable, str(Path(src) / DRIVER_REL), "harness", "--src", src,
                   "--run-dir", str(rd)], src, env, rd / "harness.stdout", rd / "harness.stderr",
                  HARNESS_TIMEOUT_S)
    hres = _read_json(rd / "harness_result.json") or {}
    rec["harness"] = {"exit_status": h["rc"], "timed_out": h["timed_out"], "seconds": h["seconds"]}
    facts.update({"harness_exit_status": h["rc"], "harness_timed_out": h["timed_out"],
                  "harness_exception": hres.get("exception") if hres else
                  ("no harness_result.json" if h["rc"] not in (0, 3) else None)})
    cc = _read_json(rd / "crosscheck.json")
    facts["capture_ok"] = bool(cc and cc.get("ok") is True)
    facts["capture_detail"] = "; ".join((cc or {}).get("failures") or ["no cross-check result"])[:2000]
    cap = dict(hres.get("capture") or {})
    cap["ok"] = facts["capture_ok"]
    rec["capture"] = cap
    facts["reset_verify_rcs"] = [c["reset_verify_rc"] for c in hres.get("cycles") or []]

    have_evidence = bool(hres.get("run_dir")) and (Path(hres["run_dir"]) / "commands.jsonl").is_file()
    capture_records: List[dict] = []
    if (rd / "capture.jsonl").is_file():
        capture_records = _load_jsonl(rd / "capture.jsonl")

    # Steps 3-5 run best-effort whenever evidence exists (also after a capture failure).
    if have_evidence:
        control = Path(hres["control_dir"])
        run_dir = hres["run_dir"]
        base = ["--sha256sums", "derived-SHA256SUMS"]
        for cy in hres["cycles"]:
            name, rep, arm = cy["name"], cy["rep"], cy["arm"]
            crec = {"name": name, "rep": rep, "arm": arm, "epoch": cy["epoch"],
                    "reset_verify_rc": cy["reset_verify_rc"]}
            full_out = control / f"between_{name}.json"
            argv_full = ["--ev-dir", run_dir, *base, "--control-dir", str(control), "--cycle", name,
                         "--rep", str(rep), "--arm", arm, "--arm-map", hres["arm_map"],
                         "--expected-host-sha", pin, "--required-supervisor-programs", "reachy-sdk-server",
                         "--expected-bridge-sha-a", A_SHA, "--expected-bridge-sha-b", B_SHA,
                         "--out", str(full_out)]
            r_full = _run_proc([sys.executable, "-m", "tools.goalfix_cmp.cycle", *argv_full], src, env,
                               rd / "cli" / f"full_{name}.stdout", rd / "cli" / f"full_{name}.stderr",
                               CLI_TIMEOUT_S)
            val_out = rd / "cli" / f"validation_{name}.json"
            argv_val = ["--ev-dir", run_dir, *base, "--control-dir", str(control), "--cycle", name,
                        "--rep", str(rep), "--arm", arm, "--out", str(val_out), "--validation-mode"]
            r_val = _run_proc([sys.executable, "-m", "tools.goalfix_cmp.cycle", *argv_val], src, env,
                              rd / "cli" / f"validation_{name}.stdout", rd / "cli" / f"validation_{name}.stderr",
                              CLI_TIMEOUT_S)
            fj, vj = _read_json(full_out), _read_json(val_out)
            crec["full_gate"] = {"argv": argv_full, "rc": r_full["rc"], "out": str(full_out),
                                 "verdict": (fj or {}).get("verdict"),
                                 "compliance": ((fj or {}).get("checks") or {}).get("compliance"),
                                 "reasons": (fj or {}).get("reasons")} if fj is not None else \
                {"argv": argv_full, "rc": r_full["rc"], "out": str(full_out), "verdict": None}
            crec["validation_mode"] = {"argv": argv_val, "rc": r_val["rc"], "out": str(val_out),
                                       "verdict": (vj or {}).get("verdict"),
                                       "reasons": (vj or {}).get("reasons")}
            for tag, path in (("full", full_out), ("validation", val_out)):
                if not path.is_file():
                    facts["missing_artifacts"].append(f"{name} {tag} CLI output")

            neck_out = rd / "neck" / f"neck_{name}.json"
            neck_out.parent.mkdir(exist_ok=True)
            argv_neck = ["--ev-dir", run_dir, *base, "--control-dir", str(control), "--cycle", name,
                         "--arm", arm, "--epoch", str(cy["epoch"]),
                         "--keyframe-xml", str(Path(src) / KEYFRAME_REL), "--keyframe-name", KEYFRAME_NAME,
                         "--keyframe-sha256", keyframe_sha256, "--keyframe-source-rev", M_SHA,
                         "--state-every", str(_native_state_every()), "--fixed-sim-step", str(FIXED_SIM_STEP),
                         "--out", str(neck_out)]
            r_neck = _run_proc([sys.executable, "-m", "tools.goalfix_cmp_report.neck_seed", *argv_neck],
                               src, env, rd / "neck" / f"neck_{name}.stdout", rd / "neck" / f"neck_{name}.stderr",
                               CLI_TIMEOUT_S)
            nj = _read_json(neck_out)
            crec["neck_seed"] = {"argv": argv_neck, "rc": r_neck["rc"], "out": str(neck_out),
                                 "all_ok": (nj or {}).get("all_ok")}
            if nj is None:
                facts["missing_artifacts"].append(f"{name} neck/seed output")
            rec["cycles"].append(crec)

        cp_out = rd / "cli" / "checkpoint_n4.json"
        argv_cp = ["checkpoint", "--control-dir", str(control), "--arm-map", hres["arm_map"],
                   "--expected-bridge-sha-a", A_SHA, "--expected-bridge-sha-b", B_SHA, "--n", "4",
                   "--native-log", hres["native_log"], "--states", str(Path(run_dir) / "states.jsonl"),
                   "--out", str(cp_out)]
        r_cp = _run_proc([sys.executable, "-m", "tools.goalfix_cmp.summary", *argv_cp], src, env,
                         rd / "cli" / "checkpoint_n4.stdout", rd / "cli" / "checkpoint_n4.stderr", CLI_TIMEOUT_S)
        rec["checkpoint"] = {"argv": argv_cp, "rc": r_cp["rc"], "out": str(cp_out)}
        if not cp_out.is_file():
            facts["missing_artifacts"].append("summary checkpoint output")

        # Step 4: analysis (in-process, report-only).
        try:
            analysis = analyze_run(rd, hres, capture_records)
            by_name = {c["name"]: c for c in analysis["cycles"]}
            for crec in rec["cycles"]:
                a = by_name.get(crec["name"])
                if a and "error" in a:
                    crec["analysis_error"] = a["error"]
                elif a:
                    crec["legs"], crec["c2"] = a["legs"], {
                        k: {kk: vv for kk, vv in v.items() if kk != "joins"} for k, v in a["c2"].items()}
        except BaseException:                                   # noqa: BLE001
            (rd / "analysis_exception.txt").write_text(traceback.format_exc())
            facts["missing_artifacts"].append("analysis.json")
    else:
        facts["missing_artifacts"].append("evidence (commands.jsonl)")

    for need in ("capture.jsonl", "crosscheck.json", "harness_result.json"):
        if not (rd / need).is_file():
            facts["missing_artifacts"].append(need)
    free = shutil.disk_usage(str(out)).free
    facts["disk_free_ok"] = free >= MIN_FREE_BYTES
    failure = classify_failure(facts)
    if failure:
        rec["status"], rec["failure"] = "failed", failure
    rec["finished_utc"] = utc_now()
    (rd / "run_manifest.json").write_text(json.dumps(rec, indent=2, sort_keys=True, default=str))
    _write_sha256sums(rd)
    return rec


def _native_state_every() -> int:
    """States are published every ``NativeStub.SUBSTEPS`` sim steps (native_stub.py), the
    state_every the plan's P4 refers to for this harness."""
    from native_stub import NativeStub
    return int(NativeStub.SUBSTEPS)


def environment_record(pin: str, out: Path, venv: str, facts: Dict[str, Any], start_utc: str) -> Dict[str, Any]:
    def run(argv):
        try:
            return subprocess.run(argv, capture_output=True, text=True, timeout=120).stdout
        except Exception as exc:                                    # noqa: BLE001
            return f"unavailable: {exc}"
    return {
        "python_version": sys.version, "python_executable": sys.executable,
        "pip_freeze": run([sys.executable, "-m", "pip", "freeze"]).splitlines(),
        "uname_a": run(["uname", "-a"]).strip(), "venv": venv,
        "REQUIRE_REACHY_SDK": os.environ.get("REQUIRE_REACHY_SDK"), "TMPDIR": os.environ.get("TMPDIR"),
        "start_utc": start_utc, "end_utc": None,
        "free_disk_bytes_start": shutil.disk_usage(str(out)).free, "free_disk_bytes_end": None,
        "disk_floor_bytes": MIN_FREE_BYTES, "pin_facts": facts,
    }


def run_loop(process, n_runs: int = N_RUNS, disk_ok=lambda: True) -> List[dict]:
    """The batch's control flow, in isolation: run ``run-01 .. run-NN`` in order via
    ``process(i) -> record``; stop early ONLY when a record has ``status == "failed"`` (an
    infrastructure/capture failure, see ``classify_failure``) or the disk floor is hit.  The
    remaining runs are recorded ``not_run`` and never executed; there is no retry, no
    replacement and no extension.  A run whose analysis shows C0-C8 failures is a normal
    ``ok`` record and never stops anything."""
    records: List[dict] = []
    stopped = False
    for i in range(1, n_runs + 1):
        if stopped:
            records.append({"run": f"run-{i:02d}", "index": i, "status": "not_run", "failure": None,
                            "cycles": [], "note": "not run: an earlier run stopped the batch"})
            continue
        if not disk_ok():
            records.append({"run": f"run-{i:02d}", "index": i, "status": "not_run", "cycles": [],
                            "failure": classify_failure({"disk_free_ok": False})})
            stopped = True
            continue
        rec = process(i)
        records.append(rec)
        if rec.get("status") == "failed":
            stopped = True
    return records


def cmd_run(args) -> int:
    out = Path(args.out).resolve()
    src = str((out / "src").resolve())
    start = utc_now()
    try:
        facts = verify_source(src, args.pin)
        seen = assert_imports_from(src)
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    facts["imports_from_clone"] = seen
    env_rec = environment_record(args.pin, out, args.venv, facts, start)
    (out / "environment.json").write_text(json.dumps(env_rec, indent=2, sort_keys=True))
    kf = facts["keyframe_sha256"]
    records = run_loop(lambda i: process_run(i, out, src, args.pin, kf),
                       disk_ok=lambda: shutil.disk_usage(str(out)).free >= MIN_FREE_BYTES)
    stopped = any(r["status"] != "ok" for r in records)
    env_rec["end_utc"] = utc_now()
    env_rec["free_disk_bytes_end"] = shutil.disk_usage(str(out)).free
    (out / "environment.json").write_text(json.dumps(env_rec, indent=2, sort_keys=True))
    summary, md = build_summary(records, {"pin": args.pin, "start_utc": start, "end_utc": env_rec["end_utc"]})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    (out / "SUMMARY.md").write_text(md)
    _write_sha256sums(out, exclude_top=("src", "tmp"))
    return 1 if stopped else 0


def cmd_preflight(args) -> int:
    try:
        facts = preflight(args.pin, args.out, args.repo, args.runs, args.venv)
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print("\n".join(plan_lines(args.pin, facts["out"])))
    print("\nPREFLIGHT OK (tools/goalfix_cmp tree and the three bridge blobs equal; nothing executed)")
    print(f"  P:tools/goalfix_cmp = {facts['P_tools_goalfix_cmp_tree']}  (== M's)")
    for f, s in facts["opt_file_sha256"].items():
        print(f"  {f} sha256 = {s}  (== B's blob)")
    print(f"  {facts['keyframe_rel']} sha256 = {facts['keyframe_sha256']}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--pin", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--venv", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pin", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--venv", required=True)
    h = sub.add_parser("harness")
    h.add_argument("--src", required=True)
    h.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "preflight":
        return cmd_preflight(args)
    if args.cmd == "run":
        return cmd_run(args)
    return cmd_harness(args)


if __name__ == "__main__":
    raise SystemExit(main())
