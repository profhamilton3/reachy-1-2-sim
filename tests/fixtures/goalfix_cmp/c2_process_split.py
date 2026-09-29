"""The fixed 7-run, U-only, process-separated C2 diagnostic: driver.

PREPARATION ONLY (written, unit-tested on synthetic inputs and dry-run only; NOT executed).
Neither execution nor decision D1 is authorized by the existence of this file.

QUESTION (fixed): with the real ``reachy_sdk`` 0.7.0 client in its own process, the gRPC server +
bridge in a second process, and NativeStub + the evidence writer (Observer) in a third, do B
setup legs still show C2 per-command tau spreads > 30 ms?  Outputs: occurrence, spread and C0-C8
check results only.  NO ATTRIBUTION: without capture, nothing here can say where a spread formed,
and the driver does not attempt it.

Sub-commands (module import is stdlib + ``c2ps_common`` only; numpy, ``tools.goalfix_cmp`` and the
SDK are imported lazily, and only by ``run`` -- ``preflight`` and ``--dry-run`` import none of
``reachy_sdk``, ``grpc``, ``websockets``, ``native_stub``, ``mujoco_remote_backend``,
``fake_reachy_server``, and start no process other than ``git``):

* ``preflight``   arguments, pins, trees, blobs, output path, free space; prints the plan.
* ``run``         the batch controller: verifies the clone, runs run-01..run-07, analyses, summarises.
* ``coordinate``  one run's coordinator (spawned by ``run`` in its own session): spawns N, B_k, C_k,
                  drives the slice's 22 steps, records provenance; never constructs a stub,
                  observer, backend, server or SDK client.

PROCESS ARRANGEMENT (per run): N = NativeStub + Observer + request server (whole run);
B_k = one bridge generation; C_k = the SDK clients of cycle k (setup + flight); plus the
coordinator as a fourth, mostly idle process.  No timestamp taken in one process is compared with
or merged into one taken in another: ``reset_t`` is taken in N, the move-wrapper counts and
seq/sim_step reads are requests to N.

ADAPTED FROM PR #146's ``validation_batch.py`` (read-only reference, nothing imported from it):
``check_out_path``/``check_runs``/``check_pin_format``/``_git``/``source_pin_facts`` (now also the
16 runtime trees, ancestry and the added-files-only diff), ``classify_failure`` (new fact set),
``c2_records``/``c2_diagnostic`` (unchanged), ``annotate_full_gate`` (unchanged rule),
``analyze_run`` (legs located by ``cycle.resolve_leg`` as the ``cycle`` CLI does, not the extractor;
the capture join and neck/seed extractor removed), ``process_run`` (coordinator process instead of
the in-process harness), ``run_loop`` (adds a pre-run cap check), ``build_summary`` (counts only,
setup and flight separate, no statistics).

NATIVESTUB COMPLIANCE LIMITATION: see ``NATIVESTUB_LIMITATION`` (stated in every SUMMARY header).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import c2ps_common as cc  # noqa: E402
from c2ps_common import (  # noqa: E402
    A_SHA, ALLOWED_NEW_PREFIXES, ARM_LABELS, B_SHA, BRIDGE_OPT_FILES, CHILD_EXIT_TIMEOUT_S,
    CLI_TIMEOUT_S, COORDINATOR_TERM_GRACE_S, CYCLE_NAMES, ECHO_SEED_BASE, EMIT_TIMEOUT_S,
    FORBIDDEN_OUT_PREFIXES, InfraFailure, LEG_TIMEOUT_S, MIN_FREE_BYTES, M_PRIME_SHA,
    M_PRIME_TOOLS_TREE, N_REQUEST_TIMEOUT_S, N_RUNS, OUT_CAP_BYTES, PROCESS_READY_TIMEOUT_S,
    RESET_TIMEOUT_S, RESET_VERIFY_TIMEOUT_S, RUN_CAP_S, RUN_NAME, RUNTIME_TREES, StepTimeout,
    TERM_GRACE_S, TOTAL_CAP_S, utc_now,
)

DRIVER_REL = "tests/fixtures/goalfix_cmp/c2_process_split.py"
CHILDREN_REL = "tests/fixtures/goalfix_cmp/c2ps_children.py"
C2_TOL_S = 0.030                         # pathcheck.C2_SKEW_TOL_S (asserted at analysis time)
CHECKS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
LEG_KINDS = ("setup", "flight")

NATIVESTUB_LIMITATION = (
    "NativeStub sets one compliance flag for all 21 joints (tests/integration/native_stub.py:33-43); "
    "the full-gate compliance check fails on every cycle; this is a harness limitation; no passing "
    "native-compliance result is ever reported; path-only (validation-mode) success is not "
    "full-gate success.")

REFERENCE_SENTENCE = (
    "Reference, stated for orientation only (earlier single-process harness, capture on): 5 of 14 "
    "B setup legs and 0 of 14 B flight legs had a per-command C2 spread over 30 ms, with events in "
    "4 of 7 runs. This diagnostic is not a replication of that batch.")
LIMITATION_SENTENCES = (
    "Zero events here would not prove that shared-process contention caused the earlier failures.",
    "No native-stack rate is given: NativeStub replaces native MuJoCo and the bridge runs natively on "
    "the host, not in the container.",
    "No justification for the 30 ms C2 threshold is provided, and none may be derived from this run.",
)
ATTRIBUTION_SENTENCE = (
    "Attribution is unavailable: this diagnostic takes no capture, so it cannot say where any "
    "spread formed.")
SCOPE_SENTENCE = (
    f"Scope: exactly {N_RUNS} runs, all uninstrumented, each one complete four-cycle slice (A B B A); "
    "A cycles are recorded with C0-C8 and excluded from the C2 counts (they carry injected echoes). "
    "Legs within a run share processes, a bridge history and a host state; the counts below are "
    "descriptive event counts with the number of evaluated legs shown beside each.")

#: The slice's per-cycle step order, as implemented here (process in brackets).
STEP_PLAN: Tuple[str, ...] = (
    "1. [coordinator, B_{k-1}] bridge-log capture for cycle k begins (file truncated, B_{k-1} attaches an append handler)",
    "2. [C_{k-1}, B_{k-1}] C_{k-1} closes its clients' gRPC channels and exits; THEN B_{k-1} stops server and backend and exits",
    "3. [N] recreate_timestamp = time.time()",
    "4. [coordinator] spawn B_k (backend start, READY wait, gRPC server on 127.0.0.1:0), then C_k (fully imported, idle)",
    "5. [N] pre_reset_sim_step = observer.last_sim_step()",
    "6. [N] reset_t = time.monotonic()  (same process as NativeStub.commands[i]['t'])",
    "7. [N, coordinator] flush partial run dir; real reset_verify.py snapshot",
    "8. [B_k] bridge request_reset() + ack wait + _last_target wait",
    "9. [N] commit reset event (reset_t, gen, pre_reset_sim_step)",
    "10. [coordinator] sleep 0.15",
    "11. [N, coordinator] flush; real reset_verify.py verify; reset record written",
    "12. [N] setup_first_seq = observer.last_seq()",
    "13. [C_k] new ReachySDK client (sleep 0.2), turn_on('r_arm'), sleep 0.3",
    "14. [C_k -> N] turn_on_state_msg = observer.snapshot()[-1]",
    "15. [C_k] fly_route(PLACE_ROUTE) with the slice's logging move wrapper (counts requested from N)",
    "16. [C_k -> N] setup_last_seq = observer.last_seq()",
    "17. [N] arm A only: inject_synthetic_echoes on the stub's log (seed 20260925 + rep)",
    "18. [coordinator] sleep 0.3",
    "19. [N] flight_first_seq = observer.last_seq()",
    "20. [C_k] new ReachySDK client (sleep 0.2), turn_on('r_arm'), sleep 0.3",
    "21. [C_k] fly_route(LIFT_TO_PRESENT) with the same wrapper",
    "22. [C_k -> N] flight_last_seq = observer.last_seq()",
)


class Refusal(Exception):
    """A refusal to start (bad argument, pin, tree or path). Never a result."""


class Interrupted(BaseException):
    """SIGINT/SIGTERM delivered to the batch controller."""


# ---------------------------------------------------------------------------
# Pure helpers (adapted from P's driver)
# ---------------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_out_path(path: str, forbidden: Sequence[str] = FORBIDDEN_OUT_PREFIXES) -> str:
    """Absolute after ``~`` expansion, outside ``/tmp`` and ``/private/tmp``, must not exist."""
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


def check_free_space(path: str, floor: int = MIN_FREE_BYTES) -> int:
    free = shutil.disk_usage(path).free
    if free < floor:
        raise Refusal(f"free space {free} bytes at {path} is below the {floor} byte precondition")
    return free


def _git_run(repo, *args: str) -> "subprocess.CompletedProcess":
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _git(repo, *args: str, check: bool = True) -> str:
    proc = _git_run(repo, *args)
    if check and proc.returncode != 0:
        raise Refusal(f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip()}")
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _git_show_bytes(repo, rev_path: str) -> bytes:
    proc = subprocess.run(["git", "-C", str(repo), "show", rev_path], capture_output=True)
    if proc.returncode != 0:
        raise Refusal(f"git show {rev_path} failed in {repo}: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def source_pin_facts(repo, pin: str, *, m_sha: str = M_PRIME_SHA, b_sha: str = B_SHA,
                     m_tools_tree: str = M_PRIME_TOOLS_TREE,
                     runtime_trees: Sequence[str] = RUNTIME_TREES) -> Dict[str, Any]:
    """Pin equalities evaluated against a git repo (the source repo before the clone, the clone
    after).  Raises ``Refusal`` on any mismatch.  Only ``git`` is run."""
    for label, rev in (("pin", pin), ("M'", m_sha), ("B", b_sha)):
        if not _git(repo, "cat-file", "-t", f"{rev}^{{commit}}", check=False):
            raise Refusal(f"commit {label}={rev} is not present in {repo}")
    p_tools = _git(repo, "rev-parse", f"{pin}:tools/goalfix_cmp")
    m_tools = _git(repo, "rev-parse", f"{m_sha}:tools/goalfix_cmp")
    if p_tools != m_tools or p_tools != m_tools_tree:
        raise Refusal(f"tools/goalfix_cmp tree differs: pin has {p_tools}, M' has {m_tools} "
                      f"(expected {m_tools_tree})")
    blobs = {}
    for f in BRIDGE_OPT_FILES:
        sp = sha256_bytes(_git_show_bytes(repo, f"{pin}:{f}"))
        sb = sha256_bytes(_git_show_bytes(repo, f"{b_sha}:{f}"))
        if sp != sb:
            raise Refusal(f"{f} at the pin differs from B's blob (sha256 {sp} vs {sb})")
        blobs[f] = sp
    trees = {}
    for t in runtime_trees:
        tp = _git(repo, "rev-parse", f"{pin}:{t}")
        tb = _git(repo, "rev-parse", f"{b_sha}:{t}")
        if tp != tb:
            raise Refusal(f"runtime tree {t} at the pin ({tp}) differs from B's ({tb})")
        trees[t] = tp
    if _git_run(repo, "merge-base", "--is-ancestor", m_sha, pin).returncode != 0:
        raise Refusal(f"the pin {pin} does not descend from M' {m_sha}")
    names = _git(repo, "diff", "--name-status", m_sha, pin)
    for line in names.splitlines():
        status, _, path = line.partition("\t")
        if status != "A" or not path.startswith(ALLOWED_NEW_PREFIXES):
            raise Refusal(f"git diff M'..pin lists {line!r}: only NEW files under "
                          f"{', '.join(ALLOWED_NEW_PREFIXES)} are allowed")
    for rel in (DRIVER_REL, CHILDREN_REL, "tests/fixtures/goalfix_cmp/c2ps_common.py"):
        if not _git(repo, "cat-file", "-t", f"{pin}:{rel}", check=False):
            raise Refusal(f"{rel} is not present at the pin")
    return {"pin": pin, "M_prime": m_sha, "B": b_sha, "A": A_SHA,
            "pin_tools_goalfix_cmp_tree": p_tools, "M_prime_tools_goalfix_cmp_tree": m_tools,
            "opt_file_sha256": blobs, "runtime_trees": trees,
            "added_files": [l.partition("\t")[2] for l in names.splitlines()]}


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
        raise Refusal("the pin is not reachable from any ref of the source repo; "
                      "`git clone --local` would not carry it")
    facts["free_bytes"] = check_free_space(os.path.dirname(out_abs))
    facts.update({"out": out_abs, "venv": os.path.expanduser(venv), "runs": N_RUNS,
                  "source_repo": repo})
    return facts


def constants_table() -> Dict[str, Any]:
    return {
        "process_ready_timeout_s": PROCESS_READY_TIMEOUT_S, "reset_timeout_s": RESET_TIMEOUT_S,
        "leg_timeout_s": LEG_TIMEOUT_S, "n_request_timeout_s": N_REQUEST_TIMEOUT_S,
        "emit_timeout_s": EMIT_TIMEOUT_S, "child_exit_timeout_s": CHILD_EXIT_TIMEOUT_S,
        "term_grace_s": TERM_GRACE_S, "reset_verify_timeout_s": RESET_VERIFY_TIMEOUT_S,
        "cli_timeout_s": CLI_TIMEOUT_S, "coordinator_term_grace_s": COORDINATOR_TERM_GRACE_S,
        "run_cap_s": RUN_CAP_S, "total_cap_s": TOTAL_CAP_S, "out_cap_bytes": OUT_CAP_BYTES,
        "min_free_bytes": MIN_FREE_BYTES,
    }


def plan_lines(pin: str, out: str) -> List[str]:
    k = constants_table()
    lines = [
        "C2 PROCESS-SPLIT (U-ONLY) PLAN (fixed scope; nothing below has been executed by --dry-run)",
        f"  pin (reviewed preparation head): {pin}",
        f"  M' (base; tools/goalfix_cmp tree must equal {M_PRIME_TOOLS_TREE}): {M_PRIME_SHA}",
        f"  B (bridge blobs and the {len(RUNTIME_TREES)} runtime trees must equal): {B_SHA}     A (label only): {A_SHA}",
        f"  out: {out}",
        f"  source: git clone --local --no-checkout <repo> {out}/src ; git checkout --detach {pin}",
        f"  runs: exactly {N_RUNS} (run-01..run-{N_RUNS:02d}), all uninstrumented (U), in order; no retry, "
        "no replacement, no extension",
        f"  cycles per run: {len(ARM_LABELS)} in order {' '.join(ARM_LABELS)} ({', '.join(CYCLE_NAMES)}); "
        f"B legs per run = {2 * ARM_LABELS.count('B')}; total B leg-evaluations = {N_RUNS * 2 * ARM_LABELS.count('B')}",
        "  caps: "
        f"{k['run_cap_s'] // 60} min wall per run (controller kills the run's process group and every recorded child); "
        f"{k['total_cap_s'] // 60} min wall total; {k['out_cap_bytes'] / 2 ** 30:.1f} GiB total output "
        f"(excluding src/, checked after each run); >= {k['min_free_bytes'] // 2 ** 30} GiB free before the batch and before each run",
        "  step timeouts (s): "
        f"process ready {k['process_ready_timeout_s']}, reset {k['reset_timeout_s']}, each leg / client step "
        f"{k['leg_timeout_s']}, request to N {k['n_request_timeout_s']}, evidence emit {k['emit_timeout_s']}, "
        f"child exit {k['child_exit_timeout_s']}, terminate grace {k['term_grace_s']}, reset_verify "
        f"{k['reset_verify_timeout_s']}, each CLI {k['cli_timeout_s']}",
        "  processes per run: coordinator (orchestration only) ; N (NativeStub + Observer + request server, whole run) ;",
        "    B_0..B_4 (one bridge generation each; B_0 = Stage 0) ; C_0..C_4 (SDK clients of one cycle each; C_0 = Stage 0)",
        "    every child: fresh interpreter, own session/process group, stdout/stderr to files, provenance in <run>/proc/",
        "  stop: only an infrastructure/evidence failure (child start/death, step timeout, separation or provenance "
        "assertion, gRPC/websocket failure, reset_verify non-zero, missing/empty/malformed evidence or CLI output, "
        "checksum mismatch, C2 diagnostic not computable, leftover process, cap); C0-C8 failures never stop anything",
    ]
    for i in range(1, N_RUNS + 1):
        rd = f"{out}/run-{i:02d}"
        lines += [
            f"  run-{i:02d}:",
            f"    a. coordinator subprocess -> {rd}/proc/ (children, provenance, pids, exit codes); evidence -> {rd}/ev, {rd}/control",
            "    b. Stage 0: B_0 up, C_0 up, C_0 client + turn_on('r_arm') + sleep 0.3; then per cycle k = 1..4 (A B B A):",
        ]
        lines += [f"         step {s}" for s in STEP_PLAN]
        lines += [
            "    c. end of run: emit evidence while the last C/B are alive, then close C, stop B, shut down N",
            f"    d. no leftover process check; cycle CLI per cycle (full gates --expected-host-sha {pin}, and --validation-mode); "
            f"summary checkpoint --n 4 -> {rd}/cli/",
            f"    e. analysis (report-only): shipped C0-C8 per leg, per-command C2 spread -> {rd}/analysis.json, c2_per_command.jsonl",
            f"    f. {rd}/run_manifest.json and {rd}/SHA256SUMS",
        ]
    lines += [
        f"  then: {out}/SUMMARY.md, {out}/summary.json, {out}/environment.json, root {out}/SHA256SUMS, "
        f"archive {out}.tar.zst (+ .sha256), never overwriting; nothing in {out} is ever deleted",
        "  NativeStub compliance limitation: " + NATIVESTUB_LIMITATION,
    ]
    return lines


# ---------------------------------------------------------------------------
# Stop-condition logic (pure)
# ---------------------------------------------------------------------------

def classify_failure(facts: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """The ONLY thing that can stop the batch: an infrastructure or evidence failure.  Returns
    ``{"kind", "detail"}`` or ``None``.  Any other key in ``facts`` (a C0-C8 failure, a C2
    failure, a verdict, a spread, an event count) is deliberately never consulted."""
    if facts.get("cap"):
        return {"kind": str(facts["cap"]), "detail": str(facts.get("cap_detail") or "cap reached")}
    if facts.get("interrupted"):
        return {"kind": "interrupted", "detail": "SIGINT/SIGTERM delivered to the batch controller"}
    cf = facts.get("coordinator_failure")
    if cf:
        return {"kind": str(cf.get("kind", "coordinator")), "detail": str(cf.get("detail", ""))}
    if facts.get("leftover_pids"):
        return {"kind": "leftover_process", "detail": f"pids still alive: {facts['leftover_pids']}"}
    if facts.get("provenance_problems"):
        return {"kind": "separation", "detail": "; ".join(map(str, facts["provenance_problems"]))[:2000]}
    bad_exit = {k: v for k, v in (facts.get("child_exit_codes") or {}).items() if v != 0}
    if bad_exit:
        return {"kind": "child_died", "detail": f"non-zero child exit codes: {bad_exit}"}
    bad = [rc for rc in (facts.get("reset_verify_rcs") or []) if rc != 0]
    if bad:
        return {"kind": "reset_verify", "detail": f"reset_verify rc {bad}"}
    if facts.get("checksum_mismatch"):
        return {"kind": "evidence_checksum", "detail": str(facts["checksum_mismatch"])}
    missing = list(facts.get("missing_artifacts") or [])
    if missing:
        return {"kind": "missing_artifact", "detail": ", ".join(missing)}
    malformed = list(facts.get("malformed_artifacts") or [])
    if malformed:
        return {"kind": "malformed_artifact", "detail": ", ".join(malformed)}
    unc = list(facts.get("incomplete_legs") or [])
    if unc:
        return {"kind": "c2_not_computable", "detail": ", ".join(unc)}
    return None


_CONNECTION_MARKERS = ("grpc", "websocket", "unavailable", "connectionclosed", "connection refused",
                       "did not connect", "connection reset")


def classify_exception(exc: BaseException) -> Dict[str, str]:
    """Maps an exception raised while driving a run to a stop record."""
    if isinstance(exc, InfraFailure):
        return {"kind": exc.kind, "detail": exc.detail}
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, cc.RpcError) and any(m in text.lower() for m in _CONNECTION_MARKERS):
        return {"kind": "connection", "detail": text}
    if isinstance(exc, cc.RpcError):
        return {"kind": "child_error", "detail": text}
    return {"kind": "coordinator_exception", "detail": text}


# ---------------------------------------------------------------------------
# Coordinator: the slice's 22 steps over three process roles
# ---------------------------------------------------------------------------

def echo_injection_range(flown_setup: Sequence[str], setup_moves: Sequence[dict]) -> Optional[Tuple[int, int]]:
    """``stage_a_slice.run_stage_a_session``'s own target for the A-cycle echo injection (its
    lines 441-460, re-expressed over dict move records): the plan §7.1 affected segment (HOVER
    through REST_SHUT) when both were flown, else the whole leg; ``None`` when no move was made."""
    if not setup_moves:
        return None
    if "HOVER" in flown_setup and "REST_SHUT" in flown_setup:
        h, r = flown_setup.index("HOVER"), flown_setup.index("REST_SHUT")
        return int(setup_moves[h]["first_command_index"]), int(setup_moves[r]["last_command_index"])
    return int(setup_moves[0]["first_command_index"]), int(setup_moves[-1]["last_command_index"])


def reset_record_text(verify_out: str, verify_rc: int, verify_err: str) -> str:
    """``stage_a_slice``'s record text (lines 426-429)."""
    text = verify_out
    if verify_rc != 0:
        text += f"# reset_verify.py exited {verify_rc}: {verify_err.strip()}\n"
    return text


def make_run_reset_verify(clone: str, timeout: float = RESET_VERIFY_TIMEOUT_S) -> Callable[[Sequence[str]], Tuple[int, str, str]]:
    """``stage_a_slice.run_reset_verify`` re-expressed (same script, same PYTHONPATH recipe, same
    cwd) without importing the slice into the coordinator."""
    def run(argv: Sequence[str]) -> Tuple[int, str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path(clone) / "src"), str(Path(clone) / "native_mujoco"), str(Path(clone) / "scripts"),
             env.get("PYTHONPATH", "")])
        try:
            proc = subprocess.run(
                [sys.executable, str(Path(clone) / "scripts" / "e1_stage1" / "reset_verify.py")] + list(argv),
                cwd=str(clone), env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise StepTimeout(f"reset_verify.py {list(argv)[:1]} exceeded {timeout} s") from exc
        return proc.returncode, proc.stdout, proc.stderr
    return run


def expected_process_names(n_cycles: int = len(ARM_LABELS)) -> List[str]:
    return ["N"] + [f"B{k}" for k in range(n_cycles + 1)] + [f"C{k}" for k in range(n_cycles + 1)]


def verify_provenance_files(proc_dir: Path, names: Sequence[str]) -> List[str]:
    """Every expected process has a provenance file with BOTH fixed points recorded, ``ok``."""
    problems: List[str] = []
    for n in names:
        p = Path(proc_dir) / f"{n}.provenance.json"
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            problems.append(f"{n}: missing or malformed provenance file")
            continue
        recs = doc.get("records") or {}
        for point in ("after_setup", "before_exit"):
            r = recs.get(point)
            if not r:
                problems.append(f"{n}: provenance point {point} not recorded")
            elif not r.get("ok", False):
                problems.append(f"{n}@{point}: {r.get('problems')}")
    return problems


class RunOrchestrator:
    """Drives one run.  ``argv_for(role, name, **kw)`` builds a child's argv (tests supply fake
    children); ``reset_verify(argv) -> (rc, stdout, stderr)`` runs the real script.  The
    coordinator only orchestrates: it holds no stub, observer, backend, server or client."""

    def __init__(self, run_dir: Path, supervisor: "cc.Supervisor",
                 argv_for: Callable[..., List[str]],
                 reset_verify: Callable[[Sequence[str]], Tuple[int, str, str]], *,
                 sleep: Callable[[float], None] = time.sleep,
                 labels: Sequence[str] = ARM_LABELS, names: Sequence[str] = CYCLE_NAMES,
                 echo_seed: int = ECHO_SEED_BASE):
        self.run_dir = Path(run_dir)
        self.ev_dir, self.control_dir = self.run_dir / "ev", self.run_dir / "control"
        self.sup, self.argv_for, self.reset_verify = supervisor, argv_for, reset_verify
        self.sleep, self.labels, self.names, self.echo_seed = sleep, list(labels), list(names), echo_seed
        self.steps: List[dict] = []
        self._t0 = time.monotonic()

    def _step(self, cycle: Optional[str], step: str, **extra: Any) -> None:
        # Coordinator-local monotonic offsets, for the record only; never merged with a child's.
        self.steps.append({"cycle": cycle, "step": step, "t_coord_s": round(time.monotonic() - self._t0, 3), **extra})

    def _clean_exit(self, ch: "cc.Child") -> None:
        code = self.sup.wait_exit(ch, CHILD_EXIT_TIMEOUT_S)
        if code != 0:
            raise cc.ChildDied(f"{ch.name} exited with {code}")

    def _start_generation(self, k: int, n: "cc.Child", stub_port: int, log_path: Optional[Path]):
        b = self.sup.spawn(f"B{k}", self.argv_for("B", f"B{k}", stub_port=stub_port,
                                                  log_path=str(log_path) if log_path else None))
        rb = self.sup.wait_ready(b)
        c = self.sup.spawn(f"C{k}", self.argv_for("C", f"C{k}", n_port=n.port, grpc_port=rb["grpc_port"]))
        self.sup.wait_ready(c)
        return b, c

    def execute(self) -> Dict[str, Any]:
        sup = self.sup
        self.ev_dir.mkdir(parents=True, exist_ok=True)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        run_dir_s = str(self.ev_dir / "e1_server_runs" / RUN_NAME)

        n = sup.spawn("N", self.argv_for("N", "N"))
        ready_n = sup.wait_ready(n)
        stub_port = int(ready_n["stub_port"])

        def call_n(method: str, timeout: float = N_REQUEST_TIMEOUT_S, **params: Any) -> Any:
            return sup.call(n, method, params, timeout)

        # Stage 0: turn_on on generation 0 (B_0 has no bridge-log capture, as in the slice).
        b, c = self._start_generation(0, n, stub_port, None)
        sup.call(c, "client", {"key": None, "record_turn_on_state": False}, LEG_TIMEOUT_S)
        self._step(None, "stage0")

        cycles: List[dict] = []
        findings: Dict[str, Any] = {}
        for rep, (name, arm) in enumerate(zip(self.names, self.labels), start=1):
            log_path = self.control_dir / f"bridge_log_{name}.txt"
            # 1
            log_path.write_text("")
            sup.call(b, "begin_capture", {"path": str(log_path)})
            self._step(name, "1")
            # 2: A4 -- the client's channels close first, then the bridge stops
            sup.call(c, "close", {}, CHILD_EXIT_TIMEOUT_S)
            self._clean_exit(c)
            self._step(name, "2a-client-closed")
            sup.call(b, "stop", {}, CHILD_EXIT_TIMEOUT_S)
            self._clean_exit(b)
            self._step(name, "2b-bridge-stopped")
            # 3
            call_n("recreate_mark", name=name)
            # 4
            b, c = self._start_generation(rep, n, stub_port, log_path)
            self._step(name, "4")
            # 5 + 6 (in N)
            pre_reset_sim_step = call_n("begin_reset")["pre_reset_sim_step"]
            self._step(name, "5-6")
            # 7
            call_n("flush_run_dir")
            snap_rc, snap_out, snap_err = self.reset_verify(["snapshot", run_dir_s])
            if snap_rc != 0:
                raise InfraFailure("reset_verify", f"snapshot rc={snap_rc}: {snap_out}{snap_err}")
            self._step(name, "7")
            # 8
            sup.call(b, "reset", {}, RESET_TIMEOUT_S)
            self._step(name, "8")
            # 9, 10
            call_n("commit_reset", reset_gen=rep)
            self.sleep(0.15)
            # 11
            call_n("flush_run_dir")
            gen_label = str(rep)
            verify_rc, verify_out, verify_err = self.reset_verify(
                ["verify", run_dir_s, gen_label, gen_label, snap_out.strip()])
            reset_record_path = self.control_dir / f"reset_{gen_label}.txt"
            reset_record_path.write_text(reset_record_text(verify_out, verify_rc, verify_err))
            if verify_rc != 0:
                raise InfraFailure("reset_verify", f"verify rc={verify_rc}: {verify_err.strip()}")
            self._step(name, "11")
            # 12
            setup_first_seq = call_n("last_seq")
            # 13, 14
            sup.call(c, "client", {"key": name, "record_turn_on_state": True}, LEG_TIMEOUT_S)
            self._step(name, "13-14")
            # 15, 16
            res_s = sup.call(c, "fly", {"route": "PLACE_ROUTE", "key": f"{name}-setup"}, LEG_TIMEOUT_S)
            setup_last_seq = res_s["last_seq"]
            self._step(name, "15-16")
            # 17
            n_injected = 0
            if arm == "A":
                rng = echo_injection_range(res_s["flown"], res_s["moves"])
                if rng is not None:
                    n_injected = call_n("inject_echoes", name=name, first_command_index=rng[0],
                                        last_command_index=rng[1], seed=self.echo_seed + rep)["n_injected"]
            # 18
            self.sleep(0.3)
            # 19
            flight_first_seq = call_n("last_seq")
            # 20
            sup.call(c, "client", {"key": name, "record_turn_on_state": False}, LEG_TIMEOUT_S)
            # 21, 22
            res_f = sup.call(c, "fly", {"route": "LIFT_TO_PRESENT", "key": f"{name}-flight"}, LEG_TIMEOUT_S)
            flight_last_seq = res_f["last_seq"]
            self._step(name, "21-22")
            findings[name] = {"setup": res_s.get("findings"), "flight": res_f.get("findings")}
            # the slice's capture context ends here
            sup.call(b, "end_capture")
            call_n("finish_cycle", name=name, rep=rep, arm=arm, reset_gen=rep,
                   bridge_log_path=str(log_path), reset_record_path=str(reset_record_path),
                   snap_out=snap_out, verify_out=verify_out, verify_rc=verify_rc,
                   setup={"first_seq": setup_first_seq, "last_seq": setup_last_seq,
                          "flown": res_s["flown"], "moves": res_s["moves"]},
                   flight={"first_seq": flight_first_seq, "last_seq": flight_last_seq,
                           "flown": res_f["flown"], "moves": res_f["moves"]},
                   pre_reset_sim_step=pre_reset_sim_step)
            cycles.append({"name": name, "rep": rep, "arm": arm, "epoch": rep,
                           "reset_verify_rc": verify_rc, "n_injected_echoes": n_injected})

        # End of run: evidence emission while the last C/B are still alive, then C, B, N.
        emitted = call_n("emit_evidence", timeout=EMIT_TIMEOUT_S)
        self._step(None, "emit")
        sup.call(c, "close", {}, CHILD_EXIT_TIMEOUT_S)
        self._clean_exit(c)
        sup.call(b, "stop", {}, CHILD_EXIT_TIMEOUT_S)
        self._clean_exit(b)
        call_n("shutdown")
        self._clean_exit(n)
        return {"cycles": cycles, "emitted": emitted, "findings": findings,
                "run_dir": emitted["run_dir"], "control_dir": emitted["control_dir"],
                "arm_map": emitted["arm_map"], "native_log": emitted["native_log"]}


def default_argv_for(clone: str, proc_dir: Path, ev_dir: Path, control_dir: Path) -> Callable[..., List[str]]:
    script = str(Path(clone) / CHILDREN_REL)

    def argv_for(role: str, name: str, **kw: Any) -> List[str]:
        argv = [sys.executable, script, role, "--name", name, "--clone", clone, "--proc-dir", str(proc_dir)]
        if role == "N":
            argv += ["--ev-dir", str(ev_dir), "--control-dir", str(control_dir)]
        if role == "B":
            argv += ["--stub-port", str(kw["stub_port"])]
            if kw.get("log_path"):
                argv += ["--log-path", kw["log_path"]]
        if role == "C":
            argv += ["--n-port", str(kw["n_port"]), "--grpc-port", str(kw["grpc_port"])]
        return argv
    return argv_for


def child_env(src: str) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [f"{src}/src", f"{src}/native_mujoco", f"{src}/scripts", f"{src}/tests/integration",
         f"{src}/tests/fixtures/goalfix_cmp", src])
    env["REQUIRE_REACHY_SDK"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def cmd_coordinate(args) -> int:
    run_dir = Path(args.run_dir)
    proc_dir = run_dir / "proc"
    proc_dir.mkdir(parents=True, exist_ok=True)
    src = str(Path(args.src).resolve())
    start_utc = utc_now()
    sup = cc.Supervisor(proc_dir, child_env(src), src)
    orch = RunOrchestrator(run_dir, sup, default_argv_for(src, proc_dir, run_dir / "ev", run_dir / "control"),
                           make_run_reset_verify(src))
    result: Dict[str, Any] = {"ok": False, "failure": None, "started_utc": start_utc}
    state = {"written": False}

    def write_result(failure: Optional[dict]) -> None:
        if state["written"]:
            return
        state["written"] = True
        down = sup.shutdown_all()
        result["shutdown"] = down
        result["steps"] = orch.steps
        result["supervisor_events"] = sup.events
        result["finished_utc"] = utc_now()
        result["failure"] = failure
        result["ok"] = failure is None
        cc.write_json_atomic(proc_dir / "coordinator_result.json", result)

    def on_signal() -> None:
        write_result({"kind": "interrupted", "detail": "coordinator received SIGINT/SIGTERM"})

    restore = cc.install_signal_cleanup(on_signal)
    failure: Optional[dict] = None
    records: Dict[str, Any] = {}
    try:
        records["after_setup"] = cc.collect_provenance("coordinator", "coordinator", src, start_utc,
                                                       point="after_setup")
        cc.write_json_atomic(proc_dir / "coordinator.provenance.json", {"records": records})
        if not records["after_setup"]["ok"]:
            raise InfraFailure("separation", f"coordinator: {records['after_setup']['problems']}")
        result.update(orch.execute())
        prob = verify_provenance_files(proc_dir, expected_process_names())
        if prob:
            raise InfraFailure("separation", "; ".join(prob)[:2000])
    except BaseException as exc:                        # noqa: BLE001 -- recorded, never swallowed
        failure = classify_exception(exc)
        (proc_dir / "coordinator_exception.txt").write_text(traceback.format_exc())
    finally:
        restore()
        records["before_exit"] = cc.collect_provenance("coordinator", "coordinator", src, start_utc,
                                                       point="before_exit")
        cc.write_json_atomic(proc_dir / "coordinator.provenance.json", {"records": records})
        if failure is None and not records["before_exit"]["ok"]:
            failure = {"kind": "separation", "detail": f"coordinator: {records['before_exit']['problems']}"}
        write_result(failure)
    return 0 if failure is None else 1


# ---------------------------------------------------------------------------
# C2 diagnostic (report-only; shipped helpers) -- adapted unchanged from P's driver
# ---------------------------------------------------------------------------

def c2_records(targets8, start_pose8, route, assignment, withdraw=None) -> Dict[str, Any]:
    """Per-command tau figures for every command with a goal assignment, mirroring
    ``pathcheck.check_c2_c3``'s C2 loop with the shipped helpers.  ``first_fail`` is the first
    local index where C2 (a constant joint moved, or a tau spread > 30 ms + 1e-6 s) would fail."""
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
    """``c2_records`` plain and (only if anything is withdrawn on this leg, exactly as
    ``cycle.evaluate_cycle`` does) with the R-carry' withdrawals; the diagnostic C2 is the plain
    result if it fails, else the withdrawn result (the shipped first-pass-then-recheck order)."""
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


def annotate_full_gate(fg: Optional[dict]) -> Dict[str, Any]:
    """Full-gate output verbatim, plus the fixed annotation.  A passing native-compliance result
    is NEVER reported: an unexpected pass is withheld and flagged for inspection (P's rule)."""
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


def _leg_c2_figures(evd, idx: Sequence[int], spec, lr, echo_results, echo_mod, pc, np) -> Tuple[dict, List[dict]]:
    withdraw_rows = []
    for gi in idx:
        cr = echo_results[gi]
        withdraw_rows.append({j: (cr is not None and cr.joints[j].label == echo_mod.ECHO_CARRY)
                              for j in pc.R_JOINTS})
    t21 = evd.commands.target_rad[np.asarray(list(idx))]
    t8 = [{j: float(row[k]) for k, j in enumerate(pc.R_JOINTS)} for row in t21]
    diag = c2_diagnostic(t8, spec.start_pose8, spec.route_rad, lr.assignment, withdraw_rows)
    shipped = lr.pathcheck["C2"]
    reproduced = (diag["passed"] == shipped.passed and diag["first_fail"] == shipped.first_violation_index)
    lines: List[dict] = []
    over: List[int] = []
    max_spread = None
    for mode in ("plain", "withdrawn"):
        d = diag[mode]
        if d is None:
            continue
        for r in d["records"]:
            if r["n_tau"] >= 2:
                gi = idx[r["local_index"]]
                lines.append({"global_index": gi, "local_index": r["local_index"], "mode": mode,
                              "goal": r["goal"], "n_tau": r["n_tau"], "taus": r["taus"],
                              "spread_ms": r["spread_ms"], "verified": reproduced})
                if max_spread is None or r["spread_ms"] > max_spread:
                    max_spread = r["spread_ms"]
                if r["spread_ms"] > (pc.C2_SKEW_TOL_S + 1e-6) * 1000.0 and gi not in over:
                    over.append(gi)
    fig = {"reproduced": reproduced, "diag_passed": diag["passed"], "diag_index": diag["first_fail"],
           "shipped_passed": shipped.passed, "shipped_index": shipped.first_violation_index,
           "max_spread_ms": max_spread,
           "n_multi_tau_commands": sum(1 for r in diag["plain"]["records"] if r["n_tau"] >= 2),
           "n_gt_30ms": len(over), "gt_30ms_global_indices": over,
           "figures_status": "verified" if reproduced else "UNVERIFIED (diagnostic disagrees with shipped C2)"}
    return fig, lines


def analyze_run(run_dir: Path, ev_run_dir: str, control_dir: str, cycles: Sequence[dict]) -> Dict[str, Any]:
    """Report-only analysis: shipped C0-C8 per leg (``cycle.evaluate_cycle(..., skip_gates=True)``
    on legs located by ``cycle.resolve_leg`` exactly as the ``cycle`` CLI does), and per-command
    C2 figures with a ``reproduced`` flag.  A missing/malformed leg is ``incomplete`` (never a
    zero); a disagreement with the shipped C2 marks that leg's figures UNVERIFIED (never a stop)."""
    import numpy as np
    from reachy_ai.motion import rig_routes as R
    from tools.goalfix_cmp import cycle as cyc
    from tools.goalfix_cmp import echo
    from tools.goalfix_cmp import evidence as ev
    from tools.goalfix_cmp import pathcheck as pc

    assert abs(pc.C2_SKEW_TOL_S - C2_TOL_S) < 1e-12, "pathcheck.C2_SKEW_TOL_S is no longer 0.030"
    run_dir = Path(run_dir)
    control = Path(control_dir)
    per_cmd_lines: List[str] = []
    out_cycles: List[dict] = []
    try:
        evd = ev.verify_and_load(ev_run_dir, "states.jsonl", "commands.jsonl", "derived-SHA256SUMS")
    except (ev.EvidenceError, OSError, ValueError) as exc:
        evd = None
        load_error = f"{type(exc).__name__}: {exc}"
    for cy in cycles:
        name, arm = cy["name"], cy["arm"]
        if evd is None:
            out_cycles.append({"name": name, "arm": arm, "error": load_error,
                               "legs": {k: {"status": "incomplete", "error": load_error} for k in LEG_KINDS}})
            continue
        try:
            manifest = json.loads((control / f"cycle_{name}.json").read_text())
            run_dir_resolved = str(Path(ev_run_dir).resolve())
            setup_sidecar = str(control / manifest["setup_sidecar"])
            flight_sidecar = str(control / manifest["flight_sidecar"])
            legs = [
                cyc.resolve_leg(evd, setup_sidecar, "setup", R.PLACE_ROUTE, R.CRITICAL_JOINTS,
                                expected_route_name="PLACE_ROUTE", expected_server_run_dir=run_dir_resolved),
                cyc.resolve_leg(evd, flight_sidecar, "flight", R.LIFT_TO_PRESENT, R._PRESENT_GUARD,
                                expected_route_name="LIFT_TO_PRESENT", expected_server_run_dir=run_dir_resolved),
            ]
            scene = board_object_ids = None
            sdoc = json.loads(Path(setup_sidecar).read_text())
            if sdoc.get("scene"):
                scene = cyc.load_verified_scene(setup_sidecar)
                board_object_ids = sdoc["scene"].get("board_object_ids")
            cv, leg_results = cyc.evaluate_cycle(name, arm, evd, legs, scene=scene,
                                                 board_object_ids=board_object_ids, skip_gates=True)
        except (ev.EvidenceError, OSError, ValueError, KeyError) as exc:
            err = f"{type(exc).__name__}: {exc}"
            out_cycles.append({"name": name, "arm": arm, "error": err,
                               "legs": {k: {"status": "incomplete", "error": err} for k in LEG_KINDS}})
            continue

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
        for spec in legs:
            lr = leg_results[spec.name]
            checks = {cid: {"passed": lr.pathcheck[cid].passed, "index": lr.pathcheck[cid].first_violation_index,
                            "detail": lr.pathcheck[cid].detail} for cid in pc.ALL_CHECKS}
            try:
                fig, lines = _leg_c2_figures(evd, list(spec.command_indices), spec, lr, echo_results, echo, pc, np)
            except Exception as exc:                        # noqa: BLE001 -- a leg we cannot compute is incomplete
                leg_out[spec.name] = {"status": "incomplete", "checks": checks,
                                      "error": f"C2 diagnostic: {type(exc).__name__}: {exc}"}
                continue
            for ln in lines:
                per_cmd_lines.append(json.dumps({"cycle": name, "arm": arm, "leg": spec.name, **ln}, sort_keys=True))
            leg_out[spec.name] = {"status": "evaluated", "checks": checks, "c2": fig}
        out_cycles.append({"name": name, "arm": arm, "legs": leg_out,
                           "verdict_skip_gates": cv.verdict, "reasons_skip_gates": cv.reasons})
    (run_dir / "c2_per_command.jsonl").write_text("\n".join(per_cmd_lines) + ("\n" if per_cmd_lines else ""))
    doc = {"cycles": out_cycles, "report_only": True}
    (run_dir / "analysis.json").write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    return doc


# ---------------------------------------------------------------------------
# Summary (pure): counts only, setup and flight separate, evaluated denominators
# ---------------------------------------------------------------------------

def _iter_legs(rec: dict):
    for cy in rec.get("cycles") or []:
        for kind, leg in (cy.get("legs") or {}).items():
            yield cy, kind, leg


def expected_legs(rec: dict) -> List[Tuple[str, str, str]]:
    return [(n, a, k) for n, a in zip(CYCLE_NAMES, ARM_LABELS) for k in LEG_KINDS]


def summarize_run(rec: dict) -> Dict[str, Any]:
    """Per-run figures.  ``incomplete`` lists every expected leg with no evaluated result."""
    present = {}
    for cy, kind, leg in _iter_legs(rec):
        present[(cy.get("name"), kind)] = (cy, leg)
    per_kind: Dict[str, Dict[str, Any]] = {}
    for kind in LEG_KINDS:
        per_kind[kind] = {"evaluated": 0, "c2_shipped_fail": 0, "c2_verified": 0, "c2_unverified": 0,
                          "legs_gt_30ms": 0, "max_spread_ms": [],
                          "check_fail_counts": {c: 0 for c in CHECKS}}
    incomplete: List[str] = []
    unverified: List[str] = []
    a_legs: List[dict] = []
    any_event = False
    if rec.get("status") in ("ok", "failed"):
        for name, arm, kind in expected_legs(rec):
            got = present.get((name, kind))
            if got is None or got[1].get("status") != "evaluated":
                if rec.get("status") == "ok" or got is not None:
                    incomplete.append(f"{rec.get('run')}/{name}/{kind}")
                continue
            leg = got[1]
            checks, c2 = leg["checks"], leg["c2"]
            if arm == "A":
                a_legs.append({"cycle": name, "leg": kind,
                               "checks": {c: checks[c]["passed"] for c in CHECKS},
                               "note": "recorded; excluded from C2 counts (injected echoes)"})
                continue
            pk = per_kind[kind]
            pk["evaluated"] += 1
            for c in CHECKS:
                if checks[c]["passed"] is False:
                    pk["check_fail_counts"][c] += 1
            shipped_fail = checks["C2"]["passed"] is False
            if shipped_fail:
                pk["c2_shipped_fail"] += 1
            if c2.get("reproduced") is True:
                pk["c2_verified"] += 1
                if c2.get("max_spread_ms") is not None:
                    pk["max_spread_ms"].append({"cycle": name, "spread_ms": c2["max_spread_ms"]})
                if c2.get("n_gt_30ms", 0) >= 1:
                    pk["legs_gt_30ms"] += 1
                    any_event = True
            else:
                pk["c2_unverified"] += 1
                unverified.append(f"{rec.get('run')}/{name}/{kind}")
            if shipped_fail:
                any_event = True
    return {"per_kind": per_kind, "incomplete": incomplete, "unverified": unverified, "a_legs": a_legs,
            "any_b_event": any_event}


def build_summary(records: Sequence[dict], meta: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Compact summary over per-run records (``ok`` / ``failed`` / ``not_run``).  A failed run or
    an infrastructure stop gives a PARTIAL summary; nothing is replaced or extended."""
    runs_out = []
    tot = {k: {"evaluated": 0, "c2_shipped_fail": 0, "c2_verified": 0, "c2_unverified": 0, "legs_gt_30ms": 0,
               "check_fail_counts": {c: 0 for c in CHECKS}} for k in LEG_KINDS}
    counts = {"runs_ok": 0, "runs_failed": 0, "runs_not_run": 0, "runs_with_any_b_event": 0}
    incomplete_all: List[str] = []
    unverified_all: List[str] = []
    for rec in records:
        sr = summarize_run(rec)
        status = rec.get("status")
        counts[{"ok": "runs_ok", "failed": "runs_failed"}.get(status, "runs_not_run")] += 1
        for kind in LEG_KINDS:
            for k in ("evaluated", "c2_shipped_fail", "c2_verified", "c2_unverified", "legs_gt_30ms"):
                tot[kind][k] += sr["per_kind"][kind][k]
            for c in CHECKS:
                tot[kind]["check_fail_counts"][c] += sr["per_kind"][kind]["check_fail_counts"][c]
        if sr["any_b_event"]:
            counts["runs_with_any_b_event"] += 1
        incomplete_all += sr["incomplete"]
        unverified_all += sr["unverified"]
        cycles = []
        for cy in rec.get("cycles") or []:
            cycles.append({"cycle": cy.get("name"), "arm": cy.get("arm"),
                           "full_gate": annotate_full_gate(cy.get("full_gate")),
                           "validation_mode_path_only": {
                               "verdict": (cy.get("validation_mode") or {}).get("verdict"),
                               "rc": (cy.get("validation_mode") or {}).get("rc"),
                               "note": "path-only (--validation-mode); NOT full-gate success"}})
        runs_out.append({"run": rec.get("run"), "status": status, "failure": rec.get("failure"),
                         "cycles": cycles, "figures": sr})

    partial = counts["runs_failed"] > 0 or counts["runs_not_run"] > 0 or len(records) < N_RUNS or bool(incomplete_all)
    planned_b = N_RUNS * ARM_LABELS.count("B")
    summary = {
        "scope": {"planned_runs": N_RUNS, "planned_b_setup_legs": planned_b, "planned_b_flight_legs": planned_b,
                  "partial": partial, "d1_authorized": False, "execution_note": "prepared for owner-authorized execution only"},
        "meta": meta, "runs": runs_out, "totals": {"by_leg_kind": tot, **counts},
        "incomplete_legs": incomplete_all, "unverified_legs": unverified_all,
        "nativestub_compliance_limitation": NATIVESTUB_LIMITATION,
    }

    md: List[str] = ["# C2 process-separated (U-only) diagnostic summary", ""]
    md += [f"**NativeStub compliance limitation.** {NATIVESTUB_LIMITATION}", ""]
    if partial:
        md += ["**PARTIAL: not every planned run/leg was evaluated. Nothing was replaced or extended.**", ""]
    md += [f"- Pin: `{meta.get('pin')}`; M' `{M_PRIME_SHA}`; bridge blobs and runtime trees equal B `{B_SHA}`.",
           "- D1 and the simulator comparison are NOT authorized by this diagnostic.",
           f"- {SCOPE_SENTENCE}", f"- {REFERENCE_SENTENCE}", f"- {ATTRIBUTION_SENTENCE}", ""]
    md += ["## Counts (B legs; each with the number of evaluated legs)", "",
           "| leg kind | evaluated | shipped C2 failures | figures verified | figures UNVERIFIED | legs with >= 1 command > 30 ms (of verified) |",
           "|---|---|---|---|---|---|"]
    for kind in LEG_KINDS:
        t = tot[kind]
        md.append(f"| B {kind} | {t['evaluated']} | {t['c2_shipped_fail']} of {t['evaluated']} | {t['c2_verified']} | "
                  f"{t['c2_unverified']} | {t['legs_gt_30ms']} of {t['c2_verified']} |")
    md += ["", f"- Runs with any B event (a shipped C2 failure or a verified command > 30 ms): "
               f"{counts['runs_with_any_b_event']} of {counts['runs_ok'] + counts['runs_failed']} runs executed.",
           f"- Runs ok / failed / not run: {counts['runs_ok']} / {counts['runs_failed']} / {counts['runs_not_run']}.", ""]
    md += ["## C0-C8 failing B legs (count of evaluated)", "",
           "| leg kind | " + " | ".join(CHECKS) + " |", "|---|" + "---|" * len(CHECKS)]
    for kind in LEG_KINDS:
        t = tot[kind]
        md.append(f"| B {kind} (of {t['evaluated']}) | " + " | ".join(str(t["check_fail_counts"][c]) for c in CHECKS) + " |")
    md += ["", "## Per-leg maximum C2 spread (B legs, verified figures, ms)", "",
           "| run | leg kind | cycle | max spread (ms) |", "|---|---|---|---|"]
    for r in runs_out:
        for kind in LEG_KINDS:
            for e in r["figures"]["per_kind"][kind]["max_spread_ms"]:
                md.append(f"| {r['run']} | {kind} | {e['cycle']} | {round(e['spread_ms'], 3)} |")
    md += ["", "## A legs (recorded with C0-C8; excluded from every C2 count above: they carry injected echoes)", "",
           "| run | cycle | leg | failing checks |", "|---|---|---|---|"]
    for r in runs_out:
        for x in r["figures"]["a_legs"]:
            failing = [c for c in CHECKS if x["checks"].get(c) is False]
            md.append(f"| {r['run']} | {x['cycle']} | {x['leg']} | {', '.join(failing) if failing else 'none'} |")
    md += ["", "## Incomplete and unverified legs", ""]
    md += [f"- incomplete (not counted as zero): {', '.join(incomplete_all) if incomplete_all else 'none'}",
           f"- UNVERIFIED figures (excluded from the counts above): {', '.join(unverified_all) if unverified_all else 'none'}", ""]
    md += ["## Per run", "", "| run | status | stop reason |", "|---|---|---|"]
    for r in runs_out:
        f = r["failure"]
        md.append(f"| {r['run']} | {r['status']} | {f['kind'] + ': ' + f['detail'] if f else ''} |")
    md += ["", "## Per cycle (full gates verbatim; validation mode is path-only)", "",
           "| run | cycle | arm | full-gate rc | full-gate verdict | full-gate compliance check | validation-mode verdict (path-only) |",
           "|---|---|---|---|---|---|---|"]
    for r in runs_out:
        for c in r["cycles"]:
            fg = c["full_gate"]
            md.append(f"| {r['run']} | {c['cycle']} | {c['arm']} | {fg['rc']} | {fg['verdict']} | "
                      f"{fg['compliance_gate']} ({fg['annotation']}) | {c['validation_mode_path_only']['verdict']} |")
    md += ["", "## Limits", ""] + [f"- {s}" for s in LIMITATION_SENTENCES] + [""]
    return summary, "\n".join(md)


# ---------------------------------------------------------------------------
# Batch controller
# ---------------------------------------------------------------------------

def _read_json(path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _write_sha256sums(directory: Path, name: str = "SHA256SUMS", exclude_top: Sequence[str] = ()) -> None:
    lines = []
    for p in sorted(directory.rglob("*")):
        if not p.is_file() or (p.name == name and p.parent == directory):
            continue
        rel = p.relative_to(directory)
        if rel.parts and rel.parts[0] in exclude_top:
            continue
        lines.append(f"{sha256_file(p)}  ./{rel.as_posix()}")
    (directory / name).write_text("\n".join(lines) + ("\n" if lines else ""))


def dir_size(path: Path, exclude_top: Sequence[str] = ()) -> int:
    total = 0
    for p in Path(path).rglob("*"):
        try:
            rel = p.relative_to(path)
            if rel.parts and rel.parts[0] in exclude_top:
                continue
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def verify_evidence_checksums(ev_dir: Path, run_dir: Path) -> Optional[str]:
    """Root ``SHA256SUMS`` (keyed ``./e1_server_runs/<run>/...``) and the run directory's derived
    digest file, both against the real bytes.  Returns a description of the first mismatch."""
    for base, sums in ((ev_dir, ev_dir / "SHA256SUMS"), (run_dir, run_dir / "derived-SHA256SUMS")):
        try:
            text = sums.read_text()
        except OSError as exc:
            return f"{sums}: {exc}"
        for line in text.splitlines():
            if not line.strip():
                continue
            digest, _, rel = line.partition("  ")
            target = base / rel.strip()
            if not target.is_file():
                return f"{sums}: {rel.strip()} is missing"
            if sha256_file(target) != digest.strip():
                return f"{sums}: {rel.strip()} sha256 mismatch"
    return None


def kill_run_group(pop: "subprocess.Popen", proc_dir: Path, run_dir: Path) -> List[int]:
    """Terminate then kill the coordinator's process group, then every recorded child group, then
    anything still carrying the run directory on its command line.  Returns pids still alive."""
    def _killpg(pgid: int, sig: int) -> None:
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass
    if pop.poll() is None:
        _killpg(pop.pid, signal.SIGTERM)
        try:
            pop.wait(timeout=COORDINATOR_TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            pass
    _killpg(pop.pid, signal.SIGKILL)
    if pop.poll() is None:
        try:
            pop.wait(timeout=TERM_GRACE_S)
        except subprocess.TimeoutExpired:                        # pragma: no cover
            pass
    return sweep_children(proc_dir, run_dir)


def sweep_children(proc_dir: Path, run_dir: Path, kill: bool = True) -> List[int]:
    """Pids of processes still carrying the run directory on their command line (every child's
    argv does).  With ``kill``: terminate then kill them (process group first, when recorded) and
    report what is STILL alive.  Never signals a pid that does not carry the run directory."""
    needle = str(run_dir)
    pids = _read_json(Path(proc_dir) / "pids.json")
    recorded = {}
    for e in pids if isinstance(pids, list) else []:
        try:
            recorded[int(e["pid"])] = int(e.get("pgid", e["pid"]))
        except (KeyError, ValueError, TypeError):
            pass
    if kill:
        for sig, pause in ((signal.SIGTERM, 0.5), (signal.SIGKILL, 0.2)):
            live = cc.ps_leftovers(needle)
            if not live:
                break
            for pid in live:
                for fn, target in ((os.killpg, recorded.get(pid, pid)), (os.kill, pid)):
                    try:
                        fn(target, sig)
                    except (ProcessLookupError, PermissionError):
                        pass
            time.sleep(pause)
    return cc.ps_leftovers(needle)


def _run_proc(argv: Sequence[str], cwd: str, env: Dict[str, str], stdout: Path, stderr: Path,
              timeout: int) -> Dict[str, Any]:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    timed_out = False
    with open(stdout, "wb") as fo, open(stderr, "wb") as fe:
        try:
            rc = subprocess.run(list(argv), cwd=cwd, env=env, stdout=fo, stderr=fe, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = None, True
    return {"argv": list(argv), "rc": rc, "timed_out": timed_out, "seconds": time.monotonic() - t0,
            "stdout": stdout.name, "stderr": stderr.name}


def run_clis(rd: Path, src: str, pin: str, coord: dict, facts: dict, env: dict) -> List[dict]:
    """The ``cycle`` CLI per cycle with full gates and with ``--validation-mode``, then
    ``summary checkpoint --n 4``.  argv, rc, verdict, reasons recorded verbatim."""
    run_dir, control = coord["run_dir"], Path(coord["control_dir"])
    base = ["--sha256sums", "derived-SHA256SUMS"]
    out: List[dict] = []
    for cy in coord["cycles"]:
        name, rep, arm = cy["name"], cy["rep"], cy["arm"]
        crec: Dict[str, Any] = {"name": name, "rep": rep, "arm": arm, "epoch": cy["epoch"],
                                "reset_verify_rc": cy["reset_verify_rc"]}
        full_out = control / f"between_{name}.json"
        argv_full = ["--ev-dir", run_dir, *base, "--control-dir", str(control), "--cycle", name, "--rep", str(rep),
                     "--arm", arm, "--arm-map", coord["arm_map"], "--expected-host-sha", pin,
                     "--required-supervisor-programs", "reachy-sdk-server",
                     "--expected-bridge-sha-a", A_SHA, "--expected-bridge-sha-b", B_SHA, "--out", str(full_out)]
        r_full = _run_proc([sys.executable, "-m", "tools.goalfix_cmp.cycle", *argv_full], src, env,
                           rd / "cli" / f"full_{name}.stdout", rd / "cli" / f"full_{name}.stderr", CLI_TIMEOUT_S)
        val_out = rd / "cli" / f"validation_{name}.json"
        argv_val = ["--ev-dir", run_dir, *base, "--control-dir", str(control), "--cycle", name, "--rep", str(rep),
                    "--arm", arm, "--out", str(val_out), "--validation-mode"]
        r_val = _run_proc([sys.executable, "-m", "tools.goalfix_cmp.cycle", *argv_val], src, env,
                          rd / "cli" / f"validation_{name}.stdout", rd / "cli" / f"validation_{name}.stderr", CLI_TIMEOUT_S)
        for tag, path, r in (("full", full_out, r_full), ("validation", val_out, r_val)):
            if r["timed_out"] or not path.is_file():
                facts["missing_artifacts"].append(f"{name} {tag} CLI output")
            elif _read_json(path) is None:
                facts["malformed_artifacts"].append(f"{name} {tag} CLI output")
        fj, vj = _read_json(full_out), _read_json(val_out)
        crec["full_gate"] = {"argv": argv_full, "rc": r_full["rc"], "out": str(full_out),
                             "verdict": (fj or {}).get("verdict"),
                             "compliance": ((fj or {}).get("checks") or {}).get("compliance"),
                             "reasons": (fj or {}).get("reasons")}
        crec["validation_mode"] = {"argv": argv_val, "rc": r_val["rc"], "out": str(val_out),
                                   "verdict": (vj or {}).get("verdict"), "reasons": (vj or {}).get("reasons")}
        out.append(crec)
    cp_out = rd / "cli" / "checkpoint_n4.json"
    argv_cp = ["checkpoint", "--control-dir", str(control), "--arm-map", coord["arm_map"],
               "--expected-bridge-sha-a", A_SHA, "--expected-bridge-sha-b", B_SHA, "--n", "4",
               "--native-log", coord["native_log"], "--states", str(Path(run_dir) / "states.jsonl"),
               "--out", str(cp_out)]
    r_cp = _run_proc([sys.executable, "-m", "tools.goalfix_cmp.summary", *argv_cp], src, env,
                     rd / "cli" / "checkpoint_n4.stdout", rd / "cli" / "checkpoint_n4.stderr", CLI_TIMEOUT_S)
    facts["checkpoint"] = {"argv": argv_cp, "rc": r_cp["rc"], "out": str(cp_out)}
    if r_cp["timed_out"] or not cp_out.is_file():
        facts["missing_artifacts"].append("summary checkpoint output")
    elif _read_json(cp_out) is None:
        facts["malformed_artifacts"].append("summary checkpoint output")
    return out


def process_run(idx: int, out: Path, src: str, pin: str, *, time_left: Callable[[], float],
                interrupted: Callable[[], bool] = lambda: False) -> Dict[str, Any]:
    rd = out / f"run-{idx:02d}"
    for sub in ("ev", "control", "cli", "proc"):
        (rd / sub).mkdir(parents=True)
    proc_dir = rd / "proc"
    env = child_env(src)
    rec: Dict[str, Any] = {"run": rd.name, "index": idx, "status": "ok", "failure": None,
                           "started_utc": utc_now(), "cycles": []}
    facts: Dict[str, Any] = {"missing_artifacts": [], "malformed_artifacts": [], "incomplete_legs": []}

    budget = min(RUN_CAP_S, max(1.0, time_left()))
    argv = [sys.executable, str(Path(src) / DRIVER_REL), "coordinate", "--run-dir", str(rd), "--src", src, "--pin", pin]
    t0 = time.monotonic()
    timed_out = was_interrupted = False
    with open(proc_dir / "coordinator.stdout", "wb") as fo, open(proc_dir / "coordinator.stderr", "wb") as fe:
        pop = subprocess.Popen(argv, cwd=src, env=env, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe,
                               start_new_session=True)
        try:
            pop.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            timed_out = True
        except Interrupted:
            was_interrupted = True
        finally:
            if pop.poll() is None or timed_out or was_interrupted:
                survivors = kill_run_group(pop, proc_dir, rd)
            else:
                survivors = sweep_children(proc_dir, rd, kill=False)
                if survivors:
                    sweep_children(proc_dir, rd, kill=True)
    rec["coordinator"] = {"rc": pop.returncode, "seconds": time.monotonic() - t0, "timed_out": timed_out}
    if timed_out:
        facts["cap"] = "run_cap" if budget >= RUN_CAP_S else "total_cap"
        facts["cap_detail"] = f"run exceeded {budget:.0f} s"
    if was_interrupted or interrupted():
        facts["interrupted"] = True
    if survivors:
        facts["leftover_pids"] = survivors

    coord = _read_json(proc_dir / "coordinator_result.json")
    if coord is None:
        facts["coordinator_failure"] = {"kind": "coordinator_no_result",
                                        "detail": f"no coordinator_result.json (rc {pop.returncode})"} \
            if not (timed_out or was_interrupted) else None
    elif not coord.get("ok"):
        facts["coordinator_failure"] = coord.get("failure") or {"kind": "coordinator", "detail": "not ok"}
    exit_codes = _read_json(proc_dir / "exit_codes.json") or {}
    facts["child_exit_codes"] = ({k: v.get("exit_code") for k, v in exit_codes.items()}
                                 if coord and coord.get("ok") else {})
    if coord and coord.get("ok"):
        prob = verify_provenance_files(proc_dir, expected_process_names() + ["coordinator"])
        if prob:
            facts["provenance_problems"] = prob
        facts["reset_verify_rcs"] = [c["reset_verify_rc"] for c in coord["cycles"]]

    failure = classify_failure(facts)
    if failure is None and coord and coord.get("ok"):
        ev_dir = rd / "ev"
        run_ev = Path(coord["run_dir"])
        if not (run_ev / "commands.jsonl").is_file() or not (run_ev / "states.jsonl").is_file():
            facts["missing_artifacts"].append("evidence (commands.jsonl/states.jsonl)")
        elif (run_ev / "commands.jsonl").stat().st_size == 0 or (run_ev / "states.jsonl").stat().st_size == 0:
            facts["malformed_artifacts"].append("evidence is empty")
        else:
            mismatch = verify_evidence_checksums(ev_dir, run_ev)
            if mismatch:
                facts["checksum_mismatch"] = mismatch
        failure = classify_failure(facts)
        if failure is None:
            rec["cycles"] = run_clis(rd, src, pin, coord, facts, env)
            failure = classify_failure(facts)
        if failure is None:
            try:
                analysis = analyze_run(rd, str(run_ev), coord["control_dir"], coord["cycles"])
                by_name = {c["name"]: c for c in analysis["cycles"]}
                for crec in rec["cycles"]:
                    a = by_name.get(crec["name"], {})
                    crec["legs"] = a.get("legs", {})
                    if "error" in a:
                        crec["analysis_error"] = a["error"]
                    for kind, leg in crec["legs"].items():
                        if leg.get("status") != "evaluated":
                            facts["incomplete_legs"].append(f"{rd.name}/{crec['name']}/{kind}")
            except BaseException:                               # noqa: BLE001
                (rd / "analysis_exception.txt").write_text(traceback.format_exc())
                facts["missing_artifacts"].append("analysis.json")
            failure = classify_failure(facts)
    if failure:
        rec["status"], rec["failure"] = "failed", failure
    rec["finished_utc"] = utc_now()
    (rd / "run_manifest.json").write_text(json.dumps(rec, indent=2, sort_keys=True, default=str))
    _write_sha256sums(rd)
    return rec


def run_loop(process: Callable[[int], dict], n_runs: int = N_RUNS,
             precheck: Callable[[], Optional[dict]] = lambda: None) -> List[dict]:
    """The batch's control flow, in isolation: ``run-01 .. run-NN`` in order via ``process(i) ->
    record``; stop early ONLY when a record has ``status == "failed"`` (an infrastructure/evidence
    failure, see ``classify_failure``) or a pre-run cap/precondition check fails.  The remaining
    runs are recorded ``not_run`` and never executed.  Each index is processed at most once: there
    is no retry, no replacement and no extension.  A run whose analysis shows C0-C8 failures is a
    normal ``ok`` record and never stops anything."""
    check_runs(n_runs)
    records: List[dict] = []
    stopped = False
    for i in range(1, n_runs + 1):
        if stopped:
            records.append({"run": f"run-{i:02d}", "index": i, "status": "not_run", "failure": None,
                            "cycles": [], "note": "not run: an earlier run stopped the batch"})
            continue
        blocked = precheck()
        if blocked:
            records.append({"run": f"run-{i:02d}", "index": i, "status": "not_run", "cycles": [],
                            "failure": blocked})
            stopped = True
            continue
        rec = process(i)
        records.append(rec)
        if rec.get("status") == "failed":
            stopped = True
    return records


def verify_source(src: str, pin: str) -> Dict[str, Any]:
    src_p = Path(src)
    if _git(src, "status", "--porcelain", check=False) != "":
        raise Refusal(f"clone {src} has a dirty tree")
    head = _git(src, "rev-parse", "HEAD")
    if head != pin:
        raise Refusal(f"clone HEAD {head} != pin {pin}")
    facts = source_pin_facts(src, pin)
    for f in BRIDGE_OPT_FILES:
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
    for name in ("tools.goalfix_cmp", "reachy_ai", "c2ps_common"):
        mod = importlib.import_module(name)
        f = Path(getattr(mod, "__file__", None) or list(mod.__path__)[0]).resolve()
        if root not in f.parents:
            raise Refusal(f"module {name} imported from {f}, not from {root}")
        seen[name] = str(f)
    return seen


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
        "start_utc": start_utc, "end_utc": None, "free_disk_bytes_start": shutil.disk_usage(str(out)).free,
        "free_disk_bytes_end": None, "constants": constants_table(), "pin_facts": facts,
        "dry_run_plan": plan_lines(pin, str(out)),
    }


def make_archive(out: Path) -> Dict[str, Any]:
    """``tar --zstd`` of ``<out>`` (excluding ``tmp``) to ``<out>.tar.zst`` plus ``.sha256``,
    never overwriting."""
    arc = Path(str(out) + ".tar.zst")
    shaf = Path(str(arc) + ".sha256")
    if arc.exists() or shaf.exists():
        return {"ok": False, "error": f"{arc} or {shaf} already exists; not overwriting"}
    proc = subprocess.run(["tar", "--zstd", "-cf", str(arc), "--exclude", f"{out.name}/tmp",
                           "-C", str(out.parent), out.name], capture_output=True, text=True)
    if proc.returncode != 0:
        return {"ok": False, "error": f"tar rc={proc.returncode}: {proc.stderr.strip()[:500]}"}
    shaf.write_text(f"{sha256_file(arc)}  {arc.name}\n")
    return {"ok": True, "archive": str(arc), "sha256": str(shaf)}


def cmd_run(args) -> int:
    out = Path(args.out).resolve()
    src = str((out / "src").resolve())
    start = utc_now()
    t_start = time.monotonic()
    try:
        facts = verify_source(src, args.pin)
        seen = assert_imports_from(src)
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    facts["imports_from_clone"] = seen
    env_rec = environment_record(args.pin, out, args.venv, facts, start)
    (out / "environment.json").write_text(json.dumps(env_rec, indent=2, sort_keys=True))
    state = {"interrupted": False}

    def on_signal(signum, _frame):
        state["interrupted"] = True
        raise Interrupted()
    old = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}

    def time_left() -> float:
        return TOTAL_CAP_S - (time.monotonic() - t_start)

    def precheck() -> Optional[dict]:
        if state["interrupted"]:
            return {"kind": "interrupted", "detail": "signal received"}
        if time_left() <= 0:
            return {"kind": "total_cap", "detail": f"{TOTAL_CAP_S} s total wall reached"}
        if dir_size(out, exclude_top=("src",)) > OUT_CAP_BYTES:
            return {"kind": "output_cap", "detail": f"output exceeds {OUT_CAP_BYTES} bytes"}
        free = shutil.disk_usage(str(out)).free
        if free < MIN_FREE_BYTES:
            return {"kind": "disk_low", "detail": f"free {free} bytes < {MIN_FREE_BYTES}"}
        return None

    collected: List[dict] = []

    def process(i: int) -> dict:
        rec = process_run(i, out, src, args.pin, time_left=time_left, interrupted=lambda: state["interrupted"])
        collected.append(rec)
        return rec

    try:
        records = run_loop(process, precheck=precheck)
    except Interrupted:
        # A signal outside process_run's own wait: the run in progress is failed, the rest not run.
        records = list(collected)
        stop = {"kind": "interrupted", "detail": "SIGINT/SIGTERM delivered to the batch controller"}
        for i in range(len(records) + 1, N_RUNS + 1):
            records.append({"run": f"run-{i:02d}", "index": i, "status": "failed" if i == len(collected) + 1 else "not_run",
                            "failure": stop if i == len(collected) + 1 else None, "cycles": []})
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    stopped = any(r["status"] != "ok" for r in records)
    final_size = dir_size(out, exclude_top=("src",))
    if final_size > OUT_CAP_BYTES:
        stopped = True
        records[-1].setdefault("notes", []).append(f"output cap exceeded after the final run: {final_size} bytes")
    env_rec["end_utc"] = utc_now()
    env_rec["free_disk_bytes_end"] = shutil.disk_usage(str(out)).free
    (out / "environment.json").write_text(json.dumps(env_rec, indent=2, sort_keys=True))
    summary, md = build_summary(records, {"pin": args.pin, "start_utc": start, "end_utc": env_rec["end_utc"]})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    (out / "SUMMARY.md").write_text(md)
    _write_sha256sums(out, exclude_top=("src", "tmp"))
    arc = make_archive(out)
    print(json.dumps({"archive": arc}))
    return 1 if (stopped or not arc["ok"]) else 0


def cmd_preflight(args) -> int:
    try:
        facts = preflight(args.pin, args.out, args.repo, args.runs, args.venv)
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print("\n".join(plan_lines(args.pin, facts["out"])))
    print("\nPREFLIGHT OK (tools/goalfix_cmp tree, the three bridge blobs and the 16 runtime trees equal; "
          "pin descends from M'; only new files added; nothing executed)")
    print(f"  pin:tools/goalfix_cmp = {facts['pin_tools_goalfix_cmp_tree']}  (== M' 's)")
    for f, s in facts["opt_file_sha256"].items():
        print(f"  {f} sha256 = {s}  (== B's blob)")
    for t, s in facts["runtime_trees"].items():
        print(f"  runtime tree {t} = {s}  (== B's)")
    print(f"  files added over M': {', '.join(facts['added_files'])}")
    print(f"  free space at the parent of --out: {facts['free_bytes']} bytes")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    for a in ("--pin", "--out", "--repo", "--runs", "--venv"):
        p.add_argument(a, required=True)
    r = sub.add_parser("run")
    for a in ("--pin", "--out", "--venv"):
        r.add_argument(a, required=True)
    c = sub.add_parser("coordinate")
    for a in ("--run-dir", "--src", "--pin"):
        c.add_argument(a, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "preflight":
        return cmd_preflight(args)
    if args.cmd == "run":
        return cmd_run(args)
    return cmd_coordinate(args)


if __name__ == "__main__":
    raise SystemExit(main())
