"""Generate one marker-gated Stage 1 execution notebook per cycle (fail-
closed baseline gate, fresh-server provenance). Cell shapes and marker
protocol (`go_<leg>`, `recorder_<leg>.log`, `<leg>_done`, `binding_ok_<cycle>`
/ `binding_FAIL_<cycle>`) carried over from PR #117's reference generator
(read-only, `~/reachy-1-2-sim-stage1/docs/reviews/probes-2026-09-15-e1-stage1-b4/
control/tools/make_cycle_notebook.py`); the motion cell adds the W2
fail-closed `cmd_seq` baseline, and cells 1-2 add the W3 timing constants and
W4 tree-provenance checks. This module only writes files: it never imports
or executes a notebook, connects to an SDK, or starts a server.

Usage:
  python -m scripts.e1_stage1.make_cycle_notebook <cycle> --repo <path> \\
      --evidence-dir <path> [--lead-in-s 3.0] [--compliance-timeout-s 3.0]

Writes <evidence-dir>/e1_stage1_<cycle>.ipynb and
<evidence-dir>/plan_<cycle>.json. Refuses, with nothing written, if
`--repo`'s HEAD does not descend from `plan.REQUIRED_SHA` (W4): a native
server built off an older tree never writes `cmd_seq` into `State`, and the
gate would refuse forever without explaining why. Also refuses if
`--evidence-dir` is under `docs/reviews/` or anywhere inside `--repo`
itself (M3, PR #120 review): runtime artifacts written under it would
dirty the server's own checkout.

`--repo` must point at the exact checkout the native server is running
from, not merely a tree that happens to descend from `plan.REQUIRED_SHA`
(W4 note 1): the generated notebook's binding cell now requires the
server's manifest `code_sha` to equal this call's `--repo` HEAD exactly,
not just be a descendant of it, so a server on a different (even if
related) tree fails closed at binding time rather than silently.

Stage 2 (decision note outputs/e1-stage2-decision-2026-09-15.md) adds two
more entry points -- `generate_repetition` (board/shape/repetition-aware
cycles) and `generate_stage0` (the policy-A arm-on setup, once per
board/session) -- and a `--board` scene parameter shared by all three.
`generate` (this legacy path) is unchanged in behaviour for its default
`board="B4"`. See README.md's "Stage 2 additions" section for the CLI and
the reused-identity/stale-marker refusals these add.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
from typing import Callable, List, Sequence

from . import plan

RunGit = Callable[[List[str]], subprocess.CompletedProcess]


class GenerationRefused(Exception):
    """Raised instead of writing any file."""


def _run_git(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True)


# ── Reused-identity refusal (decision note
# outputs/e1-stage2-decision-2026-09-15.md §6 item 2) ───────────────────────
#
# Stage 1 generated exactly one notebook per cycle, ever, so nothing
# checked for a pre-existing plan file or marker. Stage 2 generates 18
# cycles per board from the same three leg names' pattern repeated across
# repetitions; `plan.stage2_legs` makes every leg name unique per
# board+shape+repetition, but that only helps if generation itself refuses
# to reuse an identity that already has a plan file or markers on disk --
# otherwise a second `generate_repetition` call with the same
# board/shape/rep (a typo, a re-run after a crash, an operator mistake)
# would silently overwrite `plan_<cycle_id>.json` and could regenerate a
# notebook whose binding cell finds an already-populated `go_*`/`*_done`
# set from the first attempt still on disk.

def _session_ledger_path(evidence_dir: str) -> pathlib.Path:
    return pathlib.Path(evidence_dir) / "control" / "e1_stage2_sessions.json"


def _check_and_record_session(evidence_dir: str, *, board: str, session: str) -> None:
    """One server session per board (decision note §5: "one server run per
    board"). Refuses a generation call that names a different session id
    for a board this evidence directory has already recorded one for --
    the reused-execution-directory shape that let a stale run
    directory/marker satisfy a later attempt's checks by coincidence
    (decision note §6 item 2). Recording the first session id seen for a
    board is not itself a claim that session ran cleanly; it only pins
    what this evidence directory is allowed to keep meaning by "this
    board" from here on."""
    ledger_path = _session_ledger_path(evidence_dir)
    ledger: dict = {}
    if ledger_path.is_file():
        try:
            ledger = json.loads(ledger_path.read_text())
        except json.JSONDecodeError as exc:
            raise GenerationRefused(
                f"{ledger_path} is corrupt ({exc}) -- refusing to guess "
                "which session this evidence directory already recorded "
                f"for board {board!r}")
    existing = ledger.get(board)
    if existing is not None and existing != session:
        raise GenerationRefused(
            f"evidence dir {evidence_dir!r} already recorded session "
            f"{existing!r} for board {board!r}; this call names session "
            f"{session!r} -- a second session for the same board in the "
            "same evidence directory is the reused-execution-directory "
            "shape decision note §6 item 2 warns about; use a fresh "
            "evidence directory for a new session")
    ledger[board] = session
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True))


def _refuse_if_identity_reused(
    evidence_dir: str, *, identity: str, leg_names: Sequence[str],
) -> None:
    """Refuse generation if `identity`'s plan file, or any marker any of
    its legs would use, already exists in this evidence directory. An
    earlier repetition, cycle, or Stage 0 setup must never be regenerated
    or overwritten in place -- "no re-fly, no recovery" (decision note
    §5), and never allowed to let a leftover marker satisfy a later
    attempt's readiness gate (§6 item 2)."""
    evidence_path = pathlib.Path(evidence_dir)
    plan_path = evidence_path / f"plan_{identity}.json"
    if plan_path.is_file():
        raise GenerationRefused(
            f"{plan_path} already exists -- identity {identity!r} was "
            "already generated in this evidence directory; a repetition, "
            "cycle, or Stage 0 setup must never be regenerated in place "
            "(decision note §6 item 2)")
    control = evidence_path / "control"
    if control.is_dir():
        marker_names = [f"binding_ok_{identity}", f"binding_FAIL_{identity}"]
        for leg in leg_names:
            marker_names += [f"go_{leg}", f"recorder_{leg}.log", f"{leg}_done"]
        stale = sorted(name for name in marker_names if (control / name).exists())
        if stale:
            raise GenerationRefused(
                f"stale marker(s) for identity {identity!r} already present "
                f"in {control}: {stale} -- an earlier repetition's marker "
                "must never be able to satisfy this identity's readiness "
                "gate (decision note §6 item 2); use a fresh evidence "
                "directory or a new, never-used identity")


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source}


def markdown_cell_source(cycle: str, repo: str, repo_sha: str,
                          legs: List[plan.Leg]) -> str:
    leg_desc = ", ".join(
        f"`{leg.name}` {leg.route} via `{leg.tool}` ({leg.desc})" for leg in legs)
    return (
        f"# E1 Stage 1 cycle {cycle} -- fail-closed baseline, fresh-server "
        "provenance\n\n"
        f"Code `{repo}` = `{repo_sha}`. Legs: {leg_desc}. One attempt each; "
        "a `stop` marker, a failed start check, an invalid baseline "
        "`cmd_seq`, or a failed compliance/provenance check means no "
        "motion.")


def connect_cell_source(*, repo: str, evidence_dir: str, repo_sha: str,
                         cycle: str, lead_in_s: float,
                         compliance_timeout_s: float, required_sha: str,
                         merge_time_iso: str, scene_rel: str) -> str:
    return f'''# Cell 1 -- connect with literals; hygiene shown; tree identity pinned at generation time
import os, subprocess, sys, time, json, pathlib, traceback
sys.path.insert(0, "{repo}/src"); sys.path.insert(0, "{repo}/scripts")
sys.path.insert(0, "{repo}/scripts/e1_stage1")
CTRL = pathlib.Path("{evidence_dir}/control"); RECORD_ROOT = "{evidence_dir}/e1_server_runs"
SCENE = "{repo}/{scene_rel}"
LEAD_IN_S = {lead_in_s!r}; COMPLIANCE_TIMEOUT_S = {compliance_timeout_s!r}; CYCLE = "{cycle}"
REPO = "{repo}"; GENERATED_AT_SHA = "{repo_sha}"
REQUIRED_SHA = "{required_sha}"; MERGE_TIME_ISO = "{merge_time_iso}"
_current_sha = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
assert _current_sha == GENERATED_AT_SHA, f"kernel tree {{_current_sha}} != generated-at tree {{GENERATED_AT_SHA}} -- regenerate the notebook"
print("REACHY env in this kernel:", {{k: v for k, v in os.environ.items() if k.upper().startswith("REACHY")}})
assert "REACHY_IP" not in os.environ and "REACHY_ENABLE_MOTION" not in os.environ, "shell hygiene violated"
from reachy_sdk import ReachySDK
HOST, PORT = "localhost", 50051
reachy = ReachySDK(host=HOST, sdk_port=PORT)
print(f"ReachySDK(host={{HOST!r}}, sdk_port={{PORT}}) connected at wall {{time.time_ns()}} mono {{time.monotonic_ns()}}")
print("python:", sys.executable)
'''


def binding_cell_source() -> str:
    return '''# Cell 2 -- motion-client binding check + W4 fresh-server provenance
import e1_identity, provenance, gating
from reachy_ai.motion import rig_routes as R
from reachy_ai.motion import primitives
from reachy_ai.tasks import rig_motion
def _pose(): return {name: float(getattr(reachy.r_arm, name).present_position) for name in R.R_JOINTS}
ident = e1_identity.verify_simulator_identity(host=HOST, port=PORT, scene_path=SCENE, record_root=RECORD_ROOT, read_sdk_joints=_pose)
d = ident.as_dict()
manifest = {}
if ident.run_dir:
    manifest_path = pathlib.Path(ident.run_dir) / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
def _git_is_ancestor(a, b):
    return subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", a, b]).returncode == 0
prov_ok, prov_reasons = provenance.check_binding_provenance(manifest, git_is_ancestor=_git_is_ancestor, required_sha=REQUIRED_SHA, merge_time_iso=MERGE_TIME_ISO, generated_at_sha=GENERATED_AT_SHA)
d["provenance_ok"] = prov_ok; d["provenance_reasons"] = prov_reasons
print(json.dumps(d, indent=2, default=str))
BINDING_OK = bool(ident.ok) and prov_ok
(CTRL / (f"binding_ok_{CYCLE}" if BINDING_OK else f"binding_FAIL_{CYCLE}")).write_text(json.dumps(d, default=str))
print("BINDING_OK =", BINDING_OK); print("present:", {k: round(v, 1) for k, v in _pose().items()})
def wait_for(pred, timeout_s, period=0.25):
    return gating.wait_for(CTRL, pred, timeout_s, period)
def start_check(kind):
    p = _pose(); here = R.posture_of(p)
    if kind == "PLACE_ROUTE_start": ok, why = rig_motion.check_start(reachy.r_arm, R.PLACE_ROUTE)
    elif kind == "PRESENT": ok = R.at_pose(p, R.PRESENT, tol=12.0, joints=list(R.GROSS_JOINTS)); why = "" if ok else "not at PRESENT (gross, 12 deg)"
    elif kind == "REST": ok = R.at_pose(p, R.REST, tol=12.0, joints=list(R.GROSS_JOINTS)); why = "" if ok else "not at REST (gross, 12 deg)"
    return {"kind": kind, "ok": bool(ok), "why": why, "posture_of": here, "pose": {k: round(v, 1) for k, v in p.items()}}
PREV_OK = BINDING_OK
'''


def motion_cell_source(leg: plan.Leg) -> str:
    fn = plan.TOOL_TO_CALL[leg.tool]
    return f'''# Leg {leg.name}: {leg.route} via {leg.tool} ({leg.desc}) -- one attempt, gated on go_{leg.name} + the recorder
LEG = {{"leg": "{leg.name}", "route": "{leg.route}", "tool": "{leg.tool}", "cycle": CYCLE}}
go = wait_for(lambda: (CTRL / "go_{leg.name}").exists(), 1800) if PREV_OK else "not_eligible"
LEG["go"] = go; print("go:", go)
rec = wait_for(lambda: (CTRL / "recorder_{leg.name}.log").exists() and "fly the route now" in (CTRL / "recorder_{leg.name}.log").read_text(), 900) if go == "ready" else go
LEG["recorder_status"] = rec; print("recorder status:", rec)
if rec == "ready":
    time.sleep(LEAD_IN_S)
    LEG["start_check"] = start_check("{leg.check}"); print("start check:", LEG["start_check"])
if rec == "ready" and LEG["start_check"]["ok"]:
    LEG["t_start_mono_ns"] = time.monotonic_ns(); LEG["t_start_wall_ns"] = time.time_ns()
    phases = []
    baseline = e1_identity._read_last_state(pathlib.Path(ident.run_dir) / "states.jsonl")
    baseline_cmd_seq = (baseline or {{}}).get("cmd_seq")
    if isinstance(baseline_cmd_seq, bool) or not isinstance(baseline_cmd_seq, int):
        LEG["outcome"] = "STOP no_valid_baseline_cmd_seq"; LEG["baseline"] = repr(baseline_cmd_seq)
        (CTRL / "stop").write_text(json.dumps(LEG, default=str))
    else:
        reachy.turn_on("r_arm")
        chk = e1_identity.require_compliance(ident.run_dir, R.R_JOINTS, compliant=False,
                                             timeout_s=COMPLIANCE_TIMEOUT_S, min_cmd_seq=baseline_cmd_seq)
        LEG["compliance_check"] = chk.as_dict(); print("compliance:", chk.as_dict())
        if chk.ok:
            try:
                ret = {fn}
                LEG["outcome"] = "returned"; LEG["returned"] = ret
            except Exception as exc:
                LEG["outcome"] = f"EXC {{type(exc).__name__}}: {{exc}}"; traceback.print_exc()
        else:
            LEG["outcome"] = "STOP compliance_check"
            (CTRL / "stop").write_text(json.dumps(chk.as_dict()))
    LEG["phases"] = phases
    LEG["t_end_mono_ns"] = time.monotonic_ns(); LEG["t_end_wall_ns"] = time.time_ns()
    LEG["elapsed_s"] = (LEG["t_end_mono_ns"] - LEG["t_start_mono_ns"]) / 1e9
    LEG["end_pose"] = {{k: round(v, 1) for k, v in _pose().items()}}
    print("outcome:", LEG["outcome"], "elapsed %.1f s" % LEG["elapsed_s"]); print("returned:", LEG.get("returned")); print("end pose:", LEG["end_pose"])
else:
    LEG["outcome"] = "not_attempted"
PREV_OK = LEG["outcome"] == "returned"
(CTRL / "{leg.name}_done").write_text(json.dumps(LEG, default=str)); print(json.dumps(LEG, default=str))
'''


def armon_cell_source() -> str:
    """Stage 0's policy-A preparation (decision note §4/§6): turn the right
    arm stiff BEFORE reset #1, recorded and gated exactly like any other
    leg -- `turn_on` is NOT treated as motion-free here. Going stiff pins
    `goal_position` to `present_position` for each joint (see this
    package's README, "Already-stiff arm" investigation, point 1's side
    effect note), so nothing here asserts the arm cannot move; the
    parallel recorder this cell waits on is what proves whether it did.
    No route call anywhere in this cell -- the only commanded action is
    `turn_on`, gated behind the same go-marker/recorder/baseline-cmd_seq
    chain every leg uses."""
    return '''# Stage 0 -- explicit recorded setup action (policy A): arm stiff before reset #1
LEG = {"leg": "armon", "route": None, "tool": "turn_on", "cycle": CYCLE}
go = wait_for(lambda: (CTRL / "go_armon").exists(), 1800) if PREV_OK else "not_eligible"
LEG["go"] = go; print("go:", go)
rec = wait_for(lambda: (CTRL / "recorder_armon.log").exists() and "fly the route now" in (CTRL / "recorder_armon.log").read_text(), 900) if go == "ready" else go
LEG["recorder_status"] = rec; print("recorder status:", rec)
if rec == "ready":
    time.sleep(LEAD_IN_S)
    LEG["t_start_mono_ns"] = time.monotonic_ns(); LEG["t_start_wall_ns"] = time.time_ns()
    baseline = e1_identity._read_last_state(pathlib.Path(ident.run_dir) / "states.jsonl")
    baseline_cmd_seq = (baseline or {}).get("cmd_seq")
    if isinstance(baseline_cmd_seq, bool) or not isinstance(baseline_cmd_seq, int):
        LEG["outcome"] = "STOP no_valid_baseline_cmd_seq"; LEG["baseline"] = repr(baseline_cmd_seq)
        (CTRL / "stop").write_text(json.dumps(LEG, default=str))
    else:
        reachy.turn_on("r_arm")
        chk = e1_identity.require_compliance(ident.run_dir, R.R_JOINTS, compliant=False,
                                             timeout_s=COMPLIANCE_TIMEOUT_S, min_cmd_seq=baseline_cmd_seq)
        LEG["compliance_check"] = chk.as_dict(); print("compliance:", chk.as_dict())
        LEG["outcome"] = "returned" if chk.ok else "STOP compliance_check"
        if not chk.ok:
            (CTRL / "stop").write_text(json.dumps(chk.as_dict()))
    LEG["t_end_mono_ns"] = time.monotonic_ns(); LEG["t_end_wall_ns"] = time.time_ns()
    LEG["elapsed_s"] = (LEG["t_end_mono_ns"] - LEG["t_start_mono_ns"]) / 1e9
    LEG["end_pose"] = {k: round(v, 1) for k, v in _pose().items()}
    print("outcome:", LEG["outcome"], "elapsed %.1f s" % LEG["elapsed_s"]); print("end pose:", LEG["end_pose"])
else:
    LEG["outcome"] = "not_attempted"
PREV_OK = LEG["outcome"] == "returned"
(CTRL / "armon_done").write_text(json.dumps(LEG, default=str)); print(json.dumps(LEG, default=str))
'''


def final_cell_source() -> str:
    return '''# Final read-only state; no further motion
print("final pose:", {k: round(v, 1) for k, v in _pose().items()}); print("done at wall", time.time_ns())
'''


def _validate_repo_and_evidence_dir(
    *, repo: str, evidence_dir: str, required_sha: str, run_git: RunGit,
) -> str:
    """Shared generation-time refusals (W4/M3, unchanged from PR #120):
    a real `--evidence-dir`, not under `docs/reviews/` or inside `--repo`,
    and a `--repo` HEAD that descends from `required_sha`. Returns the
    repo's current HEAD sha."""
    if not evidence_dir:
        raise GenerationRefused(
            "--evidence-dir is required (no default; must not point into "
            "docs/reviews/)")
    if "docs/reviews" in evidence_dir.replace("\\", "/"):
        raise GenerationRefused(
            f"--evidence-dir {evidence_dir!r} must not be under docs/reviews/")

    repo_path = pathlib.Path(repo).resolve()
    evidence_path_check = pathlib.Path(evidence_dir).resolve()
    if evidence_path_check == repo_path or repo_path in evidence_path_check.parents:
        raise GenerationRefused(
            f"--evidence-dir {evidence_dir!r} is inside --repo {repo!r} -- "
            "untracked runtime artifacts written under it (recorder runs, "
            "logs, the plan/notebook themselves) would show up in `git "
            "status --porcelain` and make the server's own code_sha_dirty "
            "true on every run (PR #120 review, M3); use a directory "
            "outside the repo's checkout entirely")

    sha_proc = run_git(["-C", repo, "rev-parse", "HEAD"])
    if sha_proc.returncode != 0:
        raise GenerationRefused(
            f"git rev-parse HEAD failed in {repo!r}: {sha_proc.stderr.strip()}")
    repo_sha = sha_proc.stdout.strip()

    anc_proc = run_git(["-C", repo, "merge-base", "--is-ancestor", required_sha, repo_sha])
    if anc_proc.returncode != 0:
        raise GenerationRefused(
            f"{repo!r} HEAD {repo_sha} does not contain {required_sha} -- "
            "restart the native server from a tree at or after that merge "
            "before generating")
    return repo_sha


def _write_notebook_and_plan(
    *, identity: str, cells: List[dict], evidence_dir: str, plan_doc: dict,
) -> pathlib.Path:
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i}"
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"name": "e1venv", "display_name": "e1venv",
                                       "language": "python"}},
          "nbformat": 4, "nbformat_minor": 5}

    evidence_path = pathlib.Path(evidence_dir)
    evidence_path.mkdir(parents=True, exist_ok=True)
    nb_path = evidence_path / f"e1_stage1_{identity}.ipynb"
    plan_path = evidence_path / f"plan_{identity}.json"
    plan_path.write_text(json.dumps(plan_doc, indent=2))
    nb_path.write_text(json.dumps(nb, indent=1))
    return nb_path


def generate(cycle: str, *, repo: str, evidence_dir: str,
             required_sha: str = plan.REQUIRED_SHA,
             merge_time_iso: str = plan.MERGE_TIME_ISO,
             lead_in_s: float = plan.LEAD_IN_S,
             compliance_timeout_s: float = plan.COMPLIANCE_TIMEOUT_S,
             board: str = "B4",
             run_git: RunGit = _run_git) -> pathlib.Path:
    """Legacy Stage 1 entry point (`S1a`/`S1b`/`S1c`) -- unchanged
    behaviour and output for `board="B4"` (its default), which is the only
    board Stage 1 ever ran. `board` is accepted so this shares
    `connect_cell_source`'s scene parameter with the Stage 2 entry points
    below rather than the two paths carrying independent scene literals;
    it does not add the Stage 2 reused-identity refusal (`generate_repetition`
    and `generate_stage0` below), since Stage 1's one-notebook-per-cycle
    shape never had a repetition to collide with."""
    if cycle not in plan.CYCLES:
        raise GenerationRefused(
            f"unknown cycle {cycle!r}; choose from {sorted(plan.CYCLES)}")
    repo_sha = _validate_repo_and_evidence_dir(
        repo=repo, evidence_dir=evidence_dir, required_sha=required_sha,
        run_git=run_git)
    scene_rel = plan.board_scene_rel(board)

    legs = list(plan.CYCLES[cycle])
    durations = plan.cycle_durations(
        cycle, lead_in_s=lead_in_s, compliance_timeout_s=compliance_timeout_s)

    cells = [
        _md_cell(markdown_cell_source(cycle, repo, repo_sha, legs)),
        _code_cell(connect_cell_source(
            repo=repo, evidence_dir=evidence_dir, repo_sha=repo_sha,
            cycle=cycle, lead_in_s=lead_in_s,
            compliance_timeout_s=compliance_timeout_s,
            required_sha=required_sha, merge_time_iso=merge_time_iso,
            scene_rel=scene_rel)),
        _code_cell(binding_cell_source()),
    ]
    cells.extend(_code_cell(motion_cell_source(leg)) for leg in legs)
    cells.append(_code_cell(final_cell_source()))

    plan_doc = {
        "cycle": cycle, "board": board, "scene_rel": scene_rel,
        "repo": repo, "repo_sha": repo_sha,
        "required_sha": required_sha, "merge_time_iso": merge_time_iso,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "lead_in_s": lead_in_s, "compliance_timeout_s": compliance_timeout_s,
        "margin_s": plan.MARGIN_S, "route_budget_s": plan.ROUTE_BUDGET_S,
        "unflown_routes": list(plan.UNFLOWN_ROUTES),
        "legs": {leg.name: {"route": leg.route, "dur_s": durations[leg.name]}
                 for leg in legs},
    }
    return _write_notebook_and_plan(
        identity=cycle, cells=cells, evidence_dir=evidence_dir, plan_doc=plan_doc)


def generate_repetition(
    board: str, shape: str, rep: int, *, session: str, repo: str,
    evidence_dir: str, required_sha: str = plan.REQUIRED_SHA,
    merge_time_iso: str = plan.MERGE_TIME_ISO,
    lead_in_s: float = plan.LEAD_IN_S,
    compliance_timeout_s: float = plan.COMPLIANCE_TIMEOUT_S,
    run_git: RunGit = _run_git,
) -> pathlib.Path:
    """Stage 2 entry point (decision note §5/§6): one repetition of one
    shape on one board, within one board-session. `plan.stage2_legs` gives
    every leg a name unique to this board+shape+repetition; generation
    additionally refuses (before writing anything) if this board's session
    in `evidence_dir` has already been recorded as a different session, or
    if this exact identity (or any of its legs' markers) already exists --
    the two checks decision note §6 item 2 calls for."""
    legs = list(plan.stage2_legs(board, shape, rep))
    identity = plan.cycle_id(board, shape, rep)
    _check_and_record_session(evidence_dir, board=board, session=session)
    _refuse_if_identity_reused(
        evidence_dir, identity=identity, leg_names=[leg.name for leg in legs])
    repo_sha = _validate_repo_and_evidence_dir(
        repo=repo, evidence_dir=evidence_dir, required_sha=required_sha,
        run_git=run_git)
    scene_rel = plan.board_scene_rel(board)

    durations = {
        leg.name: plan.leg_duration_s(
            leg.route, lead_in_s=lead_in_s,
            compliance_timeout_s=compliance_timeout_s)
        for leg in legs
    }

    cells = [
        _md_cell(markdown_cell_source(identity, repo, repo_sha, legs)),
        _code_cell(connect_cell_source(
            repo=repo, evidence_dir=evidence_dir, repo_sha=repo_sha,
            cycle=identity, lead_in_s=lead_in_s,
            compliance_timeout_s=compliance_timeout_s,
            required_sha=required_sha, merge_time_iso=merge_time_iso,
            scene_rel=scene_rel)),
        _code_cell(binding_cell_source()),
    ]
    cells.extend(_code_cell(motion_cell_source(leg)) for leg in legs)
    cells.append(_code_cell(final_cell_source()))

    plan_doc = {
        "cycle": identity, "board": board, "shape": shape, "rep": rep,
        "session": session, "scene_rel": scene_rel,
        "repo": repo, "repo_sha": repo_sha,
        "required_sha": required_sha, "merge_time_iso": merge_time_iso,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "lead_in_s": lead_in_s, "compliance_timeout_s": compliance_timeout_s,
        "margin_s": plan.MARGIN_S, "route_budget_s": plan.ROUTE_BUDGET_S,
        "unflown_routes": list(plan.UNFLOWN_ROUTES),
        "legs": {leg.name: {"route": leg.route, "dur_s": durations[leg.name]}
                 for leg in legs},
    }
    return _write_notebook_and_plan(
        identity=identity, cells=cells, evidence_dir=evidence_dir,
        plan_doc=plan_doc)


def generate_stage0(
    board: str, session: str, *, repo: str, evidence_dir: str,
    required_sha: str = plan.REQUIRED_SHA,
    merge_time_iso: str = plan.MERGE_TIME_ISO,
    lead_in_s: float = plan.LEAD_IN_S,
    compliance_timeout_s: float = plan.COMPLIANCE_TIMEOUT_S,
    run_git: RunGit = _run_git,
) -> pathlib.Path:
    """Stage 2 policy A's one-time-per-board-session setup (decision note
    §4/§6): `turn_on("r_arm")` + `require_compliance`, before reset #1, as
    its own recorded action (`armon_cell_source`) -- not asserted
    motion-free. Shares the reused-identity refusal with
    `generate_repetition`; `identity` here is `stage0-<board>-<session>`,
    so a repeated Stage 0 call for the same board+session is refused the
    same way a repeated cycle would be."""
    plan.board_scene_rel(board)  # raises on an unauthorized/unknown board
    identity = f"stage0-{board}-{session}"
    _check_and_record_session(evidence_dir, board=board, session=session)
    _refuse_if_identity_reused(
        evidence_dir, identity=identity, leg_names=["armon"])
    repo_sha = _validate_repo_and_evidence_dir(
        repo=repo, evidence_dir=evidence_dir, required_sha=required_sha,
        run_git=run_git)
    scene_rel = plan.board_scene_rel(board)

    # The armon step flies no route; this label is only the recorder's
    # planned-vs-realised clearance reference, matching how a parked
    # recording's own ROUTE argument is used (never flown).
    route_label = plan.SHAPES["a"][0].route
    dur_s = plan.armon_duration_s(
        lead_in_s=lead_in_s, compliance_timeout_s=compliance_timeout_s)

    cells = [
        _md_cell(
            f"# E1 Stage 2 Stage 0 -- board `{board}`, session `{session}` "
            f"-- policy A arm-on setup, no route\n\n"
            f"Code `{repo}` = `{repo_sha}`. One recorded action: "
            "`turn_on(\"r_arm\")` + `require_compliance`, gated exactly "
            "like a leg; not asserted motion-free."),
        _code_cell(connect_cell_source(
            repo=repo, evidence_dir=evidence_dir, repo_sha=repo_sha,
            cycle=identity, lead_in_s=lead_in_s,
            compliance_timeout_s=compliance_timeout_s,
            required_sha=required_sha, merge_time_iso=merge_time_iso,
            scene_rel=scene_rel)),
        _code_cell(binding_cell_source()),
        _code_cell(armon_cell_source()),
        _code_cell(final_cell_source()),
    ]

    plan_doc = {
        "cycle": identity, "board": board, "session": session,
        "scene_rel": scene_rel, "repo": repo, "repo_sha": repo_sha,
        "required_sha": required_sha, "merge_time_iso": merge_time_iso,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "lead_in_s": lead_in_s, "compliance_timeout_s": compliance_timeout_s,
        "margin_s": plan.MARGIN_S,
        "legs": {"armon": {"route": route_label, "dur_s": dur_s}},
    }
    return _write_notebook_and_plan(
        identity=identity, cells=cells, evidence_dir=evidence_dir,
        plan_doc=plan_doc)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cycle", nargs="?", choices=sorted(plan.CYCLES),
                    help="legacy Stage 1 cycle id (S1a/S1b/S1c)")
    p.add_argument("--repo", required=True)
    p.add_argument("--evidence-dir", required=True)
    p.add_argument("--lead-in-s", type=float, default=plan.LEAD_IN_S)
    p.add_argument("--compliance-timeout-s", type=float,
                    default=plan.COMPLIANCE_TIMEOUT_S)
    p.add_argument("--board", choices=plan.BOARD_ORDER,
                    help="Stage 2: board for --stage0 or --shape/--rep")
    p.add_argument("--shape", choices=plan.SHAPE_ORDER,
                    help="Stage 2: shape for a repetition (with --board/--rep/--session)")
    p.add_argument("--rep", type=int,
                    help="Stage 2: repetition number, 1-based (with --board/--shape/--session)")
    p.add_argument("--session",
                    help="Stage 2: session id, one per board server run")
    p.add_argument("--stage0", action="store_true",
                    help="generate the Stage 2 policy-A arm-on setup notebook "
                         "(with --board/--session)")
    return p.parse_args(argv)


def main(argv: List[str] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    common = dict(repo=args.repo, evidence_dir=args.evidence_dir,
                  lead_in_s=args.lead_in_s,
                  compliance_timeout_s=args.compliance_timeout_s)
    try:
        if args.stage0:
            if not (args.board and args.session):
                raise GenerationRefused(
                    "--stage0 requires --board and --session")
            if args.cycle or args.shape or args.rep:
                raise GenerationRefused(
                    "--stage0 takes no cycle/--shape/--rep")
            out = generate_stage0(args.board, args.session, **common)
        elif args.cycle:
            if args.board or args.shape or args.rep or args.session:
                raise GenerationRefused(
                    "a legacy cycle id takes no --board/--shape/--rep/--session")
            out = generate(args.cycle, **common)
        elif args.board and args.shape and args.rep and args.session:
            out = generate_repetition(
                args.board, args.shape, args.rep, session=args.session,
                **common)
        else:
            raise GenerationRefused(
                "specify either a legacy cycle id, --stage0 with "
                "--board/--session, or --board/--shape/--rep/--session")
    except GenerationRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
