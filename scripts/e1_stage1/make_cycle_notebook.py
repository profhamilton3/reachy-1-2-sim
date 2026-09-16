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
gate would refuse forever without explaining why.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
from typing import Callable, List

from . import plan

RunGit = Callable[[List[str]], subprocess.CompletedProcess]


class GenerationRefused(Exception):
    """Raised instead of writing any file."""


def _run_git(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True)


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
                         merge_time_iso: str) -> str:
    return f'''# Cell 1 -- connect with literals; hygiene shown; tree identity pinned at generation time
import os, subprocess, sys, time, json, pathlib, traceback
sys.path.insert(0, "{repo}/src"); sys.path.insert(0, "{repo}/scripts")
sys.path.insert(0, "{repo}/scripts/e1_stage1")
CTRL = pathlib.Path("{evidence_dir}/control"); RECORD_ROOT = "{evidence_dir}/e1_server_runs"
SCENE = "{repo}/scenes/e1_boards/B4_pool_box_1_r2c3.yaml"
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
import e1_identity, provenance
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
prov_ok, prov_reasons = provenance.check_binding_provenance(manifest, git_is_ancestor=_git_is_ancestor, required_sha=REQUIRED_SHA, merge_time_iso=MERGE_TIME_ISO)
d["provenance_ok"] = prov_ok; d["provenance_reasons"] = prov_reasons
print(json.dumps(d, indent=2, default=str))
BINDING_OK = bool(ident.ok) and prov_ok
(CTRL / (f"binding_ok_{CYCLE}" if BINDING_OK else f"binding_FAIL_{CYCLE}")).write_text(json.dumps(d, default=str))
print("BINDING_OK =", BINDING_OK); print("present:", {k: round(v, 1) for k, v in _pose().items()})
def wait_for(pred, timeout_s, period=0.25):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if (CTRL / "stop").exists(): return "stop"
        if pred(): return "ready"
        time.sleep(period)
    return "timeout"
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


def final_cell_source() -> str:
    return '''# Final read-only state; no further motion
print("final pose:", {k: round(v, 1) for k, v in _pose().items()}); print("done at wall", time.time_ns())
'''


def generate(cycle: str, *, repo: str, evidence_dir: str,
             required_sha: str = plan.REQUIRED_SHA,
             merge_time_iso: str = plan.MERGE_TIME_ISO,
             lead_in_s: float = plan.LEAD_IN_S,
             compliance_timeout_s: float = plan.COMPLIANCE_TIMEOUT_S,
             run_git: RunGit = _run_git) -> pathlib.Path:
    if cycle not in plan.CYCLES:
        raise GenerationRefused(
            f"unknown cycle {cycle!r}; choose from {sorted(plan.CYCLES)}")
    if not evidence_dir:
        raise GenerationRefused(
            "--evidence-dir is required (no default; must not point into "
            "docs/reviews/)")
    if "docs/reviews" in evidence_dir.replace("\\", "/"):
        raise GenerationRefused(
            f"--evidence-dir {evidence_dir!r} must not be under docs/reviews/")

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

    legs = list(plan.CYCLES[cycle])
    durations = plan.cycle_durations(
        cycle, lead_in_s=lead_in_s, compliance_timeout_s=compliance_timeout_s)

    cells = [
        _md_cell(markdown_cell_source(cycle, repo, repo_sha, legs)),
        _code_cell(connect_cell_source(
            repo=repo, evidence_dir=evidence_dir, repo_sha=repo_sha,
            cycle=cycle, lead_in_s=lead_in_s,
            compliance_timeout_s=compliance_timeout_s,
            required_sha=required_sha, merge_time_iso=merge_time_iso)),
        _code_cell(binding_cell_source()),
    ]
    cells.extend(_code_cell(motion_cell_source(leg)) for leg in legs)
    cells.append(_code_cell(final_cell_source()))
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i}"

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"name": "e1venv", "display_name": "e1venv",
                                       "language": "python"}},
          "nbformat": 4, "nbformat_minor": 5}

    evidence_path = pathlib.Path(evidence_dir)
    evidence_path.mkdir(parents=True, exist_ok=True)
    nb_path = evidence_path / f"e1_stage1_{cycle}.ipynb"
    plan_path = evidence_path / f"plan_{cycle}.json"

    plan_doc = {
        "cycle": cycle, "repo": repo, "repo_sha": repo_sha,
        "required_sha": required_sha, "merge_time_iso": merge_time_iso,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "lead_in_s": lead_in_s, "compliance_timeout_s": compliance_timeout_s,
        "margin_s": plan.MARGIN_S, "route_budget_s": plan.ROUTE_BUDGET_S,
        "unflown_routes": list(plan.UNFLOWN_ROUTES),
        "legs": {leg.name: {"route": leg.route, "dur_s": durations[leg.name]}
                 for leg in legs},
    }
    plan_path.write_text(json.dumps(plan_doc, indent=2))
    nb_path.write_text(json.dumps(nb, indent=1))
    return nb_path


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cycle", choices=sorted(plan.CYCLES))
    p.add_argument("--repo", required=True)
    p.add_argument("--evidence-dir", required=True)
    p.add_argument("--lead-in-s", type=float, default=plan.LEAD_IN_S)
    p.add_argument("--compliance-timeout-s", type=float,
                    default=plan.COMPLIANCE_TIMEOUT_S)
    return p.parse_args(argv)


def main(argv: List[str] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        out = generate(args.cycle, repo=args.repo, evidence_dir=args.evidence_dir,
                        lead_in_s=args.lead_in_s,
                        compliance_timeout_s=args.compliance_timeout_s)
    except GenerationRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
