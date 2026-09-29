"""Synthetic-only tests of the prepared (NOT executed) U-only process-separated C2 diagnostic:
``tests/fixtures/goalfix_cmp/{c2ps_common,c2ps_children,c2_process_split}.py`` and
``run_c2_process_split.sh``.

FORBIDDEN HERE (and not done): starting NativeStub, a bridge, a gRPC server, an SDK client, or
the real driver end to end.  What IS used: fake child processes (``c2ps_fake_child.py``: tiny
scripts speaking the same request protocol), in-process fake stub/observer objects, throw-away git
repositories, and ``make_fixtures`` synthetic legs.  Importing ``stage_a_slice``/``real_bridge``
(which imports the SDK modules) starts nothing.
"""
from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_FIX = os.path.join(_ROOT, "tests", "fixtures", "goalfix_cmp")
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "native_mujoco"),
           os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "tests", "integration"), _FIX):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import c2_process_split as d  # noqa: E402
import c2ps_children as ch  # noqa: E402
import c2ps_common as cc  # noqa: E402

FAKE_CHILD = os.path.join(_HERE, "c2ps_fake_child.py")
SH = os.path.join(_FIX, "run_c2_process_split.sh")
DRIVER_FILES = [os.path.join(_FIX, n) for n in ("c2_process_split.py", "c2ps_children.py", "c2ps_common.py")]
FORBIDDEN_HEAVY = ("reachy_sdk", "grpc", "websockets", "native_stub", "mujoco_remote_backend",
                   "fake_reachy_server")


# ---------------------------------------------------------------------------
# throw-away git repo for pin/tree/blob tests
# ---------------------------------------------------------------------------

def _g(repo, *args, check=True):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo), *args],
                          check=check, capture_output=True, text=True).stdout.strip()


def _write(repo, rel, text):
    p = Path(repo) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


DRIVER_RELS = ("tests/fixtures/goalfix_cmp/c2_process_split.py", "tests/fixtures/goalfix_cmp/c2ps_children.py",
               "tests/fixtures/goalfix_cmp/c2ps_common.py")


def make_repo(tmp_path, *, mutate=None, sibling=False):
    """B (bridge blobs + a runtime tree) -> M' (adds tools/goalfix_cmp and one existing test file)
    -> pin (adds the three driver files).  ``mutate(repo)`` edits the working tree before the pin
    commit.  ``sibling``: the pin descends from B instead of M' (same tools tree)."""
    repo = tmp_path / f"repo-{uuid.uuid4().hex[:6]}"
    repo.mkdir()
    _g(repo, "init", "-q")
    for f in d.BRIDGE_OPT_FILES:
        _write(repo, f, f"# {f}\n")
    _write(repo, "kinematic_backend.py", "# k\n")
    _write(repo, "src/a.py", "a = 1\n")
    _g(repo, "add", "-A")
    _g(repo, "commit", "-qm", "B")
    b = _g(repo, "rev-parse", "HEAD")
    _write(repo, "tools/goalfix_cmp/x.py", "x = 1\n")
    _write(repo, "tests/unit/existing.py", "e = 1\n")
    _g(repo, "add", "-A")
    _g(repo, "commit", "-qm", "M")
    m = _g(repo, "rev-parse", "HEAD")
    tools_tree = _g(repo, "rev-parse", "HEAD:tools/goalfix_cmp")
    if sibling:
        _g(repo, "checkout", "-q", "-b", "sib", b)
        _write(repo, "tools/goalfix_cmp/x.py", "x = 1\n")
    for rel in DRIVER_RELS:
        _write(repo, rel, "# driver\n")
    if mutate:
        mutate(repo)
    _g(repo, "add", "-A")
    _g(repo, "commit", "-qm", "pin")
    pin = _g(repo, "rev-parse", "HEAD")
    kw = dict(m_sha=m, b_sha=b, m_tools_tree=tools_tree, runtime_trees=["src", "kinematic_backend.py"])
    return repo, pin, kw


def _home_out(tag="out"):
    return str(Path.home() / f"c2ps-test-{tag}-{uuid.uuid4().hex[:8]}")


# ---------------------------------------------------------------------------
# 1. argument / pin refusals
# ---------------------------------------------------------------------------

class TestRefusals:
    def test_runs_must_be_exactly_seven(self):
        for bad in (6, 8, "0", "x", None):
            with pytest.raises(d.Refusal):
                d.check_runs(bad)
        assert d.check_runs(7) == 7 and d.check_runs("7") == 7

    @pytest.mark.parametrize("pin", ["", "abc", "A" * 40, "g" * 40, "0" * 39, "0" * 41])
    def test_bad_pin_format(self, pin):
        with pytest.raises(d.Refusal):
            d.check_pin_format(pin)

    @pytest.mark.parametrize("out", ["/tmp/c2ps-x", "/private/tmp/c2ps-x", "/tmp", "relative/out", ""])
    def test_bad_out_path(self, out):
        with pytest.raises(d.Refusal):
            d.check_out_path(out)

    def test_existing_out_refused(self):
        p = Path(_home_out("exists"))
        p.mkdir()
        try:
            with pytest.raises(d.Refusal, match="already exists"):
                d.check_out_path(str(p))
        finally:
            p.rmdir()

    def test_missing_parent_refused(self):
        with pytest.raises(d.Refusal, match="parent"):
            d.check_out_path(str(Path.home() / f"no-such-{uuid.uuid4().hex}" / "out"))

    def test_free_space_precondition(self):
        with pytest.raises(d.Refusal, match="free space"):
            d.check_free_space(str(Path.home()), floor=10 ** 18)
        assert d.check_free_space(str(Path.home()), floor=1) > 0

    def test_cli_returns_2_and_names_the_refusal(self, capsys):
        base = ["preflight", "--repo", _ROOT, "--venv", str(Path(sys.executable).parent.parent)]
        for extra, needle in (
                (["--pin", "abc", "--out", _home_out(), "--runs", "7"], "40-hex"),
                (["--pin", "0" * 40, "--out", _home_out(), "--runs", "6"], "exactly 7"),
                (["--pin", "0" * 40, "--out", "/tmp/c2ps-x", "--runs", "7"], "outside /tmp")):
            assert d.main(base + extra) == 2
            assert needle in capsys.readouterr().err

    def test_facts_ok_on_a_clean_repo(self, tmp_path):
        repo, pin, kw = make_repo(tmp_path)
        facts = d.source_pin_facts(repo, pin, **kw)
        assert facts["pin"] == pin and set(facts["opt_file_sha256"]) == set(d.BRIDGE_OPT_FILES)
        assert sorted(facts["added_files"]) == sorted(DRIVER_RELS)

    def test_tools_tree_mismatch(self, tmp_path):
        repo, pin, kw = make_repo(tmp_path, mutate=lambda r: _write(r, "tools/goalfix_cmp/x.py", "x = 2\n"))
        with pytest.raises(d.Refusal, match="tools/goalfix_cmp tree differs"):
            d.source_pin_facts(repo, pin, **kw)

    def test_expected_tools_tree_constant_is_enforced(self, tmp_path):
        repo, pin, kw = make_repo(tmp_path)
        kw = dict(kw, m_tools_tree="0" * 40)
        with pytest.raises(d.Refusal, match="tools/goalfix_cmp tree differs"):
            d.source_pin_facts(repo, pin, **kw)

    def test_bridge_blob_mismatch(self, tmp_path):
        repo, pin, kw = make_repo(tmp_path, mutate=lambda r: _write(r, "mujoco_remote_backend.py", "# changed\n"))
        with pytest.raises(d.Refusal, match="mujoco_remote_backend.py.*differs from B"):
            d.source_pin_facts(repo, pin, **kw)

    def test_runtime_tree_mismatch(self, tmp_path):
        repo, pin, kw = make_repo(tmp_path, mutate=lambda r: _write(r, "src/a.py", "a = 2\n"))
        with pytest.raises(d.Refusal, match="runtime tree src"):
            d.source_pin_facts(repo, pin, **kw)

    def test_pin_must_descend_from_m_prime(self, tmp_path):
        repo, pin, kw = make_repo(tmp_path, sibling=True)
        with pytest.raises(d.Refusal, match="does not descend"):
            d.source_pin_facts(repo, pin, **kw)

    def test_modified_existing_file_refused(self, tmp_path):
        repo, pin, kw = make_repo(tmp_path, mutate=lambda r: _write(r, "tests/unit/existing.py", "e = 2\n"))
        with pytest.raises(d.Refusal, match="only NEW files"):
            d.source_pin_facts(repo, pin, **kw)

    def test_new_file_outside_allowed_prefixes_refused(self, tmp_path):
        repo, pin, kw = make_repo(tmp_path, mutate=lambda r: _write(r, "docs/x.md", "x\n"))
        with pytest.raises(d.Refusal, match="only NEW files"):
            d.source_pin_facts(repo, pin, **kw)

    def test_missing_driver_file_refused(self, tmp_path):
        def drop(r):
            (Path(r) / DRIVER_RELS[1]).unlink()
        repo, pin, kw = make_repo(tmp_path, mutate=drop)
        with pytest.raises(d.Refusal, match="not present at the pin"):
            d.source_pin_facts(repo, pin, **kw)

    def test_absent_commit_refused(self, tmp_path):
        repo, pin, kw = make_repo(tmp_path)
        with pytest.raises(d.Refusal, match="not present"):
            d.source_pin_facts(repo, "1" * 40, **kw)

    def test_shell_refusals_exit_2(self):
        venv = str(Path(sys.executable).parent.parent)
        for args in (["--pin", "abc", "--out", _home_out(), "--runs", "7"],
                     ["--pin", "0" * 40, "--out", _home_out(), "--runs", "6"],
                     ["--pin", "0" * 40, "--out", "/private/tmp/c2ps-x", "--runs", "7"],
                     ["--pin", "0" * 40, "--runs", "7"],
                     ["--bogus"]):
            r = subprocess.run(["bash", SH, "--venv", venv, *args], capture_output=True, text=True)
            assert r.returncode == 2, (args, r.stdout, r.stderr)
            assert "REFUSED" in r.stderr


# ---------------------------------------------------------------------------
# 2. dry run: no forbidden import, nothing spawned but git, nothing created
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_preflight_imports_nothing_heavy_and_spawns_only_git(self, tmp_path):
        repo, pin, kw = make_repo(tmp_path)
        out = _home_out("dry")
        script = f"""
import json, subprocess, sys
sys.path.insert(0, {_FIX!r})
calls = []
_orig = subprocess.Popen
class Spy(_orig):
    def __init__(self, args, *a, **k):
        calls.append(list(args)[0] if not isinstance(args, str) else args)
        super().__init__(args, *a, **k)
subprocess.Popen = Spy
import c2_process_split as d
d.source_pin_facts.__kwdefaults__.update(m_sha={kw['m_sha']!r}, b_sha={kw['b_sha']!r},
    m_tools_tree={kw['m_tools_tree']!r}, runtime_trees={kw['runtime_trees']!r})
rc = d.main(["preflight", "--pin", {pin!r}, "--out", {out!r}, "--repo", {str(repo)!r}, "--runs", "7",
             "--venv", {str(Path(sys.executable).parent.parent)!r}])
heavy = [m for m in {FORBIDDEN_HEAVY!r} if m in sys.modules]
print(json.dumps({{"rc": rc, "calls": calls, "heavy": heavy, "numpy": "numpy" in sys.modules}}))
"""
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        doc = json.loads(r.stdout.strip().splitlines()[-1])
        assert doc["rc"] == 0
        assert doc["heavy"] == []
        assert doc["calls"] and set(doc["calls"]) == {"git"}
        assert not os.path.lexists(out)
        assert "PREFLIGHT OK" in r.stdout and "step 22." in r.stdout

    def test_module_import_is_stdlib_only(self):
        script = f"""
import json, sys
sys.path.insert(0, {_FIX!r})
import c2_process_split, c2ps_common, c2ps_children
print(json.dumps([m for m in {FORBIDDEN_HEAVY!r} + ('numpy', 'stage_a_slice', 'real_bridge') if m in sys.modules]))
"""
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout) == []

    def test_plan_names_processes_steps_caps_and_timeouts(self):
        text = "\n".join(d.plan_lines("a" * 40, "/x/out"))
        assert len(d.STEP_PLAN) == 22
        for i in range(1, 8):
            assert f"run-{i:02d}:" in text
        for needle in ("15 min wall per run", "90 min wall total", "1.0 GiB", "20 GiB", "step 22.",
                       "process ready 60", "reset 30", "each leg / client step 300", "coordinator", "N (NativeStub",
                       "B_0..B_4", "C_0..C_4"):
            assert needle in text, needle
        assert d.NATIVESTUB_LIMITATION in text


# ---------------------------------------------------------------------------
# 3. run_loop
# ---------------------------------------------------------------------------

def _rec(i, status="ok"):
    return {"run": f"run-{i:02d}", "index": i, "status": status, "failure": None if status == "ok" else {"kind": "k", "detail": "d"},
            "cycles": []}


class TestRunLoop:
    def test_exactly_seven_in_order(self):
        seen = []
        recs = d.run_loop(lambda i: (seen.append(i), _rec(i))[1])
        assert seen == [1, 2, 3, 4, 5, 6, 7]
        assert [r["run"] for r in recs] == [f"run-{i:02d}" for i in range(1, 8)]

    def test_n_runs_other_than_seven_refused(self):
        with pytest.raises(d.Refusal):
            d.run_loop(lambda i: _rec(i), n_runs=6)

    def test_infrastructure_failure_stops_and_marks_rest_not_run(self):
        seen = []
        recs = d.run_loop(lambda i: (seen.append(i), _rec(i, "failed" if i == 3 else "ok"))[1])
        assert seen == [1, 2, 3]
        assert [r["status"] for r in recs] == ["ok", "ok", "failed", "not_run", "not_run", "not_run", "not_run"]

    def test_c0_c8_failures_never_stop_and_nothing_is_retried(self):
        calls = []

        def process(i):
            calls.append(i)
            rec = _rec(i)
            rec["cycles"] = [{"name": "S2-B4-c-r2", "arm": "B", "legs": {"setup": {
                "status": "evaluated", "checks": {c: {"passed": False, "index": 0, "detail": "x"} for c in d.CHECKS},
                "c2": {"reproduced": True, "n_gt_30ms": 5, "max_spread_ms": 99.0}}}}]
            return rec
        recs = d.run_loop(process)
        assert calls == list(range(1, 8)) and len(set(calls)) == 7      # each index once: no retry
        assert all(r["status"] == "ok" for r in recs)

    def test_precheck_stops_before_the_run_and_records_the_reason(self):
        seen = []
        state = {"n": 0}

        def precheck():
            state["n"] += 1
            return {"kind": "total_cap", "detail": "x"} if state["n"] == 3 else None
        recs = d.run_loop(lambda i: (seen.append(i), _rec(i))[1], precheck=precheck)
        assert seen == [1, 2]
        assert recs[2]["status"] == "not_run" and recs[2]["failure"]["kind"] == "total_cap"
        assert all(r["status"] == "not_run" for r in recs[2:])

    def test_no_retry_construct_in_the_loop_source(self):
        src = Path(DRIVER_FILES[0]).read_text()
        body = src[src.index("def run_loop("):src.index("def verify_source(")]
        code = "\n".join(l for l in body.splitlines() if not l.strip().startswith("#"))
        assert "while " not in code and "retry" not in code.split('"""')[-1]


# ---------------------------------------------------------------------------
# 4. stop classifier
# ---------------------------------------------------------------------------

class TestClassifier:
    @pytest.mark.parametrize("facts,kind", [
        ({"coordinator_failure": {"kind": "child_died", "detail": "N"}}, "child_died"),
        ({"coordinator_failure": {"kind": "step_timeout", "detail": "leg"}}, "step_timeout"),
        ({"coordinator_failure": {"kind": "connection", "detail": "grpc UNAVAILABLE"}}, "connection"),
        ({"provenance_problems": ["C0: holds NativeStub"]}, "separation"),
        ({"child_exit_codes": {"B1": 0, "C1": 9}}, "child_died"),
        ({"reset_verify_rcs": [0, 1]}, "reset_verify"),
        ({"checksum_mismatch": "states.jsonl sha256 mismatch"}, "evidence_checksum"),
        ({"missing_artifacts": ["S2-B4-c-r1 full CLI output"]}, "missing_artifact"),
        ({"malformed_artifacts": ["checkpoint output"]}, "malformed_artifact"),
        ({"incomplete_legs": ["run-01/S2-B4-c-r2/setup"]}, "c2_not_computable"),
        ({"leftover_pids": [123]}, "leftover_process"),
        ({"cap": "run_cap", "cap_detail": "900 s"}, "run_cap"),
        ({"cap": "total_cap"}, "total_cap"),
        ({"cap": "output_cap"}, "output_cap"),
        ({"interrupted": True}, "interrupted"),
    ])
    def test_infrastructure_and_evidence_cases_stop(self, facts, kind):
        got = d.classify_failure(facts)
        assert got is not None and got["kind"] == kind

    @pytest.mark.parametrize("facts", [
        {}, {"c2_failed": True}, {"verdict": "stop"},
        {"legs": {"C2": {"passed": False}}, "max_spread_ms": 99.0, "n_gt_30ms": 12},
        {"checks_failed": ["C0", "C2", "C5", "C7"], "unverified_legs": ["run-01/S2-B4-c-r2/setup"]},
        {"child_exit_codes": {"N": 0, "B0": 0}, "reset_verify_rcs": [0, 0, 0, 0], "incomplete_legs": [],
         "missing_artifacts": [], "malformed_artifacts": []},
    ])
    def test_observations_never_stop(self, facts):
        assert d.classify_failure(facts) is None

    def test_exception_mapping(self):
        assert d.classify_exception(cc.StepTimeout("x"))["kind"] == "step_timeout"
        assert d.classify_exception(cc.ChildDied("x"))["kind"] == "child_died"
        assert d.classify_exception(cc.InfraFailure("reset_verify", "rc"))["kind"] == "reset_verify"
        assert d.classify_exception(cc.RpcError("grpc._channel._InactiveRpcError UNAVAILABLE"))["kind"] == "connection"
        assert d.classify_exception(cc.RpcError("websockets ConnectionClosed"))["kind"] == "connection"
        assert d.classify_exception(cc.RpcError("KeyError: 'x'"))["kind"] == "child_error"
        assert d.classify_exception(ValueError("boom"))["kind"] == "coordinator_exception"

    def test_missing_leg_is_incomplete_and_never_counted_as_zero(self):
        rec = _rec(1)
        rec["cycles"] = [{"name": "S2-B4-c-r2", "arm": "B", "legs": {
            "setup": {"status": "incomplete", "error": "no leg"},
            "flight": {"status": "evaluated", "checks": {c: {"passed": True} for c in d.CHECKS},
                       "c2": {"reproduced": True, "n_gt_30ms": 0, "max_spread_ms": 4.0}}}}]
        sr = d.summarize_run(rec)
        assert "run-01/S2-B4-c-r2/setup" in sr["incomplete"]
        assert sr["per_kind"]["setup"]["evaluated"] == 0          # not a zero-event evaluated leg
        assert sr["per_kind"]["flight"]["evaluated"] == 1
        # every other expected leg of an "ok" run is also reported incomplete, not silently zero
        assert len(sr["incomplete"]) == 8 - 1


# ---------------------------------------------------------------------------
# 5. lifecycle with FAKE children
# ---------------------------------------------------------------------------

def _fake_reset_verify(argv):
    if argv[0] == "snapshot":
        return 0, "snap\n", ""
    return 0, f"reset gen={argv[2]} ack={argv[3]} resets_recorded 0->1 sim_step 0->0\n", ""


class FakeRig:
    def __init__(self, tmp_path, modes="", term_grace=1.0):
        self.tmp = tmp_path
        self.run_dir = tmp_path / "run-01"
        self.proc = self.run_dir / "proc"
        self.events = tmp_path / "events.jsonl"
        env = dict(os.environ, C2PS_FAKE_EVENTS=str(self.events), C2PS_FAKE_MODE=modes)
        self.sup = cc.Supervisor(self.proc, env, str(tmp_path), term_grace=term_grace)
        self.orch = d.RunOrchestrator(self.run_dir, self.sup, self.argv_for, _fake_reset_verify, sleep=lambda s: None)

    def argv_for(self, role, name, **kw):
        return [sys.executable, FAKE_CHILD, role, "--name", name, "--proc-dir", str(self.proc),
                "--ev-dir", str(self.run_dir / "ev"), "--control-dir", str(self.run_dir / "control")]

    def ev(self):
        return [json.loads(l) for l in self.events.read_text().splitlines()]

    def pos(self, who, what, nth=0):
        hits = [i for i, e in enumerate(self.ev()) if e["who"] == who and e["what"] == what]
        return hits[nth]


class TestLifecycle:
    def test_full_run_with_fake_children_orders_steps_and_leaves_nothing(self, tmp_path):
        rig = FakeRig(tmp_path)
        try:
            out = rig.orch.execute()
        finally:
            down = rig.sup.shutdown_all()
        assert down["survivors"] == [] and all(v == 0 for v in down["exit_codes"].values())
        assert [c["arm"] for c in out["cycles"]] == ["A", "B", "B", "A"]
        for k in range(4):
            # A4: C_k's channels close and it exits BEFORE B_k stops; B_k is gone BEFORE B_{k+1} starts
            assert rig.pos(f"C{k}", "close") < rig.pos(f"C{k}", "exit") < rig.pos(f"B{k}", "stop") \
                < rig.pos(f"B{k}", "exit") < rig.pos(f"B{k + 1}", "start")
            # C_{k+1} is fully up before its cycle's reset is even requested, well before step 12
            assert rig.pos(f"B{k + 1}", "ready") < rig.pos(f"C{k + 1}", "ready") < rig.pos(f"B{k + 1}", "reset")
            # steps 5/6 (reset_t in N) after the new bridge is up and before the bridge reset; commit after it
            assert rig.pos(f"B{k + 1}", "ready") < rig.pos("N", "begin_reset", k) < rig.pos(f"B{k + 1}", "reset") \
                < rig.pos("N", "commit_reset", k)
            # the recreate mark (step 3) is taken after the old bridge is gone
            assert rig.pos(f"B{k}", "exit") < rig.pos("N", "recreate_mark", k)
        # the slice's A/B cycles: echo injection for A only, over HOVER..REST_SHUT
        inj = [e for e in rig.ev() if e["what"] == "inject_echoes"]
        assert len(inj) == 2
        assert all(e["first_command_index"] == 10 and e["last_command_index"] == 40 for e in inj)
        assert [e["seed"] for e in inj] == [d.ECHO_SEED_BASE + 1, d.ECHO_SEED_BASE + 4]
        # evidence is emitted while the last C/B are alive, then C, B, N leave
        assert rig.pos("N", "emit_evidence") < rig.pos("C4", "close") < rig.pos("B4", "stop") < rig.pos("N", "shutdown")
        assert len(json.loads((rig.proc / "pids.json").read_text())) == 11      # N + 5 B + 5 C
        codes = json.loads((rig.proc / "exit_codes.json").read_text())
        assert set(codes) == set(d.expected_process_names()) and all(v["exit_code"] == 0 for v in codes.values())
        assert not d.verify_provenance_files(rig.proc, d.expected_process_names())

    def test_step_timeout_terminates_then_kills_and_records_exit_codes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(d, "LEG_TIMEOUT_S", 1.0)
        rig = FakeRig(tmp_path, modes="hang_fly,ignore_term", term_grace=0.5)
        with pytest.raises(cc.StepTimeout):
            try:
                rig.orch.execute()
            finally:
                down = rig.sup.shutdown_all()
        assert down["survivors"] == []
        pids = [c.pid for c in rig.sup.children]
        assert pids and not any(cc.pid_alive(p) for p in pids)
        codes = json.loads((rig.proc / "exit_codes.json").read_text())
        assert all(v["exit_code"] is not None for v in codes.values())
        assert any(v["stopped_by"] == "kill" for v in codes.values())      # SIGTERM ignored -> SIGKILL

    def test_child_that_dies_mid_step_is_a_child_died_failure(self, tmp_path):
        rig = FakeRig(tmp_path, modes="die_on_reset")
        with pytest.raises(cc.ChildDied) as ei:
            try:
                rig.orch.execute()
            finally:
                down = rig.sup.shutdown_all()
        assert d.classify_exception(ei.value)["kind"] == "child_died"
        assert down["survivors"] == []

    def test_nonzero_child_exit_is_a_failure(self, tmp_path):
        rig = FakeRig(tmp_path, modes="exit_nonzero")
        with pytest.raises(cc.ChildDied, match="exited with 3"):
            try:
                rig.orch.execute()
            finally:
                rig.sup.shutdown_all()

    def test_separation_problem_reported_at_ready_is_a_failure(self, tmp_path):
        rig = FakeRig(tmp_path, modes="bad_provenance")
        with pytest.raises(cc.InfraFailure) as ei:
            try:
                rig.orch.execute()
            finally:
                rig.sup.shutdown_all()
        assert ei.value.kind == "separation"

    def test_child_that_never_becomes_ready_times_out(self, tmp_path):
        rig = FakeRig(tmp_path, modes="no_ready")
        try:
            ch_ = rig.sup.spawn("N", rig.argv_for("N", "N"))
            with pytest.raises(cc.StepTimeout):
                rig.sup.wait_ready(ch_, timeout=1.0)
        finally:
            down = rig.sup.shutdown_all()
        assert down["survivors"] == []

    def test_spawn_failure_is_child_start(self, tmp_path):
        sup = cc.Supervisor(tmp_path / "proc", dict(os.environ), str(tmp_path))
        with pytest.raises(cc.InfraFailure) as ei:
            sup.spawn("N", ["/nonexistent/interpreter"])
        assert ei.value.kind == "child_start"

    def test_sigterm_to_the_controller_cleans_up_children(self, tmp_path):
        script = f"""
import os, sys, time
sys.path.insert(0, {_FIX!r})
import c2ps_common as cc
sup = cc.Supervisor(__import__('pathlib').Path({str(tmp_path / 'proc')!r}), dict(os.environ), {str(tmp_path)!r}, term_grace=1)
for i in range(2):
    sup.spawn('K%d' % i, [sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)'])
cc.install_signal_cleanup(lambda: sup.shutdown_all())
print('READY', flush=True)
time.sleep(600)
"""
        p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        try:
            assert p.stdout.readline().strip() == "READY"
            pids = [e["pid"] for e in json.loads((tmp_path / "proc" / "pids.json").read_text())]
            assert len(pids) == 2 and all(cc.pid_alive(x) for x in pids)
            p.send_signal(signal.SIGTERM)
            assert p.wait(timeout=20) == 128 + signal.SIGTERM
        finally:
            if p.poll() is None:
                p.kill()
        deadline = time.time() + 5
        while time.time() < deadline and any(cc.pid_alive(x) for x in pids):
            time.sleep(0.1)
        assert not any(cc.pid_alive(x) for x in pids)
        codes = json.loads((tmp_path / "proc" / "exit_codes.json").read_text())
        assert set(codes) == {"K0", "K1"} and all(v["exit_code"] is not None for v in codes.values())

    def test_controller_kills_the_run_group_and_every_recorded_child(self, tmp_path, monkeypatch):
        monkeypatch.setattr(d, "COORDINATOR_TERM_GRACE_S", 1.0)
        rd = tmp_path / "run-01"
        proc = rd / "proc"
        proc.mkdir(parents=True)
        # a "coordinator" and a detached "child" (own session), both carrying the run directory
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", str(rd)], start_new_session=True)
        coord = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)", str(rd)], start_new_session=True)
        cc.write_json_atomic(proc / "pids.json", [{"name": "K", "pid": child.pid, "pgid": child.pid, "argv": []}])
        try:
            left = d.kill_run_group(coord, proc, rd)
            assert left == []
            child.wait(timeout=5)
            assert coord.poll() is not None and child.poll() is not None
        finally:
            for x in (child, coord):
                if x.poll() is None:
                    x.kill()

    def test_leftover_scan_finds_an_unrecorded_process(self, tmp_path):
        rd = tmp_path / "run-01"
        proc = rd / "proc"
        proc.mkdir(parents=True)
        stray = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", str(rd)], start_new_session=True)
        try:
            assert d.sweep_children(proc, rd, kill=False) == [stray.pid]
            assert d.sweep_children(proc, rd, kill=True) == []
            stray.wait(timeout=5)
        finally:
            if stray.poll() is None:
                stray.kill()

    def test_coordinator_source_never_constructs_the_heavy_objects(self):
        tree = ast.parse(Path(DRIVER_FILES[0]).read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for banned in ("NativeStub", "Observer", "BridgeGen", "MujocoRemoteBackend", "KinematicBridge",
                       "ReachySDK", "FakeJointService"):
            assert banned not in names, banned


# ---------------------------------------------------------------------------
# 6. N request server (real handlers over an in-process fake stub/observer)
# ---------------------------------------------------------------------------

sas = pytest.importorskip("stage_a_slice", reason="needs the SDK modules importable (import only)")
rb = pytest.importorskip("real_bridge")


class _Lock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeStub:
    def __init__(self):
        self.lock = threading.Lock()
        self.commands = []
        self.states = []


class FakeObserver:
    def __init__(self):
        self.states = []

    def snapshot(self):
        return list(self.states)

    def last_seq(self):
        return self.states[-1]["seq"] if self.states else None

    def last_sim_step(self):
        return self.states[-1]["sim_step"] if self.states else None


def _state_msg(seq, sim_step):
    from make_fixtures import JOINT_ORDER
    return {"type": "state", "seq": seq, "sim_step": sim_step, "sim_time_s": sim_step * 0.002, "cmd_seq": 0,
            "_observed_wall_time_ns": 10 ** 9 + seq,
            "joints": [{"name": n, "uid": i, "position_rad": 0.0, "velocity_rad_s": 0.0, "effort": 0.0,
                        "compliant": False} for i, n in enumerate(JOINT_ORDER)]}


def _cmd(t, seq, val=0.1):
    return {"t": t, "seq": seq, "target": [val] * 21, "compliant": [None] * 21}


@pytest.fixture()
def n_server(tmp_path):
    stub, obs = FakeStub(), FakeObserver()
    session = rb.V3Session(stub, obs)
    session.close = lambda: None
    exit_evt = threading.Event()
    handlers, post = ch.make_n_handlers(session, sas, rb, tmp_path / "ev", tmp_path / "control", exit_evt)
    (tmp_path / "ev" / "e1_server_runs" / sas.RUN_NAME).mkdir(parents=True)
    server = cc.ControlServer(handlers, post_reply=post).start()
    yield session, stub, obs, server, tmp_path, exit_evt
    server.stop()


class TestNServer:
    def test_counts_last_seq_and_sim_step_are_ns_own(self, n_server):
        session, stub, obs, server, *_ = n_server
        assert cc.rpc(server.port, "count_commands") == 0
        stub.commands += [_cmd(1.0, 0), _cmd(1.1, 1)]
        obs.states += [_state_msg(4, 40), _state_msg(5, 50)]
        assert cc.rpc(server.port, "count_commands") == 2
        assert cc.rpc(server.port, "last_seq") == 5 and cc.rpc(server.port, "last_sim_step") == 50

    def test_remote_session_proxy_is_equivalent_to_a_local_stub(self, n_server, monkeypatch):
        """``real_bridge._make_logging_move`` over the proxy records the same MoveCall as over the
        stub itself (the proxy's ``len(stub.commands)`` is the request to N)."""
        from reachy_ai.tasks import rig_motion
        session, stub, obs, server, *_ = n_server

        def fake_sdk_move(arm, pose, seconds):
            stub.commands.extend(_cmd(9.0, 100 + i) for i in range(6))
        monkeypatch.setattr(rig_motion, "sdk_move", fake_sdk_move)
        monkeypatch.setattr(time, "sleep", lambda s: None)
        stub.commands.extend(_cmd(1.0, i) for i in range(3))
        local_calls, remote_calls = [], []
        rb._make_logging_move(session, local_calls)("arm", {"j": 1.0}, 2.0)
        remote = ch.RemoteSession(server.port)
        assert isinstance(len(remote.stub.commands), int)
        rb._make_logging_move(remote, remote_calls)("arm", {"j": 1.0}, 2.0)
        assert (local_calls[0].first_command_index, local_calls[0].last_command_index) == (3, 9)
        assert (remote_calls[0].first_command_index, remote_calls[0].last_command_index) == (9, 15)
        assert remote_calls[0].last_command_index - remote_calls[0].first_command_index == \
            local_calls[0].last_command_index - local_calls[0].first_command_index
        with remote.stub.lock:
            pass
        assert remote.findings == {}

    def test_reset_t_is_taken_in_n_and_merges_with_command_times(self, n_server):
        session, stub, obs, server, tmp, _ = n_server
        run_dir = tmp / "ev" / "e1_server_runs" / sas.RUN_NAME
        # epoch 0: steps 0..30; a command before the reset; the epoch closes at 30 and restarts at 0
        obs.states += [_state_msg(1, 10), _state_msg(2, 20), _state_msg(3, 30)]
        stub.commands.append(_cmd(time.monotonic() - 5.0, 0, 0.1))
        pre = cc.rpc(server.port, "begin_reset")["pre_reset_sim_step"]
        assert pre == 30
        assert session.reset_events == []                       # nothing committed before the ack (as the slice)
        time.sleep(0.01)
        obs.states += [_state_msg(4, 0), _state_msg(5, 10)]      # the sim_step drop = the reset landing
        stub.commands.append(_cmd(time.monotonic(), 1, 0.2))      # a command after reset_t
        cc.rpc(server.port, "flush_run_dir")
        rows0 = [json.loads(l) for l in (run_dir / "commands.jsonl").read_text().splitlines()]
        assert [r["type"] for r in rows0] == ["joint_command", "joint_command"]   # uncommitted: not merged yet
        assert cc.rpc(server.port, "commit_reset", {"reset_gen": 1})["n_reset_events"] == 1
        assert session.reset_events[0][1] == 1 and session.reset_events[0][2] == 30
        cc.rpc(server.port, "flush_run_dir")
        rows = [json.loads(l) for l in (run_dir / "commands.jsonl").read_text().splitlines()]
        assert [r["type"] for r in rows] == ["joint_command", "reset", "joint_command"]
        assert rows[1]["seed"] == 1 and rows[1]["sim_step"] == 30
        assert rows[0]["target_rad"][0] == 0.1 and rows[2]["target_rad"][0] == 0.2
        # reset_t itself is on the stub's clock (time.monotonic), i.e. between the two command times
        t_reset = session.reset_events[0][0]
        assert stub.commands[0]["t"] < t_reset < stub.commands[1]["t"]

    def test_echo_injection_range_and_n_side_injection(self, n_server):
        session, stub, obs, server, *_ = n_server
        # range: HOVER..REST_SHUT segment when both flown, else the whole leg, else none
        moves = [{"first_command_index": 0, "last_command_index": 10}, {"first_command_index": 10, "last_command_index": 20},
                 {"first_command_index": 20, "last_command_index": 30}]
        assert d.echo_injection_range(["A", "HOVER", "REST_SHUT"], moves) == (10, 30)
        assert d.echo_injection_range(["A", "B", "C"], moves) == (0, 30)
        assert d.echo_injection_range(["HOVER"], moves[:1]) == (0, 10)
        assert d.echo_injection_range([], []) is None
        # N-side: the slice's own inject_synthetic_echoes over the stub's log, inside [lo, hi)
        stub.states += [{"t": 0.01 * i, "sim_step": i, "pos": [0.01 * i * (j + 1) for j in range(21)]} for i in range(200)]
        stub.commands += [_cmd(0.5 + 0.01 * i, i, 0.5) for i in range(100)]
        before = [list(c["target"]) for c in stub.commands]
        res = cc.rpc(server.port, "inject_echoes", {"name": "S2-B4-c-r1", "first_command_index": 20,
                                                    "last_command_index": 80, "seed": d.ECHO_SEED_BASE + 1})
        assert res["n_injected"] > 0
        changed = [i for i, c in enumerate(stub.commands) if c["target"] != before[i]]
        assert changed and all(20 + sas.ECHO_MARGIN_TICKS <= i < 80 - sas.ECHO_MARGIN_TICKS for i in changed)

    def test_shutdown_replies_then_signals_exit(self, n_server):
        *_, server, tmp, exit_evt = n_server
        assert cc.rpc(server.port, "shutdown") == {}
        assert exit_evt.wait(2)

    def test_unknown_method_and_dead_port(self, n_server):
        *_, server, tmp, exit_evt = n_server
        with pytest.raises(cc.RpcError, match="unknown method"):
            cc.rpc(server.port, "nope")
        with pytest.raises(cc.ChildDied):
            cc.rpc(1, "ping", timeout=1)

    def test_bridge_log_capture_matches_the_slice_format(self, tmp_path):
        import logging
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        lg = logging.getLogger("mujoco_remote_backend")
        with sas.BridgeLogCapture(a):
            lg.warning("hello %d", 1)
        b.write_text("")
        with ch.AppendLogCapture(b):
            lg.warning("hello %d", 1)
        assert a.read_text() == b.read_text() == "WARNING mujoco_remote_backend: hello 1\n"
        with ch.AppendLogCapture(b):
            lg.warning("again")
        assert b.read_text().count("\n") == 2                    # appended, not truncated

    def test_stub_url_proxy(self):
        assert ch.StubUrl(1234).url() == "ws://127.0.0.1:1234"

    def test_slice_helpers_the_children_rely_on_exist_with_the_used_signatures(self):
        import inspect
        assert list(inspect.signature(sas._flush_partial_run_dir).parameters) == ["session", "run_dir", "reset_events_committed"]
        assert list(inspect.signature(sas.inject_synthetic_echoes).parameters)[:3] == ["session", "first_command_index", "last_command_index"]
        assert list(inspect.signature(rb._fly_or_record).parameters) == ["session", "arm", "route", "key", "move_calls"]
        assert list(inspect.signature(rb.BridgeGen.__init__).parameters) == ["self", "stub"]
        assert sas.RUN_NAME == cc.RUN_NAME and sas.ARM_LABELS == cc.ARM_LABELS
        assert sas.BRIDGE_SHA == {"A": cc.A_SHA, "B": cc.B_SHA}


# ---------------------------------------------------------------------------
# 7. separation and provenance
# ---------------------------------------------------------------------------

def _fake_instance(module, qualname):
    cls = type(qualname, (), {})
    cls.__module__, cls.__qualname__ = module, qualname
    return cls()


class TestSeparation:
    def test_c_holding_a_stub_fails(self):
        assert cc.find_forbidden_instances("C", [_fake_instance("native_stub", "NativeStub")])
        assert cc.find_forbidden_instances("C", [_fake_instance("real_bridge", "Observer")])
        assert cc.find_forbidden_instances("C", [_fake_instance("grpc._server", "_Server")])
        assert cc.find_forbidden_instances("C", [_fake_instance("mujoco_remote_backend", "MujocoRemoteBackend")])
        assert not cc.find_forbidden_instances("C", [_fake_instance("reachy_sdk.reachy_sdk", "ReachySDK")])

    def test_b_holding_an_sdk_client_or_stub_fails(self):
        assert cc.find_forbidden_instances("B", [_fake_instance("reachy_sdk.reachy_sdk", "ReachySDK")])
        assert cc.find_forbidden_instances("B", [_fake_instance("native_stub", "NativeStub")])
        assert not cc.find_forbidden_instances("B", [_fake_instance("grpc._server", "_Server"),
                                                     _fake_instance("mujoco_remote_backend", "KinematicBridge")])

    def test_n_holding_sdk_server_or_backend_fails(self):
        for m, q in (("reachy_sdk.reachy_sdk", "ReachySDK"), ("grpc._server", "_Server"),
                     ("mujoco_remote_backend", "MujocoRemoteBackend"), ("fake_reachy_server", "FakeJointService")):
            assert cc.find_forbidden_instances("N", [_fake_instance(m, q)]), (m, q)
        assert not cc.find_forbidden_instances("N", [_fake_instance("native_stub", "NativeStub"),
                                                     _fake_instance("real_bridge", "Observer")])

    def test_subclass_of_a_forbidden_class_is_caught(self):
        base = _fake_instance("native_stub", "NativeStub").__class__
        sub = type("Sub", (base,), {})
        assert cc.find_forbidden_instances("C", [sub()])

    def test_coordinator_holds_nothing_and_imports_nothing_heavy(self):
        for m, q in (("reachy_sdk.reachy_sdk", "ReachySDK"), ("native_stub", "NativeStub"), ("real_bridge", "Observer"),
                     ("grpc._server", "_Server"), ("mujoco_remote_backend", "KinematicBridge")):
            assert cc.check_separation("coordinator", [_fake_instance(m, q)], module_names=[])
        assert cc.check_separation("coordinator", [], module_names=["os", "grpc._channel"])
        assert cc.check_separation("coordinator", [], module_names=["os", "json"]) == []

    def test_module_imported_from_outside_the_clone_fails(self, tmp_path):
        clone = tmp_path / "clone"
        (clone / "src").mkdir(parents=True)
        inside = clone / "src" / "mujoco_remote_backend.py"
        outside = tmp_path / "primary" / "mujoco_remote_backend.py"
        assert cc.check_module_origins(str(clone), {"mujoco_remote_backend": str(inside)}) == []
        bad = cc.check_module_origins(str(clone), {"mujoco_remote_backend": str(outside)})
        assert bad and "outside the clone" in bad[0]
        assert cc.check_module_origins(str(clone), {"tools.goalfix_cmp.cycle": str(outside)})
        assert cc.check_module_origins(str(clone), {"numpy": "/venv/numpy/__init__.py"}) == []

    def test_collect_provenance_records_the_required_fields(self, tmp_path):
        rec = cc.collect_provenance("coordinator", "coordinator", str(_ROOT), "2026-01-01T00:00:00Z", point="after_setup")
        for k in ("sys_executable", "sys_version", "pid", "ppid", "start_utc", "record_utc", "sys_modules",
                  "product_module_files", "threads", "problems", "ok"):
            assert k in rec
        assert rec["pid"] == os.getpid() and rec["sys_modules"] == sorted(rec["sys_modules"])
        assert isinstance(rec["threads"], list) and "c2ps_common" in rec["product_module_files"]
        # this test process imported the SDK modules (importorskip above), so the COORDINATOR check must flag it
        assert rec["ok"] is False and any("forbidden module" in p for p in rec["problems"])

    def test_a_process_record_with_a_separation_problem_fails_verification(self, tmp_path):
        ok = {"ok": True, "problems": []}
        cc.write_json_atomic(tmp_path / "N.provenance.json", {"records": {"after_setup": ok, "before_exit": ok}})
        cc.write_json_atomic(tmp_path / "B0.provenance.json", {"records": {
            "after_setup": ok, "before_exit": {"ok": False, "problems": ["B holds an instance of reachy_sdk.reachy_sdk.ReachySDK"]}}})
        cc.write_json_atomic(tmp_path / "C0.provenance.json", {"records": {"after_setup": ok}})
        probs = d.verify_provenance_files(tmp_path, ["N", "B0", "C0", "C1"])
        assert any("B0@before_exit" in p for p in probs)
        assert any("C0: provenance point before_exit not recorded" in p for p in probs)
        assert any("C1: missing or malformed" in p for p in probs)
        assert not d.verify_provenance_files(tmp_path, ["N"])


# ---------------------------------------------------------------------------
# 8. C2 figures reproduce the shipped C2 on make_fixtures legs
# ---------------------------------------------------------------------------

import make_fixtures as mf  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402


def _build_run(tmp_path, *, skews=None, cycles=None, names=d.CYCLE_NAMES, arms=d.ARM_LABELS):
    """Synthetic evidence in the slice's layout: one epoch per cycle (reset rows and sim_step
    drops), sidecars naming the route and the run directory, manifests, derived digest."""
    rd = tmp_path / "run-01"
    ev_dir, control = rd / "ev", rd / "control"
    run_dir = ev_dir / "e1_server_runs" / d.RUN_NAME
    control.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    sim = mf.FlightSim(dict(R.HOME), pose_units="deg", restream_passes=1)
    skews = skews or {}
    bounds = []
    for k, name in enumerate(names, start=1):
        sim.pose = dict(R.HOME)
        sim._hold_ticks(1, dict(sim.pose))
        sim.reset(seed=k)
        s0 = sim.state_rows[-1]["seq"]
        sim.skew_s = {j: v / 1000.0 for j, v in skews.get(name, {}).items()}
        sim.fly(R.PLACE_ROUTE)
        s1 = sim.state_rows[-1]["seq"]
        for _ in range(25):
            sim._hold_ticks(1, dict(sim.pose))
        f0 = sim.state_rows[-1]["seq"]
        sim.pose = dict(R.REST)
        sim.skew_s = {}
        sim.fly(R.LIFT_TO_PRESENT)
        f1 = sim.state_rows[-1]["seq"]
        bounds.append((name, k, s0, s1, f0, f1))
    res = sim.result()
    mf.write_evidence(run_dir, res.state_rows, res.command_rows)
    sums = (run_dir / "SHA256SUMS").read_text()
    (run_dir / "derived-SHA256SUMS").write_text(sums)
    (ev_dir / "SHA256SUMS").write_text("".join(
        f"{line.split('  ')[0]}  ./e1_server_runs/{d.RUN_NAME}/{line.split('  ')[1]}\n" for line in sums.splitlines()))
    resolved = str(run_dir.resolve())
    for name, k, s0, s1, f0, f1 in bounds:
        for kind, a, b, route in (("setup", s0, s1, "PLACE_ROUTE"), ("flight", f0, f1, "LIFT_TO_PRESENT")):
            (control / f"route_clearance_{route}_{name}.link.json").write_text(json.dumps({
                "alignment": [{"server_seq": a}, {"server_seq": b}],
                "log": f"route_clearance_{route}_{name}.log", "server_run_dir": resolved}))
        (control / f"cycle_{name}.json").write_text(json.dumps({
            "rep": k, "cycle": name, "setup_sidecar": f"route_clearance_PLACE_ROUTE_{name}.link.json",
            "flight_sidecar": f"route_clearance_LIFT_TO_PRESENT_{name}.link.json"}))
    cyc_list = [{"name": n, "rep": k, "arm": a, "epoch": k, "reset_verify_rc": 0}
                for (n, k, *_), a in zip(bounds, arms)]
    return rd, run_dir, control, cyc_list


@pytest.fixture()
def zero_lag(monkeypatch):
    monkeypatch.setattr(mf, "_LAG_RAD", [0.0] * 8)


class TestC2Figures:
    def test_pass_and_fail_reproduce_the_shipped_c2(self, tmp_path, zero_lag):
        rd, run_dir, control, cycles = _build_run(tmp_path, skews={"S2-B4-c-r2": {"r_wrist_pitch": 90.0}})
        doc = d.analyze_run(rd, str(run_dir), str(control), cycles)
        by = {c["name"]: c for c in doc["cycles"]}
        clean = by["S2-B4-c-r3"]["legs"]["setup"]
        assert clean["status"] == "evaluated" and clean["c2"]["reproduced"] is True
        assert clean["checks"]["C2"]["passed"] is True and clean["c2"]["n_gt_30ms"] == 0
        assert clean["c2"]["figures_status"] == "verified"
        bad = by["S2-B4-c-r2"]["legs"]["setup"]
        assert bad["status"] == "evaluated" and bad["checks"]["C2"]["passed"] is False
        assert bad["c2"]["reproduced"] is True and bad["c2"]["diag_index"] == bad["c2"]["shipped_index"]
        assert bad["c2"]["n_gt_30ms"] >= 1 and bad["c2"]["max_spread_ms"] > 30.0
        assert set(bad["checks"]) == set(d.CHECKS)
        lines = [json.loads(l) for l in (rd / "c2_per_command.jsonl").read_text().splitlines()]
        assert any(l["cycle"] == "S2-B4-c-r2" and l["leg"] == "setup" and l["spread_ms"] > 30 and l["verified"] for l in lines)
        assert (rd / "analysis.json").is_file()
        # all 8 legs evaluated
        assert sum(1 for c in doc["cycles"] for l in c["legs"].values() if l["status"] == "evaluated") == 8

    def test_injected_disagreement_is_unverified_excluded_and_flagged(self, tmp_path, zero_lag, monkeypatch):
        rd, run_dir, control, cycles = _build_run(tmp_path, skews={"S2-B4-c-r2": {"r_wrist_pitch": 90.0}})
        real = d.c2_diagnostic

        def wrong(*a, **k):
            out = real(*a, **k)
            out["passed"] = not out["passed"]
            return out
        monkeypatch.setattr(d, "c2_diagnostic", wrong)
        doc = d.analyze_run(rd, str(run_dir), str(control), cycles)
        leg = {c["name"]: c for c in doc["cycles"]}["S2-B4-c-r2"]["legs"]["setup"]
        assert leg["c2"]["reproduced"] is False and leg["c2"]["figures_status"].startswith("UNVERIFIED")
        rec = {"run": "run-01", "index": 1, "status": "ok", "failure": None, "cycles": [
            {"name": c["name"], "arm": c["arm"], "legs": c["legs"]} for c in doc["cycles"]]}
        sr = d.summarize_run(rec)
        assert sr["per_kind"]["setup"]["c2_verified"] == 0 and sr["per_kind"]["setup"]["c2_unverified"] == 2
        assert sr["per_kind"]["setup"]["legs_gt_30ms"] == 0           # not counted from unverified figures
        assert "run-01/S2-B4-c-r2/setup" in sr["unverified"]
        summary, md = d.build_summary([rec], {"pin": "p"})
        assert "UNVERIFIED figures (excluded from the counts above): run-01/S2-B4-c-r2/setup" in md

    def test_missing_sidecar_makes_both_legs_incomplete(self, tmp_path, zero_lag):
        rd, run_dir, control, cycles = _build_run(tmp_path)
        (control / "route_clearance_PLACE_ROUTE_S2-B4-c-r2.link.json").unlink()
        doc = d.analyze_run(rd, str(run_dir), str(control), cycles)
        c = {x["name"]: x for x in doc["cycles"]}["S2-B4-c-r2"]
        assert c["error"] and all(l["status"] == "incomplete" for l in c["legs"].values())

    def test_tampered_evidence_makes_every_leg_incomplete(self, tmp_path, zero_lag):
        rd, run_dir, control, cycles = _build_run(tmp_path)
        (run_dir / "states.jsonl").write_text((run_dir / "states.jsonl").read_text() + " \n")
        doc = d.analyze_run(rd, str(run_dir), str(control), cycles)
        assert all(l["status"] == "incomplete" for c in doc["cycles"] for l in c["legs"].values())
        assert d.verify_evidence_checksums(rd / "ev", run_dir)

    def test_c2_tolerance_constant_is_the_shipped_one(self):
        from tools.goalfix_cmp import pathcheck as pc
        assert pc.C2_SKEW_TOL_S == 0.030 == d.C2_TOL_S

    def test_process_run_end_to_end_on_a_fake_coordinator(self, tmp_path, zero_lag, monkeypatch):
        """process_run over synthetic evidence: a fake coordinator process (the driver's DRIVER_REL
        is pointed at a tiny script) reports a finished run; the shipped CLIs are replaced by a
        recording fake; the REAL analysis and summary run."""
        out = tmp_path / "out"
        out.mkdir()
        rd = out / "run-01"
        src = tmp_path / "src"
        src.mkdir()
        # build the synthetic evidence exactly where process_run will look, before the fake coordinator "runs"
        fake_driver = tmp_path / "fake_driver.py"
        built = tmp_path / "built"
        built.mkdir()
        b_rd, b_run_dir, b_control, cyc_list = _build_run(built, skews={"S2-B4-c-r3": {"r_wrist_pitch": 90.0}})
        fake_driver.write_text(f"""
import json, os, shutil, sys
from pathlib import Path
rd = Path(sys.argv[sys.argv.index('--run-dir') + 1])
for sub in ('ev', 'control'):
    shutil.copytree({str(b_rd)!r} + '/' + sub, rd / sub, dirs_exist_ok=True)
proc = rd / 'proc'
ok = {{'ok': True, 'problems': []}}
names = ['N'] + ['B%d' % k for k in range(5)] + ['C%d' % k for k in range(5)] + ['coordinator']
for n in names:
    (proc / (n + '.provenance.json')).write_text(json.dumps({{'records': {{'after_setup': ok, 'before_exit': ok}}}}))
(proc / 'exit_codes.json').write_text(json.dumps({{n: {{'exit_code': 0}} for n in names}}))
(proc / 'pids.json').write_text('[]')
run_dir = str(rd / 'ev' / 'e1_server_runs' / {d.RUN_NAME!r})
(proc / 'coordinator_result.json').write_text(json.dumps({{
    'ok': True, 'failure': None, 'run_dir': run_dir, 'control_dir': str(rd / 'control'),
    'arm_map': str(rd / 'control' / 'arm_map.json'), 'native_log': str(rd / 'control' / 'native_session.log'),
    'cycles': {json.dumps(cyc_list)}}}))
""")
        # the synthetic control dir/sidecars record server_run_dir = the BUILD location; rewrite for the run location
        for f in (b_control).glob("*.link.json"):
            doc = json.loads(f.read_text())
            doc["server_run_dir"] = str((rd / "ev" / "e1_server_runs" / d.RUN_NAME).resolve())
            f.write_text(json.dumps(doc))
        monkeypatch.setattr(d, "DRIVER_REL", str(fake_driver))
        calls = []

        def fake_run_proc(argv, cwd, env, stdout, stderr, timeout):
            calls.append(list(argv))
            o = argv[argv.index("--out") + 1]
            Path(o).write_text(json.dumps({"verdict": "stop", "checks": {"compliance": {"ok": False, "detail": "antennas"}}}))
            stdout.parent.mkdir(parents=True, exist_ok=True)
            stdout.write_text("")
            stderr.write_text("")
            return {"argv": list(argv), "rc": 0, "timed_out": False, "seconds": 0.0, "stdout": stdout.name, "stderr": stderr.name}
        monkeypatch.setattr(d, "_run_proc", fake_run_proc)
        rec = d.process_run(1, out, str(src), "f" * 40, time_left=lambda: 900.0)
        assert rec["status"] == "ok", rec["failure"]
        # 4 cycles x (full + validation) + checkpoint
        assert len(calls) == 9
        full = [c for c in calls if "--expected-host-sha" in c]
        val = [c for c in calls if "--validation-mode" in c]
        assert len(full) == 4 and len(val) == 4
        assert all("--expected-host-sha" in c and "f" * 40 in c for c in full)
        assert all("--required-supervisor-programs" in c and "reachy-sdk-server" in c for c in full)
        assert all(d.A_SHA in c and d.B_SHA in c for c in full)
        assert any(c[0:1] == ["checkpoint"] or "checkpoint" in c for c in calls)
        assert [c["arm"] for c in rec["cycles"]] == ["A", "B", "B", "A"]
        assert (rd / "run_manifest.json").is_file() and (rd / "SHA256SUMS").is_file()
        summary, md = d.build_summary([rec], {"pin": "f" * 40})
        setup = summary["totals"]["by_leg_kind"]["setup"]
        assert setup["evaluated"] == 2 and setup["c2_shipped_fail"] == 1 and setup["legs_gt_30ms"] == 1
        assert summary["totals"]["by_leg_kind"]["flight"]["evaluated"] == 2
        assert summary["totals"]["runs_with_any_b_event"] == 1
        # every cycle's full-gate compliance failure is reported as the harness limitation, never as a pass
        assert all(c["full_gate"]["compliance_gate"].startswith("FAILED") for c in summary["runs"][0]["cycles"])

    def test_process_run_coordinator_failure_stops(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        fake_driver = tmp_path / "fake_driver.py"
        fake_driver.write_text("""
import json, sys
from pathlib import Path
rd = Path(sys.argv[sys.argv.index('--run-dir') + 1])
(rd / 'proc' / 'coordinator_result.json').write_text(json.dumps(
    {'ok': False, 'failure': {'kind': 'step_timeout', 'detail': 'leg'}}))
""")
        monkeypatch.setattr(d, "DRIVER_REL", str(fake_driver))
        rec = d.process_run(1, out, str(tmp_path), "f" * 40, time_left=lambda: 900.0)
        assert rec["status"] == "failed" and rec["failure"]["kind"] == "step_timeout"

    def test_process_run_cap_kills_the_group_and_reports_the_cap(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        fake_driver = tmp_path / "fake_driver.py"
        fake_driver.write_text("import time; time.sleep(600)\n")
        monkeypatch.setattr(d, "DRIVER_REL", str(fake_driver))
        monkeypatch.setattr(d, "RUN_CAP_S", 1)
        monkeypatch.setattr(d, "COORDINATOR_TERM_GRACE_S", 1.0)
        rec = d.process_run(1, out, str(tmp_path), "f" * 40, time_left=lambda: 900.0)
        assert rec["status"] == "failed" and rec["failure"]["kind"] == "run_cap"
        assert rec["coordinator"]["timed_out"] is True

    def test_process_run_leftover_process_is_a_failure_and_is_killed(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        fake_driver = tmp_path / "fake_driver.py"
        fake_driver.write_text("""
import json, subprocess, sys
from pathlib import Path
rd = Path(sys.argv[sys.argv.index('--run-dir') + 1])
subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)', str(rd)], start_new_session=True)
(rd / 'proc' / 'coordinator_result.json').write_text(json.dumps({'ok': True, 'failure': None, 'cycles': []}))
""")
        monkeypatch.setattr(d, "DRIVER_REL", str(fake_driver))
        rec = d.process_run(1, out, str(tmp_path), "f" * 40, time_left=lambda: 900.0)
        assert rec["status"] == "failed" and rec["failure"]["kind"] == "leftover_process"
        time.sleep(0.5)
        assert d.cc.ps_leftovers(str(out / "run-01")) == []


# ---------------------------------------------------------------------------
# 9. summary
# ---------------------------------------------------------------------------

def _leg(passed_c2=True, gt=0, reproduced=True, mx=5.0, status="evaluated"):
    checks = {c: {"passed": True, "index": None, "detail": ""} for c in d.CHECKS}
    checks["C2"]["passed"] = passed_c2
    return {"status": status, "checks": checks,
            "c2": {"reproduced": reproduced, "n_gt_30ms": gt, "max_spread_ms": mx}}


def _run_rec(i, *, setup_bad_cycles=(), flight_bad_cycles=(), status="ok"):
    cycles = []
    for name, arm in zip(d.CYCLE_NAMES, d.ARM_LABELS):
        s = _leg(False, 3, mx=44.0) if name in setup_bad_cycles else _leg()
        f = _leg(False, 1, mx=33.0) if name in flight_bad_cycles else _leg()
        if arm == "A":
            s = _leg(False, 9, mx=120.0)              # A legs: injected echoes, C2 expected to fail
        cycles.append({"name": name, "arm": arm, "legs": {"setup": s, "flight": f},
                       "full_gate": {"rc": 1, "verdict": "stop",
                                     "compliance": {"ok": False, "detail": "antennas"}},
                       "validation_mode": {"rc": 0, "verdict": "ok"}})
    return {"run": f"run-{i:02d}", "index": i, "status": status, "failure": None, "cycles": cycles}


_BANNED_ATTRIBUTION = ("within-message", "within message", "fragment", "split", "mixed")
_BANNED_STATS = ("%", "upper bound", "probabilit", "power", "independen", "confidence", "one-sided", "significan")


class TestSummary:
    def test_setup_and_flight_counted_separately_with_denominators_and_a_legs_excluded(self):
        recs = [_run_rec(1, setup_bad_cycles=("S2-B4-c-r2",)), _run_rec(2, flight_bad_cycles=("S2-B4-c-r3",)),
                _run_rec(3)] + [_rec(i, "not_run") for i in range(4, 8)]
        summary, md = d.build_summary(recs, {"pin": "p" * 40})
        t = summary["totals"]
        assert t["by_leg_kind"]["setup"]["evaluated"] == 6 and t["by_leg_kind"]["setup"]["c2_shipped_fail"] == 1
        assert t["by_leg_kind"]["setup"]["legs_gt_30ms"] == 1
        assert t["by_leg_kind"]["flight"]["evaluated"] == 6 and t["by_leg_kind"]["flight"]["c2_shipped_fail"] == 1
        assert t["by_leg_kind"]["flight"]["legs_gt_30ms"] == 1
        assert t["runs_with_any_b_event"] == 2 and t["runs_ok"] == 3 and t["runs_not_run"] == 4
        # A legs (injected echoes; C2 fails there by construction) never enter the counts
        assert t["by_leg_kind"]["setup"]["c2_shipped_fail"] == 1
        a_legs = summary["runs"][0]["figures"]["a_legs"]
        assert len(a_legs) == 4 and all("excluded from C2 counts" in x["note"] for x in a_legs)
        assert "| run-01 | S2-B4-c-r1 | setup | C2 |" in md and "excluded from every C2 count" in md
        assert "| B setup | 6 | 1 of 6 |" in md and "| B flight | 6 | 1 of 6 |" in md
        assert summary["scope"]["partial"] is True and "PARTIAL" in md

    def test_header_carries_the_limitation_verbatim_and_no_compliance_pass(self):
        summary, md = d.build_summary([_run_rec(i) for i in range(1, 8)], {"pin": "p" * 40})
        assert d.NATIVESTUB_LIMITATION in md
        assert "path-only (validation-mode) success is not full-gate success" in md
        assert md.count("FAILED: antennas") == 28 and "WITHHELD" not in md
        assert "compliance check | validation-mode" in md
        assert summary["scope"]["partial"] is False and "PARTIAL" not in md

    def test_unexpected_full_gate_compliance_pass_is_withheld(self):
        got = d.annotate_full_gate({"rc": 0, "verdict": "ok", "compliance": {"ok": True}})
        assert got["compliance_gate"].startswith("WITHHELD") and "UNEXPECTED" in got["annotation"]
        assert d.annotate_full_gate({"rc": 1, "verdict": "stop", "compliance": {"ok": False, "detail": "x"}})[
            "compliance_gate"].startswith("FAILED")
        assert d.annotate_full_gate(None)["compliance_gate"] == "no output"
        assert d.annotate_full_gate({"rc": 1, "verdict": "stop"})["compliance_gate"] == "not reported by the CLI"

    def test_wording_rules(self):
        recs = [_run_rec(1, setup_bad_cycles=("S2-B4-c-r2",))] + [_run_rec(i) for i in range(2, 8)]
        _, md = d.build_summary(recs, {"pin": "p" * 40})
        low = md.lower()
        # reference figures, descriptively
        assert "5 of 14 b setup legs and 0 of 14 b flight legs" in low and "4 of 7 runs" in low
        # required limitation sentences
        for s in d.LIMITATION_SENTENCES:
            assert s in md
        assert "would not prove that shared-process contention caused the earlier failures" in md
        # no statistics vocabulary
        stripped = md.replace(d.LIMITATION_SENTENCES[1], "")
        for w in _BANNED_STATS:
            assert w not in stripped.lower(), w
        import re
        assert not re.search(r"\brate\b", stripped.lower())
        # no attribution vocabulary outside a sentence saying attribution is unavailable
        for sentence in re.split(r"(?<=[.!?])\s+|\n", md):
            if any(w in sentence.lower() for w in _BANNED_ATTRIBUTION):
                assert "unavailable" in sentence.lower(), sentence
        assert d.ATTRIBUTION_SENTENCE in md
        assert "D1" in md and "NOT authorized" in md

    def test_incomplete_runs_are_listed_separately_and_make_the_summary_partial(self):
        rec = _run_rec(1)
        rec["cycles"][1]["legs"]["setup"] = {"status": "incomplete", "error": "x"}
        summary, md = d.build_summary([rec] + [_rec(i, "not_run") for i in range(2, 8)], {"pin": "p"})
        assert "run-01/S2-B4-c-r2/setup" in summary["incomplete_legs"]
        assert "incomplete (not counted as zero): run-01/S2-B4-c-r2/setup" in md
        assert summary["totals"]["by_leg_kind"]["setup"]["evaluated"] == 1

    def test_failed_run_reports_its_stop_reason_in_a_partial_summary(self):
        recs = [_run_rec(1), {**_rec(2, "failed"), "failure": {"kind": "step_timeout", "detail": "leg 300 s"}}] + \
               [_rec(i, "not_run") for i in range(3, 8)]
        summary, md = d.build_summary(recs, {"pin": "p"})
        assert "step_timeout: leg 300 s" in md and summary["scope"]["partial"] is True
        assert summary["totals"]["runs_failed"] == 1 and summary["totals"]["runs_not_run"] == 5


# ---------------------------------------------------------------------------
# 10. static checks on the driver files
# ---------------------------------------------------------------------------

class TestStatic:
    @pytest.mark.parametrize("path", DRIVER_FILES + [FAKE_CHILD])
    def test_no_capture_import_no_report_import_no_patching(self, path):
        tree = ast.parse(Path(path).read_text())
        mods = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                mods.append(n.module or "")
        for m in mods:
            assert m.split(".")[0] != "capture", m
            assert not m.startswith("tools.goalfix_cmp_report"), m
            assert "mock" not in m and m != "pytest", m
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                fn = n.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                assert name not in ("setattr", "patch", "patch_object"), (path, name)
            if isinstance(n, (ast.Assign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                        assert t.value.id not in ("grpc", "reachy_sdk", "fake_reachy_server", "mujoco_remote_backend",
                                                  "kinematic_backend", "native_stub", "rb", "sas", "subprocess"), \
                            (path, t.value.id, t.attr)

    def test_no_pr146_or_capture_names_anywhere_in_code(self):
        for path in DRIVER_FILES:
            code = Path(path).read_text()
            assert "import capture" not in code and "from capture" not in code
            assert "goalfix_cmp_report" not in "\n".join(
                l for l in code.splitlines() if l.lstrip().startswith(("import ", "from ")))

    def test_no_reachy2_imports(self):
        for path in DRIVER_FILES:
            tree = ast.parse(Path(path).read_text())
            for n in ast.walk(tree):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
                    assert not any(x.startswith(("reachy2_sdk", "reachy2_core")) for x in names)

    def test_children_import_only_the_documented_product_modules(self):
        src = Path(DRIVER_FILES[1]).read_text()
        tree = ast.parse(src)
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                imported.add(n.module or "")
        assert {"real_bridge", "stage_a_slice", "reachy_sdk", "reachy_ai.motion"} <= imported
        for banned in ("native_stub", "mujoco_remote_backend", "fake_reachy_server", "kinematic_backend", "grpc", "websockets"):
            assert banned not in imported, banned
