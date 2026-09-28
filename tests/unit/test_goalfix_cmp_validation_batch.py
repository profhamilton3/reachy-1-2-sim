"""Unit tests of the validation batch driver's pure parts and its ``--dry-run``
(tests/fixtures/goalfix_cmp/{validation_batch.py,run_validation_batch.sh}).

Nothing here runs the batch, the Stage A slice, the real bridge or any SDK code;
the driver is stdlib-only at import.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_FIX = os.path.join(_ROOT, "tests", "fixtures", "goalfix_cmp")
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "native_mujoco"), _FIX):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import validation_batch as vb  # noqa: E402

SCRIPT = os.path.join(_FIX, "run_validation_batch.sh")


# ---------------------------------------------------------------------------
# Bound arithmetic
# ---------------------------------------------------------------------------

def test_one_sided_upper_bounds_match_the_brief():
    assert vb.one_sided_upper_bound(28) == pytest.approx(1 - 0.05 ** (1 / 28))
    assert round(vb.one_sided_upper_bound(28) * 100, 1) == 10.1
    assert round(vb.one_sided_upper_bound(7) * 100, 1) == 34.8
    assert vb.one_sided_upper_bound(1) == pytest.approx(0.95)
    with pytest.raises(ValueError):
        vb.one_sided_upper_bound(0)


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------

def _facts(**kw):
    f = {"disk_free_ok": True, "harness_exit_status": 0, "harness_timed_out": False,
         "harness_exception": None, "capture_ok": True, "reset_verify_rcs": [0, 0, 0, 0],
         "missing_artifacts": []}
    f.update(kw)
    return f


def test_a_c2_failure_or_any_result_never_stops_the_batch():
    assert vb.classify_failure(_facts()) is None
    # result-like keys are never consulted
    for k in ("c2_failed", "c0_failed", "verdict_stop", "max_spread_ms", "n_events", "full_gate_rc"):
        assert vb.classify_failure(_facts(**{k: True})) is None
        assert vb.classify_failure(_facts(**{k: 999})) is None


def test_capture_mismatch_and_infrastructure_failures_do_stop():
    f = vb.classify_failure(_facts(capture_ok=False, capture_detail="gen 1: seq mismatch"))
    assert f["kind"] == "capture_crosscheck" and "seq mismatch" in f["detail"]
    # harness exit 3 == the child's capture-failure code, not a generic harness failure
    assert vb.classify_failure(_facts(capture_ok=False, harness_exit_status=3))["kind"] == "capture_crosscheck"
    assert vb.classify_failure(_facts(harness_exit_status=1, harness_exception="boom"))["kind"] == "harness_exception"
    assert vb.classify_failure(_facts(harness_exit_status=7))["kind"] == "harness_exit"
    assert vb.classify_failure(_facts(harness_exit_status=None, harness_timed_out=True))["kind"] == "harness_timeout"
    assert vb.classify_failure(_facts(reset_verify_rcs=[0, 2, 0, 0]))["kind"] == "reset_verify"
    assert vb.classify_failure(_facts(missing_artifacts=["analysis.json"]))["kind"] == "missing_artifact"
    assert vb.classify_failure(_facts(disk_free_ok=False))["kind"] == "disk_low"


def _rec(i, status="ok", *, c2_fail=False, mixed=0, capture_ok=True, failure=None, arm_fail_c0=False):
    def legs(fail):
        base = {c: {"passed": True, "index": None, "detail": ""} for c in
                ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")}
        if fail:
            base["C2"] = {"passed": False, "index": 12, "detail": "per-joint tau spread 84.3 ms > 30 ms"}
        return base

    cycles = []
    for k, arm in enumerate(vb.ARM_LABELS, start=1):
        fail = c2_fail and arm == "B" and k == 2
        cycles.append({
            "name": f"S2-B4-c-r{k}", "rep": k, "arm": arm, "epoch": k, "reset_verify_rc": 0,
            "full_gate": {"rc": 2, "verdict": "stop",
                          "compliance": {"ok": False, "detail": "compliance mismatch on ['l_antenna']"}},
            "validation_mode": {"rc": 3, "verdict": "stop" if fail else "ok"},
            "legs": {"setup": legs(fail), "flight": legs(False)},
            "c2": {"setup": {"reproduced": True, "max_spread_ms": 84.3 if fail else 4.0,
                             "classes": {"mixed_batch": 1 if fail else 0, "split_involved_not_mixed": 0,
                                         "neither": 0, "unjoined": 0}},
                   "flight": {"reproduced": True, "max_spread_ms": 3.0,
                              "classes": {"mixed_batch": 0, "split_involved_not_mixed": 0,
                                          "neither": 0, "unjoined": 0}}}})
    return {"run": f"run-{i:02d}", "index": i, "status": status, "failure": failure,
            "harness": {"exit_status": 0}, "cycles": cycles if status != "not_run" else [],
            "capture": {"ok": capture_ok, "n_batches": 900, "n_builds": 1000, "n_split_batches": 2,
                        "n_mixed": mixed, "n_split_involved_not_mixed": 1}}


def test_run_loop_a_c2_failure_run_does_not_stop_and_all_seven_run():
    calls = []

    def process(i):
        calls.append(i)
        return _rec(i, c2_fail=(i == 3))          # run 3 has a failing C2 on a B leg
    recs = vb.run_loop(process)
    assert calls == list(range(1, 8)) and [r["status"] for r in recs] == ["ok"] * 7


def test_run_loop_capture_failure_stops_records_rest_not_run_no_replacement():
    calls = []

    def process(i):
        calls.append(i)
        if i == 3:
            return _rec(i, "failed", capture_ok=False,
                        failure={"kind": "capture_crosscheck", "detail": "row mismatch"})
        return _rec(i)
    recs = vb.run_loop(process)
    assert calls == [1, 2, 3]                                   # 4..7 never executed, none re-run
    assert [r["status"] for r in recs] == ["ok", "ok", "failed", "not_run", "not_run", "not_run", "not_run"]
    assert len(recs) == 7 and [r["index"] for r in recs] == list(range(1, 8))


def test_run_loop_disk_floor_stops_before_the_run():
    state = {"n": 0}

    def disk_ok():
        state["n"] += 1
        return state["n"] <= 2                                  # third check fails
    calls = []
    recs = vb.run_loop(lambda i: (calls.append(i), _rec(i))[1], disk_ok=disk_ok)
    assert calls == [1, 2]
    assert recs[2]["status"] == "not_run" and recs[2]["failure"]["kind"] == "disk_low"
    assert all(r["status"] == "not_run" for r in recs[2:])


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def test_summary_over_seven_ok_runs_is_complete_with_zero_event_bounds():
    recs = [_rec(i) for i in range(1, 8)]
    summ, md = vb.build_summary(recs, {"pin": "p" * 40})
    assert summ["scope"]["partial"] is False
    t = summ["totals"]
    assert t["runs_ok"] == 7 and t["b_leg_evaluations"] == 28 and t["fails"]["C2"] == 0
    assert summ["bounds"]["C2"]["one_sided_95pct_upper_bound"] == pytest.approx(1 - 0.05 ** (1 / 28))
    assert summ["bounds"]["C2"]["assumption"].startswith("independent legs (NOT established)")
    assert summ["bounds"]["per_run_any_event"]["n"] == 7
    assert summ["bounds"]["per_run_any_event"]["one_sided_95pct_upper_bound"] == pytest.approx(1 - 0.05 ** (1 / 7))
    assert "10.1%" in md and "34.8%" in md and "NOT established" in md
    assert "PARTIAL" not in md


def test_summary_header_states_the_nativestub_limitation_and_never_a_compliance_pass():
    summ, md = vb.build_summary([_rec(i) for i in range(1, 8)], {"pin": "p" * 40})
    assert "NativeStub compliance limitation" in md
    assert "documented HARNESS LIMITATION" in md and "no passing native-compliance result" in md
    assert "NOT full-gate success" in md
    for run in summ["runs"]:
        for c in run["cycles"]:
            assert c["full_gate"]["compliance_gate"].startswith("FAILED")
            assert "documented harness limitation" in c["full_gate"]["annotation"]
            assert "NOT full-gate success" in c["validation_mode_path_only"]["note"]


def test_an_unexpected_compliance_pass_is_withheld_never_reported_as_a_pass():
    a = vb.annotate_full_gate({"rc": 0, "verdict": "ok", "compliance": {"ok": True, "detail": ""}})
    assert a["compliance_gate"].startswith("WITHHELD") and "UNEXPECTED" in a["annotation"]
    assert "PASS" not in a["compliance_gate"].upper().replace("UNEXPECTED PASS", "")
    assert vb.annotate_full_gate(None)["compliance_gate"] == "no output"


def test_summary_counts_events_and_gives_no_bound_where_events_were_observed():
    recs = [_rec(i, c2_fail=(i == 4), mixed=(1 if i == 4 else 0)) for i in range(1, 8)]
    summ, md = vb.build_summary(recs, {"pin": "p" * 40})
    t = summ["totals"]
    assert t["fails"]["C2"] == 1 and t["fails"]["C0"] == 0
    assert t["classes_gt30"]["mixed_batch"] == 1
    assert t["mixed_commands"] == 1 and t["runs_with_any_event"] == 1
    assert summ["bounds"]["C2"]["one_sided_95pct_upper_bound"] is None
    assert summ["bounds"]["C0"]["one_sided_95pct_upper_bound"] is not None
    assert summ["bounds"]["per_run_any_event"]["one_sided_95pct_upper_bound"] is None
    run4 = summ["runs"][3]
    assert run4["b_legs"]["max_c2_spread_ms"] == 84.3
    assert "84.3" in md


def test_partial_summary_after_a_capture_failure_no_replacement():
    recs = vb.run_loop(lambda i: _rec(i, "failed", capture_ok=False,
                                       failure={"kind": "capture_crosscheck", "detail": "row mismatch"})
                       if i == 2 else _rec(i))
    summ, md = vb.build_summary(recs, {"pin": "p" * 40})
    assert summ["scope"]["partial"] is True and md.startswith("# Validation batch summary")
    assert "PARTIAL" in md and "Nothing was replaced or extended" in md
    assert [r["status"] for r in summ["runs"]] == ["ok", "failed"] + ["not_run"] * 5
    assert summ["totals"]["runs_failed"] == 1 and summ["totals"]["runs_not_run"] == 5
    assert summ["totals"]["b_leg_evaluations"] == 8            # only executed runs counted
    assert "capture_crosscheck" in md and "row mismatch" in md
    assert summ["bounds"]["C2"]["n"] == 8                      # n is the executed n, not 28


def test_summary_with_a_failed_harness_run_and_no_cycles_is_still_built():
    rec = {"run": "run-01", "index": 1, "status": "failed",
           "failure": {"kind": "harness_exception", "detail": "RuntimeError: x"},
           "harness": {"exit_status": 1}, "cycles": [], "capture": {}}
    summ, md = vb.build_summary([rec], {"pin": "p" * 40})
    assert summ["runs"][0]["b_legs"]["b_leg_evaluations"] == 0
    assert summ["bounds"]["C2"]["one_sided_95pct_upper_bound"] is None      # no evaluation, no bound
    assert "harness_exception" in md


# ---------------------------------------------------------------------------
# C2 diagnostic against the shipped check
# ---------------------------------------------------------------------------

def _route_and_start():
    from reachy_ai.motion.rig_routes import R_JOINTS
    from tools.goalfix_cmp import segments as seg

    class WP:
        def __init__(self, name, pose, seconds):
            self.name, self.pose, self.seconds = name, pose, seconds

    zero = {j: 0.0 for j in R_JOINTS}
    goal = dict(zero, r_shoulder_pitch=0.5, r_elbow_pitch=-0.8)
    return [WP("G0", goal, 2.0)], dict(zero), R_JOINTS, seg


def _mj(a, b, tau):
    from tools.goalfix_cmp._minjerk import s_of_tau
    return a + s_of_tau(tau) * (b - a)


def test_c2_diagnostic_reproduces_shipped_c2_on_a_split_like_spread():
    from tools.goalfix_cmp import pathcheck as pc
    from tools.goalfix_cmp import segments as seg
    route, start, R_JOINTS, _ = _route_and_start()
    def cmd(t_sp, t_el):
        d = dict(start)
        d["r_shoulder_pitch"] = _mj(0.0, 0.5, t_sp)
        d["r_elbow_pitch"] = _mj(0.0, -0.8, t_el)
        return d
    # cmd 0: equal tau (clean); cmd 1: elbow lags by 0.025 tau * 2 s = 50 ms
    targets8 = [cmd(0.5, 0.5), cmd(0.55, 0.575), cmd(0.7, 0.7)]
    assignment = seg.GoalAssignment(goal_index=[0, 0, 0], ok=True)
    diag = vb.c2_diagnostic(targets8, start, route, assignment, None)
    shipped, _c3 = pc.check_c2_c3(targets8, [0.0] * 3, start, route, assignment)
    assert shipped.passed is False and diag["passed"] is False
    assert diag["first_fail"] == shipped.first_violation_index == 1
    recs = {r["local_index"]: r for r in diag["plain"]["records"]}
    assert recs[0]["spread_ms"] == pytest.approx(0.0, abs=1e-6)
    assert recs[1]["spread_ms"] == pytest.approx(50.0, abs=1e-3)
    assert recs[1]["n_tau"] == 2 and set(recs[1]["taus"]) == {"r_shoulder_pitch", "r_elbow_pitch"}
    assert diag["withdrawn"] is None


def test_c2_diagnostic_passes_when_the_shipped_check_passes():
    from tools.goalfix_cmp import pathcheck as pc
    from tools.goalfix_cmp import segments as seg
    route, start, R_JOINTS, _ = _route_and_start()
    targets8 = []
    for t in (0.3, 0.5, 0.8):
        d = dict(start)
        d["r_shoulder_pitch"] = _mj(0.0, 0.5, t)
        d["r_elbow_pitch"] = _mj(0.0, -0.8, t + 0.005)      # 10 ms apart: within tolerance
        targets8.append(d)
    assignment = seg.GoalAssignment(goal_index=[0, 0, 0], ok=True)
    diag = vb.c2_diagnostic(targets8, start, route, assignment, None)
    shipped, _ = pc.check_c2_c3(targets8, [0.0] * 3, start, route, assignment)
    assert shipped.passed and diag["passed"] and diag["first_fail"] is None


def test_c2_diagnostic_applies_rcarry_withdrawal_like_the_shipped_recheck():
    from tools.goalfix_cmp import pathcheck as pc
    from tools.goalfix_cmp import segments as seg
    route, start, R_JOINTS, _ = _route_and_start()
    d0 = dict(start)
    d0["r_shoulder_pitch"] = _mj(0.0, 0.5, 0.5)
    d0["r_elbow_pitch"] = _mj(0.0, -0.8, 0.5)
    d1 = dict(d0)                                            # elbow: bit-exact carry of cmd 0
    d1["r_shoulder_pitch"] = _mj(0.0, 0.5, 0.7)
    targets8 = [d0, d1]
    assignment = seg.GoalAssignment(goal_index=[0, 0], ok=True)
    rows = [{j: False for j in R_JOINTS}, {j: (j == "r_elbow_pitch") for j in R_JOINTS}]
    shipped_plain, _ = pc.check_c2_c3(targets8, [0.0] * 2, start, route, assignment)
    shipped_wd, _ = pc.check_c2_c3(targets8, [0.0] * 2, start, route, assignment, carry_withdraw=rows)
    assert shipped_plain.passed and not shipped_wd.passed and shipped_wd.first_violation_index == 1
    diag = vb.c2_diagnostic(targets8, start, route, assignment, rows)
    assert diag["passed"] is False and diag["first_fail"] == 1 and diag["detail"].startswith("R-carry′")
    # the plain pass had a single tau-bearing joint at cmd 1 (the carry is excluded)
    assert {r["local_index"]: r["n_tau"] for r in diag["plain"]["records"]}[1] == 1
    assert {r["local_index"]: r["n_tau"] for r in diag["withdrawn"]["records"]}[1] == 2


# ---------------------------------------------------------------------------
# Joining commands > 30 ms to capture classes
# ---------------------------------------------------------------------------

def test_join_commands_to_capture_classes_and_gaps():
    kinds = ["reset", "joint_command", "joint_command", "joint_command", "joint_command"]
    joined = {(0, 1): 0, (0, 2): 1, (0, 3): 2, (1, 1): 3}       # (gen, seq) -> jc ordinal
    classification = {
        "builds": [
            {"gen": 0, "seq": 1, "class": "neither", "source_batches": [1], "source_span_ns": None},
            {"gen": 0, "seq": 2, "class": "mixed_batch", "source_batches": [1, 2], "source_span_ns": 5_000_000},
            {"gen": 0, "seq": 3, "class": "split_involved_not_mixed", "source_batches": [2], "source_span_ns": None},
            {"gen": 1, "seq": 1, "class": "neither", "source_batches": [3], "source_span_ns": None}],
        "batch_gaps_ns": [{"gen": 0, "batch_id": 2, "prev_batch_id": 1, "gap_ns": 18_000_000}]}
    # global indices: row 1 -> ordinal 0, row 2 -> 1, row 3 -> 2, row 4 -> 3 ; 99 is unknown
    out = vb.join_commands_to_capture([2, 3, 4, 99], kinds, joined, classification)
    assert [o["class"] for o in out] == ["mixed_batch", "split_involved_not_mixed", "neither", "unjoined"]
    assert out[0]["source_span_ns"] == 5_000_000 and out[0]["newest_source_batch_gap_ns"] == 18_000_000
    assert out[3]["gen"] is None


# ---------------------------------------------------------------------------
# Path / runs / pin checks
# ---------------------------------------------------------------------------

def test_check_out_path_refusals(tmp_path):
    ok = tmp_path / "fresh"
    assert vb.check_out_path(str(ok), forbidden=()) == str(ok)
    with pytest.raises(vb.Refusal, match="outside /tmp"):
        vb.check_out_path("/tmp/goalfix-cmp-harness-validation-x")
    with pytest.raises(vb.Refusal, match="outside /tmp"):
        vb.check_out_path("/private/tmp/goalfix-cmp-harness-validation-x")
    with pytest.raises(vb.Refusal, match="already exists"):
        vb.check_out_path(str(tmp_path), forbidden=())
    with pytest.raises(vb.Refusal, match="absolute"):
        vb.check_out_path("relative/dir")
    with pytest.raises(vb.Refusal, match="parent directory"):
        vb.check_out_path(str(tmp_path / "no" / "such" / "dir"), forbidden=())


def test_check_runs_and_pin_format():
    assert vb.check_runs("7") == 7 and vb.check_runs(7) == 7
    for bad in ("8", 6, "seven", None, 0):
        with pytest.raises(vb.Refusal):
            vb.check_runs(bad)
    assert vb.check_pin_format("a" * 40) == "a" * 40
    for bad in ("abc", "A" * 40, "g" * 40, "a" * 39):
        with pytest.raises(vb.Refusal):
            vb.check_pin_format(bad)


def test_plan_lines_states_the_fixed_scope():
    txt = "\n".join(vb.plan_lines("p" * 40, "/x/out"))
    for i in range(1, 8):
        assert f"run-{i:02d}:" in txt
    assert "run-08" not in txt
    assert "total B leg-evaluations = 28" in txt and "no replacement, no extension" in txt
    assert "harness (subprocess)" in txt and "neck/seed extractor" in txt


def test_write_sha256sums_covers_every_file_but_itself_and_excluded_tops(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_text("x")
    (tmp_path / "SUMMARY.md").write_text("s")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "big").write_text("no")
    vb._write_sha256sums(tmp_path, exclude_top=("src", "tmp"))
    lines = (tmp_path / "SHA256SUMS").read_text().splitlines()
    names = sorted(l.split("  ", 1)[1] for l in lines)
    assert names == ["./SUMMARY.md", "./a/x.txt"]
    assert lines[0].split("  ")[0] == vb.sha256_bytes(b"s") or lines[1].split("  ")[0] == vb.sha256_bytes(b"s")


# ---------------------------------------------------------------------------
# Pin / tree / blob equalities on a throwaway git repo
# ---------------------------------------------------------------------------

def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _mini_repo(tmp_path):
    repo = tmp_path / "mini"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "tools" / "goalfix_cmp").mkdir(parents=True)
    (repo / "tools" / "goalfix_cmp" / "x.py").write_text("x = 1\n")
    for f in vb.OPT_FILES:
        (repo / f).write_text(f"# {f}\n")
    (repo / "native_mujoco" / "model").mkdir(parents=True)
    (repo / vb.KEYFRAME_REL).write_text("<mujoco/>\n")
    (repo / vb.DRIVER_REL).parent.mkdir(parents=True)
    (repo / vb.DRIVER_REL).write_text("# driver\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _commit(repo, msg="c"):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def test_source_pin_facts_pass_and_each_mismatch_refuses(tmp_path, monkeypatch):
    repo, base = _mini_repo(tmp_path)
    monkeypatch.setattr(vb, "M_SHA", base)
    monkeypatch.setattr(vb, "B_SHA", base)
    p_ok = _commit(repo, "same trees")
    facts = vb.source_pin_facts(repo, p_ok)
    assert facts["P_tools_goalfix_cmp_tree"] == facts["M_tools_goalfix_cmp_tree"]
    assert set(facts["opt_file_sha256"]) == set(vb.OPT_FILES)

    (repo / "tools" / "goalfix_cmp" / "x.py").write_text("x = 2\n")
    p_tools = _commit(repo, "tools differ")
    with pytest.raises(vb.Refusal, match="tools/goalfix_cmp tree differs"):
        vb.source_pin_facts(repo, p_tools)
    _git(repo, "checkout", "-q", p_ok, "--", "tools")

    (repo / "fake_reachy_server.py").write_text("# changed\n")
    p_blob = _commit(repo, "blob differs")
    with pytest.raises(vb.Refusal, match="fake_reachy_server.py at P differs"):
        vb.source_pin_facts(repo, p_blob)
    _git(repo, "checkout", "-q", p_ok, "--", "fake_reachy_server.py")

    (repo / vb.KEYFRAME_REL).write_text("<mujoco>x</mujoco>\n")
    p_kf = _commit(repo, "keyframe differs")
    with pytest.raises(vb.Refusal, match="differs from M's"):
        vb.source_pin_facts(repo, p_kf)
    _git(repo, "checkout", "-q", p_ok, "--", "native_mujoco")

    _git(repo, "rm", "-q", vb.DRIVER_REL)
    p_nodriver = _commit(repo, "no driver")
    with pytest.raises(vb.Refusal, match="not present at P"):
        vb.source_pin_facts(repo, p_nodriver)

    with pytest.raises(vb.Refusal, match="not present"):
        vb.source_pin_facts(repo, "0" * 40)


def test_verify_source_refuses_dirty_tree_and_wrong_head(tmp_path, monkeypatch):
    repo, base = _mini_repo(tmp_path)
    monkeypatch.setattr(vb, "M_SHA", base)
    monkeypatch.setattr(vb, "B_SHA", base)
    assert vb.verify_source(str(repo), base)["worktree_clean"] is True
    p2 = _commit(repo)
    with pytest.raises(vb.Refusal, match="HEAD"):
        vb.verify_source(str(repo), base)
    (repo / "stray.txt").write_text("x")
    with pytest.raises(vb.Refusal, match="dirty"):
        vb.verify_source(str(repo), p2)


# ---------------------------------------------------------------------------
# --dry-run through the shell entry point
# ---------------------------------------------------------------------------

def _have_pins():
    for rev in (vb.M_SHA, vb.B_SHA):
        if subprocess.run(["git", "-C", _ROOT, "cat-file", "-e", f"{rev}^{{commit}}"],
                          capture_output=True).returncode != 0:
            return False
    return True


def _fake_venv(tmp_path):
    v = tmp_path / "venv"
    if not (v / "bin" / "python").exists():
        (v / "bin").mkdir(parents=True)
        os.symlink(sys.executable, v / "bin" / "python")
    return v


def _head():
    return subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def _dry(tmp_path, *extra, out=None, pin=None, runs="7", venv=None):
    out = out if out is not None else str(tmp_path / "goalfix-cmp-harness-validation-test")
    argv = ["bash", SCRIPT, "--pin", pin or _head(), "--out", out, "--venv", str(venv or _fake_venv(tmp_path)),
            "--runs", runs, "--repo", _ROOT, "--dry-run", *extra]
    return subprocess.run(argv, capture_output=True, text=True, cwd=str(tmp_path))


def test_script_is_executable_and_uses_the_required_safety_options():
    assert os.access(SCRIPT, os.X_OK)
    text = Path(SCRIPT).read_text()
    assert "set -euo pipefail" in text
    assert "git clone --local --no-checkout" in text and "checkout --detach" in text
    assert "worktree" not in text.replace("not a worktree", "")
    assert 'export TMPDIR="$OUT/tmp"' in text and "export REQUIRE_REACHY_SDK=1" in text


@pytest.mark.skipif(not _have_pins(), reason="M/B objects not present in this clone")
def test_dry_run_succeeds_prints_plan_and_touches_nothing(tmp_path):
    if os.path.realpath(str(tmp_path)).startswith(("/tmp", "/private/tmp")):
        pytest.skip("pytest tmp_path is under /tmp on this platform, which the script refuses by design")
    out = tmp_path / "goalfix-cmp-harness-validation-test"
    r = _dry(tmp_path, out=str(out))
    assert r.returncode == 0, r.stderr
    assert "PREFLIGHT OK" in r.stdout and "DRY RUN: nothing was cloned" in r.stdout
    for i in range(1, 8):
        assert f"run-{i:02d}:" in r.stdout
    assert "total B leg-evaluations = 28" in r.stdout
    assert not out.exists()                                     # nothing created


@pytest.mark.skipif(not _have_pins(), reason="M/B objects not present in this clone")
def test_dry_run_refuses_tmp_out_existing_out_bad_runs_and_tree_mismatch(tmp_path):
    r = _dry(tmp_path, out="/tmp/goalfix-cmp-harness-validation-refuse-me")
    assert r.returncode == 2 and "outside /tmp" in r.stderr
    r = _dry(tmp_path, out="/private/tmp/goalfix-cmp-harness-validation-refuse-me")
    assert r.returncode == 2 and "outside /tmp" in r.stderr
    existing = tmp_path / "already"
    existing.mkdir()
    r = _dry(tmp_path, out=str(existing))
    assert r.returncode == 2 and "already exists" in r.stderr
    r = _dry(tmp_path, runs="8")
    assert r.returncode == 2 and "exactly 7" in r.stderr
    r = _dry(tmp_path, pin="abc")
    assert r.returncode == 2 and "40-hex" in r.stderr
    r = _dry(tmp_path, venv=tmp_path / "no-venv")
    assert r.returncode == 2 and "bin/python" in r.stderr
    # tree mismatch: M's own first parent predates tools/goalfix_cmp
    m_parent = subprocess.run(["git", "-C", _ROOT, "rev-parse", f"{vb.M_SHA}^1"],
                              capture_output=True, text=True).stdout.strip()
    r = _dry(tmp_path, pin=m_parent)
    assert r.returncode == 2 and "tools/goalfix_cmp" in r.stderr
    # none of the refusals created anything
    assert not (tmp_path / "goalfix-cmp-harness-validation-test").exists()


def test_preflight_imports_no_sdk_no_numpy():
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {_FIX!r})
        import validation_batch as vb
        try:
            vb.preflight("a"*40, "/tmp/x", {_ROOT!r}, 7, "/nonexistent")
        except vb.Refusal:
            pass
        bad = [m for m in ("reachy_sdk", "grpc", "websockets", "numpy", "scipy", "stage_a_slice",
                           "real_bridge", "capture") if m in sys.modules]
        print("BAD", bad)
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "BAD []" in r.stdout
