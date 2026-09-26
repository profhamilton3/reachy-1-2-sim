"""Stage A vertical-slice regeneration test (assignment
``outputs/assignment-2026-09-25-sonnet-pr144-slice-then-h1-h10.md``, Stage A).

Drives ``tests/fixtures/goalfix_cmp/stage_a_slice.py`` end to end: 4 real
V3-harness cycles (A B B A), plan-§6-shaped evidence (root ``SHA256SUMS``,
linker sidecars with a real B4 scene block, ``control/`` manifests,
``reset_<gen>.txt`` produced by ACTUALLY running
``scripts/e1_stage1/reset_verify.py`` against the fixture run directory),
then the shipped ``cycle``/``summary`` CLIs in non-validation mode.

This is the SLOW path: 4 real cycles over a real (loopback) gRPC/websocket
stack take real wall-clock minutes, plus 4 real ``reset_verify.py``
subprocess round-trips. There is no marker registry in this repo (no
``conftest.py``/``pytest.ini``), so this is gated the same, sole way
``test_goalfix_cmp_v3_real_bridge.py`` already is: ``reachy_sdk``/``grpc``/
``websockets``/``scipy`` -- skip on system ``python3``, fail under
``REQUIRE_REACHY_SDK=1`` / ``~/goalfix-venv``.

This test does not assert every finding in the Stage A handoff (those were
established once, by hand, and are not re-derived here); it re-runs the
SAME generation path and pins the structural facts that must keep holding:
the evidence lays out as plan §6 describes, ``reset_verify.py`` really
ran and really agreed with the harness's own reset count, the H6 layout
conflict reproduces on the real root ``SHA256SUMS``, and the documented
workaround lets the shipped CLI produce a real (non-validation-mode)
verdict for all 4 cycles plus a checkpoint.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "native_mujoco"),
           os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "tests", "integration"),
           os.path.join(_ROOT, "tests", "fixtures", "goalfix_cmp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if os.environ.get("REQUIRE_REACHY_SDK") == "1":
    import grpc  # noqa: F401
    import numpy as np  # noqa: F401
    import reachy_sdk  # noqa: F401
    import websockets  # noqa: F401
    import scipy  # noqa: F401
else:
    pytest.importorskip("grpc")
    pytest.importorskip("numpy")
    pytest.importorskip("reachy_sdk")
    pytest.importorskip("websockets")
    pytest.importorskip("scipy")

import stage_a_slice as sas  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import summary as summ  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp._io import read_result  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")

EXPECTED_HOST_SHA = "e54be0e65335af4a6317e287a6be0c22fad753c9"


@pytest.fixture(scope="module")
def stage_a(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stage_a_slice")
    ev_dir = tmp / "ev"
    control_dir = tmp / "control"
    stage = sas.run_stage_a_session(ev_dir, control_dir)
    try:
        evidence = sas.emit_stage_a_evidence(stage, ev_dir)
        yield stage, evidence
    finally:
        stage.session.close()


def test_four_cycles_order_a_b_b_a(stage_a):
    stage, _evidence = stage_a
    assert [c.arm for c in stage.cycles] == ["A", "B", "B", "A"]
    assert [c.rep for c in stage.cycles] == [1, 2, 3, 4]


def test_reset_verify_actually_ran_and_agreed(stage_a):
    """A1/A2: reset_<gen>.txt is produced by ACTUALLY running
    reset_verify.py, not synthesised -- each cycle's real subprocess call
    must have exited 0 and reported the expected resets_recorded
    before->after increment (gen == this cycle's own rep)."""
    stage, _evidence = stage_a
    for cyc_ in stage.cycles:
        assert cyc_.reset_verify_rc == 0, cyc_.reset_verify_stdout
        assert f"reset gen={cyc_.reset_gen} ack={cyc_.reset_gen}" in cyc_.reset_verify_stdout
        before_after = f"resets_recorded {cyc_.reset_gen - 1}->{cyc_.reset_gen}"
        assert before_after in cyc_.reset_verify_stdout


def test_only_a_cycles_carry_injected_echoes(stage_a):
    stage, _evidence = stage_a
    by_name = {c.name: c for c in stage.cycles}
    for c in stage.cycles:
        if c.arm == "A":
            assert len(c.injected_echoes) > 0, "A cycle should carry synthetic echo injection"
        else:
            assert c.injected_echoes == []


def test_evidence_layout_matches_plan_section_6(stage_a):
    """Root SHA256SUMS keyed ./e1_server_runs/<run>/..., run dir holds the
    real commands/states files, sidecars are server_run_dir-bound to the
    run directory and carry a real B4 scene block."""
    _stage, evidence = stage_a
    root_sums = evidence.sha256sums_root.read_text()
    assert f"./e1_server_runs/{evidence.run_dir.name}/states.jsonl" in root_sums
    assert f"./e1_server_runs/{evidence.run_dir.name}/commands.jsonl" in root_sums
    assert (evidence.run_dir / "states.jsonl").is_file()
    assert (evidence.run_dir / "commands.jsonl").is_file()
    for cyc_name, paths in evidence.sidecars.items():
        setup_doc = json.loads(paths["setup"].read_text())
        assert setup_doc["server_run_dir"] == str(evidence.run_dir.resolve())
        assert "PLACE_ROUTE" in setup_doc["log"]
        assert setup_doc["scene"]["chain_sha256"]
        flight_doc = json.loads(paths["flight"].read_text())
        assert "LIFT_TO_PRESENT" in flight_doc["log"]


def test_h6_layout_conflict_reproduces_on_real_root_sha256sums(stage_a):
    """H6 (merge verdict / assignment A4): --ev-dir == evidence root (so
    verify_and_load's SHA256SUMS check passes against the REAL root file)
    makes resolve_leg's server_run_dir check fail, because the sidecar's
    server_run_dir is the run directory, not the root -- exactly the
    conflict the review predicted."""
    _stage, evidence = stage_a
    run_name = evidence.run_dir.name
    out = evidence.control_dir / "between_h6_root_attempt.json"
    cyc_name = "S2-B4-c-r2"
    argv = [
        "--ev-dir", str(evidence.ev_dir),
        "--states", f"e1_server_runs/{run_name}/states.jsonl",
        "--commands", f"e1_server_runs/{run_name}/commands.jsonl",
        "--sha256sums", "SHA256SUMS",
        "--control-dir", str(evidence.control_dir),
        "--cycle", cyc_name, "--rep", "2", "--arm", "B",
        "--arm-map", str(evidence.arm_map_path),
        "--expected-host-sha", EXPECTED_HOST_SHA,
        "--required-supervisor-programs", "reachy-sdk-server",
        "--expected-bridge-sha-a", sas.BRIDGE_SHA["A"],
        "--expected-bridge-sha-b", sas.BRIDGE_SHA["B"],
        "--out", str(out),
    ]
    rc = cyc._cli(argv)
    assert rc == 3
    payload = read_result(out)
    assert "server_run_dir" in payload["reason"]


def test_h6_workaround_lets_cli_run_in_non_validation_mode(stage_a):
    """The documented fixture-side workaround (a derived, prefix-stripped
    SHA256SUMS inside the run directory, --ev-dir == the run directory)
    lets cycle._cli produce a real, non-validation-mode verdict for every
    cycle -- exercising C0-C8, echo classification, the provenance/
    compliance/start_variant gates and the reset-binding check for real."""
    stage, evidence = stage_a
    outs = {}
    for c in stage.cycles:
        out = evidence.control_dir / f"between_{c.name}.json"
        argv = [
            "--ev-dir", str(evidence.run_dir),
            "--sha256sums", "derived-SHA256SUMS",
            "--control-dir", str(evidence.control_dir),
            "--cycle", c.name, "--rep", str(c.rep), "--arm", c.arm,
            "--arm-map", str(evidence.arm_map_path),
            "--expected-host-sha", EXPECTED_HOST_SHA,
            "--required-supervisor-programs", "reachy-sdk-server",
            "--expected-bridge-sha-a", sas.BRIDGE_SHA["A"],
            "--expected-bridge-sha-b", sas.BRIDGE_SHA["B"],
            "--out", str(out),
        ]
        rc = cyc._cli(argv)
        payload = read_result(out)
        assert payload["validation_only"] is False
        assert rc == payload["rc"]
        outs[c.name] = payload

    out = evidence.control_dir / "checkpoint_n4.json"
    argv = [
        "--control-dir", str(evidence.control_dir), "--arm-map", str(evidence.arm_map_path),
        "--expected-bridge-sha-a", sas.BRIDGE_SHA["A"], "--expected-bridge-sha-b", sas.BRIDGE_SHA["B"],
        "--n", "4", "--native-log", str(evidence.native_log_path),
        "--states", str(evidence.run_dir / "states.jsonl"),
        "--control-stop-absent", "--out", str(out),
    ]
    rc = summ._cli(["checkpoint"] + argv)
    payload = read_result(out)
    assert rc == payload["rc"]
    # Real reset-protocol tripwires are clean on this harness (a container
    # recreate + a real reset() each cycle, no lease/control_held/pause
    # sources triggered) -- never asserted as 0 by fiat, this is what a
    # clean generate/reset cycle on the harness actually produces.
    tw = payload["tripwires"]
    assert tw["reset_ack_timeout_total"] == 0
    assert tw["control_held_refusal_total"] == 0
