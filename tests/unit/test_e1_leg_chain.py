"""End-to-end regressions for the REAL `leg.sh`/`parked.sh` bash chain
(PR #124 re-review, R3): a recorded contact and a 20 mm board displacement
both used to reach `LEG ok`/`PARKED ok` because nothing in the chain
looked at what the linker's own evidence said happened, only whether the
evidence was *complete* (`contacts_recorded`). The fix adds one shared
gate step (`scripts/experiment_gate.py`) that both scripts now call right
after the linker.

These tests run the actual `bash scripts/e1_stage1/leg.sh ...` and
`parked.sh ...` invocations documented in `e1_stage1/README.md`, not just
the Python functions underneath -- every step except the recorder itself
(`scripts/measure_route_clearance.py`, which needs `reachy_sdk` and a
live native server) is the real, unmodified code. The recorder step is
replaced by `e1_leg_chain_stub_recorder.py` (`E1_PYTHON` points at a tiny
wrapper that delegates every other call to the real `python3`) -- see
that file's docstring for exactly what it reproduces and what the test
controls via environment variables.

Offline throughout: no server, no SDK, no motors.
"""
import json
import os
import pathlib
import re
import stat
import subprocess
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_LEG_SH = _REPO / "scripts/e1_stage1/leg.sh"
_PARKED_SH = _REPO / "scripts/e1_stage1/parked.sh"
_STUB_RECORDER = _HERE / "e1_leg_chain_stub_recorder.py"

sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))
from e1_stage1 import plan  # noqa: E402

_B4_SCENE_REL = plan.BOARDS["B4"]

_FAKE_PYTHON = """#!/usr/bin/env bash
# Built by tests/unit/test_e1_leg_chain.py: intercepts only the recorder
# invocation (measure_route_clearance.py -- needs reachy_sdk and a live
# server) and delegates every other call -- the linker, the experiment
# gate, the tail check, start_variant, and leg.sh/parked.sh's own -c
# reads -- to the real python3 unchanged, so leg.sh/parked.sh's own bash
# logic and every downstream script run for real.
set -u
for a in "$@"; do
  case "$a" in
    *measure_route_clearance.py) exec python3 "$STUB_RECORDER" "$@" ;;
  esac
done
exec python3 "$@"
"""


@pytest.fixture
def fake_python(tmp_path):
    path = tmp_path / "fake_python.sh"
    path.write_text(_FAKE_PYTHON)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _base_env(fake_python, **test_vars):
    env = dict(os.environ)
    env["E1_PYTHON"] = str(fake_python)
    env["STUB_RECORDER"] = str(_STUB_RECORDER)
    for k, v in test_vars.items():
        env[k] = str(v)
    return env


def _write_leg_plan(evidence_dir, cycle, name, dur_s):
    (evidence_dir / "control").mkdir(parents=True, exist_ok=True)
    (evidence_dir / f"plan_{cycle}.json").write_text(json.dumps({
        "scene_rel": _B4_SCENE_REL,
        "legs": {name: {"dur_s": dur_s}},
    }))


def _run_leg(tmp_path, fake_python, *, mode="end", name="test-leg",
            cycle="S2-TEST-legchain-r1", **test_vars):
    evidence_dir = tmp_path / "evidence"
    _write_leg_plan(evidence_dir, cycle, name, plan.armon_duration_s())
    env = _base_env(fake_python, **test_vars)
    result = subprocess.run(
        ["bash", str(_LEG_SH), str(_REPO), str(evidence_dir), cycle, name,
         "RAISE_TO_SIDE", "INIT", mode],
        env=env, capture_output=True, text=True)
    return result, evidence_dir


def _run_parked(tmp_path, fake_python, *, cycle, name="test-parked",
                pose="stiff", **test_vars):
    evidence_dir = tmp_path / "evidence"
    (evidence_dir / "control").mkdir(parents=True, exist_ok=True)
    env = _base_env(fake_python, E1_TEST_POSE=pose, **test_vars)
    result = subprocess.run(
        ["bash", str(_PARKED_SH), str(_REPO), str(evidence_dir), name, "B4",
         "RAISE_TO_SIDE", cycle],
        env=env, capture_output=True, text=True)
    return result, evidence_dir


class TestLegShCleanPass:

    def test_clean_recording_reaches_leg_ok(self, tmp_path, fake_python):
        result, evidence_dir = _run_leg(tmp_path, fake_python)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "LEG test-leg ok" in result.stdout
        assert not (evidence_dir / "control/stop").exists()
        archived = list((evidence_dir / "recorder_logs").glob("*.json"))
        assert any(p.suffix == ".json" and not p.name.endswith(".link.json")
                   for p in archived)
        assert any(p.name.endswith(".link.json") for p in archived)


class TestLegShRecordedContact:
    """The review's own demonstrated case: a contact recorded during the
    flight used to still reach `LEG ok`."""

    def test_stops_and_never_prints_leg_ok(self, tmp_path, fake_python):
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_CONTACT_AT_S=5.0)
        assert result.returncode != 0
        assert "LEG test-leg ok" not in result.stdout
        assert "ok" not in result.stdout.strip().splitlines()[-1].lower()
        assert (evidence_dir / "control/stop").exists()
        stop_text = (evidence_dir / "control/stop").read_text()
        assert "experiment gate" in stop_text

    def test_gate_output_names_the_contact(self, tmp_path, fake_python):
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_CONTACT_AT_S=5.0)
        gate_txt = (evidence_dir / "control/gate_test-leg.txt").read_text()
        assert "EXPERIMENT_ACCEPTED=NO" in gate_txt
        assert "contact" in gate_txt

    def test_evidence_is_still_archived(self, tmp_path, fake_python):
        """Preserving the results: the rejected recording's log and
        sidecar must still be in recorder_logs/ for review."""
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_CONTACT_AT_S=5.0)
        archived = list((evidence_dir / "recorder_logs").glob("*"))
        assert any(p.name.endswith(".link.json") for p in archived)
        sidecar = next(p for p in archived if p.name.endswith(".link.json"))
        doc = json.loads(sidecar.read_text())
        assert doc["contacts_recorded"] is True
        assert len(doc["contacts"]) >= 1

    def test_gate_stops_regardless_of_mode(self, tmp_path, fake_python):
        """Unlike the tail check, the gate is unconditional -- a contact
        must STOP even in `start` mode."""
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, mode="start", E1_TEST_CONTACT_AT_S=5.0)
        assert result.returncode != 0
        assert "LEG test-leg ok" not in result.stdout
        assert (evidence_dir / "control/stop").exists()


class TestLegShDisplacedBoard:
    """The review's other demonstrated case: the board displaced 20 mm
    during the recording, no contact recorded."""

    def test_stops_and_never_prints_leg_ok(self, tmp_path, fake_python):
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_DISPLACE_M=0.020)
        assert result.returncode != 0
        assert "LEG test-leg ok" not in result.stdout
        assert (evidence_dir / "control/stop").exists()
        stop_text = (evidence_dir / "control/stop").read_text()
        assert "experiment gate" in stop_text

    def test_gate_output_names_the_displacement(self, tmp_path, fake_python):
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_DISPLACE_M=0.020)
        gate_txt = (evidence_dir / "control/gate_test-leg.txt").read_text()
        assert "EXPERIMENT_ACCEPTED=NO" in gate_txt
        assert "pool_box_1" in gate_txt
        # alignment picks the nearest server state to each sample, so the
        # reported displacement is close to but not bit-exact with the
        # 20 mm the stub moved the board -- assert the magnitude, not an
        # exact string.
        match = re.search(r"displaced ([\d.]+) mm", gate_txt)
        assert match, gate_txt
        assert 19.0 <= float(match.group(1)) <= 21.0

    def test_small_displacement_within_tolerance_still_passes(self, tmp_path, fake_python):
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_DISPLACE_M=0.0003)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "LEG test-leg ok" in result.stdout


class TestLegShResetInWindow:
    """#126: a scene reset mid-recording (`sim_step` drops back to 0
    inside the contact window) must STOP through the real gate step, the
    same as a recorded contact or a displaced board -- the reset makes
    the window's own contact/displacement evidence suspect, so it is
    refused rather than trusted."""

    def test_stops_and_never_prints_leg_ok(self, tmp_path, fake_python):
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_RESET_MID_FLIGHT="1")
        assert result.returncode != 0
        assert "LEG test-leg ok" not in result.stdout
        assert (evidence_dir / "control/stop").exists()
        stop_text = (evidence_dir / "control/stop").read_text()
        assert "experiment gate" in stop_text

    def test_gate_output_names_reset_in_window(self, tmp_path, fake_python):
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_RESET_MID_FLIGHT="1")
        gate_txt = (evidence_dir / "control/gate_test-leg.txt").read_text()
        assert "EXPERIMENT_ACCEPTED=NO" in gate_txt
        assert "reset_in_window" in gate_txt

    def test_gate_stops_regardless_of_mode(self, tmp_path, fake_python):
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, mode="start", E1_TEST_RESET_MID_FLIGHT="1")
        assert result.returncode != 0
        assert "LEG test-leg ok" not in result.stdout
        assert (evidence_dir / "control/stop").exists()

    def test_evidence_is_still_archived(self, tmp_path, fake_python):
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_RESET_MID_FLIGHT="1")
        archived = list((evidence_dir / "recorder_logs").glob("*.link.json"))
        assert archived
        doc = json.loads(archived[0].read_text())
        assert doc["contacts_window"]["reset_in_window"] is True

    def test_no_reset_does_not_trigger_this_refusal(self, tmp_path, fake_python):
        """Control: the same chain without the reset reaches `LEG ok`, so
        the STOP above is attributable to `reset_in_window` and not to
        some other side effect of the stub."""
        result, evidence_dir = _run_leg(tmp_path, fake_python)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "LEG test-leg ok" in result.stdout


class TestLegShMissingBoardDisplacement:
    """#126: a board object present in the leaf board's own
    `scene.board_object_ids` but never tracked in server state (so it has
    no `displacement_m` entry) must STOP -- the old gate silently read the
    missing key as zero displacement instead of missing evidence."""

    def test_stops_and_never_prints_leg_ok(self, tmp_path, fake_python):
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_OMIT_OBJECT="1")
        assert result.returncode != 0
        assert "LEG test-leg ok" not in result.stdout
        assert (evidence_dir / "control/stop").exists()
        stop_text = (evidence_dir / "control/stop").read_text()
        assert "experiment gate" in stop_text

    def test_gate_output_names_the_missing_object(self, tmp_path, fake_python):
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_OMIT_OBJECT="1")
        gate_txt = (evidence_dir / "control/gate_test-leg.txt").read_text()
        assert "EXPERIMENT_ACCEPTED=NO" in gate_txt
        assert "pool_box_1" in gate_txt

    def test_evidence_is_still_archived(self, tmp_path, fake_python):
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_OMIT_OBJECT="1")
        archived = list((evidence_dir / "recorder_logs").glob("*.link.json"))
        assert archived
        doc = json.loads(archived[0].read_text())
        assert "pool_box_1" not in doc["displacement_m"]
        assert "pool_box_1" in doc["scene"]["board_object_ids"]


class TestParkedShResetInWindowAndBoardCoverage:
    """Both scripts share the same gate call -- `parked.sh`'s own
    docstring says a disturbed board during a parked interval "must STOP
    the same as during a flight leg", and #126 applies equally here."""

    def test_reset_in_window_stops(self, tmp_path, fake_python):
        result, evidence_dir = _run_parked(
            tmp_path, fake_python, cycle="S2-TEST-parked-reset-r1",
            E1_TEST_RESET_MID_FLIGHT="1")
        assert result.returncode != 0
        assert "PARKED test-parked ok" not in result.stdout
        assert (evidence_dir / "control/stop").exists()
        assert "experiment gate" in (evidence_dir / "control/stop").read_text()

    def test_missing_board_displacement_stops(self, tmp_path, fake_python):
        result, evidence_dir = _run_parked(
            tmp_path, fake_python, cycle="S2-TEST-parked-nodisp-r1",
            E1_TEST_OMIT_OBJECT="1")
        assert result.returncode != 0
        assert "PARKED test-parked ok" not in result.stdout
        assert (evidence_dir / "control/stop").exists()
        assert "experiment gate" in (evidence_dir / "control/stop").read_text()


class TestLegShMissingOrMalformedEvidence:

    def test_no_run_directory_at_all_stops(self, tmp_path, fake_python):
        """Missing evidence: the native server's run directory never
        existed (e.g. the server died before the recorder connected)."""
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_NO_RUN_DIR="1")
        assert result.returncode != 0
        assert "LEG test-leg ok" not in result.stdout
        assert (evidence_dir / "control/stop").exists()

    def test_corrupt_states_jsonl_stops(self, tmp_path, fake_python):
        """Malformed evidence: a non-terminal line of states.jsonl is not
        valid JSON -- not a single-run stream, refuse."""
        result, evidence_dir = _run_leg(
            tmp_path, fake_python, E1_TEST_CORRUPT_LINE=2)
        assert result.returncode != 0
        assert "LEG test-leg ok" not in result.stdout
        assert (evidence_dir / "control/stop").exists()


class TestParkedShGate:

    def test_clean_parked_recording_reaches_parked_ok(self, tmp_path, fake_python):
        result, evidence_dir = _run_parked(
            tmp_path, fake_python, cycle="S2-TEST-parked-clean-r1")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PARKED test-parked ok" in result.stdout
        assert not (evidence_dir / "control/stop").exists()
        archived = list((evidence_dir / "recorder_logs").glob("*.link.json"))
        assert archived

    def test_recorded_contact_stops_before_start_variant(self, tmp_path, fake_python):
        result, evidence_dir = _run_parked(
            tmp_path, fake_python, cycle="S2-TEST-parked-contact-r1",
            E1_TEST_CONTACT_AT_S=1.0)
        assert result.returncode != 0
        assert "PARKED test-parked ok" not in result.stdout
        assert (evidence_dir / "control/stop").exists()
        assert "experiment gate" in (evidence_dir / "control/stop").read_text()
        # the gate STOPs before start_variant ever runs for this cycle.
        assert not (evidence_dir / f"control/start_variant_S2-TEST-parked-contact-r1.json").exists()

    def test_displaced_board_stops(self, tmp_path, fake_python):
        result, evidence_dir = _run_parked(
            tmp_path, fake_python, cycle="S2-TEST-parked-disp-r1",
            E1_TEST_DISPLACE_M=0.020)
        assert result.returncode != 0
        assert "PARKED test-parked ok" not in result.stdout
        assert (evidence_dir / "control/stop").exists()

    def test_evidence_preserved_on_rejection(self, tmp_path, fake_python):
        result, evidence_dir = _run_parked(
            tmp_path, fake_python, cycle="S2-TEST-parked-preserve-r1",
            E1_TEST_CONTACT_AT_S=1.0)
        archived = list((evidence_dir / "recorder_logs").glob("*.link.json"))
        assert archived
        doc = json.loads(archived[0].read_text())
        assert doc["contacts_recorded"] is True
        assert len(doc["contacts"]) >= 1
