"""Pilot item (2026-09-15 matrix readiness): scripts/e1_tail_check.py,
moved unchanged from the E1 pilot evidence directory into scripts/ so it
has a real unit test and cannot silently drift from `rig_routes`.

This is the same 11-case table `e1_tail_check.selftest()` has always run,
now as individually-visible pytest cases (each failure names ITS case,
rather than a single 11/11 pass/fail line) plus the module's own
invariant assertions (REST vs REST_SHUT differing only in `r_gripper`).
Offline throughout: no samples come from a live server or SDK.

Also covers `check_init`/`INIT` (2026-09-16 re-review, R1): the dedicated
Stage 0 initialization acceptance check that replaced the unpassable
`HOME` tail check documented in `e1_stage1/README.md`.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))

import e1_tail_check as etc  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402

_CASES = [
    ("parked at REST, judged REST", etc._synthetic(R.REST), "REST", True),
    ("parked at REST_SHUT, judged REST", etc._synthetic(R.REST_SHUT), "REST", False),
    ("parked at REST, judged REST_SHUT", etc._synthetic(R.REST), "REST_SHUT", False),
    ("parked at REST_SHUT, judged REST_SHUT", etc._synthetic(R.REST_SHUT), "REST_SHUT", True),
    ("parked at PRESENT, judged PRESENT", etc._synthetic(R.PRESENT), "PRESENT", True),
    ("parked at PRESENT, judged REST", etc._synthetic(R.PRESENT), "REST", False),
    ("parked at HOME, judged PRESENT", etc._synthetic(R.HOME), "PRESENT", False),
    ("moving at the end, judged REST",
     etc._synthetic(R.REST, moving_from_s=17.0), "REST", False),
    ("REST arm joints, gripper half-open (-20)",
     etc._synthetic(R.REST, gripper=-20.0), "REST", False),
    ("REST, gripper -47 (within 3)", etc._synthetic(R.REST, gripper=-47.0), "REST", True),
    ("REST, wrist_pitch drooped 18 deg (info only)",
     [dict(s, joints=dict(s["joints"], r_wrist_pitch=R.REST["r_wrist_pitch"] - 18.0))
      for s in etc._synthetic(R.REST)], "REST", True),
]


@pytest.mark.parametrize("label,samples,target,expect", _CASES, ids=[c[0] for c in _CASES])
def test_selftest_case(label, samples, target, expect):
    assert etc.check(samples, target)["ok"] == expect, label


def test_rest_and_rest_shut_differ_only_in_gripper():
    """The check's whole design rests on this: if REST and REST_SHUT ever
    diverge on an arm joint too, the gripper criterion stops being the
    only thing separating them and `--selftest`'s premise breaks."""
    diff = {j for j in R.R_JOINTS if R.REST[j] != R.REST_SHUT[j]}
    assert diff == {"r_gripper"}
    assert R.REST["r_gripper"] == R.OPEN == -45.0
    assert R.REST_SHUT["r_gripper"] == R.SHUT == 20.0


def test_gross_joint_posture_alone_does_not_separate_rest_from_rest_shut():
    assert R.at_pose(R.REST_SHUT, R.REST, tol=8.0, joints=list(R.GROSS_JOINTS))


def test_selftest_entrypoint_still_passes():
    assert etc.selftest() is True


# ── check_init / INIT (2026-09-16 re-review, R1) ────────────────────────────
#
# HOME was the wrong acceptance criterion for the Stage 0 armon recording
# (README, "Policy A: Stage 0 arm-on"): it requires r_gripper within 3 deg
# of HOME's OPEN (-45 deg), but on a fresh, compliant server the gripper
# is still sagging toward keyframe-sag (~[-40, 0] deg) throughout the
# recording, disjoint from that window -- every Stage 1 parked HOME tail
# check on a comparably fresh arm failed on exactly this criterion, which
# in `end` mode stops the board session before cycle 1. `check_init`
# replaces it: allow the turn_on transient (no first-to-last invariance),
# require a stable final window, require complete contact evidence.

def _synthetic_settling(pose, seconds=14.0, still_from_s=6.0, drift_deg_per_s=6.0):
    """Samples that drift on `r_gripper` (standing in for the pre-turn_on
    compliant sag) until `still_from_s`, then hold `pose` exactly -- the
    shape check_init's "allow the transient, require a stable final
    window" rule is meant to accept. With the default `window_s=3.0` and
    `seconds=14.0` (== `plan.armon_duration_s()`), the tail covers
    [11, 14)s, well after the default `still_from_s=6.0`."""
    out = []
    for k in range(int(seconds * etc.SAMPLE_HZ)):
        t = k / etc.SAMPLE_HZ
        p = dict(pose)
        if t < still_from_s:
            p["r_gripper"] = pose["r_gripper"] + (still_from_s - t) * drift_deg_per_s
        out.append({"t": t, "wall_time_ns": 10**9 + k * 50_000_000, "joints": p})
    return out


def _write_log_and_sidecar(tmp_path, samples, *, sidecar_present=True,
                            contacts_recorded=True, corrupt_sidecar=False,
                            contacts=(), displacement_m=None, log_name=None,
                            board_object_ids=None, reset_in_window=False):
    """`board_object_ids` defaults to `displacement_m`'s own keys and
    `reset_in_window` defaults to `False` (#126) -- the shape a
    successful `link_e1_flight` run always produces -- so a caller that
    only sets `displacement_m`/`contacts` still gets a sidecar the gate
    can evaluate past its evidence-validity checks."""
    log_path = tmp_path / "flight.json"
    log_path.write_text(json.dumps({"samples": samples}))
    if sidecar_present:
        sidecar_path = log_path.with_suffix(".link.json")
        if corrupt_sidecar:
            sidecar_path.write_text("{not json")
        else:
            displacement_m = displacement_m if displacement_m is not None else {}
            board_object_ids = (
                list(displacement_m) if board_object_ids is None
                else board_object_ids)
            sidecar_path.write_text(json.dumps({
                "log": log_name if log_name is not None else log_path.name,
                "contacts_recorded": contacts_recorded,
                "contacts": list(contacts),
                "displacement_m": displacement_m,
                "scene": {"board_object_ids": board_object_ids},
                "contacts_window": {"reset_in_window": reset_in_window},
            }))
    return log_path


class TestCheckInit:

    def test_settling_recording_with_complete_evidence_passes(self, tmp_path):
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        # No first-to-last invariance: the first sample is well off HOME
        # (mid-sag), the last is exactly HOME -- and that is fine.
        assert abs(samples[0]["joints"]["r_gripper"]
                   - samples[-1]["joints"]["r_gripper"]) > 30.0
        log_path = _write_log_and_sidecar(tmp_path, samples)
        res = etc.check_init(str(log_path), samples)
        assert res["still_ok"] is True
        assert res["samples_ok"] is True
        assert res["evidence_ok"] is True
        assert res["ok"] is True

    def test_moving_in_the_final_window_fails(self, tmp_path):
        """The transient is allowed everywhere EXCEPT the final window --
        still moving there must fail, same stillness bar check() uses."""
        samples = _synthetic_settling(R.HOME, still_from_s=12.0)
        log_path = _write_log_and_sidecar(tmp_path, samples)
        res = etc.check_init(str(log_path), samples)
        assert res["still_ok"] is False
        assert res["ok"] is False

    def test_missing_sidecar_refuses(self, tmp_path):
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        log_path = _write_log_and_sidecar(tmp_path, samples, sidecar_present=False)
        res = etc.check_init(str(log_path), samples)
        assert res["still_ok"] is True  # the motion criterion alone would pass
        assert res["evidence_ok"] is False
        assert "no linked sidecar" in res["evidence_reason"]
        assert res["ok"] is False

    def test_incomplete_contact_evidence_refuses(self, tmp_path):
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        log_path = _write_log_and_sidecar(tmp_path, samples, contacts_recorded=False)
        res = etc.check_init(str(log_path), samples)
        assert res["evidence_ok"] is False
        assert res["ok"] is False

    def test_malformed_sidecar_refuses(self, tmp_path):
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        log_path = _write_log_and_sidecar(tmp_path, samples, corrupt_sidecar=True)
        res = etc.check_init(str(log_path), samples)
        assert res["evidence_ok"] is False
        assert res["ok"] is False

    def test_stage1_measured_gripper_values_that_failed_home_now_pass(self, tmp_path):
        """The re-review's own evidence table: real Stage 1 parked-HOME
        tail checks measured r_gripper at -38.1, -37.7, -40.1, and -0.0
        deg (stiff-zero legs), with posture_ok/still_ok/samples_ok all
        True and only HOME's gripper criterion failing. check_init has no
        gripper (or any posture) criterion, so a still, evidence-complete
        recording at any of these poses now passes."""
        for gripper in (-38.1, -37.7, -40.1, -0.0):
            pose = dict(R.HOME, r_gripper=gripper)
            samples = etc._synthetic(pose, seconds=14.0)  # still throughout
            log_path = _write_log_and_sidecar(tmp_path, samples)
            res = etc.check_init(str(log_path), samples)
            assert res["ok"] is True, gripper

    def test_readme_documents_INIT_not_HOME_for_the_armon_invocation(self):
        readme = (pathlib.Path(_HERE) / "../../scripts/e1_stage1/README.md").read_text()
        assert "RAISE_TO_SIDE INIT end" in readme
        assert "RAISE_TO_SIDE HOME end" not in readme

    def test_cli_matches_the_exact_documented_invocation(self, tmp_path):
        """`E1_PYTHON=<e1venv python> scripts/e1_stage1/leg.sh ... RAISE_TO_SIDE
        INIT end` runs `PYTHONPATH=src <python> scripts/e1_tail_check.py
        <log> INIT` (leg.sh:39) -- exercise that exact CLI form end to
        end, not just the Python function."""
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        log_path = _write_log_and_sidecar(tmp_path, samples)
        script = (pathlib.Path(_HERE) / "../../scripts/e1_tail_check.py").resolve()
        src_dir = (pathlib.Path(_HERE) / "../../src").resolve()
        env = dict(os.environ, PYTHONPATH=str(src_dir))
        result = subprocess.run(
            [sys.executable, str(script), str(log_path), "INIT"],
            capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "STAGE0_INIT_OK=yes" in result.stdout

    def test_cli_reports_failure_for_the_old_HOME_target_on_the_same_recording(
            self, tmp_path):
        """The failure the re-review demonstrated, reproduced end to end:
        the same still, evidence-complete recording that passes INIT
        fails HOME, non-zero exit -- which is what made the previously
        documented invocation stop the board session before cycle 1."""
        pose = dict(R.HOME, r_gripper=-38.1)
        samples = etc._synthetic(pose, seconds=14.0)
        log_path = _write_log_and_sidecar(tmp_path, samples)
        script = (pathlib.Path(_HERE) / "../../scripts/e1_tail_check.py").resolve()
        src_dir = (pathlib.Path(_HERE) / "../../src").resolve()
        env = dict(os.environ, PYTHONPATH=str(src_dir))
        result = subprocess.run(
            [sys.executable, str(script), str(log_path), "HOME"],
            capture_output=True, text=True, env=env)
        assert result.returncode != 0
        assert "PARKED_AT_HOME=NO" in result.stdout


# ── check_init / experiment acceptance (2026-09-16 re-review, R3) ──────────
#
# check_init used to accept any recording whose sidecar said
# `contacts_recorded: True` -- a completeness verdict, not a no-contact
# verdict. The re-review demonstrated a recorded contact and a 20 mm board
# displacement both passing INIT (and `leg.sh ... end`'s "LEG ok") anyway.
# check_init now delegates to the shared `experiment_gate.evaluate`, the
# same gate `leg.sh`/`parked.sh` call -- these cases exercise that gate
# through check_init directly (`test_experiment_gate.py` covers the gate
# module and its CLI on their own).

class TestCheckInitExperimentAcceptance:

    def test_recorded_contact_refuses(self, tmp_path):
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        log_path = _write_log_and_sidecar(
            tmp_path, samples,
            contacts=[{"arm_geom": "r_hand_tube", "object_id": "pool_box_1"}])
        res = etc.check_init(str(log_path), samples)
        assert res["still_ok"] is True
        assert res["evidence_ok"] is True
        assert res["experiment_accepted"] is False
        assert "contact" in res["experiment_reason"]
        assert res["ok"] is False

    def test_20mm_displacement_refuses(self, tmp_path):
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        log_path = _write_log_and_sidecar(
            tmp_path, samples, displacement_m={"pool_box_1": 0.020})
        res = etc.check_init(str(log_path), samples)
        assert res["evidence_ok"] is True
        assert res["experiment_accepted"] is False
        assert "pool_box_1" in res["experiment_reason"]
        assert "20.00 mm" in res["experiment_reason"]
        assert res["ok"] is False

    def test_displacement_at_the_1mm_tolerance_boundary_passes(self, tmp_path):
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        log_path = _write_log_and_sidecar(
            tmp_path, samples, displacement_m={"pool_box_1": 0.001})
        res = etc.check_init(str(log_path), samples)
        assert res["experiment_accepted"] is True
        assert res["ok"] is True

    def test_displacement_just_over_1mm_refuses(self, tmp_path):
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        log_path = _write_log_and_sidecar(
            tmp_path, samples, displacement_m={"pool_box_1": 0.0010001})
        res = etc.check_init(str(log_path), samples)
        assert res["experiment_accepted"] is False
        assert res["ok"] is False

    def test_non_finite_displacement_refuses_as_invalid_evidence(self, tmp_path):
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        log_path = _write_log_and_sidecar(tmp_path, samples)
        sidecar_path = log_path.with_suffix(".link.json")
        sidecar_path.write_text(json.dumps({
            "log": log_path.name, "contacts_recorded": True, "contacts": [],
            "displacement_m": {"pool_box_1": float("nan")},
        }))
        res = etc.check_init(str(log_path), samples)
        assert res["evidence_ok"] is False
        assert res["experiment_accepted"] is None
        assert res["ok"] is False

    def test_sidecar_naming_a_different_recording_refuses(self, tmp_path):
        """A hand-copied or stale sidecar for another flight must not
        authorize this one."""
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        log_path = _write_log_and_sidecar(
            tmp_path, samples, log_name="some_other_recording.json")
        res = etc.check_init(str(log_path), samples)
        assert res["evidence_ok"] is False
        assert "wrong-recording" in res["evidence_reason"]
        assert res["ok"] is False

    def test_missing_contacts_field_refuses_as_invalid_evidence(self, tmp_path):
        """contacts_recorded=True with no 'contacts' key at all is not the
        shape a successful linker run produces -- refuse, do not treat as
        zero contacts."""
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        log_path = _write_log_and_sidecar(tmp_path, samples)
        sidecar_path = log_path.with_suffix(".link.json")
        sidecar_path.write_text(json.dumps(
            {"log": log_path.name, "contacts_recorded": True}))
        res = etc.check_init(str(log_path), samples)
        assert res["evidence_ok"] is False
        assert res["ok"] is False

    def test_clean_recording_with_full_evidence_passes(self, tmp_path):
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        log_path = _write_log_and_sidecar(
            tmp_path, samples, contacts=[],
            displacement_m={"pool_box_1": 0.0002})
        res = etc.check_init(str(log_path), samples)
        assert res["evidence_ok"] is True
        assert res["experiment_accepted"] is True
        assert res["ok"] is True

    def test_cli_reports_stage0_init_not_ok_on_recorded_contact(self, tmp_path):
        """The literal `scripts/e1_tail_check.py <log> INIT` CLI leg.sh
        invokes -- non-zero exit and STAGE0_INIT_OK=NO on a recorded
        contact, not just the Python-level result dict."""
        samples = _synthetic_settling(R.HOME, still_from_s=6.0)
        log_path = _write_log_and_sidecar(
            tmp_path, samples,
            contacts=[{"arm_geom": "r_hand_tube", "object_id": "pool_box_1"}])
        script = (pathlib.Path(_HERE) / "../../scripts/e1_tail_check.py").resolve()
        src_dir = (pathlib.Path(_HERE) / "../../src").resolve()
        env = dict(os.environ, PYTHONPATH=str(src_dir))
        result = subprocess.run(
            [sys.executable, str(script), str(log_path), "INIT"],
            capture_output=True, text=True, env=env)
        assert result.returncode != 0
        assert "STAGE0_INIT_OK=NO" in result.stdout
