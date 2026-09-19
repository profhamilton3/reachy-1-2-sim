"""scripts/experiment_gate.py (PR #124 re-review, R3): the shared
experiment-acceptance gate `leg.sh`, `parked.sh`, and
`e1_tail_check.check_init` all call after the linker succeeds.

Offline throughout: no samples or sidecars come from a live server, SDK,
or the real `link_e1_flight.py` -- sidecars are written directly, the
same shape `link_e1_flight.build_base_sidecar`/`align_and_recompute`
produce on a real run.
"""

import json
import math
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))

import experiment_gate as gate  # noqa: E402


def _write_sidecar(tmp_path, log_name="flight.json", **fields):
    """Writes a sidecar with the full shape a successful `link_e1_flight`
    run guarantees -- including `scene.board_object_ids` and
    `contacts_window.reset_in_window` (#126) -- so a test that overrides
    only `displacement_m`/`contacts` still exercises a fully-valid
    baseline. `board_object_ids` defaults to `displacement_m`'s own keys
    so an override of one alone still passes #126's coverage check unless
    a test deliberately sets `scene` to something that does not."""
    log_path = tmp_path / log_name
    log_path.write_text("{}")
    sidecar = {
        "log": log_name,
        "contacts_recorded": True,
        "contacts": [],
        "displacement_m": {},
        "contacts_window": {"reset_in_window": False},
    }
    sidecar.update(fields)
    if "scene" not in sidecar:
        disp = sidecar.get("displacement_m")
        sidecar["scene"] = {
            "board_object_ids": list(disp) if isinstance(disp, dict) else [],
        }
    log_path.with_suffix(".link.json").write_text(json.dumps(sidecar))
    return log_path


class TestEvidenceValidity:

    def test_missing_sidecar_refuses(self, tmp_path):
        log_path = tmp_path / "flight.json"
        log_path.write_text("{}")
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["accepted"] is None
        assert res["ok"] is False
        assert "no linked sidecar" in res["evidence_reason"]

    def test_empty_sidecar_file_refuses(self, tmp_path):
        log_path = tmp_path / "flight.json"
        log_path.write_text("{}")
        log_path.with_suffix(".link.json").write_text("")
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["ok"] is False

    def test_malformed_json_refuses(self, tmp_path):
        log_path = tmp_path / "flight.json"
        log_path.write_text("{}")
        log_path.with_suffix(".link.json").write_text("{not json")
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["ok"] is False

    def test_sidecar_that_is_a_json_list_refuses(self, tmp_path):
        log_path = tmp_path / "flight.json"
        log_path.write_text("{}")
        log_path.with_suffix(".link.json").write_text("[true]")
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["ok"] is False

    def test_wrong_recording_sidecar_refuses(self, tmp_path):
        log_path = _write_sidecar(tmp_path, log_name="flight.json")
        sidecar_path = log_path.with_suffix(".link.json")
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["log"] = "some_other_recording.json"
        sidecar_path.write_text(json.dumps(sidecar))
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["accepted"] is None
        assert "wrong-recording" in res["evidence_reason"]
        assert res["ok"] is False

    def test_sidecar_missing_log_field_refuses(self, tmp_path):
        log_path = tmp_path / "flight.json"
        log_path.write_text("{}")
        log_path.with_suffix(".link.json").write_text(json.dumps(
            {"contacts_recorded": True, "contacts": [], "displacement_m": {}}))
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert "wrong-recording" in res["evidence_reason"]

    @pytest.mark.parametrize("contacts_recorded", [False, None, "true", 1, "1"])
    def test_contacts_not_recorded_refuses(self, tmp_path, contacts_recorded):
        log_path = _write_sidecar(tmp_path, contacts_recorded=contacts_recorded)
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["accepted"] is None
        assert res["ok"] is False

    def test_missing_contacts_recorded_key_refuses(self, tmp_path):
        log_path = tmp_path / "flight.json"
        log_path.write_text("{}")
        log_path.with_suffix(".link.json").write_text(json.dumps(
            {"log": "flight.json", "contacts": [], "displacement_m": {}}))
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False

    def test_contacts_not_a_list_refuses(self, tmp_path):
        log_path = _write_sidecar(tmp_path, contacts=None)
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["accepted"] is None

    def test_missing_contacts_key_refuses(self, tmp_path):
        log_path = tmp_path / "flight.json"
        log_path.write_text("{}")
        log_path.with_suffix(".link.json").write_text(json.dumps(
            {"log": "flight.json", "contacts_recorded": True,
             "displacement_m": {}}))
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False

    def test_displacement_m_not_an_object_refuses(self, tmp_path):
        log_path = _write_sidecar(tmp_path, displacement_m=None)
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False

    def test_missing_displacement_m_key_refuses(self, tmp_path):
        log_path = tmp_path / "flight.json"
        log_path.write_text("{}")
        log_path.with_suffix(".link.json").write_text(json.dumps(
            {"log": "flight.json", "contacts_recorded": True, "contacts": []}))
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"),
                                     "0.02", True, None, [0.02]])
    def test_non_finite_or_wrong_type_displacement_refuses(self, tmp_path, bad):
        log_path = _write_sidecar(tmp_path, displacement_m={"pool_box_1": bad})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["accepted"] is None


class TestResetInWindow:
    """#126: `contacts_window.reset_in_window` must be the exact value
    `False`. A mid-recording scene reset makes the window's contact and
    displacement attribution suspect (see
    `link_e1_flight.contacts_in_flight_window`'s docstring), so anything
    else -- `True`, a missing key, or a malformed `contacts_window` --
    must refuse, not just a truthy check."""

    def test_reset_in_window_true_refuses(self, tmp_path):
        log_path = _write_sidecar(
            tmp_path, contacts_window={"reset_in_window": True})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["accepted"] is None
        assert res["ok"] is False
        assert "reset_in_window" in res["evidence_reason"]

    def test_missing_reset_in_window_key_refuses(self, tmp_path):
        log_path = _write_sidecar(tmp_path, contacts_window={})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert "reset_in_window" in res["evidence_reason"]

    def test_missing_contacts_window_key_refuses(self, tmp_path):
        log_path = _write_sidecar(tmp_path)
        sidecar_path = log_path.with_suffix(".link.json")
        sidecar = json.loads(sidecar_path.read_text())
        del sidecar["contacts_window"]
        sidecar_path.write_text(json.dumps(sidecar))
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False

    @pytest.mark.parametrize("contacts_window", [None, [], "false", 1])
    def test_contacts_window_not_an_object_refuses(self, tmp_path, contacts_window):
        log_path = _write_sidecar(tmp_path, contacts_window=contacts_window)
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False

    @pytest.mark.parametrize("bad", [None, "false", "False", 0, 1, "0"])
    def test_malformed_reset_in_window_refuses(self, tmp_path, bad):
        log_path = _write_sidecar(
            tmp_path, contacts_window={"reset_in_window": bad})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["accepted"] is None

    def test_explicit_false_does_not_block_acceptance(self, tmp_path):
        log_path = _write_sidecar(
            tmp_path, contacts=[], displacement_m={},
            contacts_window={"reset_in_window": False, "states_in_window": 4})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is True
        assert res["accepted"] is True
        assert res["ok"] is True

    def test_reset_in_window_checked_even_when_experiment_would_otherwise_be_rejected(self, tmp_path):
        """A reset alone must refuse, regardless of what the (suspect)
        contacts/displacement fields say -- the point is that those
        fields are not to be trusted once reset_in_window is true."""
        log_path = _write_sidecar(
            tmp_path,
            contacts=[{"arm_geom": "r_hand_tube", "object_id": "pool_box_1"}],
            contacts_window={"reset_in_window": True})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["accepted"] is None


class TestBoardDisplacementCoverage:
    """#126: every id `scene.board_object_ids` names must have its own
    `displacement_m` entry. `displacement_m` is only ever populated for
    objects that appear in BOTH the first and last aligned server state
    (`link_e1_flight.build_base_sidecar`); a board object that dropped out
    of tracking is simply absent from the dict, and reading that as "0 mm
    displaced" (the old behaviour) hides exactly the kind of gap the gate
    exists to catch."""

    def test_missing_scene_key_refuses(self, tmp_path):
        log_path = _write_sidecar(tmp_path)
        sidecar_path = log_path.with_suffix(".link.json")
        sidecar = json.loads(sidecar_path.read_text())
        del sidecar["scene"]
        sidecar_path.write_text(json.dumps(sidecar))
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False

    @pytest.mark.parametrize("scene", [None, [], "B4", 1])
    def test_scene_not_an_object_refuses(self, tmp_path, scene):
        log_path = _write_sidecar(tmp_path, scene=scene)
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False

    @pytest.mark.parametrize("board_object_ids", [None, "pool_box_1", {"pool_box_1": True}, [1, 2]])
    def test_board_object_ids_not_a_list_of_strings_refuses(self, tmp_path, board_object_ids):
        log_path = _write_sidecar(
            tmp_path, scene={"board_object_ids": board_object_ids})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["accepted"] is None

    def test_board_object_missing_from_displacement_refuses(self, tmp_path):
        log_path = _write_sidecar(
            tmp_path, contacts=[],
            displacement_m={"pool_box_1": 0.0002},
            scene={"board_object_ids": ["pool_box_1", "pool_box_2"]})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["accepted"] is None
        assert "pool_box_2" in res["evidence_reason"]

    def test_board_object_missing_from_displacement_refuses_even_when_contacts_present(self, tmp_path):
        """Missing evidence refuses outright -- it is not folded into the
        (accepted=False) contact rejection path."""
        log_path = _write_sidecar(
            tmp_path,
            contacts=[{"arm_geom": "r_hand_tube", "object_id": "pool_box_1"}],
            displacement_m={},
            scene={"board_object_ids": ["pool_box_1"]})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is False
        assert res["accepted"] is None

    def test_every_board_object_covered_is_accepted(self, tmp_path):
        log_path = _write_sidecar(
            tmp_path, contacts=[],
            displacement_m={"pool_box_1": 0.0002, "pool_box_2": 0.0003},
            scene={"board_object_ids": ["pool_box_1", "pool_box_2"]})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is True
        assert res["accepted"] is True

    def test_extra_non_board_displacement_entries_do_not_block_acceptance(self, tmp_path):
        """`displacement_m` may legitimately carry more objects than the
        leaf board names (anything tracked in server state); only full
        coverage of `board_object_ids` is required, not an exact match."""
        log_path = _write_sidecar(
            tmp_path, contacts=[],
            displacement_m={"pool_box_1": 0.0002, "some_other_tracked_obj": 0.0003},
            scene={"board_object_ids": ["pool_box_1"]})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is True
        assert res["accepted"] is True

    def test_empty_board_object_ids_requires_no_coverage(self, tmp_path):
        log_path = _write_sidecar(
            tmp_path, contacts=[], displacement_m={},
            scene={"board_object_ids": []})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is True
        assert res["accepted"] is True


class TestExperimentAcceptance:

    def test_recorded_contact_rejects_the_experiment(self, tmp_path):
        log_path = _write_sidecar(
            tmp_path,
            contacts=[{"arm_geom": "r_hand_tube", "object_id": "pool_box_1"}],
            displacement_m={"pool_box_1": 0.0})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is True
        assert res["accepted"] is False
        assert res["ok"] is False
        assert res["contacts_count"] == 1
        assert "contact" in res["reason"]

    def test_20mm_displacement_rejects_the_experiment(self, tmp_path):
        """The review's own demonstrated case: a board displaced 20 mm
        during the recording, no contact recorded."""
        log_path = _write_sidecar(
            tmp_path, contacts=[], displacement_m={"pool_box_1": 0.020})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is True
        assert res["accepted"] is False
        assert res["ok"] is False
        assert res["max_displacement_m"] == pytest.approx(0.020)
        assert "pool_box_1" in res["reason"]
        assert "20.00 mm" in res["reason"]

    def test_displacement_exactly_at_tolerance_is_accepted(self, tmp_path):
        log_path = _write_sidecar(
            tmp_path, contacts=[],
            displacement_m={"pool_box_1": gate.DISPLACEMENT_TOL_M})
        res = gate.evaluate(str(log_path))
        assert res["accepted"] is True
        assert res["ok"] is True

    def test_displacement_a_hair_over_tolerance_is_rejected(self, tmp_path):
        log_path = _write_sidecar(
            tmp_path, contacts=[],
            displacement_m={"pool_box_1": gate.DISPLACEMENT_TOL_M + 1e-9})
        res = gate.evaluate(str(log_path))
        assert res["accepted"] is False
        assert res["ok"] is False

    def test_contact_and_displacement_both_present_still_rejects(self, tmp_path):
        log_path = _write_sidecar(
            tmp_path,
            contacts=[{"arm_geom": "r_hand_tube", "object_id": "pool_box_1"}],
            displacement_m={"pool_box_1": 0.004})
        res = gate.evaluate(str(log_path))
        assert res["ok"] is False
        assert res["accepted"] is False

    def test_multiple_tracked_objects_the_worst_one_names_the_rejection(self, tmp_path):
        log_path = _write_sidecar(
            tmp_path, contacts=[],
            displacement_m={"pool_box_1": 0.0003, "pool_box_2": 0.015})
        res = gate.evaluate(str(log_path))
        assert res["accepted"] is False
        assert "pool_box_2" in res["reason"]

    def test_clean_recording_with_no_tracked_objects_is_accepted(self, tmp_path):
        log_path = _write_sidecar(tmp_path, contacts=[], displacement_m={})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is True
        assert res["accepted"] is True
        assert res["ok"] is True

    def test_clean_recording_with_small_displacement_is_accepted(self, tmp_path):
        log_path = _write_sidecar(
            tmp_path, contacts=[], displacement_m={"pool_box_1": 0.0002})
        res = gate.evaluate(str(log_path))
        assert res["evidence_ok"] is True
        assert res["accepted"] is True
        assert res["ok"] is True


class TestReadOnly:

    def test_evaluate_never_writes_the_sidecar_or_log(self, tmp_path):
        """Preserving the results: a rejected experiment's evidence must
        be left exactly as the linker wrote it, byte for byte, so the
        recording remains analyzable after the STOP."""
        log_path = _write_sidecar(
            tmp_path,
            contacts=[{"arm_geom": "r_hand_tube", "object_id": "pool_box_1"}])
        sidecar_path = log_path.with_suffix(".link.json")
        before_sidecar = sidecar_path.read_bytes()
        before_log = log_path.read_bytes()
        res = gate.evaluate(str(log_path))
        assert res["ok"] is False
        assert sidecar_path.read_bytes() == before_sidecar
        assert log_path.read_bytes() == before_log


class TestCli:

    def _run(self, log_path):
        script = os.path.join(_HERE, "../../scripts/experiment_gate.py")
        return subprocess.run(
            [sys.executable, script, str(log_path)],
            capture_output=True, text=True)

    def test_cli_exit_0_on_acceptance(self, tmp_path):
        log_path = _write_sidecar(tmp_path, contacts=[], displacement_m={})
        result = self._run(log_path)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "EXPERIMENT_ACCEPTED=yes" in result.stdout

    def test_cli_exit_1_on_invalid_evidence(self, tmp_path):
        log_path = tmp_path / "flight.json"
        log_path.write_text("{}")
        result = self._run(log_path)
        assert result.returncode == 1
        assert "EXPERIMENT_ACCEPTED=NO" in result.stdout

    def test_cli_exit_2_on_recorded_contact(self, tmp_path):
        log_path = _write_sidecar(
            tmp_path,
            contacts=[{"arm_geom": "r_hand_tube", "object_id": "pool_box_1"}])
        result = self._run(log_path)
        assert result.returncode == 2
        assert "EXPERIMENT_ACCEPTED=NO" in result.stdout

    def test_cli_exit_2_on_20mm_displacement(self, tmp_path):
        log_path = _write_sidecar(
            tmp_path, contacts=[], displacement_m={"pool_box_1": 0.020})
        result = self._run(log_path)
        assert result.returncode == 2
        assert "EXPERIMENT_ACCEPTED=NO" in result.stdout
