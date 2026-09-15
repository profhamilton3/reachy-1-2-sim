"""E1 readiness (assignment 2026-09-14, work item 3): scripts/link_e1_flight.py
and the schema-3 changes to scripts/measure_route_clearance.py it depends on.

Offline throughout: every run directory is written with the REAL
`native_mujoco.recorder.Recorder`, and there is no live server or SDK
connection anywhere in this file.
"""

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, Tuple

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

import link_e1_flight as lef  # noqa: E402
import measure_route_clearance as mrc  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402
from recorder import Recorder  # noqa: E402

_BOARD_SCENE = os.path.join(_HERE, "../../scenes/e1_boards/B4_pool_box_1_r2c3.yaml")
_POSE_DEG = dict(R.REST)


def _rad_joints(pose_deg: Dict[str, float]) -> list:
    return [{"name": name, "position_rad": math.radians(pose_deg[name])}
            for name in R.R_JOINTS]


def _state(seq, sim_step, wall_time_ns, pose_deg=_POSE_DEG, objects=None,
          contacts=None):
    state = {
        "type": "state", "seq": seq, "sim_step": sim_step, "sim_time_s": 0.1,
        "wall_time_ns": wall_time_ns, "scene_revision": "r1",
        "joints": _rad_joints(pose_deg), "objects": objects or [], "grippers": [],
    }
    if contacts is not None:
        state["contacts"] = contacts
    return state


def _contact(sim_step, object_id="pool_box_1", arm_geom="r_forearm_col"):
    """One `ContactAccumulator.drain()`-shaped entry, tagged by the
    state's own sim_step so a test can tell which state's window it came
    from after the union."""
    return {
        "arm_geom": arm_geom, "object_id": object_id, "steps": 3,
        "max_normal_force_n": 1.5, "min_dist_m": -0.001,
        "first_sim_step": sim_step, "last_sim_step": sim_step,
        "pos_at_max_force": [0.1, 0.2, 0.3],
    }


def _write_states(tmp_path, states):
    rec = Recorder.new(tmp_path, {"scene_path": _BOARD_SCENE})
    for s in states:
        rec.record_state(s)
    rec.finalize(total_steps=len(states), duration_s=0.1)
    return rec.run_dir


# B1 fixture: reproduces the review's repro (50 states at 50 Hz, 20 recorder
# samples at ~20 Hz, 10 states carrying a contact) -- but with the aligned
# window's first/last index NOT at the ends of the state stream (2..46 of
# 0..49), so states 0-1 and 47-49 are genuinely OUTSIDE the flight window
# and double as the "flight-boundary" exclusion case.
_CW_BASE_WALL_NS = 5_000_000_000
_CW_STEP_NS = 20_000_000  # 50 Hz server push
_CW_N_STATES = 50
_CW_FIRST_ALIGNED_IDX = 2
_CW_LAST_ALIGNED_IDX = 46
_CW_CONTACT_IDXS = (2, 5, 9, 14, 20, 26, 33, 38, 42, 46)  # 10 windows


def _cw_aligned_idxs():
    lo, hi = _CW_FIRST_ALIGNED_IDX, _CW_LAST_ALIGNED_IDX
    return [lo + round(k * (hi - lo) / 19) for k in range(20)]


def _cw_run_dir(tmp_path, contact_idxs=None):
    contact_idxs = set(_CW_CONTACT_IDXS if contact_idxs is None else contact_idxs)
    states = [
        _state(i, 100 + i, _CW_BASE_WALL_NS + i * _CW_STEP_NS,
              contacts=([_contact(100 + i)] if i in contact_idxs else None))
        for i in range(_CW_N_STATES)
    ]
    return _write_states(tmp_path, states)


def _cw_samples():
    return [
        {"t": k * 0.05, "wall_time_ns": _CW_BASE_WALL_NS + idx * _CW_STEP_NS,
         "joints": dict(_POSE_DEG)}
        for k, idx in enumerate(_cw_aligned_idxs())
    ]


@dataclass(frozen=True)
class _FakeIdentity:
    """Stands in for e1_identity.SimulatorIdentityCheck -- only the
    attributes build_base_sidecar actually reads."""
    manifest: dict = field(default_factory=dict)
    scene_chain_sha256: Dict[str, str] = field(default_factory=dict)

    def as_dict(self):
        return {"ok": True, "manifest": self.manifest,
               "scene_chain_sha256": self.scene_chain_sha256}


class TestAlignment:
    def test_alignment_picks_the_intended_state(self, tmp_path):
        pose_a = dict(_POSE_DEG)
        pose_b = {**_POSE_DEG, "r_wrist_roll": _POSE_DEG["r_wrist_roll"] + 20.0}
        run_dir = _write_states(tmp_path, [
            _state(1, 100, 1_000_000_000, pose_deg=pose_b),
            _state(2, 101, 1_000_050_000, pose_deg=pose_a),
        ])
        states = lef.read_states(run_dir)
        sample = {"wall_time_ns": 1_000_050_000, "joints": pose_a}
        chosen, wall_offset_s, residual_deg = lef.align_sample(sample, states)
        assert chosen["sim_step"] == 101
        assert residual_deg == pytest.approx(0.0, abs=1e-9)
        assert wall_offset_s == pytest.approx(0.0, abs=1e-9)

    def test_40ms_wall_skew_is_corrected_by_joint_refinement(self, tmp_path):
        """The chronologically-NEAREST state has the WRONG joints; a state
        40ms further away (but still inside the +/-60ms window) has the
        RIGHT joints. Alignment must pick the right one."""
        pose_a = dict(_POSE_DEG)
        pose_wrong = {**_POSE_DEG, "r_elbow_pitch": _POSE_DEG["r_elbow_pitch"] - 30.0}
        sample_wall = 5_000_000_000
        run_dir = _write_states(tmp_path, [
            _state(1, 100, sample_wall, pose_deg=pose_wrong),        # nearest in time
            _state(2, 101, sample_wall + 40_000_000, pose_deg=pose_a),  # right joints
        ])
        states = lef.read_states(run_dir)
        sample = {"wall_time_ns": sample_wall, "joints": pose_a}
        chosen, wall_offset_s, residual_deg = lef.align_sample(sample, states)
        assert chosen["sim_step"] == 101, (
            "nearest-by-wall-time alone would have picked sim_step=100 "
            "(the wrong joints) -- refinement must override it")
        assert residual_deg == pytest.approx(0.0, abs=1e-9)
        assert wall_offset_s == pytest.approx(0.040, abs=1e-9)

    def test_align_samples_produces_one_record_per_sample(self, tmp_path):
        run_dir = _write_states(tmp_path, [_state(1, 100, 1_000_000_000)])
        states = lef.read_states(run_dir)
        samples = [{"wall_time_ns": 1_000_000_000, "joints": dict(_POSE_DEG)}
                  for _ in range(3)]
        alignment = lef.align_samples(samples, states)
        assert len(alignment) == 3
        assert all(r["server_sim_step"] == 100 for r in alignment)
        assert all(r["sample_index"] == i for i, r in enumerate(alignment))


class TestContactsFullWindow:
    """B1 regression: `align_and_recompute` must report every state's
    contacts across the whole aligned window, not just the ~20 Hz states
    an individual recorder sample happened to land on."""

    def test_all_ten_contact_windows_are_reported(self, tmp_path):
        run_dir = _cw_run_dir(tmp_path)
        block = lef.align_and_recompute(_cw_samples(), run_dir, _BOARD_SCENE)
        assert block["contacts_recorded"] is True
        reported = sorted(c["first_sim_step"] for c in block["contacts"])
        expected = sorted(100 + i for i in _CW_CONTACT_IDXS)
        assert reported == expected
        assert len(block["contacts"]) == 10

    def test_contacts_between_recorder_samples_are_not_dropped(self, tmp_path):
        """The specific bug: a contact on a state that is nobody's
        alignment target (falls strictly between two 20 Hz samples) used
        to be silently dropped -- see contacts_if_only_aligned below."""
        aligned = set(_cw_aligned_idxs())
        between = [i for i in _CW_CONTACT_IDXS if i not in aligned]
        assert between, "fixture must exercise at least one non-aligned index"

        run_dir = _cw_run_dir(tmp_path)
        block = lef.align_and_recompute(_cw_samples(), run_dir, _BOARD_SCENE)
        reported = {c["first_sim_step"] for c in block["contacts"]}
        for i in between:
            assert 100 + i in reported, f"contact at index {i} (between samples) was dropped"

    def test_flight_boundary_included_and_out_of_range_excluded(self, tmp_path):
        """Contacts exactly at the first/last ALIGNED state are included
        (inclusive bounds); contacts outside [first, last] aligned
        sim_step -- before the flight starts or after it ends -- are not."""
        boundary_idxs = {
            "before": 0, "at_first": _CW_FIRST_ALIGNED_IDX,
            "at_last": _CW_LAST_ALIGNED_IDX, "after": _CW_N_STATES - 1,
        }
        assert boundary_idxs["before"] < _CW_FIRST_ALIGNED_IDX
        assert boundary_idxs["after"] > _CW_LAST_ALIGNED_IDX

        run_dir = _cw_run_dir(tmp_path, contact_idxs=boundary_idxs.values())
        block = lef.align_and_recompute(_cw_samples(), run_dir, _BOARD_SCENE)
        reported = {c["first_sim_step"] for c in block["contacts"]}
        assert 100 + boundary_idxs["at_first"] in reported
        assert 100 + boundary_idxs["at_last"] in reported
        assert 100 + boundary_idxs["before"] not in reported
        assert 100 + boundary_idxs["after"] not in reported

    def test_no_double_counting(self, tmp_path):
        run_dir = _cw_run_dir(tmp_path)
        block = lef.align_and_recompute(_cw_samples(), run_dir, _BOARD_SCENE)
        seen = [(c["arm_geom"], c["object_id"], c["first_sim_step"])
               for c in block["contacts"]]
        assert len(seen) == len(set(seen))

    def test_contacts_if_only_aligned_states_were_used_would_miss_most(self, tmp_path):
        """Documents the bug being fixed: filtering to only the ALIGNED
        states' own contacts (the old behaviour) finds just the overlap
        between contact-bearing indices and aligned indices -- fewer than
        the full ten. Guards against a future regression back to that
        narrower read."""
        aligned = set(_cw_aligned_idxs())
        only_aligned_overlap = [i for i in _CW_CONTACT_IDXS if i in aligned]
        assert 0 < len(only_aligned_overlap) < len(_CW_CONTACT_IDXS)


class TestContactEvidenceIntegrity:
    """Missing or malformed contact evidence must never silently read as
    'no contacts' -- see `align_and_recompute`'s docstring."""

    def test_missing_contacts_key_is_flagged_not_silently_zero(self, tmp_path):
        """A run recorded before work item 4 (or with --record's contact
        tracking off) has no 'contacts' key on any state at all."""
        states = [_state(i, 100 + i, _CW_BASE_WALL_NS + i * _CW_STEP_NS)
                 for i in range(5)]
        run_dir = _write_states(tmp_path, states)
        samples = [{"wall_time_ns": _CW_BASE_WALL_NS + i * _CW_STEP_NS,
                   "joints": dict(_POSE_DEG)} for i in range(5)]
        block = lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)
        assert block["contacts"] == []
        assert block["contacts_recorded"] is False

    def test_genuine_zero_contacts_is_distinguished_from_missing(self, tmp_path):
        """The key IS present on every state (contact tracking covered
        this run) but nothing ever touched anything -- a real zero,
        distinct from the missing-evidence case above."""
        states = [_state(i, 100 + i, _CW_BASE_WALL_NS + i * _CW_STEP_NS, contacts=[])
                 for i in range(5)]
        run_dir = _write_states(tmp_path, states)
        samples = [{"wall_time_ns": _CW_BASE_WALL_NS + i * _CW_STEP_NS,
                   "joints": dict(_POSE_DEG)} for i in range(5)]
        block = lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)
        assert block["contacts"] == []
        assert block["contacts_recorded"] is True

    def test_malformed_contacts_field_type_raises(self, tmp_path):
        states = [_state(0, 100, _CW_BASE_WALL_NS, contacts="not-a-list")]
        run_dir = _write_states(tmp_path, states)
        samples = [{"wall_time_ns": _CW_BASE_WALL_NS, "joints": dict(_POSE_DEG)}]
        with pytest.raises(lef.ContactEvidenceError):
            lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)

    def test_malformed_contact_entry_raises(self, tmp_path):
        states = [_state(0, 100, _CW_BASE_WALL_NS, contacts=[{"bogus": True}])]
        run_dir = _write_states(tmp_path, states)
        samples = [{"wall_time_ns": _CW_BASE_WALL_NS, "joints": dict(_POSE_DEG)}]
        with pytest.raises(lef.ContactEvidenceError):
            lef.align_and_recompute(samples, run_dir, _BOARD_SCENE)


def _cw_write_log_and_base_sidecar(tmp_path, monkeypatch):
    """Full setup for the end-to-end tests: a real run dir (10 contact
    windows, per `_cw_run_dir`), a real schema-3 log via `save_log`, and
    the base sidecar `measure_route_clearance.main()` would have written
    right after the flight -- everything `link_flight` needs on disk."""
    monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path / "runs")
    run_dir = _cw_run_dir(tmp_path)
    samples = _cw_samples()
    log_path = mrc.save_log(samples, "LOWER_TO_REST", str(_BOARD_SCENE),
                            t0_wall_ns=samples[0]["wall_time_ns"])
    identity = _FakeIdentity(manifest={"scene_revision": "r1"},
                             scene_chain_sha256={"a": "sha_a"})
    base_sidecar = lef.build_base_sidecar(
        log_path=str(log_path), samples=samples, scene_path=_BOARD_SCENE,
        identity_check=identity, run_dir=str(run_dir))
    lef.write_sidecar(str(log_path), base_sidecar)
    return log_path, run_dir


class TestContactsThroughLinkFlight:
    """End to end: `link_flight()` (the CLI's own entry point) and the
    actual CLI subprocess, both reading only the log + its base sidecar +
    the run dir -- no live server, matching how the checklist uses it."""

    def test_link_flight_writes_all_ten_contacts(self, tmp_path, monkeypatch):
        log_path, _ = _cw_write_log_and_base_sidecar(tmp_path, monkeypatch)
        written = lef.link_flight(str(log_path))
        sidecar = json.loads(written.read_text())
        assert sidecar["contacts_recorded"] is True
        assert len(sidecar["contacts"]) == 10
        reported = {c["first_sim_step"] for c in sidecar["contacts"]}
        assert reported == {100 + i for i in _CW_CONTACT_IDXS}

    def test_link_flight_run_dir_override_also_finds_all_ten(self, tmp_path, monkeypatch):
        """`--run-dir` (retained artefacts moved to a new path) must not
        change the contact result."""
        log_path, run_dir = _cw_write_log_and_base_sidecar(tmp_path, monkeypatch)
        written = lef.link_flight(str(log_path), run_dir=str(run_dir))
        sidecar = json.loads(written.read_text())
        assert len(sidecar["contacts"]) == 10

    def test_cli_subprocess_writes_all_ten_contacts(self, tmp_path, monkeypatch):
        """The literal command the checklist runs:
        `python3 scripts/link_e1_flight.py <log_path>`."""
        log_path, _ = _cw_write_log_and_base_sidecar(tmp_path, monkeypatch)
        script = os.path.join(_HERE, "../../scripts/link_e1_flight.py")
        result = subprocess.run(
            [sys.executable, script, str(log_path)],
            capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "Wrote" in result.stdout

        sidecar_path = lef.sidecar_path_for(str(log_path))
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["contacts_recorded"] is True
        assert len(sidecar["contacts"]) == 10
        reported = {c["first_sim_step"] for c in sidecar["contacts"]}
        assert reported == {100 + i for i in _CW_CONTACT_IDXS}


class TestSettledPoseCheck:
    def test_passes_at_4mm(self):
        expected = {"pool_box_1": (0.4318, -0.1524, 0.76)}
        state = _state(1, 100, 0, objects=[
            {"object_id": "pool_box_1", "pos_xyz": [0.4318 + 0.004, -0.1524, 0.76]}])
        result = lef.settled_pose_check(["pool_box_1"], expected, state)
        assert result["pool_box_1"]["ok"] is True
        assert result["pool_box_1"]["dxy_m"] == pytest.approx(0.004, abs=1e-9)

    def test_fails_at_6mm(self):
        expected = {"pool_box_1": (0.4318, -0.1524, 0.76)}
        state = _state(1, 100, 0, objects=[
            {"object_id": "pool_box_1", "pos_xyz": [0.4318 + 0.006, -0.1524, 0.76]}])
        result = lef.settled_pose_check(["pool_box_1"], expected, state)
        assert result["pool_box_1"]["ok"] is False
        assert result["pool_box_1"]["dxy_m"] == pytest.approx(0.006, abs=1e-9)
        assert result["pool_box_1"]["yaml_xyz"] == [0.4318, -0.1524, 0.76]
        assert result["pool_box_1"]["stream_xyz"][0] == pytest.approx(0.4318 + 0.006)


class TestBuildBaseSidecar:
    def test_displacement_and_shape(self, tmp_path):
        yaml_pose = tuple(SceneModel.from_yaml(_BOARD_SCENE).get("pool_box_1").center)
        first_pose = list(yaml_pose)
        last_pose = [yaml_pose[0] + 0.01, yaml_pose[1], yaml_pose[2]]  # 1 cm drift
        run_dir = _write_states(tmp_path, [
            _state(1, 100, 1_000_000_000,
                  objects=[{"object_id": "pool_box_1", "pos_xyz": first_pose}]),
            _state(2, 101, 1_000_020_000,
                  objects=[{"object_id": "pool_box_1", "pos_xyz": last_pose}]),
        ])
        samples = [
            {"t": 0.0, "wall_time_ns": 1_000_000_000, "joints": dict(_POSE_DEG)},
            {"t": 0.02, "wall_time_ns": 1_000_020_000, "joints": dict(_POSE_DEG)},
        ]
        identity = _FakeIdentity(manifest={"scene_revision": "r1"},
                                 scene_chain_sha256={"a": "sha_a"})
        sidecar = lef.build_base_sidecar(
            log_path=str(tmp_path / "route_clearance_LOWER_TO_REST_x.json"),
            samples=samples, scene_path=_BOARD_SCENE,
            identity_check=identity, run_dir=str(run_dir))

        _assert_sidecar_shape(sidecar)
        assert sidecar["displacement_m"]["pool_box_1"] == pytest.approx(0.01, abs=1e-9)
        assert sidecar["settled_pose_check"]["pool_box_1"]["ok"] is True
        assert sidecar["scene"]["board_object_ids"] == ["pool_box_1"]
        assert sidecar["no_reshape_asserted"] is True


def _assert_sidecar_shape(sidecar: dict) -> None:
    """A small JSON-schema-like assertion set: every key the E1 readiness
    plan's sidecar example names is present, with the right shape."""
    for key in ("log", "identity", "server_run_dir", "manifest", "scene",
               "objects_at_first_sample", "objects_at_last_sample",
               "settled_pose_check", "displacement_m", "no_reshape_asserted"):
        assert key in sidecar, f"sidecar missing {key!r}"
    assert isinstance(sidecar["log"], str)
    assert isinstance(sidecar["identity"], dict)
    assert isinstance(sidecar["server_run_dir"], str)
    assert isinstance(sidecar["manifest"], dict)
    assert isinstance(sidecar["scene"], dict)
    for k in ("path", "chain_sha256", "board_object_ids"):
        assert k in sidecar["scene"]
    assert isinstance(sidecar["objects_at_first_sample"], dict)
    assert isinstance(sidecar["objects_at_last_sample"], dict)
    for oid, snap in sidecar["objects_at_first_sample"].items():
        assert set(snap) >= {"pos_xyz", "quat_wxyz", "sim_step", "wall_time_ns"}
    assert isinstance(sidecar["settled_pose_check"], dict)
    for oid, check in sidecar["settled_pose_check"].items():
        assert "ok" in check
    assert isinstance(sidecar["displacement_m"], dict)
    assert isinstance(sidecar["no_reshape_asserted"], bool)


class TestSidecarRoundTrip:
    def test_write_and_read_sidecar(self, tmp_path):
        log_path = tmp_path / "route_clearance_LOWER_TO_REST_x.json"
        log_path.write_text("{}")
        sidecar = {"log": log_path.name, "identity": {"ok": True},
                  "server_run_dir": "x", "manifest": {}, "scene": {},
                  "objects_at_first_sample": {}, "objects_at_last_sample": {},
                  "settled_pose_check": {}, "displacement_m": {},
                  "no_reshape_asserted": True}
        written = lef.write_sidecar(str(log_path), sidecar)
        assert written.name == "route_clearance_LOWER_TO_REST_x.link.json"
        loaded = lef.read_sidecar(str(log_path))
        assert loaded == sidecar


class TestSchema3RoundTrip:
    def test_save_log_load_log_validated_samples(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mrc, "RUNS_DIR", tmp_path)
        samples = [
            {"t": 0.0, "wall_time_ns": 1_000_000_000,
             "joints": {**{j: R.REST[j] for j in R.ARM7}, "r_gripper": R.REST["r_gripper"]}},
            {"t": 0.05, "wall_time_ns": 1_050_000_000,
             "joints": {**{j: R.REST[j] for j in R.ARM7}, "r_gripper": R.REST["r_gripper"]}},
        ]
        path = mrc.save_log(samples, "LOWER_TO_REST",
                           str(_BOARD_SCENE), t0_wall_ns=999_000_000)
        loaded = mrc.load_log(path)
        assert loaded["schema_version"] == mrc.LOG_SCHEMA_VERSION == 3
        assert loaded["t0_wall_ns"] == 999_000_000
        q7_and_gripper, assumed = mrc.validated_samples(
            loaded["samples"], schema_version=mrc.schema_version_of(loaded))
        assert assumed == []
        assert len(q7_and_gripper) == 2

    def test_report_unchanged_by_the_schema_bump(self):
        """report()'s NUMBERS must be identical whether the log is declared
        schema 2 (no wall_time_ns) or schema 3 (has it) -- the schema bump
        changes what is REQUIRED on disk, never the aperture/clearance
        math report() already had."""
        joints = {**{j: R.REST[j] for j in R.ARM7}, "r_gripper": R.REST["r_gripper"]}
        schema_2_log = [{"t": i / 20.0, "joints": dict(joints)} for i in range(3)]
        schema_3_log = [{"t": i / 20.0, "wall_time_ns": 1_000_000_000 + i * 50_000_000,
                        "joints": dict(joints)} for i in range(3)]

        result_2 = mrc.report(schema_2_log, "LOWER_TO_REST", _BOARD_SCENE,
                             schema_version=2)
        result_3 = mrc.report(schema_3_log, "LOWER_TO_REST", _BOARD_SCENE,
                             schema_version=3)
        assert result_2["realised"] == result_3["realised"]
        assert result_2["planned"] == result_3["planned"]
        assert result_2["n_samples"] == result_3["n_samples"]
