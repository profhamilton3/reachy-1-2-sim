"""E1 readiness (assignment 2026-09-14, work item 3): scripts/link_e1_flight.py
and the schema-3 changes to scripts/measure_route_clearance.py it depends on.

Offline throughout: every run directory is written with the REAL
`native_mujoco.recorder.Recorder`, and there is no live server or SDK
connection anywhere in this file.
"""

import json
import math
import os
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


def _state(seq, sim_step, wall_time_ns, pose_deg=_POSE_DEG, objects=None):
    return {
        "type": "state", "seq": seq, "sim_step": sim_step, "sim_time_s": 0.1,
        "wall_time_ns": wall_time_ns, "scene_revision": "r1",
        "joints": _rad_joints(pose_deg), "objects": objects or [], "grippers": [],
    }


def _write_states(tmp_path, states):
    rec = Recorder.new(tmp_path, {"scene_path": _BOARD_SCENE})
    for s in states:
        rec.record_state(s)
    rec.finalize(total_steps=len(states), duration_s=0.1)
    return rec.run_dir


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
