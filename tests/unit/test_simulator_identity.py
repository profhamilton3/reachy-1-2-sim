"""R12-801: Unit tests for SimulatorIdentity generation (identity.py)."""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from reachy_ai.experience.identity import build_simulator_identity
from reachy_ai.experience.models import SimulatorIdentity


# ---------------------------------------------------------------------------
# build_simulator_identity
# ---------------------------------------------------------------------------

class TestBuildSimulatorIdentity:
    def test_returns_simulator_identity(self):
        ident = build_simulator_identity()
        assert isinstance(ident, SimulatorIdentity)

    def test_all_19_fields_populated(self):
        ident = build_simulator_identity()
        expected = {
            "identity_version", "repository_git_sha", "working_tree_dirty",
            "model_source_path", "model_sha256", "compiled_model_sha256",
            "scene_source_path", "scene_sha256", "scene_revision",
            "scene_schema_version", "scene_compiler_version", "physics_profile_id",
            "protocol_version", "mujoco_version", "python_version", "host_os",
            "host_arch", "backend_name", "calibration_profile_id",
            "sensor_effect_profile_id",
        }
        actual = set(ident.to_dict().keys())
        assert expected <= actual, f"Missing: {expected - actual}"

    def test_no_paths_gives_empty_hashes(self):
        ident = build_simulator_identity()
        assert ident.model_sha256 == ""
        assert ident.scene_sha256 == ""

    def test_model_path_produces_sha256(self, tmp_path):
        model = tmp_path / "model.xml"
        model.write_bytes(b"<mujoco/>")
        ident = build_simulator_identity(model_path=str(model))
        assert len(ident.model_sha256) == 64  # hex sha256
        assert ident.model_source_path == str(model)

    def test_sha256_consistent_on_same_content(self, tmp_path):
        m1 = tmp_path / "a.xml"
        m2 = tmp_path / "b.xml"
        m1.write_bytes(b"<mujoco/>")
        m2.write_bytes(b"<mujoco/>")
        id1 = build_simulator_identity(model_path=str(m1))
        id2 = build_simulator_identity(model_path=str(m2))
        assert id1.model_sha256 == id2.model_sha256

    def test_sha256_differs_on_different_content(self, tmp_path):
        m1 = tmp_path / "a.xml"
        m2 = tmp_path / "b.xml"
        m1.write_bytes(b"<mujoco/>")
        m2.write_bytes(b"<mujoco version='2'/>")
        id1 = build_simulator_identity(model_path=str(m1))
        id2 = build_simulator_identity(model_path=str(m2))
        assert id1.model_sha256 != id2.model_sha256

    def test_scene_path_produces_sha256(self, tmp_path):
        scene = tmp_path / "scene.yaml"
        scene.write_text("objects: []")
        ident = build_simulator_identity(scene_path=str(scene))
        assert len(ident.scene_sha256) == 64

    def test_scene_hash_changes_on_content_change(self, tmp_path):
        scene = tmp_path / "scene.yaml"
        scene.write_text("objects: []")
        id1 = build_simulator_identity(scene_path=str(scene))
        scene.write_text("objects: []\n# changed")
        id2 = build_simulator_identity(scene_path=str(scene))
        assert id1.scene_sha256 != id2.scene_sha256

    def test_compiled_xml_hashed(self):
        xml = "<mujoco><worldbody/></mujoco>"
        ident = build_simulator_identity(compiled_xml=xml)
        assert len(ident.compiled_model_sha256) == 64

    def test_compiled_xml_deterministic(self):
        xml = "<mujoco><worldbody/></mujoco>"
        id1 = build_simulator_identity(compiled_xml=xml)
        id2 = build_simulator_identity(compiled_xml=xml)
        assert id1.compiled_model_sha256 == id2.compiled_model_sha256

    def test_compiled_xml_none_gives_empty(self):
        ident = build_simulator_identity(compiled_xml=None)
        assert ident.compiled_model_sha256 == ""

    def test_working_tree_dirty_is_bool(self):
        ident = build_simulator_identity()
        assert isinstance(ident.working_tree_dirty, bool)

    def test_host_os_and_arch_non_empty(self):
        ident = build_simulator_identity()
        assert ident.host_os != ""
        assert ident.host_arch != ""

    def test_python_version_non_empty(self):
        ident = build_simulator_identity()
        assert ident.python_version != ""

    def test_scene_revision_passed_through(self):
        ident = build_simulator_identity(scene_revision="rev-abc123")
        assert ident.scene_revision == "rev-abc123"

    def test_custom_profiles(self):
        ident = build_simulator_identity(
            physics_profile_id="high_friction",
            calibration_profile_id="lab_2025",
            sensor_effect_profile_id="no_noise",
        )
        assert ident.physics_profile_id == "high_friction"
        assert ident.calibration_profile_id == "lab_2025"
        assert ident.sensor_effect_profile_id == "no_noise"

    def test_backend_name_default(self):
        ident = build_simulator_identity()
        assert ident.backend_name == "native_mujoco"

    def test_backend_name_override(self):
        ident = build_simulator_identity(backend_name="kinematic")
        assert ident.backend_name == "kinematic"

    def test_json_round_trip(self):
        ident = build_simulator_identity(scene_revision="r1")
        recovered = SimulatorIdentity.from_json(ident.to_json())
        assert recovered == ident

    def test_missing_model_file_gives_empty_hash(self):
        ident = build_simulator_identity(model_path="/does/not/exist.xml")
        assert ident.model_sha256 == ""
        assert ident.model_source_path == "/does/not/exist.xml"

    def test_scene_schema_version_set(self):
        ident = build_simulator_identity()
        assert ident.scene_schema_version == "1.0"

    def test_scene_compiler_version_set(self):
        ident = build_simulator_identity()
        assert isinstance(ident.scene_compiler_version, str)
        assert ident.scene_compiler_version != ""
