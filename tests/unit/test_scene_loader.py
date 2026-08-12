"""Unit tests for scene_loader.py (R12-200).

Tests run offline — no ROS, no gRPC, no Docker.

Exit gate: tabletop.example.yaml validates; duplicate IDs, invalid quaternion,
missing mesh, and path traversal all fail with clear errors.
"""

from __future__ import annotations

import math
import os
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scene_loader import (
    SceneDocument,
    SceneValidationError,
    _rpy_to_wxyz,
    load_scene,
)

REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLE_SCENE = REPO_ROOT / "scenes" / "tabletop.example.yaml"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_scene(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test_scene.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def _minimal_scene(**overrides) -> str:
    """Return a valid minimal scene YAML string."""
    base = {
        "schema_version": "1.0",
        "name": "test-scene",
        "frame_id": "pedestal",
        "world": {"gravity": [0.0, 0.0, -9.81]},
        "objects": [],
    }
    base.update(overrides)
    return yaml.dump(base)


def _box_object(obj_id="box1", position=None, orientation=None, rpy=None):
    pose = {"position": position or [0.5, 0.0, 0.5]}
    if orientation is not None:
        pose["orientation_wxyz"] = orientation
    elif rpy is not None:
        pose["rpy"] = rpy
    else:
        pose["orientation_wxyz"] = [1.0, 0.0, 0.0, 0.0]
    return {
        "id": obj_id,
        "geometry": {"kind": "box", "size": [0.1, 0.1, 0.1]},
        "pose": pose,
    }


# ── R12-200: Valid scene loads ────────────────────────────────────────────────

class TestValidScene:
    def test_example_scene_validates(self):
        """tabletop.example.yaml must pass all validation checks."""
        doc = load_scene(EXAMPLE_SCENE)
        assert isinstance(doc, SceneDocument)
        assert doc.name == "tabletop-calibration"
        assert doc.frame_id == "pedestal"
        assert doc.seed == 42

    def test_example_scene_object_count(self):
        doc = load_scene(EXAMPLE_SCENE)
        assert len(doc.objects) == 4

    def test_example_scene_object_ids(self):
        doc = load_scene(EXAMPLE_SCENE)
        ids = {o.id for o in doc.objects}
        assert ids == {"table_top", "red_cube", "blue_cylinder", "calibration_board"}

    def test_example_scene_box_geometry(self):
        doc = load_scene(EXAMPLE_SCENE)
        table = next(o for o in doc.objects if o.id == "table_top")
        assert table.geometry.kind == "box"
        assert table.geometry.size == pytest.approx((0.90, 0.65, 0.04), abs=1e-6)

    def test_example_scene_cylinder_geometry(self):
        doc = load_scene(EXAMPLE_SCENE)
        cyl = next(o for o in doc.objects if o.id == "blue_cylinder")
        assert cyl.geometry.kind == "cylinder"
        assert cyl.geometry.radius == pytest.approx(0.035, abs=1e-6)
        assert cyl.geometry.length == pytest.approx(0.10, abs=1e-6)

    def test_example_scene_quaternion_pose(self):
        doc = load_scene(EXAMPLE_SCENE)
        table = next(o for o in doc.objects if o.id == "table_top")
        w, x, y, z = table.pose.orientation_wxyz
        assert abs(w - 1.0) < 0.001
        assert abs(x) < 0.001
        assert abs(y) < 0.001
        assert abs(z) < 0.001

    def test_example_scene_rpy_pose(self):
        """red_cube uses rpy=[0, 0, 0.15] — converted to quaternion."""
        doc = load_scene(EXAMPLE_SCENE)
        cube = next(o for o in doc.objects if o.id == "red_cube")
        w, x, y, z = cube.pose.orientation_wxyz
        norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
        assert abs(norm - 1.0) < 0.001

    def test_example_scene_tracked_flags(self):
        doc = load_scene(EXAMPLE_SCENE)
        tracked = {o.id for o in doc.objects if o.tracked}
        assert tracked == {"red_cube", "blue_cylinder"}

    def test_example_scene_dynamic_flags(self):
        doc = load_scene(EXAMPLE_SCENE)
        dynamic = {o.id for o in doc.objects if o.dynamic}
        assert dynamic == {"red_cube", "blue_cylinder"}

    def test_example_scene_material_rgba(self):
        doc = load_scene(EXAMPLE_SCENE)
        table = next(o for o in doc.objects if o.id == "table_top")
        r, g, b, a = table.material.rgba
        assert a == pytest.approx(1.0, abs=1e-3)

    def test_minimal_valid_scene(self, tmp_path):
        p = _write_scene(tmp_path, _minimal_scene())
        doc = load_scene(p)
        assert doc.name == "test-scene"
        assert len(doc.objects) == 0

    def test_scene_with_box_object(self, tmp_path):
        content = yaml.dump({
            "schema_version": "1.0",
            "name": "s",
            "world": {"gravity": [0.0, 0.0, -9.81]},
            "objects": [_box_object()],
        })
        doc = load_scene(_write_scene(tmp_path, content))
        assert len(doc.objects) == 1
        assert doc.objects[0].id == "box1"


# ── R12-200: Invalid inputs ───────────────────────────────────────────────────

class TestInvalidInputs:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_scene(tmp_path / "nonexistent.yaml")

    def test_duplicate_object_id(self, tmp_path):
        content = yaml.dump({
            "schema_version": "1.0",
            "name": "s",
            "world": {"gravity": [0.0, 0.0, -9.81]},
            "objects": [_box_object("dup"), _box_object("dup")],
        })
        with pytest.raises(SceneValidationError, match="[Dd]uplicate"):
            load_scene(_write_scene(tmp_path, content))

    def test_invalid_quaternion_norm(self, tmp_path):
        content = yaml.dump({
            "schema_version": "1.0",
            "name": "s",
            "world": {"gravity": [0.0, 0.0, -9.81]},
            "objects": [_box_object(orientation=[2.0, 0.0, 0.0, 0.0])],
        })
        with pytest.raises(SceneValidationError, match="[Qq]uaternion|norm"):
            load_scene(_write_scene(tmp_path, content))

    def test_zero_quaternion_rejected(self, tmp_path):
        content = yaml.dump({
            "schema_version": "1.0",
            "name": "s",
            "world": {"gravity": [0.0, 0.0, -9.81]},
            "objects": [_box_object(orientation=[0.0, 0.0, 0.0, 0.0])],
        })
        with pytest.raises(SceneValidationError, match="[Qq]uaternion|norm"):
            load_scene(_write_scene(tmp_path, content))

    def test_path_traversal_rejected(self, tmp_path):
        content = yaml.dump({
            "schema_version": "1.0",
            "name": "s",
            "world": {"gravity": [0.0, 0.0, -9.81]},
            "objects": [{
                "id": "m1",
                "geometry": {"kind": "mesh", "path": "../../../etc/passwd"},
                "pose": {"position": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0]},
            }],
        })
        with pytest.raises(SceneValidationError, match="[Pp]ath|traversal|escape"):
            load_scene(_write_scene(tmp_path, content))

    def test_remote_url_rejected(self, tmp_path):
        content = yaml.dump({
            "schema_version": "1.0",
            "name": "s",
            "world": {"gravity": [0.0, 0.0, -9.81]},
            "objects": [{
                "id": "m1",
                "geometry": {"kind": "mesh", "path": "https://evil.example/mesh.stl"},
                "pose": {"position": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0]},
            }],
        })
        with pytest.raises(SceneValidationError, match="[Rr]emote|URL"):
            load_scene(_write_scene(tmp_path, content))

    def test_unknown_schema_version(self, tmp_path):
        content = yaml.dump({
            "schema_version": "99.0",
            "name": "s",
            "world": {"gravity": [0.0, 0.0, -9.81]},
            "objects": [],
        })
        with pytest.raises(SceneValidationError):
            load_scene(_write_scene(tmp_path, content))

    def test_missing_required_name(self, tmp_path):
        content = yaml.dump({
            "schema_version": "1.0",
            "world": {"gravity": [0.0, 0.0, -9.81]},
            "objects": [],
        })
        with pytest.raises(SceneValidationError):
            load_scene(_write_scene(tmp_path, content))

    def test_missing_required_world(self, tmp_path):
        content = yaml.dump({
            "schema_version": "1.0",
            "name": "s",
            "objects": [],
        })
        with pytest.raises(SceneValidationError):
            load_scene(_write_scene(tmp_path, content))

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(": invalid: {\nnot yaml at all[[[")
        with pytest.raises(SceneValidationError, match="[Yy]AML"):
            load_scene(p)

    def test_non_mapping_yaml(self, tmp_path):
        p = tmp_path / "list.yaml"
        p.write_text("- item1\n- item2\n")
        with pytest.raises(SceneValidationError):
            load_scene(p)


# ── RPY → quaternion helper ───────────────────────────────────────────────────

class TestRpyToWxyz:
    def test_identity(self):
        w, x, y, z = _rpy_to_wxyz(0, 0, 0)
        assert abs(w - 1.0) < 1e-9
        assert abs(x) < 1e-9
        assert abs(y) < 1e-9
        assert abs(z) < 1e-9

    def test_norm_is_unity(self):
        for r, p, y in [(0.1, 0.2, 0.3), (1.5, -0.5, 2.0), (0, 0, math.pi)]:
            w, x, y_, z = _rpy_to_wxyz(r, p, y)
            norm = math.sqrt(w**2 + x**2 + y_**2 + z**2)
            assert abs(norm - 1.0) < 1e-9, f"norm={norm} for rpy=({r},{p},{y})"

    def test_yaw_90(self):
        w, x, y, z = _rpy_to_wxyz(0, 0, math.pi / 2)
        assert abs(w - math.cos(math.pi / 4)) < 1e-6
        assert abs(z - math.sin(math.pi / 4)) < 1e-6
