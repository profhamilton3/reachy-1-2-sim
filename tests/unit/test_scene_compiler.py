"""Unit tests for native_mujoco/scene_compiler.py (R12-402 / R12-503).

Field names follow the real scenes/scene.schema.json:
  geometry.size / radius / height, pose.position / orientation_wxyz / rpy,
  material.rgba, physics.dynamic / mass / friction, tracked.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../native_mujoco"))

from scene_compiler import (  # noqa: E402
    SceneCompilerError,
    compile_scene,
    compile_scene_body_fragment,
    dynamic_object_ids,
    tracked_object_ids,
)


def _box(**over):
    obj = {
        "id": "test_box",
        "geometry": {"kind": "box", "size": [0.05, 0.08, 0.03]},
        "pose": {"position": [0.5, 0.0, 0.77], "orientation_wxyz": [1, 0, 0, 0]},
        "material": {"rgba": [0.8, 0.2, 0.1, 1.0]},
    }
    obj.update(over)
    return {"objects": [obj]}


class TestGeometry:
    def test_box_half_extents(self):
        xml = compile_scene(_box())
        # size 0.05,0.08,0.03 -> half 0.025,0.040,0.015
        assert "0.025000" in xml and "0.040000" in xml and "0.015000" in xml

    def test_box_wrong_size_raises(self):
        with pytest.raises(SceneCompilerError, match="3 elements"):
            compile_scene(_box(geometry={"kind": "box", "size": [0.1, 0.1]}))

    def test_sphere(self):
        doc = {"objects": [{"id": "b", "geometry": {"kind": "sphere", "radius": 0.03},
                            "pose": {"position": [0, 0, 0]}}]}
        assert 'type="sphere"' in compile_scene(doc) and "0.030000" in compile_scene(doc)

    def test_cylinder_half_height(self):
        doc = {"objects": [{"id": "c", "geometry": {"kind": "cylinder", "radius": 0.025, "length": 0.12},
                            "pose": {"position": [0, 0, 0]}}]}
        xml = compile_scene(doc)
        assert 'type="cylinder"' in xml and "0.025000" in xml and "0.060000" in xml

    def test_mesh_asset(self):
        doc = {"objects": [{"id": "mug", "geometry": {"kind": "mesh", "path": "assets/mug.obj"},
                            "pose": {"position": [0, 0, 0]}}]}
        asset, _ = compile_scene_body_fragment(doc)
        assert "<mesh" in asset and "mug.obj" in asset

    def test_unknown_kind_raises(self):
        with pytest.raises(SceneCompilerError, match="unknown geometry kind"):
            compile_scene(_box(geometry={"kind": "torus"}))


class TestPose:
    def test_position(self):
        xml = compile_scene(_box())
        assert "0.500000" in xml and "0.770000" in xml

    def test_orientation_wxyz(self):
        xml = compile_scene(_box(pose={"position": [0, 0, 0],
                                       "orientation_wxyz": [0.707107, 0, 0, 0.707107]}))
        assert "0.707107" in xml

    def test_rpy_converted_to_quat(self):
        # yaw 90deg -> wxyz (0.7071, 0, 0, 0.7071)
        xml = compile_scene(_box(pose={"position": [0, 0, 0], "rpy": [0, 0, math.pi / 2]}))
        assert "0.707107" in xml

    def test_default_identity_quat(self):
        xml = compile_scene(_box(pose={"position": [1, 2, 3]}))
        assert 'quat="1.000000 0.000000 0.000000 0.000000"' in xml


class TestMaterial:
    def test_rgba(self):
        xml = compile_scene(_box())
        assert "0.800" in xml and "0.200" in xml and "0.100" in xml


class TestPhysics:
    def test_static_default_no_freejoint(self):
        xml = compile_scene(_box())
        assert "<freejoint/>" not in xml

    def test_dynamic_has_freejoint(self):
        xml = compile_scene(_box(physics={"dynamic": True, "mass": 0.1}))
        assert "<freejoint/>" in xml

    def test_dynamic_mass(self):
        xml = compile_scene(_box(physics={"dynamic": True, "mass": 0.123}))
        assert 'mass="0.123000"' in xml

    def test_object_collision_channel(self):
        xml = compile_scene(_box(physics={"dynamic": True, "collision": True}))
        assert 'contype="4" conaffinity="7"' in xml

    def test_no_collision_disables_contact(self):
        xml = compile_scene(_box(physics={"dynamic": False, "collision": False}))
        assert 'contype="0" conaffinity="0"' in xml

    def test_friction_emitted(self):
        xml = compile_scene(_box(physics={"dynamic": True, "collision": True,
                                          "friction": [0.8, 0.01, 0.001]}))
        assert 'friction="0.8000 0.0100 0.0010"' in xml


class TestObjectClassifiers:
    def test_tracked_ids(self):
        doc = {"objects": [
            {"id": "a", "geometry": {"kind": "sphere", "radius": 0.01},
             "pose": {"position": [0, 0, 0]}, "tracked": True},
            {"id": "b", "geometry": {"kind": "sphere", "radius": 0.01},
             "pose": {"position": [0, 0, 0]}, "tracked": False},
            {"id": "c", "geometry": {"kind": "sphere", "radius": 0.01},
             "pose": {"position": [0, 0, 0]}},
        ]}
        assert tracked_object_ids(doc) == ["a"]

    def test_dynamic_ids(self):
        doc = {"objects": [
            {"id": "a", "geometry": {"kind": "sphere", "radius": 0.01},
             "pose": {"position": [0, 0, 0]}, "physics": {"dynamic": True}},
            {"id": "b", "geometry": {"kind": "sphere", "radius": 0.01},
             "pose": {"position": [0, 0, 0]}, "physics": {"dynamic": False}},
        ]}
        assert dynamic_object_ids(doc) == ["a"]


class TestFixtureCollisionChannel:
    """Gate 8-C: 'fixture' collision value emits the correct contact channel."""

    def test_fixture_emits_fixture_contype(self):
        xml = compile_scene(_box(physics={"dynamic": False, "collision": "fixture"}))
        assert 'contype="8"' in xml, "fixture should use contype=8"
        assert 'conaffinity="2"' in xml, "fixture should use conaffinity=2"

    def test_fixture_does_not_emit_object_channel(self):
        xml = compile_scene(_box(physics={"dynamic": False, "collision": "fixture"}))
        assert 'contype="4"' not in xml
        assert 'contype="0"' not in xml

    def test_fixture_collides_with_robot(self):
        """MuJoCo contact fires when (A.contype & B.conaffinity) OR (B.contype & A.conaffinity) != 0."""
        from scene_compiler import _FIXTURE_CONTYPE, _FIXTURE_CONAFFINITY
        robot_contype, robot_conaffinity = 2, 5  # R12-500
        # Bidirectional check (MuJoCo ORs both directions).
        assert ((_FIXTURE_CONTYPE & robot_conaffinity) | (robot_contype & _FIXTURE_CONAFFINITY)) != 0, (
            "Fixture channel must collide with robot links (contype=2, conaffinity=5)"
        )

    def test_fixture_does_not_collide_with_control_conaffinity(self):
        """(fixture_contype & obj_conaffinity) == 0: fixture must not collide with controls."""
        from scene_compiler import _FIXTURE_CONTYPE, _OBJ_CONAFFINITY
        assert (_FIXTURE_CONTYPE & _OBJ_CONAFFINITY) == 0

    def test_control_does_not_collide_with_fixture_conaffinity(self):
        """(obj_contype & fixture_conaffinity) == 0: controls must not collide with fixture."""
        from scene_compiler import _FIXTURE_CONAFFINITY, _OBJ_CONTYPE
        assert (_OBJ_CONTYPE & _FIXTURE_CONAFFINITY) == 0

    def test_false_still_disables_collision(self):
        xml = compile_scene(_box(physics={"dynamic": False, "collision": False}))
        assert 'contype="0" conaffinity="0"' in xml

    def test_true_still_uses_object_channel(self):
        xml = compile_scene(_box(physics={"dynamic": False, "collision": True}))
        assert 'contype="4" conaffinity="7"' in xml


class TestStructure:
    def test_empty_objects(self):
        xml = compile_scene({"objects": []})
        assert "<worldbody>" in xml and "</worldbody>" in xml

    def test_body_name(self):
        assert 'name="test_box"' in compile_scene(_box())

    def test_multiple_objects(self):
        doc = {"objects": [
            {"id": "a", "geometry": {"kind": "sphere", "radius": 0.01}, "pose": {"position": [0, 0, 0]}},
            {"id": "b", "geometry": {"kind": "sphere", "radius": 0.01}, "pose": {"position": [1, 0, 0]}},
        ]}
        xml = compile_scene(doc)
        assert xml.count("<body ") == 2

    def test_xml_escaping(self):
        xml = compile_scene(_box(id='o<"&'))
        assert "&amp;" in xml or "&quot;" in xml or "&lt;" in xml
