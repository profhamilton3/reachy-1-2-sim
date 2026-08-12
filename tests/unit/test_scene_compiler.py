"""Unit tests for native_mujoco/scene_compiler.py (R12-402)."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../native_mujoco"))

from scene_compiler import (
    SceneCompilerError,
    compile_scene,
    compile_scene_body_fragment,
)


def _minimal_box_doc(**kwargs) -> dict:
    obj = {
        "id": "test_box",
        "pose": {
            "position": [0.5, 0.0, 0.77],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
        },
        "geometry": {"kind": "box", "dimensions": [0.05, 0.08, 0.03]},
        "visual": {"color": [0.8, 0.2, 0.1, 1.0]},
    }
    obj.update(kwargs)
    return {"objects": [obj]}


class TestCompileSceneOutput:
    def test_returns_string(self):
        xml = compile_scene({"objects": []})
        assert isinstance(xml, str)

    def test_empty_objects(self):
        xml = compile_scene({"objects": []})
        assert "<worldbody>" in xml
        assert "</worldbody>" in xml

    def test_box_geom_present(self):
        xml = compile_scene(_minimal_box_doc())
        assert 'type="box"' in xml

    def test_box_half_extents(self):
        # dimensions=[0.05,0.08,0.03] → half-extents 0.025 0.040 0.015
        xml = compile_scene(_minimal_box_doc())
        assert "0.025000" in xml
        assert "0.040000" in xml
        assert "0.015000" in xml

    def test_body_name(self):
        xml = compile_scene(_minimal_box_doc())
        assert 'name="test_box"' in xml

    def test_pos_in_body(self):
        xml = compile_scene(_minimal_box_doc())
        assert "0.500000" in xml
        assert "0.770000" in xml

    def test_rgba_in_geom(self):
        xml = compile_scene(_minimal_box_doc())
        assert "0.800" in xml
        assert "0.200" in xml
        assert "0.100" in xml

    def test_sphere_geom(self):
        doc = {"objects": [{
            "id": "ball",
            "pose": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]},
            "geometry": {"kind": "sphere", "radius": 0.03},
        }]}
        xml = compile_scene(doc)
        assert 'type="sphere"' in xml
        assert "0.030000" in xml

    def test_cylinder_geom(self):
        doc = {"objects": [{
            "id": "can",
            "pose": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]},
            "geometry": {"kind": "cylinder", "radius": 0.025, "height": 0.12},
        }]}
        xml = compile_scene(doc)
        assert 'type="cylinder"' in xml
        assert "0.025000" in xml
        # half-height = 0.06
        assert "0.060000" in xml

    def test_mesh_asset_and_geom(self):
        doc = {"objects": [{
            "id": "mug",
            "pose": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]},
            "geometry": {"kind": "mesh", "path": "assets/mug.obj"},
        }]}
        xml = compile_scene(doc)
        assert 'type="mesh"' in xml
        assert "<asset>" in xml
        assert "mug.obj" in xml

    def test_multiple_objects(self):
        doc = {"objects": [
            {"id": "a", "pose": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]},
             "geometry": {"kind": "sphere", "radius": 0.01}},
            {"id": "b", "pose": {"position": [1, 0, 0], "quaternion": [1, 0, 0, 0]},
             "geometry": {"kind": "sphere", "radius": 0.01}},
        ]}
        xml = compile_scene(doc)
        assert xml.count('<body name="a"') == 1
        assert xml.count('<body name="b"') == 1

    def test_no_objects_key(self):
        xml = compile_scene({})
        assert "<worldbody>" in xml

    def test_unknown_geometry_raises(self):
        doc = {"objects": [{
            "id": "weird",
            "pose": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]},
            "geometry": {"kind": "torus"},
        }]}
        with pytest.raises(SceneCompilerError, match="unknown geometry kind"):
            compile_scene(doc)

    def test_xml_attr_escaping(self):
        doc = {"objects": [{
            "id": 'obj<">&',
            "pose": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]},
            "geometry": {"kind": "sphere", "radius": 0.01},
        }]}
        xml = compile_scene(doc)
        assert "&amp;" in xml or "&quot;" in xml or "&lt;" in xml or "obj" in xml


class TestCompileSceneBodyFragment:
    def test_returns_tuple(self):
        asset_xml, body_xml = compile_scene_body_fragment({"objects": []})
        assert isinstance(asset_xml, str)
        assert isinstance(body_xml, str)

    def test_box_in_body_xml(self):
        _, body_xml = compile_scene_body_fragment(_minimal_box_doc())
        assert 'type="box"' in body_xml

    def test_mesh_asset_in_asset_xml(self):
        doc = {"objects": [{
            "id": "mug",
            "pose": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]},
            "geometry": {"kind": "mesh", "path": "assets/mug.obj"},
        }]}
        asset_xml, _ = compile_scene_body_fragment(doc)
        assert "<mesh" in asset_xml
