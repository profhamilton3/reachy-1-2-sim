"""R12-504: scene compiler articulation + interactive emission tests."""

import os
import sys

import pytest

_NATIVE = os.path.join(os.path.dirname(__file__), "../../native_mujoco")
sys.path.insert(0, _NATIVE)

from scene_compiler import (  # noqa: E402
    compile_scene,
    geom_name,
    interactive_specs,
    joint_name,
)


def _button_scene():
    return {"objects": [{
        "id": "btn_red",
        "geometry": {"kind": "cylinder", "radius": 0.02, "height": 0.02},
        "pose": {"position": [0.55, -0.1, 0.87]},
        "material": {"rgba": [0.9, 0.1, 0.1, 1]},
        "articulation": {"joint": "slide", "axis": [0, 0, 1],
                          "range": [-0.012, 0.0], "stiffness": 250, "damping": 4},
        "interactive": {"type": "button", "on_threshold": -0.007,
                        "off_threshold": -0.002, "lit_rgba": [1, 0.5, 0.5, 1]},
    }]}


def _lever_scene():
    return {"objects": [{
        "id": "lever_l",
        "geometry": {"kind": "box", "size": [0.016, 0.016, 0.13]},
        "pose": {"position": [0.57, 0.19, 0.85]},
        "material": {"rgba": [1, 0.6, 0, 1]},
        "articulation": {"joint": "hinge", "axis": [0, 1, 0], "range": [-0.7, 0.7],
                         "damping": 1.2, "armature": 0.01, "handle_offset": [0, 0, 0.065]},
        "interactive": {"type": "lever", "on_threshold": 0.12, "bistable": True},
    }]}


class TestArticulationEmission:
    def test_slide_button_emits_named_joint(self):
        xml = compile_scene(_button_scene())
        assert f'<joint name="{joint_name("btn_red")}"' in xml
        assert 'type="slide"' in xml
        assert 'stiffness="250' in xml

    def test_hinge_lever_emits_named_joint_and_offset(self):
        xml = compile_scene(_lever_scene())
        assert 'type="hinge"' in xml
        assert 'armature="0.010000"' in xml
        # handle_offset -> geom pos
        assert 'pos="0.000000 0.000000 0.065000"' in xml

    def test_articulated_geom_is_named_and_massed(self):
        xml = compile_scene(_lever_scene())
        assert f'name="{geom_name("lever_l")}"' in xml
        assert "mass=" in xml       # articulated bodies get a finite mass

    def test_compiles_and_loads_in_mujoco(self):
        mujoco = pytest.importorskip("mujoco")
        xml = compile_scene(_lever_scene())
        m = mujoco.MjModel.from_xml_string(xml)
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name("lever_l"))
        assert jid >= 0
        assert m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE


class TestInteractiveSpecs:
    def test_button_spec_fields(self):
        (spec,) = interactive_specs(_button_scene())
        assert spec["id"] == "btn_red"
        assert spec["joint_name"] == joint_name("btn_red")
        assert spec["geom_name"] == geom_name("btn_red")
        assert spec["type"] == "button"
        assert spec["on_threshold"] == -0.007
        assert spec["base_rgba"] == [0.9, 0.1, 0.1, 1]
        assert spec["lit_rgba"] == [1, 0.5, 0.5, 1]

    def test_lever_spec_bistable(self):
        (spec,) = interactive_specs(_lever_scene())
        assert spec["type"] == "lever"
        assert spec["bistable"] is True
        assert spec["range"] == [-0.7, 0.7]

    def test_non_interactive_objects_ignored(self):
        scene = {"objects": [{
            "id": "block", "geometry": {"kind": "box", "size": [0.1, 0.1, 0.1]},
            "pose": {"position": [0.5, 0, 0.8]},
        }]}
        assert interactive_specs(scene) == []
