"""R12-504: scene compiler articulation + interactive emission tests."""

import os
import sys

import pytest

_NATIVE = os.path.join(os.path.dirname(__file__), "../../native_mujoco")
sys.path.insert(0, _NATIVE)

from scene_compiler import (  # noqa: E402
    SceneCompilerError,
    compile_scene,
    geom_name,
    interactive_specs,
    joint_name,
)


def _button_scene():
    return {"objects": [{
        "id": "btn_red",
        "geometry": {"kind": "cylinder", "radius": 0.02, "length": 0.02},
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


# ── Gate 8-B: interactive↔articulation relational enforcement ────────────────

class TestInteractiveArticulationContract:

    def test_button_without_articulation_raises(self):
        scene = {"objects": [{
            "id": "bad",
            "geometry": {"kind": "box", "size": [0.1, 0.1, 0.1]},
            "pose": {"position": [0.5, 0, 0.8]},
            "interactive": {"type": "button", "on_threshold": -0.005},
        }]}
        with pytest.raises(SceneCompilerError, match="interactive requires articulation"):
            compile_scene(scene)

    def test_button_with_hinge_raises(self):
        scene = {"objects": [{
            "id": "bad",
            "geometry": {"kind": "box", "size": [0.02, 0.02, 0.1]},
            "pose": {"position": [0.5, 0, 0.9]},
            "articulation": {"joint": "hinge", "axis": [0, 1, 0]},
            "interactive": {"type": "button", "on_threshold": -0.005},
        }]}
        with pytest.raises(SceneCompilerError, match="requires articulation.joint='slide'"):
            compile_scene(scene)

    def test_lever_without_articulation_raises(self):
        scene = {"objects": [{
            "id": "bad",
            "geometry": {"kind": "box", "size": [0.02, 0.02, 0.1]},
            "pose": {"position": [0.5, 0, 0.9]},
            "interactive": {"type": "lever", "on_threshold": 0.1, "bistable": True},
        }]}
        with pytest.raises(SceneCompilerError, match="interactive requires articulation"):
            compile_scene(scene)

    def test_lever_with_slide_raises(self):
        scene = {"objects": [{
            "id": "bad",
            "geometry": {"kind": "box", "size": [0.02, 0.02, 0.1]},
            "pose": {"position": [0.5, 0, 0.9]},
            "articulation": {"joint": "slide", "axis": [0, 0, 1]},
            "interactive": {"type": "lever", "on_threshold": 0.1, "bistable": True},
        }]}
        with pytest.raises(SceneCompilerError, match="requires articulation.joint='hinge'"):
            compile_scene(scene)

    def test_switch_with_slide_raises(self):
        scene = {"objects": [{
            "id": "bad",
            "geometry": {"kind": "box", "size": [0.02, 0.02, 0.07]},
            "pose": {"position": [0.5, 0, 0.9]},
            "articulation": {"joint": "slide", "axis": [0, 0, 1]},
            "interactive": {"type": "switch", "on_threshold": 0.0, "bistable": True},
        }]}
        with pytest.raises(SceneCompilerError, match="requires articulation.joint='hinge'"):
            compile_scene(scene)

    def test_button_with_correct_slide_passes(self):
        """button + slide articulation must compile without error."""
        compile_scene(_button_scene())   # should not raise

    def test_lever_with_correct_hinge_passes(self):
        """lever + hinge articulation must compile without error."""
        compile_scene(_lever_scene())   # should not raise


# ── Gate 8-B: control_panel.yaml schema→compile contract ─────────────────────

class TestControlPanelSchemaContract:
    """Load control_panel.yaml through schema validation then compile and check dims."""

    @pytest.fixture(scope="class")
    def panel_xml(self):
        import pathlib, yaml as _yaml
        yaml_path = pathlib.Path(__file__).parents[2] / "scenes" / "control_panel.yaml"
        if not yaml_path.exists():
            pytest.skip("control_panel.yaml not found")
        with open(yaml_path) as f:
            doc = _yaml.safe_load(f)
        return compile_scene(doc)

    def test_all_button_cylinders_have_correct_half_length(self, panel_xml):
        """All six cylinders (radius=0.030, length=0.028) → half=0.014."""
        assert "0.030000 0.050000" not in panel_xml, (
            "At least one button compiled with 0.1 m default height"
        )
        assert "0.030000 0.014000" in panel_xml

    def test_fixture_channel_emitted_for_console_bodies(self, panel_xml):
        """Console fixtures must use contype=8 (fixture channel)."""
        assert 'contype="8"' in panel_xml
        assert 'conaffinity="2"' in panel_xml

    def test_control_collision_channel_emitted_for_buttons(self, panel_xml):
        """Interactive controls must use contype=4 (object channel)."""
        assert 'contype="4"' in panel_xml

    def test_mujoco_model_button_geom_size(self):
        """MjModel geom_size for btn_red must match the YAML (r=0.030, h/2=0.014)."""
        mujoco = pytest.importorskip("mujoco")
        import pathlib, yaml as _yaml
        yaml_path = pathlib.Path(__file__).parents[2] / "scenes" / "control_panel.yaml"
        if not yaml_path.exists():
            pytest.skip("control_panel.yaml not found")
        native_path = pathlib.Path(__file__).parents[2] / "native_mujoco"
        import sys
        sys.path.insert(0, str(native_path))
        from objects import build_scene_model_xml
        model_xml_path = native_path / "model" / "reachy_1_2.xml"
        if not model_xml_path.exists():
            pytest.skip("Robot model XML not found")
        with open(yaml_path) as f:
            doc = _yaml.safe_load(f)
        xml = build_scene_model_xml(doc, str(model_xml_path))
        m = mujoco.MjModel.from_xml_string(xml)
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "btn_red__g")
        assert gid >= 0, "btn_red__g geom not found in compiled model"
        # MuJoCo cylinder size[0]=radius, size[1]=half-length
        import pytest as _pt
        assert m.geom_size[gid, 0] == _pt.approx(0.030, abs=1e-4), "Wrong radius"
        assert m.geom_size[gid, 1] == _pt.approx(0.014, abs=1e-4), "Wrong half-length"


# ── B2 regression: cylinder length/height field contract ─────────────────────

class TestCylinderLengthContract:

    def test_cylinder_length_maps_to_mujoco_half_length(self):
        scene = {"objects": [{
            "id": "btn",
            "geometry": {"kind": "cylinder", "radius": 0.030, "length": 0.028},
            "pose": {"position": [0.55, -0.1, 0.87]},
        }]}
        xml = compile_scene(scene)
        # MuJoCo cylinder size="radius half_length": 0.028/2 = 0.014
        assert "0.030000 0.014000" in xml, (
            "Cylinder compiled with wrong half-length; expected 0.014000 (half of 0.028)"
        )

    def test_cylinder_height_field_not_used(self):
        """The 'height' field (invalid in schema) must not silently override length."""
        scene_with_length = {"objects": [{
            "id": "a",
            "geometry": {"kind": "cylinder", "radius": 0.030, "length": 0.028},
            "pose": {"position": [0.5, 0, 0.8]},
        }]}
        xml = compile_scene(scene_with_length)
        # Must NOT produce the 0.1m default half-length (0.050000)
        assert "0.030000 0.050000" not in xml, (
            "Cylinder fell back to default height 0.1m; 'length' field was not read"
        )

    def test_control_panel_buttons_compile_correct_height(self):
        import pathlib
        try:
            import yaml as _yaml
        except ImportError:
            pytest.skip("PyYAML not available")
        yaml_path = pathlib.Path(__file__).parent.parent.parent / "scenes" / "control_panel.yaml"
        if not yaml_path.exists():
            pytest.skip("control_panel.yaml not found")
        with open(yaml_path) as f:
            doc = _yaml.safe_load(f)
        xml = compile_scene(doc)
        # All six cylindrical buttons declare length: 0.028 → half = 0.014
        assert "0.030000 0.050000" not in xml, (
            "At least one button compiled with 0.1m default; 'length' was not read"
        )
        assert "0.030000 0.014000" in xml
