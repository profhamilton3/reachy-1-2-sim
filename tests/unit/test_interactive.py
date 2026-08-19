"""R12-504: interactive controller behaviour (button toggle, switch snap)."""

import os
import sys

import pytest

mujoco = pytest.importorskip("mujoco")
import numpy as np  # noqa: E402

_NATIVE = os.path.join(os.path.dirname(__file__), "../../native_mujoco")
sys.path.insert(0, _NATIVE)

from interactive import InteractiveController  # noqa: E402
from scene_compiler import compile_scene, interactive_specs, joint_name  # noqa: E402


def _build(scene):
    xml = compile_scene(scene)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ctl = InteractiveController(m, d, interactive_specs(scene))
    return m, d, ctl


_BTN = {"objects": [{
    "id": "btn", "geometry": {"kind": "cylinder", "radius": 0.02, "height": 0.02},
    "pose": {"position": [0.5, 0, 0.8]}, "material": {"rgba": [0.9, 0.1, 0.1, 1]},
    "articulation": {"joint": "slide", "axis": [0, 0, 1], "range": [-0.012, 0.0],
                     "stiffness": 250, "damping": 4},
    "interactive": {"type": "button", "on_threshold": -0.007, "off_threshold": -0.002,
                    "lit_rgba": [1, 0.5, 0.5, 1]},
}]}

_SWITCH = {"objects": [{
    "id": "sw", "geometry": {"kind": "box", "size": [0.014, 0.014, 0.075]},
    "pose": {"position": [0.5, 0, 0.9]}, "material": {"rgba": [0, 0.7, 0.8, 1]},
    "articulation": {"joint": "hinge", "axis": [0, 1, 0], "range": [-0.5, 0.5],
                     "damping": 1.0, "armature": 0.008, "handle_offset": [0, 0, 0.038]},
    "interactive": {"type": "switch", "on_threshold": 0.0, "bistable": True},
}]}


class TestButtonToggle:
    def test_full_press_toggles_once(self):
        m, d, ctl = _build(_BTN)
        qa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name("btn"))]
        assert ctl.states()[0]["on"] is False
        # press then release = one toggle
        d.qpos[qa] = -0.009; mujoco.mj_forward(m, d); ctl.update(d)
        d.qpos[qa] = 0.0;    mujoco.mj_forward(m, d); ctl.update(d)
        assert ctl.states()[0]["on"] is True

    def test_held_press_does_not_double_toggle(self):
        m, d, ctl = _build(_BTN)
        qa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name("btn"))]
        d.qpos[qa] = -0.009
        for _ in range(10):
            mujoco.mj_forward(m, d); ctl.update(d)
        assert ctl.states()[0]["on"] is True  # exactly one toggle while held

    def test_color_reflects_on_state(self):
        m, d, ctl = _build(_BTN)
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "btn__g")
        qa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name("btn"))]
        d.qpos[qa] = -0.009; mujoco.mj_forward(m, d); ctl.update(d)
        d.qpos[qa] = 0.0;    mujoco.mj_forward(m, d); ctl.update(d)
        assert np.allclose(m.geom_rgba[gid], [1, 0.5, 0.5, 1])


class TestSwitchSnap:
    def test_starts_off(self):
        m, d, ctl = _build(_SWITCH)
        for _ in range(400):
            ctl.update(d); mujoco.mj_step(m, d)
        assert ctl.states(d)[0]["on"] is False

    def test_flip_past_midpoint_snaps_on(self):
        m, d, ctl = _build(_SWITCH)
        qa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name("sw"))]
        d.qpos[qa] = 0.2; mujoco.mj_forward(m, d)
        for _ in range(500):
            ctl.update(d); mujoco.mj_step(m, d)
        s = ctl.states(d)[0]
        # Pushed past the midpoint it stays ON (does not fall back to OFF).
        assert s["on"] is True and d.qpos[qa] >= 0.0

    def test_reset_clears(self):
        m, d, ctl = _build(_SWITCH)
        qa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name("sw"))]
        d.qpos[qa] = 0.2
        for _ in range(300):
            ctl.update(d); mujoco.mj_step(m, d)
        ctl.reset(d)
        assert ctl.states(d)[0]["on"] is False
