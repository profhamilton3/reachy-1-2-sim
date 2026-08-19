"""
Gate 8-C integration tests: console fixture collision truthfulness.

Requires native MuJoCo (mjpython / Apple Silicon).  Tests are skipped when
mujoco is not importable so the offline suite remains green everywhere.

Run on the host:
    mjpython -m pytest tests/integration/test_forbidden_console_contact.py -v
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import pytest

mujoco = pytest.importorskip("mujoco", reason="native MuJoCo not available")

# Reach the native-server modules from any working directory.
_REPO = pathlib.Path(__file__).resolve().parents[2]
_NATIVE = _REPO / "native_mujoco"
for _p in (_NATIVE, _REPO / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import yaml
from objects import build_scene_model_xml
from scene_compiler import interactive_specs, tracked_object_ids

_SCENE_YAML = _REPO / "scenes" / "control_panel.yaml"
_MODEL_XML  = _NATIVE / "model" / "reachy_1_2.xml"

# Robot link contype (see R12-500 comment in reachy_1_2.xml).
_ROBOT_CONTYPE = 2
# Fixture contype (assigned by scene_compiler for collision:"fixture").
_FIXTURE_CONTYPE = 8


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_model() -> tuple[mujoco.MjModel, dict]:
    if not _SCENE_YAML.exists():
        pytest.skip(f"Scene file not found: {_SCENE_YAML}")
    if not _MODEL_XML.exists():
        pytest.skip(f"Robot model not found: {_MODEL_XML}")
    with open(_SCENE_YAML) as f:
        scene_doc = yaml.safe_load(f)
    xml = build_scene_model_xml(scene_doc, str(_MODEL_XML))
    model = mujoco.MjModel.from_xml_string(xml)
    return model, scene_doc


def _geom_contype(model: mujoco.MjModel, geom_name: str) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        pytest.fail(f"Geom not found in model: {geom_name}")
    return int(model.geom_contype[gid])


def _geom_conaffinity(model: mujoco.MjModel, geom_name: str) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        pytest.fail(f"Geom not found in model: {geom_name}")
    return int(model.geom_conaffinity[gid])


def _has_contact_between(data: mujoco.MjData, model: mujoco.MjModel,
                          contype_a: int, contype_b: int) -> bool:
    """Return True if any active contact pairs one geom of contype_a with contype_b."""
    for i in range(data.ncon):
        c = data.contact[i]
        ct1 = int(model.geom_contype[c.geom1])
        ct2 = int(model.geom_contype[c.geom2])
        if (ct1 == contype_a and ct2 == contype_b) or \
           (ct1 == contype_b and ct2 == contype_a):
            return True
    return False


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestFixtureChannelAssignment:
    """Verify the compiled model assigns the fixture channel to console geometry."""

    def test_console_base_has_fixture_contype(self):
        model, _ = _load_model()
        # The body "console_base" contains one geom; search by body name prefix.
        for i in range(model.ngeom):
            bid = model.geom_bodyid[i]
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or ""
            if bname == "console_base":
                assert int(model.geom_contype[i]) == _FIXTURE_CONTYPE, (
                    f"console_base geom contype should be {_FIXTURE_CONTYPE}, "
                    f"got {model.geom_contype[i]}"
                )
                return
        pytest.fail("console_base body not found in compiled model")

    def test_console_ledge_has_fixture_contype(self):
        model, _ = _load_model()
        for i in range(model.ngeom):
            bid = model.geom_bodyid[i]
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or ""
            if bname == "console_ledge":
                assert int(model.geom_contype[i]) == _FIXTURE_CONTYPE
                return
        pytest.fail("console_ledge body not found in compiled model")

    def test_console_slant_has_fixture_contype(self):
        model, _ = _load_model()
        for i in range(model.ngeom):
            bid = model.geom_bodyid[i]
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or ""
            if bname == "console_slant":
                assert int(model.geom_contype[i]) == _FIXTURE_CONTYPE
                return
        pytest.fail("console_slant body not found in compiled model")

    def test_fixture_collides_with_robot(self):
        """(contype_fixture & conaffinity_robot) != 0."""
        model, _ = _load_model()
        robot_conaffinity = 5   # from R12-500: robot conaffinity=5
        assert (_FIXTURE_CONTYPE & robot_conaffinity) != 0, (
            "Fixture channel must collide with robot links"
        )

    def test_fixture_does_not_collide_with_controls(self):
        """(contype_fixture & conaffinity_control) == 0 and (contype_control & conaffinity_fixture) == 0."""
        from scene_compiler import _FIXTURE_CONAFFINITY, _OBJ_CONTYPE, _OBJ_CONAFFINITY
        assert (_FIXTURE_CONTYPE & _OBJ_CONAFFINITY) == 0, (
            "Fixture contype must not collide with control conaffinity"
        )
        assert (_OBJ_CONTYPE & _FIXTURE_CONAFFINITY) == 0, (
            "Control contype must not collide with fixture conaffinity"
        )


class TestForbiddenConsoleContact:
    """Headless physics: robot link driven into console registers a contact."""

    def test_arm_driven_into_console_base_generates_contact(self):
        """Force an arm link into the console base and assert MuJoCo sees a contact."""
        model, scene_doc = _load_model()
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Settle for a few steps with gravity so the robot is in home position.
        for _ in range(20):
            mujoco.mj_step(model, data)

        # Find the right-shoulder or forearm body to teleport into the console.
        # Drive the wrist body directly into the console base (x≈0.67, z≈0.36).
        wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wrist")
        if wrist_bid < 0:
            pytest.skip("r_wrist body not found — check robot model joint names")

        # Teleport: set qpos for a reachable position inside the console base volume.
        # Instead of IK, directly move the wrist mocap or xpos and step.
        # We use mj_setSubtreeVel=0 and directly set body xpos then call mj_forward.
        data.xpos[wrist_bid] = [0.67, 0.0, 0.36]   # inside console_base
        mujoco.mj_forward(model, data)

        # With the fixture channel active, the robot ↔ fixture contact should fire.
        contact_found = _has_contact_between(data, model, _ROBOT_CONTYPE, _FIXTURE_CONTYPE)
        # Note: teleporting xpos directly may not trigger contact detection on a single
        # mj_forward call without constraint solve.  Run a full step to propagate.
        if not contact_found:
            mujoco.mj_step(model, data)
            contact_found = _has_contact_between(data, model, _ROBOT_CONTYPE, _FIXTURE_CONTYPE)

        assert contact_found, (
            "No robot↔fixture contact detected. Check fixture contype/conaffinity "
            "and that console_base has collision:\"fixture\" in the scene YAML."
        )

    def test_controls_do_not_contact_fixture(self):
        """After settling, mounted controls must not report contact with the console fixture."""
        model, scene_doc = _load_model()
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Settle 500 steps (1.0 s at 500 Hz) to reach equilibrium.
        for _ in range(500):
            mujoco.mj_step(model, data)

        from scene_compiler import _OBJ_CONTYPE
        control_fixture_contact = _has_contact_between(
            data, model, _OBJ_CONTYPE, _FIXTURE_CONTYPE
        )
        assert not control_fixture_contact, (
            "Controls are contacting the console fixture — check that the fixture "
            "conaffinity does not include the control contype bit."
        )
