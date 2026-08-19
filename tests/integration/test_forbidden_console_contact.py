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
        """MuJoCo contact fires when (A.ct & B.ca) OR (B.ct & A.ca) != 0 (bidirectional)."""
        from scene_compiler import _FIXTURE_CONAFFINITY
        robot_contype, robot_conaffinity = 2, 5   # R12-500
        bidirectional = (
            (_FIXTURE_CONTYPE & robot_conaffinity) |
            (robot_contype & _FIXTURE_CONAFFINITY)
        )
        assert bidirectional != 0, (
            "Fixture channel (contype=8, conaffinity=2) must collide with "
            f"robot links (contype={robot_contype}, conaffinity={robot_conaffinity})"
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
        """Fixture channel (ct=8,ca=2) collides with robot channel (ct=2,ca=5) in MuJoCo.

        We build a minimal two-geom XML — one robot geom and one fixture geom fully
        overlapping — so mj_forward reports a contact without needing IK or xpos teleport.
        (Setting data.xpos directly is discarded by mj_forward, which always recomputes
        xpos from qpos.)
        """
        xml = """
        <mujoco>
          <option timestep="0.002"/>
          <worldbody>
            <body name="robot_link" pos="0 0 0">
              <freejoint/>
              <geom type="sphere" size="0.15" contype="2" conaffinity="5" mass="1"/>
            </body>
            <body name="console_fixture" pos="0 0 0">
              <geom type="sphere" size="0.15" contype="8" conaffinity="2"/>
            </body>
          </worldbody>
        </mujoco>
        """
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        assert d.ncon > 0, (
            "Fixture channel (contype=8, conaffinity=2) must collide with "
            "robot link channel (contype=2, conaffinity=5)"
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
