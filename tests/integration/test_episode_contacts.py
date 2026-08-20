"""R12-802: Contact snapshot correctness.

Verifies that EvaluationSnapshot.contacts is populated with typed
ContactRecord objects and that the fixture collision channel (contype=8)
appears correctly in snapshots when the arm is driven into the console.

Skipped when native MuJoCo is not importable.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

mujoco = pytest.importorskip("mujoco", reason="native MuJoCo not available")
import numpy as np

_REPO   = pathlib.Path(__file__).resolve().parents[2]
_NATIVE = _REPO / "native_mujoco"
_SRC    = _REPO / "src"
for _p in (_NATIVE, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluation_snapshot import ContactRecord, EvaluationSnapshot
from simulation_core import SimulationCore

_MODEL_XML  = _NATIVE / "model" / "reachy_1_2.xml"
_SCENE_YAML = _REPO / "scenes" / "control_panel.yaml"

_ROBOT_CONTYPE   = 2
_FIXTURE_CONTYPE = 8


def _require_model():
    if not _MODEL_XML.exists():
        pytest.skip(f"Robot model not found: {_MODEL_XML}")


def _require_scene():
    _require_model()
    if not _SCENE_YAML.exists():
        pytest.skip(f"Scene file not found: {_SCENE_YAML}")


# ---------------------------------------------------------------------------
# ContactRecord type tests (minimal MuJoCo model — no file needed)
# ---------------------------------------------------------------------------

class TestContactRecordSchema:
    def test_contact_record_is_frozen(self):
        c = ContactRecord(
            geom1="a", geom2="b", body1="A", body2="B",
            contype1=2, contype2=8, conaffinity1=5, conaffinity2=2,
            pos=(0.1, 0.2, 0.3), normal_force=1.5, dist=-0.001,
        )
        with pytest.raises((TypeError, AttributeError)):
            c.normal_force = 99.0  # type: ignore[misc]

    def test_contacts_by_contype(self):
        snap = EvaluationSnapshot(
            sim_step=1, sim_time_s=0.002, scene_revision="r0",
            joints=[], interactive=[], objects=[], grippers=[],
            force_sensors=[],
            contacts=[
                ContactRecord("g1", "g2", "b1", "b2", 2, 8, 5, 2,
                              (0, 0, 0), 1.0, -0.001),
                ContactRecord("g3", "g4", "b3", "b4", 4, 7, 7, 4,
                              (0, 0, 0), 0.5, -0.001),
            ],
            forbidden_contact_count=1,
        )
        robot_fixture = snap.contacts_by_contype(_ROBOT_CONTYPE, _FIXTURE_CONTYPE)
        assert len(robot_fixture) == 1
        assert robot_fixture[0].geom1 == "g1"

    def test_has_forbidden_contact(self):
        snap = EvaluationSnapshot(
            sim_step=0, sim_time_s=0.0, scene_revision="r0",
            joints=[], contacts=[], interactive=[], objects=[],
            grippers=[], force_sensors=[],
            forbidden_contact_count=2,
        )
        assert snap.has_forbidden_contact() is True

    def test_no_forbidden_contact(self):
        snap = EvaluationSnapshot(
            sim_step=0, sim_time_s=0.0, scene_revision="r0",
            joints=[], contacts=[], interactive=[], objects=[],
            grippers=[], force_sensors=[],
            forbidden_contact_count=0,
        )
        assert snap.has_forbidden_contact() is False

    def test_is_on_helper(self):
        snap = EvaluationSnapshot(
            sim_step=0, sim_time_s=0.0, scene_revision="r0",
            joints=[], contacts=[], objects=[], grippers=[], force_sensors=[],
            interactive=[
                {"id": "btn_red", "type": "button", "on": True, "value": -0.01},
                {"id": "btn_green", "type": "button", "on": False, "value": 0.0},
            ],
            forbidden_contact_count=0,
        )
        assert snap.is_on("btn_red") is True
        assert snap.is_on("btn_green") is False
        assert snap.is_on("nonexistent") is False


# ---------------------------------------------------------------------------
# Minimal XML contact test (channel bitmask verification, no model file needed)
# ---------------------------------------------------------------------------

class TestContactChannelsInMuJoCo:
    def test_robot_fixture_contact_minimal_model(self):
        """Verify the fixture channel (ct=8,ca=2) produces contacts with robot (ct=2,ca=5)."""
        xml = """
        <mujoco>
          <option timestep="0.002"/>
          <worldbody>
            <body name="robot_link" pos="0 0 0">
              <freejoint/>
              <geom name="rg" type="sphere" size="0.15" contype="2" conaffinity="5" mass="1"/>
            </body>
            <body name="console_fixture" pos="0 0 0">
              <geom name="fg" type="sphere" size="0.15" contype="8" conaffinity="2"/>
            </body>
          </worldbody>
        </mujoco>
        """
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        assert d.ncon > 0, (
            "Robot channel (ct=2,ca=5) must collide with fixture channel (ct=8,ca=2)"
        )

    def test_control_does_not_contact_fixture_minimal_model(self):
        """Verify controls (ct=4,ca=7) do NOT generate contacts with fixture (ct=8,ca=2)."""
        xml = """
        <mujoco>
          <option timestep="0.002"/>
          <worldbody>
            <body name="btn" pos="0 0 0">
              <freejoint/>
              <geom name="cg" type="sphere" size="0.15" contype="4" conaffinity="7" mass="0.1"/>
            </body>
            <body name="console_fixture" pos="0 0 0">
              <geom name="fg" type="sphere" size="0.15" contype="8" conaffinity="2"/>
            </body>
          </worldbody>
        </mujoco>
        """
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        assert d.ncon == 0, (
            "Control channel (ct=4,ca=7) must NOT collide with fixture channel (ct=8,ca=2)"
        )


# ---------------------------------------------------------------------------
# SimulationCore snapshot tests
# ---------------------------------------------------------------------------

class TestSimulationCoreSnapshots:
    def test_snapshot_returns_evaluation_snapshot(self):
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        core.reset(seed=0)
        snap = core.snapshot()
        assert isinstance(snap, EvaluationSnapshot)
        assert snap.sim_step == 0
        assert isinstance(snap.joints, list)
        assert isinstance(snap.contacts, list)

    def test_snapshot_joints_have_required_fields(self):
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        core.reset()
        snap = core.snapshot()
        required = {"name", "uid", "position_rad", "velocity_rad_s", "effort",
                    "compliant", "saturated"}
        for j in snap.joints:
            assert required <= j.keys(), f"Joint missing fields: {required - j.keys()}"

    def test_snapshot_contacts_are_contact_records(self):
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        core.reset()
        for _ in range(5):
            core.advance()
        snap = core.snapshot()
        for c in snap.contacts:
            assert isinstance(c, ContactRecord)

    def test_snapshot_step_advances(self):
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        core.reset()
        for _ in range(10):
            core.advance()
        snap = core.snapshot()
        assert snap.sim_step == 10

    def test_snapshot_sim_time_advances(self):
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        core.reset()
        dt = core.model.opt.timestep
        for _ in range(50):
            core.advance()
        snap = core.snapshot()
        assert snap.sim_time_s == pytest.approx(50 * dt, rel=1e-4)


# ---------------------------------------------------------------------------
# Scene-level contact tests
# ---------------------------------------------------------------------------

class TestSceneContactSnapshots:
    def test_snapshot_has_interactive_field_with_scene(self):
        _require_scene()
        core = SimulationCore.from_paths(str(_MODEL_XML), str(_SCENE_YAML))
        core.reset(seed=0)
        snap = core.snapshot()
        assert isinstance(snap.interactive, list)
        assert len(snap.interactive) > 0, \
            "control_panel.yaml has interactive controls"

    def test_fixture_contype_in_model(self):
        """Compiled model must assign contype=8 to console_base."""
        _require_scene()
        core = SimulationCore.from_paths(str(_MODEL_XML), str(_SCENE_YAML))
        model = core.model
        fixture_found = False
        for i in range(model.ngeom):
            bid = model.geom_bodyid[i]
            bname = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or ""
            if bname == "console_base":
                assert int(model.geom_contype[i]) == _FIXTURE_CONTYPE, \
                    f"console_base geom contype should be {_FIXTURE_CONTYPE}"
                fixture_found = True
                break
        assert fixture_found, "console_base body not found in compiled scene model"
