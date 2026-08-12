"""
R12-501: Actuator and compliance model tests.

Requires `mujoco` (native side); skipped in the Python-3.8 container / CI.

The ActuatorController mutates model.actuator_gainprm/biasprm/forcerange, so
every test gets a FRESHLY compiled model (function-scoped fixture) — the
controller's documented contract is "construct from a pristine model".
"""

import os

import pytest

mujoco = pytest.importorskip("mujoco")
import numpy as np  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../native_mujoco"))
from actuator import ActuatorController, JointControlState  # noqa: E402

_MODEL = os.path.join(
    os.path.dirname(__file__), "../../native_mujoco/model/reachy_1_2.xml"
)


@pytest.fixture
def sim():
    """Fresh (model, data, controller) per test."""
    m = mujoco.MjModel.from_xml_path(_MODEL)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    ctl = ActuatorController(m)
    ctl.sync_targets_to_current(d)
    return m, d, ctl


def _roll(m, d, ctl, steps):
    dt = m.opt.timestep
    for _ in range(steps):
        ctl.apply(d, dt)
        mujoco.mj_step(m, d)


class TestDefaults:
    def test_arms_default_compliant(self, sim):
        _, _, ctl = sim
        assert ctl.state[ctl._resolve("act_r_shoulder_pitch")].compliant is True
        assert ctl.state[ctl._resolve("act_l_elbow_pitch")].compliant is True

    def test_antennas_default_stiff(self, sim):
        _, _, ctl = sim
        assert ctl.state[ctl._resolve("act_l_antenna")].compliant is False
        assert ctl.state[ctl._resolve("act_r_antenna")].compliant is False

    def test_default_torque_limit_full(self, sim):
        _, _, ctl = sim
        assert ctl.state[0].torque_limit == 100.0


class TestStiffStepResponse:
    def test_reaches_goal(self, sim):
        m, d, ctl = sim
        ctl.set_compliant("act_r_elbow_pitch", False)
        ctl.set_goal_position("act_r_elbow_pitch", -1.0)
        _roll(m, d, ctl, 1500)
        pos = ctl.present_position(d, "act_r_elbow_pitch")
        assert pos == pytest.approx(-1.0, abs=0.05)

    def test_settles_no_oscillation(self, sim):
        m, d, ctl = sim
        ctl.set_compliant("act_r_elbow_pitch", False)
        ctl.set_goal_position("act_r_elbow_pitch", -0.8)
        _roll(m, d, ctl, 2000)
        vel = abs(ctl.present_velocity(d, "act_r_elbow_pitch"))
        assert vel < 0.01  # settled

    def test_no_nan(self, sim):
        m, d, ctl = sim
        ctl.set_compliant("act_r_shoulder_pitch", False)
        ctl.set_goal_position("act_r_shoulder_pitch", 1.0)
        _roll(m, d, ctl, 1000)
        assert not np.any(np.isnan(d.qpos))


class TestCompliance:
    def test_compliant_produces_no_actuator_force(self, sim):
        m, d, ctl = sim
        ctl.set_compliant("act_r_elbow_pitch", True)
        ctl.set_goal_position("act_r_elbow_pitch", -1.5)  # goal ignored when off
        _roll(m, d, ctl, 500)
        # Motor off: actuator force ~0 regardless of the goal.
        assert abs(ctl.actuator_force(d, "act_r_elbow_pitch")) < 1e-6

    def test_compliant_joint_falls_under_gravity(self, sim):
        m, d, ctl = sim
        # Elbow compliant; give it a lifted start so gravity has somewhere to pull.
        eidx = ctl._resolve("act_r_elbow_pitch")
        d.qpos[ctl._qpos_adr[eidx]] = -1.0
        mujoco.mj_forward(m, d)
        ctl.set_compliant("act_r_elbow_pitch", True)
        _roll(m, d, ctl, 2000)
        # With the motor off, the elbow drifts from its start under gravity.
        assert ctl.present_position(d, "act_r_elbow_pitch") != pytest.approx(-1.0, abs=0.05)

    def test_restiffen_holds_new_position(self, sim):
        m, d, ctl = sim
        ctl.set_compliant("act_r_elbow_pitch", False)
        ctl.set_goal_position("act_r_elbow_pitch", -0.5)
        _roll(m, d, ctl, 1000)
        assert ctl.present_position(d, "act_r_elbow_pitch") == pytest.approx(-0.5, abs=0.05)


class TestSpeedLimit:
    def test_speed_limit_slows_approach(self, sim):
        m, d, ctl = sim
        ctl.set_compliant("act_r_shoulder_pitch", False)
        ctl.set_goal_position("act_r_shoulder_pitch", 1.0)
        ctl.set_speed_limit("act_r_shoulder_pitch", 0.5)
        _roll(m, d, ctl, 500)  # 1.0 s
        limited = ctl.present_position(d, "act_r_shoulder_pitch")
        # At 0.5 rad/s for 1 s the joint cannot have reached the 1.0 goal.
        assert 0.2 < limited < 0.6

    def test_unlimited_is_faster(self, sim):
        m, d, ctl = sim
        ctl.set_compliant("act_r_shoulder_pitch", False)
        ctl.set_goal_position("act_r_shoulder_pitch", 1.0)
        _roll(m, d, ctl, 500)
        unlimited = ctl.present_position(d, "act_r_shoulder_pitch")
        assert unlimited > 0.8  # reaches near goal quickly without a cap


class TestTorqueLimit:
    def test_low_torque_saturates_and_cannot_hold(self, sim):
        m, d, ctl = sim
        ctl.set_compliant("act_r_shoulder_pitch", False)
        ctl.set_goal_position("act_r_shoulder_pitch", 1.4)  # heavy gravity load
        ctl.set_torque_limit("act_r_shoulder_pitch", 3.0)   # 1.8 Nm
        _roll(m, d, ctl, 1500)
        assert ctl.is_saturated(d, "act_r_shoulder_pitch")
        assert ctl.present_position(d, "act_r_shoulder_pitch") < 1.0  # fell short
        # Force capped at scaled forcerange (3% of 60 Nm = 1.8 Nm).
        assert abs(ctl.actuator_force(d, "act_r_shoulder_pitch")) <= 1.8 + 1e-3

    def test_full_torque_not_saturated(self, sim):
        m, d, ctl = sim
        ctl.set_compliant("act_r_shoulder_pitch", False)
        ctl.set_goal_position("act_r_shoulder_pitch", 1.4)
        _roll(m, d, ctl, 1500)
        assert not ctl.is_saturated(d, "act_r_shoulder_pitch")
        assert ctl.present_position(d, "act_r_shoulder_pitch") == pytest.approx(1.4, abs=0.1)


class TestLimitEnforcement:
    def test_over_range_goal_is_clamped(self, sim):
        m, d, ctl = sim
        ctl.set_compliant("act_r_wrist_pitch", False)
        ctl.set_goal_position("act_r_wrist_pitch", 5.0)  # ctrlrange upper 0.785
        _roll(m, d, ctl, 1000)
        assert ctl.present_position(d, "act_r_wrist_pitch") <= 0.79

    def test_torque_limit_clamped_to_percentage(self, sim):
        _, _, ctl = sim
        ctl.set_torque_limit("act_r_gripper", 250.0)
        assert ctl.state[ctl._resolve("act_r_gripper")].torque_limit == 100.0
        ctl.set_torque_limit("act_r_gripper", -10.0)
        assert ctl.state[ctl._resolve("act_r_gripper")].torque_limit == 0.0
