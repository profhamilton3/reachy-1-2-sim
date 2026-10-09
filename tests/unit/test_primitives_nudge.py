"""Issue #107: the host smoke test's bounded command goes through
`primitives.nudge_joint`, which checks `safety.gate_check()` first.

Offline: fake joints and a fake Reachy, the real gate (patched only where a
test says so), no SDK connection.
"""
import importlib
import math
import os
import pathlib
import re
import sys
import types

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
from reachy_ai.motion import primitives as P  # noqa: E402
from reachy_ai.motion import safety  # noqa: E402


class _Joint:
    """A joint that 'converges' instantly: setting the goal moves it there."""

    def __init__(self, present=0.0):
        self.present_position = present
        self.writes = []

    @property
    def goal_position(self):
        return self.writes[-1] if self.writes else None

    @goal_position.setter
    def goal_position(self, value):
        self.writes.append(value)
        self.present_position = value


def _arm(present=-10.0):
    return types.SimpleNamespace(r_shoulder_pitch=_Joint(present))


def _refuse(monkeypatch):
    monkeypatch.setattr(safety, "gate_check", lambda: False)


def _allow(monkeypatch):
    monkeypatch.setattr(safety, "gate_check", lambda: True)


class TestNudgeJoint:
    def test_commands_present_plus_delta(self, monkeypatch):
        _allow(monkeypatch)
        arm = _arm(-10.0)
        assert P.nudge_joint(arm, "r_shoulder_pitch", 5.0) == pytest.approx(-5.0)
        assert arm.r_shoulder_pitch.writes == [pytest.approx(-5.0)]

    def test_refused_gate_raises_and_sends_nothing(self, monkeypatch):
        _refuse(monkeypatch)
        arm = _arm()
        with pytest.raises(P.MotionRefused):
            P.nudge_joint(arm, "r_shoulder_pitch", 5.0)
        assert arm.r_shoulder_pitch.writes == []

    def test_the_real_gate_refuses_a_physical_backend_without_motion_enabled(
            self, monkeypatch):
        monkeypatch.setenv("REACHY_SIM_BACKEND", "physical")
        monkeypatch.delenv("REACHY_ENABLE_MOTION", raising=False)
        arm = _arm()
        with pytest.raises(P.MotionRefused):
            P.nudge_joint(arm, "r_shoulder_pitch", 5.0)
        assert arm.r_shoulder_pitch.writes == []

    @pytest.mark.parametrize("delta", [10.01, -10.01, math.nan, math.inf])
    def test_out_of_bound_delta_is_refused_before_the_gate(self, monkeypatch, delta):
        monkeypatch.setattr(safety, "gate_check",
                            lambda: pytest.fail("gate consulted for a refused step"))
        arm = _arm()
        with pytest.raises(ValueError):
            P.nudge_joint(arm, "r_shoulder_pitch", delta)
        assert arm.r_shoulder_pitch.writes == []


@pytest.fixture
def smoke(monkeypatch):
    monkeypatch.setitem(sys.modules, "reachy_sdk",
                        types.SimpleNamespace(ReachySDK=object))
    monkeypatch.syspath_prepend(str(_REPO / "scripts"))
    sys.modules.pop("smoke_test_host", None)
    mod = importlib.import_module("smoke_test_host")
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    return mod


def _reachy():
    calls = []
    r = types.SimpleNamespace(
        r_arm=_arm(-10.0),
        turn_on=lambda part: calls.append(("on", part)),
        turn_off=lambda part: calls.append(("off", part)))
    return r, calls


class TestSmokeTestBoundedCommand:
    def test_passes_and_returns_to_start(self, smoke, monkeypatch, capsys):
        _allow(monkeypatch)
        reachy, calls = _reachy()
        assert smoke._bounded_command(reachy) is True
        assert reachy.r_arm.r_shoulder_pitch.writes == [pytest.approx(-5.0),
                                                        pytest.approx(-10.0)]
        assert calls == [("on", "r_arm"), ("off", "r_arm")]

    def test_refused_gate_fails_loudly_and_sends_nothing(self, smoke, monkeypatch, capsys):
        _refuse(monkeypatch)
        reachy, calls = _reachy()
        assert smoke._bounded_command(reachy) is False
        out = capsys.readouterr().out
        assert "[FAIL]" in out and "safety gate refused" in out
        assert reachy.r_arm.r_shoulder_pitch.writes == []
        assert ("off", "r_arm") in calls

    def test_no_raw_goal_position_write_remains(self):
        src = (_REPO / "scripts/smoke_test_host.py").read_text()
        assert not re.search(r"\.goal_position\s*=", src)
