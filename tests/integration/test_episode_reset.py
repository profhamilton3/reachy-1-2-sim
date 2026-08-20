"""R12-802: Episode reset determinism and correctness.

Tests that SimulationCore.reset() is deterministic (same seed → same state)
and that the step counter and velocities are properly cleared.

Skipped when native MuJoCo is not importable.
"""

from __future__ import annotations

import os
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

from simulation_core import SimulationCore

_MODEL_XML  = _NATIVE / "model" / "reachy_1_2.xml"
_SCENE_YAML = _REPO / "scenes" / "control_panel.yaml"


def _require_model():
    if not _MODEL_XML.exists():
        pytest.skip(f"Robot model not found: {_MODEL_XML}")


# ---------------------------------------------------------------------------
# Basic reset tests (model-only, no scene needed)
# ---------------------------------------------------------------------------

class TestResetBasic:
    def test_step_counter_zeros_on_reset(self):
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        for _ in range(20):
            core.advance()
        assert core.step == 20
        core.reset()
        assert core.step == 0

    def test_qvel_cleared_on_reset(self):
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        for _ in range(50):
            core.advance()
        core.reset()
        # After reset + mj_forward velocities should be near zero
        assert np.allclose(core.data.qvel[:21], 0.0, atol=1e-6)

    def test_time_cleared_on_reset(self):
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        for _ in range(10):
            core.advance()
        assert core.data.time > 0.0
        core.reset()
        assert core.data.time == pytest.approx(0.0, abs=1e-9)

    def test_deterministic_qpos_same_seed(self):
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        core.reset(seed=42)
        qpos1 = core.data.qpos[:21].copy()

        core.reset(seed=42)
        qpos2 = core.data.qpos[:21].copy()

        assert np.allclose(qpos1, qpos2, atol=1e-12), \
            "Two resets with the same seed must produce identical joint qpos"

    def test_different_seed_gives_different_or_equal_qpos(self):
        """Robot qpos is fixed; seed affects object placement, not robot joints."""
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        core.reset(seed=0)
        qpos0 = core.data.qpos[:21].copy()
        core.reset(seed=1)
        qpos1 = core.data.qpos[:21].copy()
        # Robot joints are home-keyframe regardless of seed; pass either way.
        # The test confirms reset doesn't crash with different seeds.
        assert qpos0.shape == qpos1.shape

    def test_multiple_resets_stay_consistent(self):
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        core.reset(seed=7)
        baseline = core.data.qpos[:21].copy()

        for _ in range(5):
            for _ in range(30):
                core.advance()
            core.reset(seed=7)
            assert np.allclose(core.data.qpos[:21], baseline, atol=1e-12)

    def test_reset_with_none_seed(self):
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        core.reset(seed=None)
        assert core.step == 0


# ---------------------------------------------------------------------------
# Scene reset tests (requires control_panel.yaml)
# ---------------------------------------------------------------------------

class TestResetWithScene:
    def test_interactive_controls_reset_to_off(self):
        if not _SCENE_YAML.exists():
            pytest.skip(f"Scene file not found: {_SCENE_YAML}")
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML), str(_SCENE_YAML))

        # Advance to let physics settle
        for _ in range(10):
            core.advance()

        core.reset(seed=0)
        snap = core.snapshot()
        # All interactive controls should start in the OFF state after reset
        for ctrl in snap.interactive:
            assert ctrl.get("on") is False or ctrl.get("on") == 0, \
                f"Control {ctrl.get('id')} should be OFF after reset"

    def test_scene_reset_is_deterministic(self):
        if not _SCENE_YAML.exists():
            pytest.skip(f"Scene file not found: {_SCENE_YAML}")
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML), str(_SCENE_YAML))

        core.reset(seed=99)
        qpos1 = core.data.qpos.copy()

        for _ in range(50):
            core.advance()
        core.reset(seed=99)
        qpos2 = core.data.qpos.copy()

        assert np.allclose(qpos1, qpos2, atol=1e-12), \
            "Scene reset must be deterministic for the same seed"
