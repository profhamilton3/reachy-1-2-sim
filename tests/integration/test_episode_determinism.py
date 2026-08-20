"""R12-802: Episode determinism — same inputs must produce identical outputs.

Two runs with the same model, scene, seed, and command sequence must produce
step-identical joint positions, contact counts, and interactive states.

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

from simulation_core import SimulationCore
from episode_runner import EpisodeRunner, StepCommand
from joint_map import NUM_JOINTS
from reachy_ai.experience.identity import build_simulator_identity
from reachy_ai.experience.models import EpisodeConfig

_MODEL_XML  = _NATIVE / "model" / "reachy_1_2.xml"
_SCENE_YAML = _REPO / "scenes" / "control_panel.yaml"


def _require_model():
    if not _MODEL_XML.exists():
        pytest.skip(f"Robot model not found: {_MODEL_XML}")


def _identity(scene_path=None):
    return build_simulator_identity(
        model_path=str(_MODEL_XML),
        scene_path=str(scene_path) if scene_path else "",
    )


def _null_commands(n_steps: int) -> list:
    """A sequence of no-op commands (hold home position)."""
    return [StepCommand(target_rad=[0.0] * NUM_JOINTS, hold_steps=n_steps)]


# ---------------------------------------------------------------------------
# Determinism: model-only
# ---------------------------------------------------------------------------

class TestDeterminismModelOnly:
    def test_same_seed_same_final_qpos(self):
        _require_model()

        def _run(seed: int) -> np.ndarray:
            core = SimulationCore.from_paths(str(_MODEL_XML))
            cfg = EpisodeConfig(simulator_identity=_identity(), seed=seed, max_steps=200)
            runner = EpisodeRunner(core, cfg)
            runner.run(_null_commands(200))
            return core.data.qpos[:21].copy()

        q1 = _run(7)
        q2 = _run(7)
        assert np.allclose(q1, q2, atol=1e-12), \
            "Two runs with seed=7 must produce identical final qpos"

    def test_different_seeds_give_same_robot_qpos(self):
        """Robot joints are home-position regardless of seed (objects differ)."""
        _require_model()

        def _run(seed: int) -> np.ndarray:
            core = SimulationCore.from_paths(str(_MODEL_XML))
            cfg = EpisodeConfig(simulator_identity=_identity(), seed=seed, max_steps=100)
            runner = EpisodeRunner(core, cfg)
            runner.run(_null_commands(100))
            return core.data.qpos[:21].copy()

        q0 = _run(0)
        q1 = _run(1)
        # Robot joints converge to the same home position regardless of seed
        assert np.allclose(q0, q1, atol=0.01)

    def test_step_count_matches_command_hold(self):
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        cfg = EpisodeConfig(simulator_identity=_identity(), seed=0, max_steps=500)
        runner = EpisodeRunner(core, cfg)
        result = runner.run([StepCommand([0.0] * NUM_JOINTS, hold_steps=150)])
        assert result.metrics["total_steps"] == pytest.approx(150)

    def test_result_status_succeeded_with_no_forbidden_contact(self):
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML))
        cfg = EpisodeConfig(simulator_identity=_identity(), seed=0, max_steps=100)
        runner = EpisodeRunner(core, cfg)
        result = runner.run(_null_commands(50))
        from reachy_ai.experience.models import EpisodeStatus
        assert result.status == EpisodeStatus.SUCCEEDED
        # Joint-limit violations may occur at home=0 rad; only fixture contact is hard failure
        assert result.metrics["forbidden_contact_count"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Determinism: with scene
# ---------------------------------------------------------------------------

class TestDeterminismWithScene:
    def test_same_seed_same_final_interactive_states(self):
        if not _SCENE_YAML.exists():
            pytest.skip(f"Scene file not found: {_SCENE_YAML}")
        _require_model()

        def _run(seed: int) -> dict:
            core = SimulationCore.from_paths(str(_MODEL_XML), str(_SCENE_YAML))
            cfg = EpisodeConfig(
                simulator_identity=_identity(_SCENE_YAML),
                seed=seed, max_steps=300,
            )
            runner = EpisodeRunner(core, cfg)
            result = runner.run(_null_commands(300))
            return result.final_control_states

        states1 = _run(42)
        states2 = _run(42)
        assert states1.keys() == states2.keys(), "Same controls must be present"
        for cid in states1:
            assert states1[cid].get("on") == states2[cid].get("on"), \
                f"Control {cid} on-state differs between deterministic runs"

    def test_step_indexed_metrics_populated(self):
        if not _SCENE_YAML.exists():
            pytest.skip(f"Scene file not found: {_SCENE_YAML}")
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML), str(_SCENE_YAML))
        cfg = EpisodeConfig(
            simulator_identity=_identity(_SCENE_YAML),
            seed=0, max_steps=200,
        )
        runner = EpisodeRunner(core, cfg)
        result = runner.run(_null_commands(200))

        assert "total_steps" in result.metrics
        assert "forbidden_contact_count" in result.metrics
        assert "sim_duration_s" in result.metrics
        assert "wall_duration_s" in result.metrics

    def test_final_control_states_present(self):
        if not _SCENE_YAML.exists():
            pytest.skip(f"Scene file not found: {_SCENE_YAML}")
        _require_model()
        core = SimulationCore.from_paths(str(_MODEL_XML), str(_SCENE_YAML))
        cfg = EpisodeConfig(
            simulator_identity=_identity(_SCENE_YAML),
            seed=0, max_steps=100,
        )
        runner = EpisodeRunner(core, cfg)
        result = runner.run(_null_commands(100))
        assert isinstance(result.final_control_states, dict)
        assert len(result.final_control_states) > 0, \
            "control_panel.yaml has interactive controls — final_control_states must be non-empty"

    def test_episode_result_serialises(self):
        if not _SCENE_YAML.exists():
            pytest.skip(f"Scene file not found: {_SCENE_YAML}")
        _require_model()
        import json
        core = SimulationCore.from_paths(str(_MODEL_XML), str(_SCENE_YAML))
        cfg = EpisodeConfig(
            simulator_identity=_identity(_SCENE_YAML),
            seed=0, max_steps=50,
        )
        runner = EpisodeRunner(core, cfg)
        result = runner.run(_null_commands(50))
        serialised = result.to_json()
        recovered = json.loads(serialised)
        assert recovered["status"] in ("SUCCEEDED", "FAILED", "INVALID", "ABORTED")
