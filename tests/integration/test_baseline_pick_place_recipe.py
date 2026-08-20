"""R12-803: Baseline pick-and-place recipe tests.

Tests:
- YAML round-trip
- Bounds validation
- Unknown primitive rejection
- No executable code in recipe
- build_commands produces correct CommandSpec list
- End-to-end episode run (skipped when native MuJoCo not available)
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_NATIVE = _REPO / "native_mujoco"
_SRC = _REPO / "src"
for _p in (_NATIVE, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reachy_ai.motion.recipe import TrajectoryRecipe, PrimitiveStep
from reachy_ai.motion.recipe_executor import (
    RecipeExecutor,
    RecipeExecutionError,
    KNOWN_PICK_PLACE_PRIMITIVES,
    CommandSpec,
)

_RECIPE_PATH = _REPO / "recipes" / "pick_place" / "baseline_v1.yaml"
_MODEL_XML = _NATIVE / "model" / "reachy_1_2.xml"


def _load_recipe() -> TrajectoryRecipe:
    return TrajectoryRecipe.load(str(_RECIPE_PATH))


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------

class TestYAMLRoundTrip:
    def test_recipe_file_exists(self):
        assert _RECIPE_PATH.exists(), f"Recipe file not found: {_RECIPE_PATH}"

    def test_load_returns_trajectory_recipe(self):
        r = _load_recipe()
        assert isinstance(r, TrajectoryRecipe)

    def test_recipe_id_preserved(self):
        r = _load_recipe()
        assert r.recipe_id == "pick_place_baseline_v1"

    def test_task_type_is_pick_and_place(self):
        r = _load_recipe()
        assert r.task_type == "pick_and_place"

    def test_yaml_roundtrip(self):
        r = _load_recipe()
        yaml_str = r.to_yaml()
        r2 = TrajectoryRecipe.from_yaml(yaml_str)
        assert r2.recipe_id == r.recipe_id
        assert len(r2.primitive_sequence) == len(r.primitive_sequence)
        for s1, s2 in zip(r.primitive_sequence, r2.primitive_sequence):
            assert s1.primitive == s2.primitive

    def test_json_roundtrip(self):
        r = _load_recipe()
        r2 = TrajectoryRecipe.from_dict(json.loads(r.to_json()))
        assert r2.recipe_id == r.recipe_id

    def test_arm_is_right(self):
        assert _load_recipe().arm == "right"

    def test_source_is_baseline(self):
        assert _load_recipe().source == "baseline"

    def test_has_home_primitive_at_start_and_end(self):
        r = _load_recipe()
        primitives = [s.primitive for s in r.primitive_sequence]
        assert primitives[0] == "home"
        assert primitives[-1] == "home"


# ---------------------------------------------------------------------------
# Bounds validation
# ---------------------------------------------------------------------------

class TestBoundsValidation:
    def test_all_step_counts_are_positive(self):
        r = _load_recipe()
        for step in r.primitive_sequence:
            hs = step.parameters.get("hold_steps")
            if hs is not None:
                assert int(hs) >= 1, \
                    f"Primitive '{step.primitive}' has non-positive hold_steps: {hs}"

    def test_all_bounded_parameter_values_within_bounds(self):
        r = _load_recipe()
        violations = []
        for name, spec in r.bounded_parameters.items():
            if not isinstance(spec, dict):
                continue
            lo, hi, val = spec.get("min"), spec.get("max"), spec.get("value")
            if lo is None or hi is None or val is None:
                continue
            try:
                if not (float(lo) <= float(val) <= float(hi)):
                    violations.append(f"{name}: {val} not in [{lo}, {hi}]")
            except (TypeError, ValueError):
                pass
        assert violations == [], f"Bound violations: {violations}"

    def test_executor_validate_passes_baseline(self):
        r = _load_recipe()
        errors = RecipeExecutor().validate(r)
        assert errors == [], f"Validation errors: {errors}"

    def test_object_id_is_declared(self):
        r = _load_recipe()
        assert "object_id" in r.bounded_parameters, \
            "Pick-place recipe must declare object_id bounded_parameter"


# ---------------------------------------------------------------------------
# Unknown primitive rejection
# ---------------------------------------------------------------------------

class TestUnknownPrimitiveRejection:
    def test_executor_rejects_unknown_primitive(self):
        bad = TrajectoryRecipe(
            recipe_id="bad",
            task_type="pick_and_place",
            primitive_sequence=[
                PrimitiveStep(primitive="teleport", parameters={})
            ],
        )
        errors = RecipeExecutor().validate(bad)
        assert any("teleport" in e for e in errors)

    def test_build_commands_raises_on_invalid_recipe(self):
        bad = TrajectoryRecipe(
            recipe_id="bad",
            task_type="pick_and_place",
            primitive_sequence=[
                PrimitiveStep(primitive="exec", parameters={})
            ],
        )
        with pytest.raises(RecipeExecutionError):
            RecipeExecutor().build_commands(bad)

    def test_all_recipe_primitives_are_known(self):
        from reachy_ai.motion.recipe_executor import KNOWN_PRIMITIVES
        r = _load_recipe()
        for step in r.primitive_sequence:
            assert step.primitive in KNOWN_PRIMITIVES, \
                f"Unknown primitive in recipe: '{step.primitive}'"


# ---------------------------------------------------------------------------
# No executable code
# ---------------------------------------------------------------------------

class TestNoExecutableCode:
    def test_parameters_contain_no_callables(self):
        r = _load_recipe()
        for step in r.primitive_sequence:
            for k, v in step.parameters.items():
                assert not callable(v), \
                    f"Primitive '{step.primitive}' parameter '{k}' is callable"

    def test_bounded_parameters_contain_no_callables(self):
        r = _load_recipe()
        for k, v in r.bounded_parameters.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    assert not callable(subv)
            else:
                assert not callable(v)

    def test_yaml_deserialization_rejects_python_tags(self):
        malicious = """
recipe_id: evil
task_type: pick_and_place
primitive_sequence:
  - primitive: home
    parameters:
      payload: !!python/object/apply:os.system ['echo pwned']
"""
        with pytest.raises(Exception):
            TrajectoryRecipe.from_yaml(malicious)


# ---------------------------------------------------------------------------
# build_commands correctness
# ---------------------------------------------------------------------------

class TestBuildCommands:
    def test_build_commands_non_empty(self):
        r = _load_recipe()
        cmds = RecipeExecutor().build_commands(r)
        assert len(cmds) > 0

    def test_all_target_rad_have_correct_length(self):
        from joint_map import NUM_JOINTS
        r = _load_recipe()
        cmds = RecipeExecutor().build_commands(r)
        for c in cmds:
            assert len(c.target_rad) == NUM_JOINTS

    def test_hold_steps_all_positive(self):
        r = _load_recipe()
        cmds = RecipeExecutor().build_commands(r)
        for c in cmds:
            assert c.hold_steps >= 1

    def test_all_target_rad_finite(self):
        import math
        r = _load_recipe()
        cmds = RecipeExecutor().build_commands(r)
        for c in cmds:
            for rad in c.target_rad:
                assert math.isfinite(rad), f"Non-finite joint target: {rad}"

    def test_home_primitive_produces_zero_qpos_for_arm(self):
        """After the first home primitive the right-arm joints should be near zero."""
        from joint_map import by_name
        r = TrajectoryRecipe(
            recipe_id="home_only",
            task_type="pick_and_place",
            primitive_sequence=[
                PrimitiveStep(primitive="home", parameters={"hold_steps": 10})
            ],
        )
        cmds = RecipeExecutor().build_commands(r)
        final = cmds[-1].target_rad
        sp_idx = by_name("r_shoulder_pitch").mjcf_index
        assert abs(final[sp_idx]) < 0.01, \
            f"r_shoulder_pitch should be ~0 rad at home, got {final[sp_idx]}"

    def test_grasp_closes_gripper(self):
        """After the grasp primitive the gripper joint should be negative (closed)."""
        from joint_map import by_name
        import math
        r = TrajectoryRecipe(
            recipe_id="grasp_only",
            task_type="pick_and_place",
            primitive_sequence=[
                PrimitiveStep(primitive="grasp", parameters={"hold_steps": 10})
            ],
        )
        cmds = RecipeExecutor().build_commands(r)
        final = cmds[-1].target_rad
        g_idx = by_name("r_gripper").mjcf_index
        assert final[g_idx] < 0, \
            f"r_gripper should be negative (closed) after grasp, got {final[g_idx]}"

    def test_release_opens_gripper(self):
        """After the release primitive the gripper should be open (negative of closed)."""
        from joint_map import by_name
        from reachy_ai.motion.primitives import _GRIPPER_CLOSED_DEG, _GRIPPER_OPEN_DEG
        r = TrajectoryRecipe(
            recipe_id="release_only",
            task_type="pick_and_place",
            primitive_sequence=[
                PrimitiveStep(primitive="release", parameters={"hold_steps": 10})
            ],
        )
        cmds = RecipeExecutor().build_commands(r)
        final = cmds[-1].target_rad
        g_idx = by_name("r_gripper").mjcf_index
        # open is more negative than closed
        assert final[g_idx] <= by_name("r_gripper").sdk_deg_to_mjcf_rad(_GRIPPER_OPEN_DEG) + 0.01


# ---------------------------------------------------------------------------
# End-to-end (skipped if native MuJoCo not available)
# ---------------------------------------------------------------------------

class TestEndToEndPickPlace:
    def _skip_no_mujoco(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            pytest.skip("native MuJoCo not available")
        if not _MODEL_XML.exists():
            pytest.skip(f"Robot model not found: {_MODEL_XML}")

    def test_recipe_runs_and_returns_episode_result(self):
        self._skip_no_mujoco()
        from simulation_core import SimulationCore
        from episode_runner import EpisodeRunner, StepCommand
        from reachy_ai.experience.models import EpisodeConfig, EpisodeStatus
        from reachy_ai.experience.identity import build_simulator_identity

        core = SimulationCore.from_paths(str(_MODEL_XML))
        r = _load_recipe()
        cmds_spec = RecipeExecutor().build_commands(r)
        commands = [
            StepCommand(target_rad=list(c.target_rad), hold_steps=c.hold_steps)
            for c in cmds_spec
        ]

        identity = build_simulator_identity(model_path=str(_MODEL_XML))
        config = EpisodeConfig(
            simulator_identity=identity,
            seed=0,
            max_steps=15000,
        )
        runner = EpisodeRunner(core, config)
        result = runner.run(commands)

        assert result is not None
        assert result.status in (
            EpisodeStatus.SUCCEEDED,
            EpisodeStatus.FAILED,
            EpisodeStatus.INVALID,
        )
        assert "total_steps" in result.metrics

    def test_episode_result_serialises(self):
        self._skip_no_mujoco()
        from simulation_core import SimulationCore
        from episode_runner import EpisodeRunner, StepCommand
        from reachy_ai.experience.models import EpisodeConfig
        from reachy_ai.experience.identity import build_simulator_identity

        core = SimulationCore.from_paths(str(_MODEL_XML))
        r = _load_recipe()
        cmds_spec = RecipeExecutor().build_commands(r)
        commands = [
            StepCommand(target_rad=list(c.target_rad), hold_steps=c.hold_steps)
            for c in cmds_spec
        ]

        identity = build_simulator_identity(model_path=str(_MODEL_XML))
        config = EpisodeConfig(
            simulator_identity=identity,
            seed=1,
            max_steps=15000,
        )
        result = EpisodeRunner(core, config).run(commands)
        data = json.loads(result.to_json())
        assert "status" in data
        assert "hard_violations" in data

    def test_episode_result_stored_in_experience_store(self):
        self._skip_no_mujoco()
        import tempfile
        from simulation_core import SimulationCore
        from episode_runner import EpisodeRunner, StepCommand
        from reachy_ai.experience.models import EpisodeConfig
        from reachy_ai.experience.identity import build_simulator_identity
        from reachy_ai.experience.store import ExperienceStore

        core = SimulationCore.from_paths(str(_MODEL_XML))
        r = _load_recipe()
        cmds_spec = RecipeExecutor().build_commands(r)
        commands = [
            StepCommand(target_rad=list(c.target_rad), hold_steps=c.hold_steps)
            for c in cmds_spec
        ]

        identity = build_simulator_identity(model_path=str(_MODEL_XML))
        config = EpisodeConfig(
            simulator_identity=identity,
            seed=5,
            max_steps=15000,
        )

        with tempfile.TemporaryDirectory() as tmp:
            db = pathlib.Path(tmp) / "exp.db"
            with ExperienceStore.open(db) as store:
                result = EpisodeRunner(core, config).run(commands)
                from reachy_ai.experience.models import TaskSpec
                task_spec = TaskSpec(task_id="t0", task_type=r.task_type)
                trial_id = store.create_trial(
                    study_id="pr83_pickplace",
                    task_spec=task_spec,
                    recipe_json=r.to_json(),
                    config=config,
                )
                store.start_trial(trial_id)
                store.complete_trial(trial_id, result)

            with ExperienceStore.open(db) as store:
                rows = store.query_compatible_trials(identity, limit=10)
                assert len(rows) >= 1

    def test_blue_cylinder_recipe_produces_result(self):
        """Verify object_id=blue_cylinder also runs without crashing."""
        self._skip_no_mujoco()
        from simulation_core import SimulationCore
        from episode_runner import EpisodeRunner, StepCommand
        from reachy_ai.experience.models import EpisodeConfig
        from reachy_ai.experience.identity import build_simulator_identity

        core = SimulationCore.from_paths(str(_MODEL_XML))
        r = _load_recipe()
        # Override object_id in each step that specifies one
        for step in r.primitive_sequence:
            if "object_id" in step.parameters:
                step.parameters["object_id"] = "blue_cylinder"

        cmds_spec = RecipeExecutor().build_commands(r)
        commands = [
            StepCommand(target_rad=list(c.target_rad), hold_steps=c.hold_steps)
            for c in cmds_spec
        ]

        identity = build_simulator_identity(model_path=str(_MODEL_XML))
        config = EpisodeConfig(
            simulator_identity=identity,
            seed=2,
            max_steps=15000,
        )
        result = EpisodeRunner(core, config).run(commands)
        assert result is not None
