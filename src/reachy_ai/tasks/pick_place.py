"""R12-803: Pick-and-place task runner.

Runs a pick_and_place recipe through the native episode runner and
(optionally) persists the result in an ExperienceStore.

Usage::

    from reachy_ai.tasks.pick_place import run_pick_place_recipe
    from simulation_core import SimulationCore

    core = SimulationCore.from_paths(model_path)
    result = run_pick_place_recipe(
        "recipes/pick_place/baseline_v1.yaml",
        core=core,
        seed=0,
        max_steps=5000,
    )
    print(result.status, result.metrics)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

_NATIVE = os.path.join(os.path.dirname(__file__), "../../../../native_mujoco")
if os.path.isdir(_NATIVE) and _NATIVE not in sys.path:
    sys.path.insert(0, _NATIVE)

from reachy_ai.motion.recipe import TrajectoryRecipe
from reachy_ai.motion.recipe_executor import RecipeExecutor, RecipeExecutionError
from reachy_ai.experience.models import EpisodeConfig, EpisodeResult
from reachy_ai.experience.identity import build_simulator_identity


def run_pick_place_recipe(
    recipe_path: str,
    *,
    core,
    store=None,
    study_id: str = "default",
    seed: int = 0,
    max_steps: int = 5000,
    on_snapshot=None,
) -> EpisodeResult:
    """Execute a pick-and-place recipe and return the EpisodeResult.

    Args:
        recipe_path: Path to a recipe YAML or JSON file.
        core:        Initialised SimulationCore.
        store:       Optional ExperienceStore for persisting trial records.
        study_id:    Study identifier used when writing to the store.
        seed:        RNG seed passed to EpisodeRunner.reset().
        max_steps:   Hard step limit passed to EpisodeConfig.
        on_snapshot: Optional per-step callback forwarded to EpisodeRunner.run().

    Returns:
        EpisodeResult from the native episode runner.
    """
    from episode_runner import EpisodeRunner, StepCommand

    recipe = TrajectoryRecipe.load(recipe_path)
    executor = RecipeExecutor()
    errors = executor.validate(recipe)
    if errors:
        raise RecipeExecutionError(
            f"Recipe validation failed ({len(errors)} error(s)):\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    command_specs = executor.build_commands(recipe)
    commands = [
        StepCommand(target_rad=list(c.target_rad), hold_steps=c.hold_steps)
        for c in command_specs
    ]

    identity = build_simulator_identity(
        model_path=getattr(core, "_model_path", ""),
        scene_path=getattr(core, "_scene_path", "") or "",
        backend_name="native_mujoco",
    )
    config = EpisodeConfig(
        simulator_identity=identity,
        seed=seed,
        max_steps=max_steps,
        render_mode="off",
    )
    runner = EpisodeRunner(core, config)
    result = runner.run(commands, on_snapshot=on_snapshot)

    if store is not None:
        _persist(store, study_id, recipe, config, result, identity)

    return result


def _persist(store, study_id, recipe, config, result, identity) -> None:
    from reachy_ai.experience.models import TaskSpec
    task_spec = TaskSpec(task_id=result.episode_id, task_type=recipe.task_type)
    trial_id = store.create_trial(
        study_id=study_id,
        task_spec=task_spec,
        recipe_json=recipe.to_json(),
        config=config,
    )
    store.start_trial(trial_id)
    store.complete_trial(trial_id, result)
