"""R12-802: CLI entry point for running a single episode natively.

Runs one episode through SimulationCore + EpisodeRunner without Docker,
WebSocket, or SDK — the high-throughput research path from EPIC-8.md Section 5.4.

Usage (dry-run, physics only):
    mjpython native_mujoco/cli/run_episode.py \\
        --model native_mujoco/model/reachy_1_2.xml \\
        --scene scenes/control_panel.yaml \\
        --steps 2000 --seed 7

Usage with a recipe:
    mjpython native_mujoco/cli/run_episode.py \\
        --model native_mujoco/model/reachy_1_2.xml \\
        --scene scenes/control_panel.yaml \\
        --recipe recipes/control_panel/baseline_v1.yaml \\
        --seed 42 --output /tmp/result.json

Requires native MuJoCo (run with mjpython on Apple Silicon).
Recipe execution is basic in PR 8.2 (hold-targets replay); the full
primitive executor ships in PR 8.3.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys

# Reach src/ from anywhere inside the repo
_REPO = pathlib.Path(__file__).resolve().parents[2]
_NATIVE = _REPO / "native_mujoco"
_SRC = _REPO / "src"
for _p in (_NATIVE, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reachy12.cli.run_episode")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run one Reachy 1.2 episode natively (no Docker/SDK)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=str(_NATIVE / "model" / "reachy_1_2.xml"),
                   help="Path to the robot MJCF model")
    p.add_argument("--scene", default=None,
                   help="Path to a scene YAML file (optional)")
    p.add_argument("--recipe", default=None,
                   help="Path to a recipe JSON/YAML (PR 8.3 full executor; "
                        "PR 8.2 uses hold-targets replay)")
    p.add_argument("--steps", type=int, default=1000,
                   help="Max simulation steps to run")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for deterministic reset")
    p.add_argument("--output", default=None,
                   help="Path to write EpisodeResult JSON (stdout if omitted)")
    p.add_argument("--record-dir", default=None,
                   help="Directory for recorder trace output (disabled by default)")
    p.add_argument("--verbose", action="store_true",
                   help="Log every 100th snapshot")
    return p


def _load_recipe_commands(recipe_path: str):
    """Load a recipe and produce StepCommands (hold-targets replay for PR 8.2)."""
    from reachy_ai.motion.recipe import TrajectoryRecipe
    from episode_runner import StepCommand
    from joint_map import NUM_JOINTS

    recipe = TrajectoryRecipe.load(recipe_path)
    log.info("Recipe: %s  (%d primitives)", recipe.recipe_id,
             len(recipe.primitive_sequence))

    # PR 8.2 hold-targets replay: apply each bounded_parameter's value as a
    # constant offset to a neutral home position for hold_steps steps.
    # The full primitive executor (PR 8.3) maps primitives to motion calls.
    home = [0.0] * NUM_JOINTS
    hold = int(recipe.bounded_parameters.get("hold_steps", {}).get("value", 100))
    return [StepCommand(target_rad=home, hold_steps=max(1, hold))]


def main() -> None:
    args = _build_arg_parser().parse_args()

    try:
        import mujoco  # noqa: F401
    except ImportError:
        log.error("MuJoCo is not installed. Run with mjpython on Apple Silicon.")
        sys.exit(1)

    from simulation_core import SimulationCore, load_world
    from episode_runner import EpisodeRunner, StepCommand
    from reachy_ai.experience.identity import build_simulator_identity
    from reachy_ai.experience.models import EpisodeConfig

    log.info("Loading world: model=%s scene=%s", args.model, args.scene)
    _, _, _, compiled_xml = load_world(args.model, args.scene)

    identity = build_simulator_identity(
        model_path=args.model,
        scene_path=args.scene or "",
        backend_name="native_mujoco",
        compiled_xml=compiled_xml,
    )
    log.info("Identity: model_sha256=%s...  scene_sha256=%s...",
             identity.model_sha256[:12] if identity.model_sha256 else "—",
             identity.scene_sha256[:12] if identity.scene_sha256 else "—")

    config = EpisodeConfig(
        simulator_identity=identity,
        seed=args.seed,
        max_steps=args.steps,
        render_mode="off",
        record_trace=bool(args.record_dir),
    )

    core = SimulationCore.from_paths(args.model, args.scene)

    if args.recipe:
        commands = _load_recipe_commands(args.recipe)
    else:
        from joint_map import NUM_JOINTS
        # Null policy: hold home position for the full episode
        commands = [StepCommand(target_rad=[0.0] * NUM_JOINTS, hold_steps=args.steps)]

    runner = EpisodeRunner(core, config)

    step_log_every = 100 if args.verbose else None

    def _on_snapshot(snap) -> None:
        if step_log_every and snap.sim_step % step_log_every == 0:
            log.info("step=%d  contacts=%d  forbidden=%d",
                     snap.sim_step, len(snap.contacts), snap.forbidden_contact_count)

    log.info("Running episode: seed=%d  max_steps=%d", args.seed, args.steps)
    result = runner.run(commands, on_snapshot=_on_snapshot)

    log.info("Episode finished: status=%s  steps=%d  wall=%.2fs  forbidden=%d",
             result.status.value,
             int(result.metrics.get("total_steps", 0)),
             result.wall_duration_s,
             int(result.contact_summary.get("forbidden_total", 0)))

    output_dict = result.to_dict()
    output_json = json.dumps(output_dict, indent=2)

    if args.output:
        pathlib.Path(args.output).write_text(output_json)
        log.info("Result written to: %s", args.output)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
