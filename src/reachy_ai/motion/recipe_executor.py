"""R12-803: Recipe executor for the native simulation path.

Maps TrajectoryRecipe primitives to CommandSpec sequences for EpisodeRunner.
This module has NO dependency on native_mujoco — callers convert CommandSpec
to StepCommand before passing to EpisodeRunner.run().

Design:
- Each primitive handler receives (step_params, recipe) and the current joint
  target as a list of 21 floats (rad), and returns (new_target, hold_steps).
- build_commands() chains handlers in sequence, propagating the running target.
- Interpolation between targets is linear, one StepCommand per sim step.
- For control-panel recipes: poses are derived from GUARD_R plus bounded
  standoff/press parameters.
- For pick-place recipes: poses are drawn from the named poses in primitives.py.

This module does not call primitives.py motion functions (those require a live
Reachy SDK arm).  It only borrows the pose dictionaries.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

from reachy_ai.motion.recipe import TrajectoryRecipe, PrimitiveStep

# Import only pose dicts and helpers from primitives — never the motion calls
from reachy_ai.motion.primitives import (
    HOME, READY, ABDUCT_LOW, SIDE_HIGH,
    OVER_RED, GRASP_RED, CARRY_RED, OVER_PLACE_RED, PLACE_RED,
    OVER_BLUE, GRASP_BLUE, CARRY_BLUE, OVER_PLACE_BLUE, PLACE_BLUE,
    _GRIPPER_OPEN_DEG, _GRIPPER_CLOSED_DEG,
    _L_GRIPPER_OPEN_DEG, _L_GRIPPER_CLOSED_DEG,
)

import sys
import os
_NATIVE = os.path.join(os.path.dirname(__file__), "../../../native_mujoco")
if os.path.isdir(_NATIVE) and _NATIVE not in sys.path:
    sys.path.insert(0, _NATIVE)

from joint_map import JOINT_TABLE, by_name, NUM_JOINTS

NUM_RIGHT_ARM = 8   # indices 0-7
NUM_BOTH_ARMS = 16  # indices 0-15
NUM_HEAD = 5        # indices 16-20


# Guard/retracted pose for the right arm when operating the control panel.
# From demo_control_panel.py: elbow-back so the forearm clears the console.
GUARD_R: Dict[str, float] = {
    "r_shoulder_pitch":  30.0,
    "r_shoulder_roll":  -10.0,
    "r_arm_yaw":          0.0,
    "r_elbow_pitch":   -115.0,
    "r_forearm_yaw":      0.0,
    "r_wrist_pitch":     15.0,
    "r_wrist_roll":       0.0,
    "r_gripper":        _GRIPPER_OPEN_DEG,
}

# Object pose tables for the pick-place executor.
_OBJECT_HOVER: Dict[str, Dict[str, float]] = {
    "red_cube":      OVER_RED,
    "blue_cylinder": OVER_BLUE,
}
_OBJECT_GRASP: Dict[str, Dict[str, float]] = {
    "red_cube":      GRASP_RED,
    "blue_cylinder": GRASP_BLUE,
}
_OBJECT_CARRY: Dict[str, Dict[str, float]] = {
    "red_cube":      CARRY_RED,
    "blue_cylinder": CARRY_BLUE,
}
_OBJECT_OVER_PLACE: Dict[str, Dict[str, float]] = {
    "red_cube":      OVER_PLACE_RED,
    "blue_cylinder": OVER_PLACE_BLUE,
}
_OBJECT_PLACE: Dict[str, Dict[str, float]] = {
    "red_cube":      PLACE_RED,
    "blue_cylinder": PLACE_BLUE,
}

# All primitives supported by this executor.
KNOWN_CONTROL_PANEL_PRIMITIVES = frozenset({
    "guard", "look_at", "approach_standoff", "approach_control",
    "press_or_sweep", "settle", "retract", "return_guard",
})
KNOWN_PICK_PLACE_PRIMITIVES = frozenset({
    "home", "ready", "transit_clear", "hover", "descend", "grasp",
    "settle_grasp", "lift", "carry", "place", "release", "retreat",
})
#: The measured rig routes the panel flies (#91).  `waypoint` is the corridor
#: routes' one primitive, because a corridor IS a list of waypoints and giving
#: each of them a name of its own would invite a recipe to use a different set.
KNOWN_PANEL_ROUTE_PRIMITIVES = frozenset({
    "waypoint", "wave_swing", "wave_return",
    "point_approach", "point_hover", "point_return",
})
KNOWN_PRIMITIVES = (KNOWN_CONTROL_PANEL_PRIMITIVES
                    | KNOWN_PICK_PLACE_PRIMITIVES
                    | KNOWN_PANEL_ROUTE_PRIMITIVES)

#: Which abilities this executor can build commands for, and which it cannot.
#:
#: POINTING IS NOT HERE ON PURPOSE.  Its hover pose is derived from the scene —
#: the tallest thing on the board, the target cell's centre, the reachability
#: of the approach — so building it needs a planner and a loaded scene, not a
#: pose table.  Inventing a hover here so that "point" appeared to be
#: searchable would produce trials about a motion the robot does not fly, which
#: is worse than the honest refusal below.  The recipe, the evaluator and the
#: integrity check for pointing all exist; only the offline driver is missing.
PANEL_ROUTES_EXECUTABLE = frozenset({"PLACE_ROUTE", "STOW_ROUTE", "WAVE"})


@dataclasses.dataclass
class CommandSpec:
    """Portable command: joint targets and hold duration in simulation steps.

    Callers convert to native_mujoco.episode_runner.StepCommand before running.
    No native_mujoco import required here.
    """
    target_rad: List[float]
    hold_steps: int = 1


class RecipeExecutionError(Exception):
    pass


def _pose_to_qpos(pose_deg: Dict[str, float], from_qpos: List[float]) -> List[float]:
    """Apply a named-joint pose (degrees) on top of from_qpos, returning a new 21-float list.

    Joints not present in pose_deg keep their from_qpos values.
    """
    result = list(from_qpos)
    for name, deg in pose_deg.items():
        entry = by_name(name)
        if entry is not None:
            result[entry.mjcf_index] = entry.sdk_deg_to_mjcf_rad(deg)
    return result


def _interpolate(
    from_qpos: List[float],
    to_qpos: List[float],
    n_steps: int,
) -> List[CommandSpec]:
    """Linear interpolation from from_qpos → to_qpos over n_steps steps."""
    n_steps = max(1, n_steps)
    cmds: List[CommandSpec] = []
    for i in range(1, n_steps + 1):
        t = i / n_steps
        target = [f + t * (to - f) for f, to in zip(from_qpos, to_qpos)]
        cmds.append(CommandSpec(target_rad=target, hold_steps=1))
    return cmds


def _param(step: PrimitiveStep, key: str, default: Any) -> Any:
    """Read a step-level parameter, falling back to default."""
    return step.parameters.get(key, default)


def _bp_value(recipe: TrajectoryRecipe, key: str, default: Any = None) -> Any:
    """Read a bounded_parameter value (handles nested {value: ...} or plain scalar)."""
    spec = recipe.bounded_parameters.get(key)
    if spec is None:
        return default
    if isinstance(spec, dict):
        return spec.get("value", default)
    return spec


# ---------------------------------------------------------------------------
# Panel ability routes (#91)
# ---------------------------------------------------------------------------

def _route_pose(name: str) -> Optional[Dict[str, float]]:
    """A waypoint's pose, looked up in `rig_routes` by the recipe's own name.

    Looked up rather than copied.  A pose table here would be a second set of
    waypoints, and the moment it disagreed with the module the arm flies, the
    search would be optimising a motion nothing performs.
    """
    from reachy_ai.motion import rig_routes as R
    return getattr(R, name, None) if isinstance(getattr(R, name, None), dict) else None


class _PanelRouteMixin:
    """Command building for the measured routes.  Mixed into RecipeExecutor."""

    def _handle_panel_route(
        self,
        step: PrimitiveStep,
        recipe: TrajectoryRecipe,
        current: List[float],
    ) -> Tuple[List[CommandSpec], List[float]]:
        from reachy_ai.motion import rig_routes as R

        p = step.primitive
        if p.startswith("point_"):
            raise RecipeExecutionError(
                "pointing cannot be built offline from a recipe alone: the "
                "hover pose is derived from the scene (the tallest object on "
                "the board, the cell's centre, the reachability of the "
                "approach), so it needs a planner and a loaded scene.  The "
                "recipe, the evaluator and the integrity check for pointing "
                "all exist; the offline driver does not, and inventing a "
                "hover here would produce trials about a motion the robot "
                "does not fly.")

        scale = float(_bp_value(recipe, "segment_duration_scale", 1.0) or 1.0)
        settle_s = float(_bp_value(recipe, "settle_pass_s", 0.8) or 0.8)
        seconds = float(_param(step, "seconds", 1.0)) * scale
        # The runner's timestep is the episode's, not the recipe's; 500 Hz is
        # the configured default and the one the two existing recipe families
        # already assume when they count steps.
        n = max(1, int(round(seconds / 0.002)))
        settle_steps = max(1, int(round(settle_s / 0.002)))

        name = str(_param(step, "name", ""))

        if p == "waypoint":
            pose_deg = _route_pose(name)
            if pose_deg is None:
                raise RecipeExecutionError(
                    f"'{name}' is not a waypoint in rig_routes, so there is no "
                    "measured pose to fly to")
            target = _pose_to_qpos(pose_deg, current)
            cmds = _interpolate(current, target, n)
            # RE-STREAMING IS NOT OPTIONAL, and the live route runner says so:
            # the arm only moves while setpoints are streaming, and a fast
            # segment finishes 10-20 degrees short with simply holding the
            # goal doing nothing to close it.  Holding the final target for a
            # settle pass is the offline stand-in for that.  It is NOT the
            # closed-loop retry the live runner does — this one cannot see
            # whether the arm arrived — and a duration searched here is
            # therefore optimistic about the live path by however much that
            # difference is worth.
            cmds.append(CommandSpec(target_rad=list(target),
                                    hold_steps=settle_steps))
            return cmds, target

        if p == "wave_swing":
            amplitude = float(_bp_value(recipe, "wave_amplitude_deg",
                                        abs(R.WAVE_A["r_forearm_yaw"])))
            swing_s = float(_bp_value(recipe, "wave_seconds", R.WAVE_SECONDS))
            n = max(1, int(round(swing_s * scale / 0.002)))
            side = name[-1].lower() if name else "a"
            base = R.WAVE_A if side == "a" else R.WAVE_B
            sign = -1.0 if side == "a" else 1.0
            pose_deg = dict(base, r_forearm_yaw=sign * amplitude)
            target = _pose_to_qpos(pose_deg, current)
            return _interpolate(current, target, n), target

        if p == "wave_return":
            swing_s = float(_bp_value(recipe, "wave_seconds", R.WAVE_SECONDS))
            n = max(1, int(round(swing_s * scale / 0.002)))
            target = _pose_to_qpos(R.PRESENT, current)
            cmds = _interpolate(current, target, n)
            cmds.append(CommandSpec(target_rad=list(target),
                                    hold_steps=settle_steps))
            return cmds, target

        raise RecipeExecutionError(f"unknown panel-route primitive '{p}'")


class RecipeExecutor(_PanelRouteMixin):
    """Converts a TrajectoryRecipe to a CommandSpec sequence for the simulation path."""

    def validate(self, recipe: TrajectoryRecipe) -> List[str]:
        """Return a list of validation errors; empty list means the recipe is valid."""
        errors: List[str] = []
        for i, step in enumerate(recipe.primitive_sequence):
            if step.primitive not in KNOWN_PRIMITIVES:
                errors.append(
                    f"Primitive '{step.primitive}' at index {i} is not recognised. "
                    f"Valid primitives: {sorted(KNOWN_PRIMITIVES)}"
                )
        # Validate bounded_parameter bounds
        for name, spec in recipe.bounded_parameters.items():
            if not isinstance(spec, dict):
                continue
            lo = spec.get("min")
            hi = spec.get("max")
            val = spec.get("value")
            if lo is not None and hi is not None and val is not None:
                try:
                    if not (float(lo) <= float(val) <= float(hi)):
                        errors.append(
                            f"bounded_parameter '{name}': value {val} outside "
                            f"[{lo}, {hi}]"
                        )
                except (TypeError, ValueError):
                    pass  # non-numeric parameter (e.g. object_id)
        return errors

    def build_commands(self, recipe: TrajectoryRecipe) -> List[CommandSpec]:
        """Convert recipe primitive_sequence to a CommandSpec list.

        Starts from the home/zero position and linearly interpolates between
        named joint targets for each primitive phase.
        """
        errors = self.validate(recipe)
        if errors:
            raise RecipeExecutionError(
                f"Recipe '{recipe.recipe_id}' has {len(errors)} error(s):\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        current = [0.0] * NUM_JOINTS  # home keyframe = all zeros
        commands: List[CommandSpec] = []

        # Dispatch to the appropriate handler family
        if any(s.primitive in KNOWN_PANEL_ROUTE_PRIMITIVES
               for s in recipe.primitive_sequence):
            handler = self._handle_panel_route
        elif recipe.task_type in ("pick_and_place", "pick_place"):
            handler = self._handle_pick_place
        else:
            handler = self._handle_control_panel

        for step in recipe.primitive_sequence:
            cmds, current = handler(step, recipe, current)
            commands.extend(cmds)

        return commands

    # ------------------------------------------------------------------ #
    # Control-panel primitive handlers                                     #
    # ------------------------------------------------------------------ #

    def _handle_control_panel(
        self,
        step: PrimitiveStep,
        recipe: TrajectoryRecipe,
        current: List[float],
    ) -> Tuple[List[CommandSpec], List[float]]:
        p = step.primitive
        n = int(_param(step, "hold_steps", 30))

        if p == "guard" or p == "return_guard":
            target = _pose_to_qpos(GUARD_R, current)
            cmds = _interpolate(current, target, n)
            return cmds, target

        if p == "look_at":
            # In simulation episodes, look_at is a no-op (no head IK needed).
            target = list(current)
            return [CommandSpec(target_rad=target, hold_steps=max(1, n))], target

        if p == "approach_standoff":
            sp_deg = float(_param(step, "shoulder_pitch_deg", -12.0))
            sr_deg = float(_param(step, "shoulder_roll_deg", -5.0))
            ya_deg = float(_param(step, "arm_yaw_deg", 5.0))
            ep_deg = float(_param(step, "elbow_pitch_deg", -88.0))
            wp_deg = float(_param(step, "wrist_pitch_deg", 10.0))
            fy_deg = float(_param(step, "forearm_yaw_deg", 0.0))
            standoff_pose = {
                "r_shoulder_pitch": sp_deg,
                "r_shoulder_roll":  sr_deg,
                "r_arm_yaw":        ya_deg,
                "r_elbow_pitch":    ep_deg,
                "r_forearm_yaw":    fy_deg,
                "r_wrist_pitch":    wp_deg,
                "r_wrist_roll":     0.0,
                "r_gripper":        _GRIPPER_CLOSED_DEG,  # closed = solid pointer
            }
            target = _pose_to_qpos(standoff_pose, current)
            cmds = _interpolate(current, target, n)
            return cmds, target

        if p == "approach_control":
            sp_delta = float(_param(step, "shoulder_pitch_delta_deg", -8.0))
            ep_delta = float(_param(step, "elbow_pitch_delta_deg", -4.0))
            _r_sp_idx = by_name("r_shoulder_pitch").mjcf_index
            _r_ep_idx = by_name("r_elbow_pitch").mjcf_index
            target = list(current)
            target[_r_sp_idx] = current[_r_sp_idx] + by_name("r_shoulder_pitch").sdk_deg_to_mjcf_rad(sp_delta)
            target[_r_ep_idx] = current[_r_ep_idx] + by_name("r_elbow_pitch").sdk_deg_to_mjcf_rad(ep_delta)
            cmds = _interpolate(current, target, n)
            return cmds, target

        if p in ("press_or_sweep", "settle"):
            # Hold at current position for the specified steps.
            target = list(current)
            return [CommandSpec(target_rad=target, hold_steps=n)], target

        if p == "retract":
            # Pull back to the previously computed standoff position.
            # We achieve this by slightly reversing the approach deltas.
            _r_sp_idx = by_name("r_shoulder_pitch").mjcf_index
            _r_ep_idx = by_name("r_elbow_pitch").mjcf_index
            sp_delta = float(_param(step, "shoulder_pitch_delta_deg", 8.0))
            ep_delta = float(_param(step, "elbow_pitch_delta_deg", 4.0))
            target = list(current)
            target[_r_sp_idx] = current[_r_sp_idx] - by_name("r_shoulder_pitch").sdk_deg_to_mjcf_rad(sp_delta)
            target[_r_ep_idx] = current[_r_ep_idx] - by_name("r_elbow_pitch").sdk_deg_to_mjcf_rad(ep_delta)
            cmds = _interpolate(current, target, n)
            return cmds, target

        # Unknown (should not reach here after validate())
        target = list(current)
        return [CommandSpec(target_rad=target, hold_steps=n)], target

    # ------------------------------------------------------------------ #
    # Pick-place primitive handlers                                        #
    # ------------------------------------------------------------------ #

    def _handle_pick_place(
        self,
        step: PrimitiveStep,
        recipe: TrajectoryRecipe,
        current: List[float],
    ) -> Tuple[List[CommandSpec], List[float]]:
        p = step.primitive
        n = int(_param(step, "hold_steps", 30))
        obj = str(_param(step, "object_id",
                         _bp_value(recipe, "object_id", "red_cube")))

        if p in ("home", "ready"):
            target = _pose_to_qpos(HOME, current)
            cmds = _interpolate(current, target, n)
            return cmds, target

        if p == "transit_clear":
            # Phase 1: swing arm out and then up (jumping-jack transit).
            mid = _pose_to_qpos(ABDUCT_LOW, current)
            target = _pose_to_qpos(SIDE_HIGH, mid)
            cmds = _interpolate(current, mid, n // 2 or 1)
            cmds += _interpolate(mid, target, n - (n // 2) or 1)
            return cmds, target

        if p == "hover":
            hover_pose = _OBJECT_HOVER.get(obj, OVER_RED)
            target = _pose_to_qpos(hover_pose, current)
            cmds = _interpolate(current, target, n)
            return cmds, target

        if p == "descend":
            grasp_pose = _OBJECT_GRASP.get(obj, GRASP_RED)
            target = _pose_to_qpos(grasp_pose, current)
            cmds = _interpolate(current, target, n)
            return cmds, target

        if p == "grasp":
            close_pose = {"r_gripper": _GRIPPER_CLOSED_DEG}
            target = _pose_to_qpos(close_pose, current)
            cmds = _interpolate(current, target, n)
            return cmds, target

        if p == "settle_grasp":
            target = list(current)
            return [CommandSpec(target_rad=target, hold_steps=n)], target

        if p == "lift":
            # Carry pose: hover position but with gripper still closed.
            carry_pose = _OBJECT_CARRY.get(obj, CARRY_RED)
            target = _pose_to_qpos(carry_pose, current)
            cmds = _interpolate(current, target, n)
            return cmds, target

        if p == "carry":
            over_place = _OBJECT_OVER_PLACE.get(obj, OVER_PLACE_RED)
            target = _pose_to_qpos(over_place, current)
            cmds = _interpolate(current, target, n)
            return cmds, target

        if p == "place":
            place_pose = _OBJECT_PLACE.get(obj, PLACE_RED)
            target = _pose_to_qpos(place_pose, current)
            cmds = _interpolate(current, target, n)
            return cmds, target

        if p == "release":
            open_pose = {"r_gripper": _GRIPPER_OPEN_DEG}
            target = _pose_to_qpos(open_pose, current)
            cmds = _interpolate(current, target, n)
            return cmds, target

        if p == "retreat":
            over_place = _OBJECT_OVER_PLACE.get(obj, OVER_PLACE_RED)
            target = _pose_to_qpos(over_place, current)
            cmds = _interpolate(current, target, n)
            return cmds, target

        # Unknown
        target = list(current)
        return [CommandSpec(target_rad=target, hold_steps=n)], target
