"""Flying a panel ability's recipe in the headless simulator, offline (#91).

This is the bridge between a recipe and a verdict: it builds the commands,
runs them through `EpisodeRunner`, and gathers the three things the evaluators
need that an `EpisodeResult` does not carry on its own —

  * where the objects STARTED, because "nothing moved" is a comparison and the
    result records only where they ended;
  * WHICH fixture bodies the arm touched, because laying the forearm on the
    table is the task for rest and the runner's forbidden-contact counter goes
    up for it exactly as it would for hitting a rail;
  * how many wave cycles actually completed.

NOTHING HERE RUNS FROM THE PANEL.  It imports `reachy_ai.search` nowhere, and
the panel imports this nowhere; a chat turn cannot reach it, which
`test_panel_recipes.py` enforces for the whole web package.  It is a study
tool, driven by `scripts/search_panel_ability.py`.

THE OTHER ARM SAGS, AND THAT IS NOT THIS ROUTE'S DOING
------------------------------------------------------
Only the acting arm is powered, which is what the physical robot does.  The
idle arm hangs compliant and settles past its joint limit under gravity, so
`l_elbow_pitch` appears as a soft joint-limit violation in essentially every
verdict this driver produces.  It is left in rather than filtered out — a
driver that quietly dropped joint-limit violations it had decided were boring
would drop a real one the day the route caused it — but a reader comparing two
recipes should know it is a constant, not a difference between them.

WHAT THIS PATH IS NOT
---------------------
It is not the live executor.  The live route runner re-streams setpoints and
re-checks the arm's actual position between waypoints, retrying with
progressively longer passes; this one streams an interpolation and holds.  A
duration found here is therefore OPTIMISTIC about the live path by however
much that difference is worth, and a recipe promoted from this study should be
read as a candidate for a live trial rather than as a finished answer.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from reachy_ai.evaluation.panel_routes import (
    ABILITY_ROUTES, COLLIDABLE_KEY, CONTACT_BODIES_KEY, PanelRoutePolicy,
    align_to_parameters, evaluate,
)
from reachy_ai.experience.models import (
    EpisodeConfig, EpisodeResult, PanelRouteTaskSpec, SimulatorIdentity,
)
from reachy_ai.motion.recipe import TrajectoryRecipe
from reachy_ai.motion.recipe_executor import (
    PANEL_ROUTES_EXECUTABLE, RecipeExecutionError, RecipeExecutor,
)

#: Robot geoms are contype 2.  The runner's own forbidden-contact counter pairs
#: that with contype 8 (fixtures) — and THAT PAIR MATCHES NOTHING IN THE SCENES
#: THIS DRIVER RUNS.  `FWDCenterLabSivaPool` has three collision groups: the
#: world and pedestal (1), the robot (2), and the manipulable objects (4).
#: There is no contype 8 geom anywhere in it, so "no forbidden contact" out of
#: an episode there means "there was nothing tagged as a fixture to hit", not
#: "the arm kept clear".  This module therefore tallies robot-to-ANYTHING
#: contact by body name and reports what was collidable, so a clean record can
#: be told from an empty question.
_ROBOT_CONTYPE = 2

#: Bodies the rest route is MEANT to touch.
#:
#: EMPTY, AND NOT BECAUSE RESTING TOUCHES NOTHING.  REST is the forearm
#: supported on the tabletop, and in the real rig it does touch.  In the
#: simulated scenes the tabletop is part of the world body rather than a
#: separate collidable fixture, and measured across the whole route the arm
#: registers no contact at all.  Naming `table_top` here would be declaring an
#: intention about a body that does not exist, which reads in a verdict as
#: though the contact happened and was allowed.
REST_INTENDED_CONTACT: Tuple[str, ...] = ()


def _native_on_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    native = os.path.abspath(os.path.join(here, "..", "..", "..",
                                          "native_mujoco"))
    if os.path.isdir(native) and native not in sys.path:
        sys.path.insert(0, native)


def executable(ability: str) -> Tuple[bool, str]:
    """Whether this ability can be flown offline, and why not when it cannot."""
    route = ABILITY_ROUTES.get(ability, "")
    if not route:
        return False, f"{ability!r} is not an ability with a measured route"
    if route not in PANEL_ROUTES_EXECUTABLE:
        return False, (
            f"{ability!r} flies {route}, whose poses are derived from the "
            "scene rather than looked up — it needs a planner and a loaded "
            "scene, which this offline driver does not have")
    return True, ""


class OfflineRouteRunner:
    """Runs one ability's recipes against one scene, seed by seed.

    One instance per study: the simulation core is expensive to build and
    `EpisodeRunner` resets it at the start of every run, so reusing it across
    candidates is both cheaper and the thing that makes seeds comparable.
    """

    def __init__(self, ability: str, model_path: str, scene_path: str = "",
                 *, policy: Optional[PanelRoutePolicy] = None,
                 max_steps: int = 200_000, recipe_arm: str = "right") -> None:
        ok, why = executable(ability)
        if not ok:
            raise RecipeExecutionError(why)
        _native_on_path()
        from simulation_core import SimulationCore

        self.ability = ability
        self.route = ABILITY_ROUTES[ability]
        self.model_path = model_path
        self.scene_path = scene_path
        self.policy = policy or PanelRoutePolicy()
        self.max_steps = max_steps
        self._core = SimulationCore.from_paths(model_path, scene_path or None)
        self._executor = RecipeExecutor()
        self._power_on(recipe_arm)
        self._collidable = self._collidable_bodies()

    def _collidable_bodies(self) -> List[str]:
        """Non-robot bodies in this scene the arm could actually collide with.

        An episode's contact record is only as meaningful as this list: in
        `FWDCenterLabSivaPool` it is the world, the pedestal and the
        manipulable objects — the rails and the board the corridor's geometry
        was measured AGAINST are not collidable bodies at all.  A route that
        reports no contact there has not been shown to keep clear of them; it
        has been shown that they were not in the question.
        """
        import mujoco

        model = self._core.model
        out: List[str] = []
        for g in range(model.ngeom):
            contype = int(model.geom_contype[g])
            if contype in (0, _ROBOT_CONTYPE):
                continue
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                     int(model.geom_bodyid[g]))
            if body and body not in out:
                out.append(body)
        return sorted(out)

    def _power_on(self, arm: str) -> None:
        """Turn the acting arm's motors on.

        THE ARM IS COMPLIANT BY DEFAULT and that default is correct: it mirrors
        the physical robot at power-on, where the joints are back-drivable
        until something stiffens them.  The live path stiffens them through the
        SDK before it flies anything.  Offline, nothing did — so every command
        this driver streamed went to a limp arm, which sat exactly where the
        home keyframe left it and reported a clean, undisturbed, entirely
        motionless episode.  A search over that would have been ranking physics
        noise and calling it a trajectory.
        """
        side = "l" if arm.startswith("l") else "r"
        for joint in ("shoulder_pitch", "shoulder_roll", "arm_yaw",
                      "elbow_pitch", "forearm_yaw", "wrist_pitch",
                      "wrist_roll", "gripper"):
            try:
                self._core.controller.set_compliant(f"act_{side}_{joint}", False)
            except KeyError:
                # A model without that actuator is a model this ability cannot
                # be flown on; say so when the route runs, not here.
                pass

    # -- the pieces an EpisodeResult does not carry -----------------------

    def _initial_objects(self) -> Dict[str, List[float]]:
        snap = self._core.snapshot()
        out: Dict[str, List[float]] = {}
        for obj in snap.objects:
            oid = obj.get("object_id") or obj.get("id") or ""
            xyz = obj.get("pos_xyz")
            if oid and xyz and len(xyz) >= 3:
                out[oid] = [float(v) for v in xyz[:3]]
        return out

    @staticmethod
    def _fixture_bodies(snap, tally: Dict[str, int]) -> None:
        """Add this step's robot-to-anything contacts to the running tally.

        Counted per BODY rather than per geom: the evaluators and the task
        specs talk about what was touched, and a geom name is an implementation
        detail of the mesh it belongs to.

        Robot-to-robot contact is excluded — that is self-collision, which the
        snapshot classifies separately — but everything else the arm meets is
        counted whatever its collision group, because the group tagging is what
        turned out not to be trustworthy.
        """
        for c in snap.contacts:
            types = {c.contype1, c.contype2}
            if _ROBOT_CONTYPE not in types or types == {_ROBOT_CONTYPE}:
                continue
            other = c.body2 if c.contype1 == _ROBOT_CONTYPE else c.body1
            if other:
                tally[other] = tally.get(other, 0) + 1

    # -- one episode ------------------------------------------------------

    def run(self, recipe: TrajectoryRecipe, seed: int,
            identity: Optional[SimulatorIdentity] = None,
            ) -> Tuple[EpisodeResult, PanelRouteTaskSpec]:
        """Fly one recipe once, and return the result with its task spec."""
        _native_on_path()
        from episode_runner import EpisodeRunner, StepCommand

        specs = self._executor.build_commands(recipe)
        commands = [StepCommand(target_rad=list(s.target_rad),
                                hold_steps=s.hold_steps) for s in specs]

        identity = identity or _identity_for(self.model_path, self.scene_path)
        config = EpisodeConfig(simulator_identity=identity, seed=seed,
                               max_steps=self.max_steps, render_mode="off",
                               contact_sampling="all")
        runner = EpisodeRunner(self._core, config)

        # The runner resets the core at the start of run(), so the starting
        # poses have to be read from INSIDE it — read before the call and they
        # would be last episode's final poses, which is how a study comes to
        # measure drift against the wrong board.
        initial: Dict[str, List[float]] = {}
        tally: Dict[str, int] = {}
        counter = _SwingCounter(_amplitude_in(recipe))

        def watch(snap) -> None:
            if not initial:
                initial.update({
                    (o.get("object_id") or o.get("id") or ""):
                        [float(v) for v in (o.get("pos_xyz") or (0, 0, 0))[:3]]
                    for o in snap.objects
                    if (o.get("object_id") or o.get("id"))
                })
            self._fixture_bodies(snap, tally)
            if self.route == "WAVE":
                counter.observe(snap)

        result = runner.run(commands, on_snapshot=watch,
                            start_pose_rad=self._start_pose(recipe))
        result.contact_summary = dict(result.contact_summary or {})
        result.contact_summary[CONTACT_BODIES_KEY] = tally
        # WHAT COULD HAVE BEEN HIT, so that an empty tally can be read as "the
        # arm kept clear" rather than "there was nothing to keep clear of".
        result.contact_summary[COLLIDABLE_KEY] = self._collidable
        if self.route == "WAVE":
            result.metrics["wave_cycles_completed"] = float(counter.cycles)

        spec = self.task_spec(recipe, initial)
        return result, spec

    def _start_pose(self, recipe: TrajectoryRecipe) -> Optional[List[float]]:
        """The posture the route requires the arm to already be at, in MuJoCo
        radians — or None when the route starts from home and there is nothing
        to place.

        The live path REFUSES a route whose start posture the arm is not at,
        rather than driving it there: the connecting move is the one thing
        nothing measured.  Offline there is no arm to refuse, so the
        precondition is established instead — and stating it as a placement
        keeps it out of the measurement.
        """
        from reachy_ai.motion import rig_routes as R

        name = recipe.expected_start_posture or ""
        pose = R.POSTURES.get(name)
        if pose is None or name == R.POSTURE_HOME:
            return None

        _native_on_path()
        from joint_map import JOINT_TABLE, NUM_JOINTS, by_name

        qpos = [0.0] * NUM_JOINTS
        for joint, deg in pose.items():
            entry = by_name(joint)
            if entry is not None:
                qpos[entry.mjcf_index] = entry.sdk_deg_to_mjcf_rad(deg)
        return qpos

    def task_spec(self, recipe: TrajectoryRecipe,
                  initial: Dict[str, List[float]]) -> PanelRouteTaskSpec:
        from reachy_ai.motion import rig_routes as R

        ends = {"PLACE_ROUTE": R.POSTURE_REST, "STOW_ROUTE": R.POSTURE_HOME,
                "WAVE": R.POSTURE_PRESENT}
        return PanelRouteTaskSpec(
            task_id=f"panel:{self.ability}",
            task_type=self.ability,
            ability=self.ability,
            route=self.route,
            route_version=recipe.route_version or 1,
            expected_start_posture=recipe.expected_start_posture,
            expected_end_posture=ends.get(self.route, ""),
            initial_object_positions=dict(initial),
            intended_contact_bodies=list(
                REST_INTENDED_CONTACT if self.ability == "rest_forearm" else ()),
            expected_wave_cycles=(_cycles_in(recipe)
                                  if self.route == "WAVE" else 0),
            arm_policy=recipe.arm or "right",
            success_definition=_success_sentence(self.ability),
            forbidden_contact_policy="fail",
        )

    def evaluate_fn(self, identity: Optional[SimulatorIdentity] = None):
        """An `evaluate_fn` for `SearchRunner.run()`.

        Bound to this runner so the search does not have to know about scenes,
        cores or task specs — it hands over a recipe and a seed and is handed
        back a verdict.
        """
        def evaluate_one(recipe: TrajectoryRecipe, seed: int):
            # The sampler varies parameters; some of them are also facts about
            # the step list (a two-cycle wave has four swings).  Reconciled
            # here, once, so the search cannot produce a recipe its own
            # integrity check refuses.
            recipe = align_to_parameters(recipe)
            result, spec = self.run(recipe, seed, identity=identity)
            return evaluate(self.ability, result, spec, recipe, self.policy)
        return evaluate_one


class _SwingCounter:
    """Counts wave cycles the ARM actually performed, not the ones it was asked
    for.

    An earlier version of this driver reported the requested count whenever the
    episode did not crash, which made "3 of 3 completed" a restatement of the
    recipe rather than a measurement of the robot — and it said exactly that
    about a run where the motors were off and the arm never moved at all.

    A cycle is one excursion to each side.  A side counts as reached when the
    forearm yaw gets within `ARRIVAL_FRACTION` of the commanded amplitude,
    which is deliberately generous: the wave is the move the notebook measured
    the lag ON, and three weak joints reversing together do not arrive in step.
    Being generous about arrival and strict about ORDER is what keeps this a
    count of completed oscillations rather than of times the joint wobbled.
    """

    #: How near the commanded amplitude counts as having reached that side.
    ARRIVAL_FRACTION = 0.6

    def __init__(self, amplitude_deg: float) -> None:
        self._threshold = max(1.0, abs(amplitude_deg) * self.ARRIVAL_FRACTION)
        self._seen_a = False
        self.cycles = 0

    def observe(self, snap) -> None:
        yaw = None
        for j in snap.joints:
            if j.get("name") == "r_forearm_yaw":
                yaw = math.degrees(float(j.get("position_rad", 0.0)))
                break
        if yaw is None:
            return
        if yaw <= -self._threshold:
            self._seen_a = True
        elif yaw >= self._threshold and self._seen_a:
            self.cycles += 1
            self._seen_a = False


def _amplitude_in(recipe: TrajectoryRecipe) -> float:
    spec = (recipe.bounded_parameters or {}).get("wave_amplitude_deg")
    if isinstance(spec, dict):
        spec = spec.get("value", 60.0)
    try:
        return float(spec)
    except (TypeError, ValueError):
        return 60.0


def _cycles_in(recipe: TrajectoryRecipe) -> int:
    spec = (recipe.bounded_parameters or {}).get("wave_cycles")
    if isinstance(spec, dict):
        spec = spec.get("value", 0)
    try:
        return int(spec)
    except (TypeError, ValueError):
        return 0


def _success_sentence(ability: str) -> str:
    """The rule in the words it is applied in, on the stored task spec.

    The issue asks for the success definition to be stated in the evaluator
    rather than only in a comment; this is the same sentence, carried into the
    record so a reader of a stored trial does not have to go and find the code
    that judged it.
    """
    return {
        "rest_forearm": "reached the supported posture, touched only the "
                        "tabletop, and moved no object more than the drift "
                        "tolerance",
        "stow_arm": "reached the rail pocket by walking the measured corridor, "
                    "touching nothing and moving nothing",
        "wave": "completed the cycles asked for within the tracking tolerance "
                "and finished recognisably at the presentation pose",
        "point_cell": "reached the selected hover within half a grid cell, "
                      "kept its clearance, and left the board undisturbed",
        "point_object": "reached the selected hover within half a grid cell, "
                        "kept its clearance, and left the board undisturbed",
    }.get(ability, "")


def _identity_for(model_path: str, scene_path: str = "") -> SimulatorIdentity:
    from reachy_ai.experience.identity import build_simulator_identity
    return build_simulator_identity(model_path=model_path,
                                    scene_path=scene_path or None,
                                    backend_name="native_mujoco")
