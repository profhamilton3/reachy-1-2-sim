"""Issue #91: the ability recipes fly, and a search over one completes offline.

Needs MuJoCo and the native model; skipped without either.  These are the
tests that would have caught the two things constructed EpisodeResults cannot:
that the arm was compliant and never moved, and that flying the stow corridor
from the home keyframe cuts straight across it.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
for _p in (_REPO / "native_mujoco", _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reachy_ai.evaluation.panel_routes import (  # noqa: E402
    ABILITY_ROUTES, check_route_integrity, evaluate,
)
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.recipe import TrajectoryRecipe  # noqa: E402
from reachy_ai.motion.recipe_executor import RecipeExecutionError  # noqa: E402

_MODEL = _REPO / "native_mujoco" / "model" / "reachy_1_2.xml"
_SCENE = _REPO / "scenes" / "FWDCenterLabSivaPool.yaml"
_RECIPES = _REPO / "recipes" / "panel"

#: Flying a corridor is ~20k physics steps, so these are the slow ones.
FLYABLE = ("wave", "stow_arm", "rest_forearm")


def _skip_without_mujoco():
    pytest.importorskip("mujoco")
    if not _MODEL.exists():
        pytest.skip(f"native model not present: {_MODEL}")


def load(ability):
    return TrajectoryRecipe.load(str(_RECIPES / f"{ability}_baseline_v1.yaml"))


def runner_for(ability):
    from reachy_ai.evaluation.panel_offline import OfflineRouteRunner
    return OfflineRouteRunner(ability, str(_MODEL), str(_SCENE))


# ---------------------------------------------------------------------------
# Pointing is refused, out loud
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ability", ["point_cell", "point_object"])
def test_pointing_cannot_be_flown_offline_and_says_why(ability):
    """The recipe, the evaluator and the integrity check for pointing all
    exist; the offline driver does not.  Inventing a hover here would produce
    trials about a motion the robot does not fly, which is worse than the
    refusal."""
    from reachy_ai.evaluation.panel_offline import executable

    ok, why = executable(ability)
    assert not ok
    assert "derived from the scene" in why


# ---------------------------------------------------------------------------
# The routes actually move the arm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ability", FLYABLE)
def test_the_route_flies_and_the_arm_ends_where_it_promised(ability):
    _skip_without_mujoco()
    recipe = load(ability)
    runner = runner_for(ability)
    result, spec = runner.run(recipe, seed=0)

    assert result.final_joint_positions_deg, \
        "the episode recorded no final pose, so nothing can be judged"

    verdict = evaluate(ability, result, spec, recipe, runner.policy)
    assert verdict.is_valid, verdict.explanation
    assert verdict.is_safe, verdict.explanation
    assert verdict.is_successful, verdict.explanation


@pytest.mark.parametrize("ability", ["stow_arm", "rest_forearm"])
def test_the_corridor_actually_carries_the_arm_somewhere(ability):
    """THE ARM IS COMPLIANT AT POWER-ON, which mirrors the physical robot, and
    nothing offline used to stiffen it.  Every command went to a limp arm that
    sat exactly where the home keyframe left it and reported a clean,
    undisturbed, entirely motionless episode — a search over which would have
    ranked physics noise and called it a trajectory.

    Asserted as "it left where it was PUT", not "it is away from home": stow
    legitimately finishes at home, having started at rest."""
    _skip_without_mujoco()
    from joint_map import by_name

    recipe = load(ability)
    runner = runner_for(ability)
    start = runner._start_pose(recipe)
    began = {j: 0.0 for j in R.GROSS_JOINTS}
    for j in R.GROSS_JOINTS:
        if j in start:
            began[j] = by_name(j).mjcf_rad_to_sdk_deg(start[j])

    result, _ = runner.run(recipe, seed=0)
    ended = {j: result.final_joint_positions_deg[j] for j in R.GROSS_JOINTS}
    moved = {j: abs(ended[j] - began[j]) for j in R.GROSS_JOINTS}
    assert any(v > 5.0 for v in moved.values()), \
        f"the arm never left its starting posture: began {began}, ended {ended}"


def test_the_wave_actually_swings():
    """The wave begins and ends at PRESENT, so its gross joints are unchanged
    by design — what moves is the forearm yaw.  With the motors off, that
    excursion is exactly what does not happen, and every other check in this
    file still passes."""
    _skip_without_mujoco()
    result, _ = runner_for("wave").run(load("wave"), seed=0)
    assert result.metrics.get("wave_cycles_completed", 0) > 0, \
        "the forearm never reached either side of the swing"


def test_the_stow_corridor_starts_at_rest_not_at_the_pocket():
    """A direct move from anywhere over the board to HOME drives the upper arm
    through the board's near edge.  Flying STOW_ROUTE from the home keyframe
    interpolates straight into REST_SHUT, which is that move."""
    _skip_without_mujoco()
    runner = runner_for("stow_arm")
    pose = runner._start_pose(load("stow_arm"))
    assert pose, "stow must be placed at REST before it flies"

    from joint_map import by_name
    assert by_name("r_shoulder_pitch").mjcf_rad_to_sdk_deg(
        pose["r_shoulder_pitch"]) == pytest.approx(R.REST["r_shoulder_pitch"])
    # Only the joints REST names, so the neck keeps its keyframe pitch.
    assert "neck_pitch" not in pose


def test_the_offline_world_has_the_board_and_the_rails():
    """#172: `_load_world` resolves `extends:`, as the server does.

    Until it did, the offline `FWDCenterLabSivaPool` world had no table and
    no rails — only the floor, the pedestal and the pool objects — so no route
    flown here could touch the board or the rig the corridor's geometry was
    measured against, and an empty contact record said nothing about the
    corridor.  Now they are collidable, and the verdict names them."""
    _skip_without_mujoco()
    from reachy_ai.evaluation.panel_routes import COLLIDABLE_KEY

    result, spec = runner_for("wave").run(load("wave"), seed=0)
    collidable = result.contact_summary.get(COLLIDABLE_KEY)
    assert "table_top" in collidable
    assert sum(b.startswith("rig_rail") for b in collidable) == 5
    assert "world" in collidable


@pytest.mark.parametrize("ability", FLYABLE)
def test_the_panel_routes_make_no_unintended_contact(ability):
    """Every contact a measured route makes in the world WITH the board and
    rails is allowed by the shared rule table: rest and stow lay the forearm
    on the board at REST, which is their declared route exception, and
    nothing else touches anything."""
    _skip_without_mujoco()
    from reachy_ai.evaluation import contact_rules as CR
    from reachy_ai.evaluation.base import ViolationKind
    from reachy_ai.evaluation.panel_routes import contact_samples

    runner = runner_for(ability)
    recipe = load(ability)
    result, spec = runner.run(recipe, seed=0)
    judged = CR.judge_all(contact_samples(result), route=spec.route)
    assert all(j.allowed for j in judged), [j.describe() for j in judged
                                            if not j.allowed]
    made = {(j.sample.body1, j.sample.body2, j.sample.pose) for j in judged}
    if ability in ("rest_forearm", "stow_arm"):
        assert made == {("r_forearm", "table_top", "rest")}
    else:
        assert made == set()

    verdict = evaluate(ability, result, spec, recipe, runner.policy)
    assert not any(v.kind is ViolationKind.FORBIDDEN_CONTACT
                   for v in verdict.violations), verdict.explanation


# ---------------------------------------------------------------------------
# A search over one ability completes, and reports the spread
# ---------------------------------------------------------------------------

def test_a_search_completes_offline_and_exports_a_best_recipe(tmp_path):
    """The acceptance run, shrunk: a budget of two over three seeds with a
    held-out finalist stage.  What is asserted is that it finishes, that the
    winner was re-evaluated on seeds it was not searched on, that the spread
    across seeds is recorded rather than a single rollout, and that the
    exported recipe is still a variation of the measured route."""
    _skip_without_mujoco()
    from reachy_ai.evaluation.panel_routes import align_to_parameters
    from reachy_ai.search.runner import (SearchConfig, SearchRunner,
                                         apply_best_to_recipe)

    baseline = load("wave")
    runner = runner_for("wave")
    config = SearchConfig(
        study_id="test-wave", baseline_recipe=baseline, budget=2,
        sampler="random", random_seed=3, eval_seeds=[0, 1, 2],
        eval_aggregation="pessimistic", finalist_k=1, finalist_seeds=[11, 12],
    )
    result = SearchRunner(config).run(runner.evaluate_fn())

    assert result.trials_run == 2
    assert result.best is not None, "no candidate was valid, safe and successful"

    # THE SPREAD, NOT JUST THE MEAN.  #20 and #19 were both single-seed bugs.
    stds = [k for k in result.best.verdict.metrics if k.endswith("_std")]
    assert stds, "a mean with no spread beside it is a single-rollout winner"

    # And the winner was re-checked on seeds the search never saw.
    assert result.finalist_verdicts

    winner = align_to_parameters(apply_best_to_recipe(result, baseline))
    assert winner is not None
    assert check_route_integrity(winner, "WAVE") == [], \
        "the search produced a recipe its own integrity check refuses"
    assert winner.source == "search_best"


def test_a_failed_trial_stays_in_the_store(tmp_path):
    """A failed trial is evidence about the bound.  Nothing prunes them."""
    _skip_without_mujoco()
    import sqlite3

    from reachy_ai.experience.store import ExperienceStore
    from reachy_ai.search.runner import SearchConfig, SearchRunner

    db = str(tmp_path / "study.db")
    baseline = load("wave")
    runner = runner_for("wave")

    calls = {"n": 0}
    real = runner.evaluate_fn()

    def sometimes_fails(recipe, seed):
        calls["n"] += 1
        verdict = real(recipe, seed)
        if calls["n"] == 1:
            verdict.is_successful = False
            verdict.is_safe = False
        return verdict

    config = SearchConfig(study_id="test-keep", baseline_recipe=baseline,
                          budget=2, sampler="random", random_seed=1,
                          eval_seeds=[0], finalist_seeds=[])
    with ExperienceStore.open(db) as store:
        SearchRunner(config).run(sometimes_fails, store=store)

    conn = sqlite3.connect(db)
    rows = list(conn.execute("SELECT status, success FROM trials"))
    conn.close()
    assert len(rows) >= 2, "trials were pruned out of the store"


def test_a_model_whose_timestep_moved_fails_loudly(monkeypatch):
    """Recipes are built in STEPS and the model decides what a step is.

    `recipe_executor` is deliberately free of any native_mujoco dependency, so
    it converts a waypoint's measured seconds using its own constant.  If the
    model's timestep moves away from that, every duration in every recipe
    quietly means something else and a study goes on reporting seconds it never
    flew."""
    _skip_without_mujoco()
    from reachy_ai.evaluation.panel_offline import OfflineRouteRunner
    from simulation_core import SimulationCore

    real = SimulationCore.from_paths

    def slower(model_path, scene_path=None):
        core = real(model_path, scene_path)
        core.model.opt.timestep = 0.004
        return core

    monkeypatch.setattr(SimulationCore, "from_paths", staticmethod(slower))
    with pytest.raises(RecipeExecutionError) as e:
        OfflineRouteRunner("wave", str(_MODEL), str(_SCENE))
    assert "2.00x what it says" in str(e.value)


@pytest.mark.parametrize("ability", ["stow_arm", "wave"])
def test_the_first_command_is_where_the_arm_actually_is(ability):
    """PLACING THE ARM AND BUILDING FROM HOME ARE TWO HALVES OF ONE THING.

    `start_pose_rad` put the arm at REST; `build_commands` then interpolated
    from the home keyframe regardless, so the FIRST commanded target was ~0
    degrees — the stiffened arm driven from over the board straight back
    toward the pocket, outside the corridor, before the corridor's first
    waypoint.  That is the exact unmeasured transit STOW_ROUTE exists to
    forbid, and every duration, clearance and drift number from those episodes
    described it.

    Doing only the placement was worse than doing neither, because it looked
    fixed.  This measures the thing that was actually wrong: the gap between
    where the arm is put and what it is first asked for.
    """
    _skip_without_mujoco()
    from joint_map import by_name
    from reachy_ai.evaluation.panel_offline import _place

    recipe = load(ability)
    runner = runner_for(ability)
    start = runner._start_pose(recipe)
    assert start, f"{ability} has a starting posture to be placed at"

    runner._core.reset(seed=0)
    _place(runner._core, start)
    commands = runner._executor.build_commands(recipe, start_qpos=runner._qpos())

    for joint in R.GROSS_JOINTS:
        entry = by_name(joint)
        placed = entry.mjcf_rad_to_sdk_deg(start[joint])
        first = entry.mjcf_rad_to_sdk_deg(commands[0].target_rad[entry.mjcf_index])
        assert abs(first - placed) < 1.0, (
            f"{joint}: the arm is placed at {placed:.1f} deg and the first "
            f"command asks for {first:.1f} deg")


def test_the_placement_leaves_the_head_where_the_keyframe_put_it():
    """Zero-filling the joints the posture does not name resets the neck, and
    the neck actuators are stiff by default so the goal is pinned there —
    which left placed episodes running head-level and unplaced ones pitched at
    the workspace.  Two populations with different head states are not
    comparable, and nothing said so."""
    _skip_without_mujoco()
    import mujoco

    from reachy_ai.evaluation.panel_offline import _place

    runner = runner_for("stow_arm")
    core = runner._core
    core.reset(seed=0)
    jid = mujoco.mj_name2id(core.model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch")
    adr = int(core.model.jnt_qposadr[jid])
    before = float(core.data.qpos[adr])

    _place(core, runner._start_pose(load("stow_arm")))
    assert float(core.data.qpos[adr]) == pytest.approx(before)


def test_the_recipe_stored_with_a_trial_is_the_one_that_ran(tmp_path):
    """A candidate sampled to one cycle was stored with `wave_cycles: 1`
    beside the baseline's six swing steps — a combination the integrity check
    rejects, describing a motion that was never flown."""
    _skip_without_mujoco()
    import json
    import sqlite3

    from reachy_ai.evaluation.panel_routes import align_to_parameters
    from reachy_ai.experience.store import ExperienceStore
    from reachy_ai.motion.recipe import TrajectoryRecipe as TR
    from reachy_ai.search.runner import SearchConfig, SearchRunner

    db = str(tmp_path / "stored.db")
    config = SearchConfig(
        study_id="test-stored", baseline_recipe=load("wave"), budget=2,
        sampler="random", random_seed=5, eval_seeds=[0], finalist_seeds=[],
        recipe_normaliser=align_to_parameters,
    )
    with ExperienceStore.open(db) as store:
        SearchRunner(config).run(runner_for("wave").evaluate_fn(), store=store)

    conn = sqlite3.connect(db)
    recipes = [r[0] for r in conn.execute("SELECT recipe_json FROM trials")]
    conn.close()
    assert recipes

    for raw in recipes:
        stored = TR.from_dict(json.loads(raw))
        assert check_route_integrity(stored, "WAVE") == [], \
            "a stored recipe describes a motion that was never flown"
