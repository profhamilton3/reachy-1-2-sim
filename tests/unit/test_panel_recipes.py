"""Issue #90: promoted recipes are retrieved; a chat turn never searches.

Offline.  The store is a real SQLite file in tmp_path, the scene and link are
stubs, and nothing here moves anything.

The headline test is the one that fails the build if a typed command could
ever reach `reachy_ai.search`.
"""

import json
import os
import sys
import time
import types

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../web"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

import panel_abilities as abilities  # noqa: E402
from panel_planner import DeterministicPlanner  # noqa: E402
from panel_recipes import RecipeLibrary, build_library  # noqa: E402
from panel_scene import apply_snapshot, scene_view_from_doc  # noqa: E402
from panel_sim_link import SimSnapshot  # noqa: E402
from tasks import PlannerRequest  # noqa: E402

from reachy_ai.experience.compatibility import (KIND_ABILITY_ROUTE,  # noqa: E402
                                                KIND_TRAJECTORY_RECIPE)
from reachy_ai.experience.identity import build_simulator_identity  # noqa: E402
from reachy_ai.experience.models import (EpisodeConfig, EpisodeResult,  # noqa: E402
                                         EpisodeStatus, TaskSpec)
from reachy_ai.experience.store import ExperienceStore  # noqa: E402


class _Cell:
    def __init__(self, name, reachable=True):
        self.name = name
        self.x = 0.3
        self.y = 0.0
        self.top_z = 0.80
        self.half_extent = 0.06
        self.reachable = reachable
        self.shoulder_distance_m = 0.4


def make_scene(live=True):
    cells = {f"r{r}c{c}": _Cell(f"r{r}c{c}") for r in (1, 2, 3) for c in (1, 2, 3)}
    doc = {"name": "TestScene",
           "objects": [{"id": "soda_can", "semantic_class": "can",
                        "tags": ["manipulable", "pickable"]}]}
    view = scene_view_from_doc(doc, cells, placeable=["soda_can"])
    if live:
        c = cells["r1c1"]
        apply_snapshot(view, SimSnapshot(
            scene_revision="rev-1",
            objects={"soda_can": (c.x, c.y, c.top_z + 0.03)},
            received_at=time.monotonic()))
    return view


SCENE_FILE = os.path.join(_HERE, "../../web/panel_recipes.py")   # a real file to hash


def an_identity(scene_revision="rev-1"):
    import dataclasses

    from panel_recipes import _robot_model

    identity = build_simulator_identity(model_path=_robot_model(),
                                        scene_path=os.path.abspath(SCENE_FILE),
                                        backend_name="sdk_bridge")
    # Clean, because these tests are about compatibility and not about the
    # checkout they run in. A trial recorded from a MODIFIED tree is refused
    # by the gate — correctly, and `test_a_trial_from_a_modified_checkout_is_
    # refused` in the gate's own file is where that belongs — but leaving it
    # here would make every test in this file pass for the wrong reason on a
    # developer's machine and fail on a clean CI checkout.
    return dataclasses.replace(identity, scene_revision=scene_revision,
                               working_tree_dirty=False)


def write_trial(db, *, promoted=True, kind=KIND_ABILITY_ROUTE, route="STOW_ROUTE",
                route_version=1, parameters=None, obstacles=("soda_can",),
                live=False, arm="right", identity=None, start_posture="rest"):
    identity = identity or an_identity()
    recipe = {"kind": kind, "arm": arm,
              "expected_start_posture": start_posture,
              "bounded_parameters": parameters or {}}
    if kind == KIND_ABILITY_ROUTE:
        recipe.update(route=route, route_version=route_version)
    else:
        recipe.update(recipe_id="rcp-A", recipe_version=1)
    with ExperienceStore.open(db) as store:
        tid = store.create_trial(
            "study-1",
            TaskSpec(task_id="t", task_type="stow_arm", arm_policy=arm),
            json.dumps(recipe),
            EpisodeConfig(simulator_identity=identity),
            live_interactive=live,
            optimizer_metadata={"obstacles": list(obstacles)},
        )
        store.start_trial(tid)
        store.complete_trial(tid, EpisodeResult(
            episode_id=tid, trial_id=tid, status=EpisodeStatus.SUCCEEDED,
            success=True))
        if promoted:
            store._conn.execute(
                "UPDATE trials SET promotion_state='promoted' WHERE trial_id=?",
                (tid,))
            store._conn.commit()
    return tid


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "recipes.db")


def library(db):
    return RecipeLibrary(SCENE_FILE, db_path=db)


def find(lib, **kw):
    base = dict(task_type="stow_arm", arm="right", route="STOW_ROUTE",
                route_version=1, start_posture="rest",
                obstacles=["soda_can"], tunable=(), scene_revision="rev-1")
    base.update(kw)
    return lib.find(**base)


# ---------------------------------------------------------------------------
# Nothing promoted is the ordinary answer, and not a failure
# ---------------------------------------------------------------------------

def test_an_empty_store_finds_nothing_and_says_nothing(db):
    recipe, reasons = find(library(db))
    assert recipe is None
    assert reasons == []


def test_a_store_that_does_not_exist_finds_nothing(tmp_path):
    lib = RecipeLibrary(SCENE_FILE, db_path=str(tmp_path / "nope.db"))
    assert find(lib) == (None, [])


def test_an_unpromoted_trial_is_not_retrieved(db):
    write_trial(db, promoted=False)
    recipe, reasons = find(library(db))
    assert recipe is None
    assert any("unpromoted" in r for r in reasons)


def test_a_live_interactive_episode_is_not_retrieved(db):
    """Excluded by the store before the gate is even asked — two independent
    refusals for one row, which is the arrangement we want for this one."""
    write_trial(db, live=True)
    recipe, reasons = find(library(db))
    assert recipe is None
    assert reasons == []

    # And if it did reach the gate, the gate refuses it too.
    from reachy_ai.experience.compatibility import (ReuseCandidate,
                                                    ReuseRequest, check_reuse)
    with ExperienceStore.open(db) as store:
        row, = store.query_compatible_trials(an_identity(), task_type="stow_arm",
                                             include_live_interactive=True)
    decision = check_reuse(ReuseCandidate.from_row(row), an_identity(),
                           ReuseRequest(task_type="stow_arm", arm="right",
                                        route="STOW_ROUTE", route_version=1,
                                        start_posture="rest",
                                        obstacles=["soda_can"]))
    assert not decision.allowed
    assert "a person was driving" in decision.reason


def test_a_promoted_compatible_trial_is_retrieved(db):
    tid = write_trial(db)
    recipe, reasons = find(library(db))
    assert recipe is not None
    assert recipe.trial_id == tid
    assert recipe.policy_version == 1


def test_a_promoted_but_incompatible_trial_is_refused_with_a_reason(db):
    import dataclasses
    write_trial(db, identity=dataclasses.replace(
        an_identity(), calibration_profile_id="lab-2026-09"))
    recipe, reasons = find(library(db))
    assert recipe is None
    assert any("calibration_profile_id" in r for r in reasons)


def test_a_board_nobody_read_matches_nothing(db):
    """A plan made against an unread board must not be matched to a recipe
    certified over a board somebody did read."""
    write_trial(db)
    recipe, reasons = find(library(db), obstacles=None)
    assert recipe is None
    assert any("could not read which objects" in r for r in reasons)


def test_a_broken_store_costs_the_default_route_and_not_the_plan(db, monkeypatch):
    write_trial(db)
    lib = library(db)

    def explode(*a, **kw):
        raise RuntimeError("the disk is on fire")

    monkeypatch.setattr(lib, "_find", explode)
    assert lib.find(task_type="stow_arm", arm="right", route="STOW_ROUTE",
                    route_version=1, start_posture="rest",
                    obstacles=["soda_can"]) == (None, [])


# ---------------------------------------------------------------------------
# Parameters an ability cannot apply are refused, not applied
# ---------------------------------------------------------------------------

def test_a_recipe_varying_something_the_ability_cannot_apply_is_refused(db):
    write_trial(db, parameters={"segment_duration_s": 1.4})
    recipe, reasons = find(library(db))
    assert recipe is None
    assert any("has no way to apply" in r for r in reasons)


def test_a_recipe_varying_a_declared_parameter_is_retrieved(db):
    write_trial(db, parameters={"segment_duration_s": 1.4})
    recipe, _ = find(library(db), tunable=("segment_duration_s",))
    assert recipe is not None
    assert recipe.parameters == {"segment_duration_s": 1.4}


def test_a_searched_trajectory_recipe_cannot_be_asked_for_by_route(db):
    """A `trajectory_recipe` is identified by an id the panel has no way to
    ask for, so the ability path can never match one. Refused on the movement
    comparison, before the parameter contract is reached."""
    write_trial(db, kind=KIND_TRAJECTORY_RECIPE)
    recipe, reasons = find(library(db))
    assert recipe is None
    assert any("recipe_id" in r for r in reasons)


def test_no_ability_declares_a_tunable_parameter_today():
    """The routes are fixed sequences of measured waypoints. When #91 changes
    that, this test is the reminder that the executor guard must change too."""
    assert all(a.tunable == () for a in abilities.REGISTRY.values())


# ---------------------------------------------------------------------------
# The plan says where its motion came from
# ---------------------------------------------------------------------------

def plan(scene, text, recipes=None):
    planner = DeterministicPlanner(lambda: scene, recipes=recipes)
    return planner(PlannerRequest(text=text, history=[]))


def test_with_nothing_promoted_the_plan_is_what_it_always_was(db):
    out = plan(make_scene(), "stow your arm", recipes=library(db))
    assert out.kind == "proposal"
    assert out.proposal.route == "STOW_ROUTE"
    assert out.proposal.recipe_trial_id == ""
    assert out.proposal.recipe_parameters == {}
    assert "recipe" not in out.proposal.summary


def test_a_retrieved_recipe_is_named_on_the_card(db):
    tid = write_trial(db, start_posture="")      # stow_arm declares none
    out = plan(make_scene(), "stow your arm", recipes=library(db))
    assert out.kind == "proposal"
    assert out.proposal.recipe_trial_id == tid
    # Named before the operator confirms.  A plan that silently flew somebody
    # else's search result is the one thing retrieval must never do.
    assert "promoted recipe" in out.proposal.summary
    assert tid[:8] in out.proposal.summary


def test_a_planner_with_no_library_never_looks(db):
    write_trial(db)
    out = plan(make_scene(), "stow your arm", recipes=None)
    assert out.proposal.recipe_trial_id == ""


# ---------------------------------------------------------------------------
# A chat turn never starts a search
# ---------------------------------------------------------------------------

def test_no_panel_module_imports_the_search_engine():
    """`reachy_ai.search` is an offline, deliberate act with a CLI of its own.
    Nothing reachable from a typed command may import it.

    Read as code rather than grepped: these modules discuss the rule in their
    own comments, and a prose mention is not an import.
    """
    import ast
    import pathlib

    web = pathlib.Path(_HERE, "../../web").resolve()
    offenders = []
    for path in sorted(web.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                names += [f"{node.module or ''}.{a.name}" for a in node.names]
            if any(n == "reachy_ai.search" or n.startswith("reachy_ai.search.")
                   for n in names):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


def test_planning_a_wave_imports_nothing_from_the_search_engine(db):
    """And dynamically, because a module can be imported by a string.

    An ordinary "Wave." must not begin exploring parameters around the rig.
    """
    write_trial(db)
    loaded = set(sys.modules)
    plan(make_scene(), "wave", recipes=library(db))
    new = {m for m in set(sys.modules) - loaded if "search" in m}
    assert new == set()


def test_the_library_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("REACHY_PANEL_RECIPES", "0")
    assert build_library(SCENE_FILE) is None
    monkeypatch.setenv("REACHY_PANEL_RECIPES", "1")
    assert build_library(SCENE_FILE) is not None
