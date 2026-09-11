"""Promoted recipes, retrieved for planning — and never a search (issue #90).

WHAT THIS MAY DO
----------------
Look in the experience store for a PROMOTED recipe that passes the reuse gate
(`reachy_ai.experience.compatibility`, #89) for the ability being planned, in
the world as it stands, and hand it to the planner so the proposal the
operator confirms can NAME it: which recipe, from which trial, and that it was
promoted.

WHAT THIS MAY NOT DO
--------------------
START A SEARCH.  An ordinary "Wave." must never begin exploring parameters
around the rig.  Search is an offline, deliberate act by someone who chose to
run it, with `reachy_ai.search` and its CLI; nothing reachable from a typed
command may import it, and `tests/unit/test_panel_recipes.py` fails the build
if that ever becomes false.

It also may not reuse an unpromoted trial however good its metrics, reuse a
live-interactive episode, reorder a route or drop a check to improve a score,
or substitute anything silently.  The gate enforces the first two; the
`tunable` contract below enforces the third; naming the recipe on the proposal
is the fourth.

THE NORMAL ANSWER IS "NOTHING"
------------------------------
No promoted recipe is the ordinary case today and for as long as promotion
does not exist as a process.  It is not a failure and must not read as one:
the ability flies its registry route exactly as it does now, and the operator
is told nothing, because nothing happened.

WHAT AN ABILITY CAN ACCEPT
--------------------------
`Ability.tunable` lists the parameter names that ability knows how to apply.
It is empty for every ability today — the measured routes have no parameters
until #91 defines them — so a recipe carrying parameters is REFUSED rather
than applied by hope.  "The search found a better segment duration" means
nothing until something can act on it, and pretending otherwise is exactly the
silent substitution this issue forbids.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("panel.recipes")

#: How many promoted candidates to examine per plan.  Bounded: this runs on
#: the planning path, which a human is waiting on.
MAX_CANDIDATES = 25


def _ensure_paths() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "..", "src"), "/opt/src"):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)


#: The robot MJCF the simulator compiles.  `model_sha256` means this file, and
#: without it the identity is degenerate — the gate refuses to answer about a
#: world it cannot tell apart from any other, and `query_compatible_trials`
#: raises rather than matching everything.  It lives in the repo, so the panel
#: can hash it even though it never loads it: the panel talks to the simulator
#: over the SDK bridge and has no model of its own.
def _robot_model() -> str:
    for candidate in (pathlib.Path(__file__).resolve().parent.parent
                      / "native_mujoco" / "model" / "reachy_1_2.xml",
                      pathlib.Path("/opt/native_mujoco/model/reachy_1_2.xml")):
        if candidate.is_file():
            return str(candidate)
    return ""


@dataclasses.dataclass(frozen=True)
class RetrievedRecipe:
    """A promoted recipe the gate allowed, and where it came from."""

    trial_id: str
    recipe_id: str
    recipe_version: int
    parameters: Dict[str, Any]
    policy_version: int
    study_id: str = ""

    def describe(self) -> str:
        """What the operator is told before they confirm."""
        what = self.recipe_id or f"trial {self.trial_id[:8]}"
        return (f"flying promoted recipe {what} "
                f"(v{self.recipe_version}, trial {self.trial_id[:8]})")


class RecipeLibrary:
    """Reads promoted recipes.  Never writes, never searches, never raises."""

    def __init__(self, scene_file: str = "", *, db_path: str = "") -> None:
        self._scene_file = scene_file
        self._db_path = db_path or str(
            pathlib.Path(__file__).resolve().parent.parent
            / "runs" / "panel_episodes.db")
        self._identity = None
        self._identity_key: Optional[Tuple[Any, ...]] = None

    def find(self, *, task_type: str, arm: str, route: str, route_version: int,
             start_posture: str, obstacles: Optional[Sequence[str]],
             tunable: Sequence[str] = (), scene_revision: str = "",
             ) -> Tuple[Optional[RetrievedRecipe], List[str]]:
        """The recipe to fly, and the reasons every candidate was turned down.

        Returns (None, []) when there is nothing to consider at all, which is
        the ordinary case and not an error.  Returns (None, reasons) when
        candidates existed and none was allowed — the reasons are what makes
        "nothing was reusable" explainable rather than merely true.
        """
        try:
            return self._find(task_type, arm, route, route_version,
                              start_posture, obstacles, tunable, scene_revision)
        except Exception:
            # A store that cannot be read costs the default route, which is
            # the route the ability flies anyway.  It must not cost a plan.
            log.exception("could not read the recipe store")
            return None, []

    def _find(self, task_type, arm, route, route_version, start_posture,
              obstacles, tunable, scene_revision):
        if not os.path.exists(self._db_path):
            return None, []

        _ensure_paths()
        from reachy_ai.experience.compatibility import (ReuseRequest,
                                                        select_reusable)
        from reachy_ai.experience.store import ExperienceStore

        identity = self._identity_for(scene_revision)
        if not identity.model_sha256:
            # The gate would refuse to answer, and rightly: an empty hash
            # matches every world.  Asking it anyway would only convert a
            # clear refusal into an exception on the planning path.
            return None, []

        request = ReuseRequest(
            task_type=task_type, arm=arm, route=route,
            route_version=route_version, start_posture=start_posture,
            obstacles=frozenset(obstacles) if obstacles is not None else frozenset(),
        )

        with ExperienceStore.open(self._db_path, recover_orphans=False) as store:
            rows = store.query_compatible_trials(
                identity, task_type=task_type, limit=MAX_CANDIDATES)
        if not rows:
            return None, []

        # A BOARD NOBODY LOOKED AT IS NOT AN EMPTY BOARD.  The gate says so
        # about the candidate; the same has to hold for the request, or a
        # plan made against an unread scene would be matched to a recipe
        # certified over a specific one.
        if obstacles is None:
            return None, ["I could not read which objects are on the board, "
                          "so I will not match this against a recorded one."]

        chosen, decisions = select_reusable(rows, identity, request)
        reasons = [d.reason for d in decisions if not d.allowed]
        if chosen is None:
            return None, reasons

        parameters = _parameters_of(chosen)
        unknown = sorted(set(parameters) - set(tunable))
        if unknown:
            # Refused, not applied.  An ability that cannot act on a parameter
            # would fly its default route while the card claimed a promoted
            # recipe — the silent substitution this issue exists to forbid.
            return None, reasons + [
                f"promoted trial {chosen.trial_id[:8]} varies "
                f"{', '.join(unknown)}, which this ability has no way to apply"]

        winner = next(d for d in decisions if d.allowed)
        return RetrievedRecipe(
            trial_id=chosen.trial_id,
            recipe_id=chosen.recipe_id,
            recipe_version=chosen.recipe_version,
            parameters=parameters,
            policy_version=winner.policy_version,
        ), reasons

    def _identity_for(self, scene_revision: str):
        from reachy_ai.experience.identity import build_simulator_identity

        absolute = os.path.abspath(self._scene_file) if self._scene_file else ""
        try:
            stat = os.stat(absolute) if absolute else None
        except OSError:
            stat = None
        key = (absolute, stat.st_mtime_ns if stat else 0,
               stat.st_size if stat else 0)
        if self._identity is None or self._identity_key != key:
            self._identity = build_simulator_identity(
                model_path=_robot_model(), scene_path=absolute,
                backend_name="sdk_bridge")
            self._identity_key = key
        return dataclasses.replace(self._identity, scene_revision=scene_revision)


def _parameters_of(candidate) -> Dict[str, Any]:
    """What this candidate varies, whatever shape it was stored in.

    An ability route varies nothing today — it is a fixed sequence of measured
    waypoints — so this is empty for every row currently in any store.  It is
    read for ability routes all the same, because that is the shape #91's
    searched ability recipes will take: a route, with bounded parameters over
    it.  A `trajectory_recipe` is identified by a recipe id the panel has no
    way to ask for, so the gate refuses those on the movement comparison
    before this is ever reached.
    """
    return dict(candidate.bounded_parameters)


def build_library(scene_file: str = "") -> Optional[RecipeLibrary]:
    """The library for this deployment, or None when retrieval is switched off.

    On by default, because with nothing promoted it changes nothing: it looks,
    finds none, and the ability flies the route it always flies.
    `REACHY_PANEL_RECIPES=0` turns it off; `REACHY_PANEL_RECIPE_DB` points it
    at a study's own file.
    """
    if os.environ.get("REACHY_PANEL_RECIPES", "1").lower() in (
            "0", "false", "no", "off"):
        return None
    return RecipeLibrary(scene_file,
                         db_path=os.environ.get("REACHY_PANEL_RECIPE_DB", ""))
