"""May the arm move on this recipe?  (issue #89)

WHY THIS IS NOT `query_compatible_trials()`
-------------------------------------------
That method compares two hashes and a status:

    WHERE model_sha256=? AND scene_sha256=? AND status='SUCCEEDED'

`SimulatorIdentity` records eleven more fields — calibration profile, physics
profile, sensor-effect profile, backend, protocol version, scene revision,
scene schema and compiler versions, compiled-model hash — and the query checks
none of them.  `TrialRecord.promotion_state` exists on every row, defaults to
"unpromoted", and is read by nothing at all.

For offline search inside one world that is the right question.  As the input
to a decision about whether a physical-looking arm moves in the operator's
current scene, it is not.  A HISTORICAL SUCCESS IS EVIDENCE FOR A CANDIDATE.
IT IS NOT AUTHORIZATION TO MOVE IN THE CURRENT SCENE.

WHAT THIS DOES INSTEAD
----------------------
One function, `check_reuse`, that returns a decision rather than a boolean:
which fields matched, which did not, and why the answer is what it is.  A
mismatch is a REJECTION, never a score — there is no "close enough" between
two calibration profiles.  Every accepted reuse carries the trial it came from
and the policy version that accepted it, so an episode that goes wrong can be
traced back to the rule that allowed it.

WHAT IT DELIBERATELY REFUSES TODAY
----------------------------------
Everything, in a store that has never promoted anything — which is the correct
answer and not a bug.  Promotion is a reviewed process (EPIC-8 PR 8.7, and
#91's evaluators before it); until one exists, no recipe has been certified
for anything, and the gate says so with a reason rather than quietly saying
yes.  Retrieval that uses this gate is #90, and it is not built here.

WHAT THIS IS NOT
----------------
Not similarity, and not warm starts.  EPIC-8 PR 8.6 proposes a four-level
context hierarchy with nearby-task geometry, in `query.py` and `similarity.py`.
This module is the floor under all of that: the exact-match, promoted-only case
that has to be settled before "nearby" can mean anything.  Nearby geometry with
no compatibility gate underneath is how a recipe measured against one
calibration ends up flown against another.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from .models import EpisodeStatus, SimulatorIdentity

#: The promotion states a recipe may be flown from.  One, on purpose: the
#: store's default is "unpromoted", and every other value would have to be
#: introduced by a reviewed process that says what it means.
PROMOTED = "promoted"

#: Identity fields that must match EXACTLY for a recipe to be reusable.
#:
#: Every one of them can change what a trajectory does to the world.  They are
#: listed here rather than checked inline so that adding or removing one is a
#: visible edit to a named constant, and so `policy_version` can be bumped in
#: the same commit.
IDENTITY_FIELDS: Tuple[str, ...] = (
    "model_sha256",             # the rig geometry itself
    "compiled_model_sha256",    # and what the compiler made of it
    "scene_sha256",
    "scene_revision",           # the board as it stands, not just the file
    "scene_schema_version",
    "scene_compiler_version",
    "physics_profile_id",       # actuator and solver settings
    "calibration_profile_id",
    "sensor_effect_profile_id",
    "backend_name",
    "protocol_version",
)


class CompatibilityError(RuntimeError):
    """Raised when a gate cannot be evaluated at all, as distinct from a
    candidate being rejected.  A rejection is an answer; this is not."""


@dataclasses.dataclass(frozen=True)
class ReusePolicy:
    """The rules in force, frozen and versioned.

    Bump `policy_version` in the same commit as any change to what these mean.
    A decision records the version that made it, and a recorded decision whose
    rules have since changed underneath it is not evidence of anything.
    """

    policy_version: int = 1
    #: Only promoted trials may drive motion.  Turning this off is not a
    #: configuration choice; it is a different gate.
    require_promoted: bool = True
    #: Reject a trial produced from a checkout that was not identifiable.  A
    #: promoted recipe from a dirty tree is a failure of the promotion process
    #: rather than something to accommodate here.
    require_clean_tree: bool = True
    #: Reject a candidate that never recorded which objects were on the board.
    #: "Not recorded" is not "none": a route certified over an empty table is
    #: not certified over one with a can in the corridor.
    require_obstacle_match: bool = True
    #: Reject an episode observed while a person was driving.  It had no seed,
    #: no controlled reset, and a scene that may have been edited mid-session.
    allow_live_interactive: bool = False


@dataclasses.dataclass(frozen=True)
class Mismatch:
    field: str
    wanted: Any
    found: Any

    def __str__(self) -> str:
        return f"{self.field}: wanted {self.wanted!r}, found {self.found!r}"


@dataclasses.dataclass(frozen=True)
class ReuseRequest:
    """What the arm is about to be asked to do, in the world as it is now."""

    task_type: str
    arm: str
    route: str
    route_version: int
    start_posture: str
    #: Object ids on the board right now.  A frozenset because which objects
    #: are present is the question; where they are is the route validator's,
    #: and conflating the two would put a pose comparison in a gate that has
    #: no tolerance to compare poses with.
    obstacles: FrozenSet[str] = frozenset()


@dataclasses.dataclass(frozen=True)
class ReuseCandidate:
    """One stored trial, reduced to the fields the gate compares."""

    trial_id: str
    identity: SimulatorIdentity
    task_type: str
    arm: str
    route: str
    route_version: int
    start_posture: str
    promotion_state: str
    status: str
    success: bool
    live_interactive: bool
    #: None means the trial never recorded one, which is not the same as
    #: having recorded an empty board.
    obstacles: Optional[FrozenSet[str]] = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ReuseCandidate":
        """Build a candidate from an ExperienceStore trial row.

        Tolerant of a row from an older store: a missing column reads as the
        conservative value, never as permission.
        """
        meta = _loads(row.get("optimizer_metadata_json"))
        spec = _loads(row.get("task_spec_json"))
        recipe = _loads(row.get("recipe_json"))
        obstacles = meta.get("obstacles")

        return cls(
            trial_id=str(row.get("trial_id", "")),
            identity=SimulatorIdentity.from_dict(_loads(row.get("identity_json"))),
            task_type=str(row.get("task_type") or spec.get("task_type") or ""),
            arm=str(spec.get("arm_policy") or recipe.get("arm") or ""),
            route=str(recipe.get("route") or ""),
            route_version=int(recipe.get("route_version") or 0),
            start_posture=str(recipe.get("expected_start_posture") or ""),
            promotion_state=str(row.get("promotion_state") or "unpromoted"),
            status=str(row.get("status") or ""),
            success=bool(row.get("success")),
            # Absent in a v1 store, where nothing live could have been
            # written.  Absent in a row is not a claim that it was offline.
            live_interactive=bool(row.get("live_interactive", 0)),
            obstacles=(frozenset(str(o) for o in obstacles)
                       if obstacles is not None else None),
        )


@dataclasses.dataclass(frozen=True)
class ReuseDecision:
    """Why the answer is what it is.  Recorded, whichever way it went."""

    allowed: bool
    trial_id: str
    policy_version: int
    reason: str
    matched: Tuple[str, ...] = ()
    mismatches: Tuple[Mismatch, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "trial_id": self.trial_id,
            "policy_version": self.policy_version,
            "reason": self.reason,
            "matched": list(self.matched),
            "mismatches": [dataclasses.asdict(m) for m in self.mismatches],
        }


def check_reuse(candidate: ReuseCandidate, identity: SimulatorIdentity,
                request: ReuseRequest,
                policy: Optional[ReusePolicy] = None) -> ReuseDecision:
    """May this stored trial's recipe drive THIS movement, in THIS world?

    Raises CompatibilityError when the current identity is degenerate — an
    empty model hash matches everything, and a gate that cannot tell worlds
    apart must refuse to answer rather than answer "yes".
    """
    policy = policy or ReusePolicy()
    if not identity.model_sha256:
        raise CompatibilityError(
            "the current SimulatorIdentity has no model_sha256, so no "
            "candidate can be judged against it. Build the identity from a "
            "real model file."
        )

    def no(reason: str, *mismatches: Mismatch) -> ReuseDecision:
        return ReuseDecision(False, candidate.trial_id, policy.policy_version,
                             reason, mismatches=tuple(mismatches))

    # Outcome first: the cheapest questions, and the ones that make the rest
    # meaningless.  A failed trial's parameters are not a candidate at all.
    if candidate.status != EpisodeStatus.SUCCEEDED.value or not candidate.success:
        return no(f"the trial did not succeed (status {candidate.status!r})",
                  Mismatch("status", EpisodeStatus.SUCCEEDED.value,
                           candidate.status))

    if candidate.live_interactive and not policy.allow_live_interactive:
        return no(
            "the episode was observed while a person was driving, not sampled "
            "under controlled conditions",
            Mismatch("live_interactive", False, True))

    if policy.require_promoted and candidate.promotion_state != PROMOTED:
        return no(
            f"the trial is {candidate.promotion_state!r}. A success is "
            "evidence for a candidate; promotion is what authorises flying it",
            Mismatch("promotion_state", PROMOTED, candidate.promotion_state))

    if policy.require_clean_tree and candidate.identity.working_tree_dirty:
        return no(
            "the trial was produced from a modified checkout, so the code that "
            "made it cannot be identified",
            Mismatch("working_tree_dirty", False, True))

    # The world.
    matched: List[str] = []
    mismatches: List[Mismatch] = []
    for field in IDENTITY_FIELDS:
        wanted = getattr(identity, field)
        found = getattr(candidate.identity, field)
        if wanted == found:
            matched.append(field)
        else:
            # Collected rather than returned on the first one: a reader
            # chasing a rejection wants every field that disagreed, not the
            # alphabetically unluckiest.
            mismatches.append(Mismatch(field, wanted, found))
    if mismatches:
        return ReuseDecision(
            False, candidate.trial_id, policy.policy_version,
            "the recorded world is not this world: "
            + "; ".join(str(m) for m in mismatches),
            tuple(matched), tuple(mismatches))

    # The movement.
    for field, wanted, found in (
        ("task_type", request.task_type, candidate.task_type),
        ("arm", request.arm, candidate.arm),
        ("route", request.route, candidate.route),
        ("route_version", request.route_version, candidate.route_version),
        ("start_posture", request.start_posture, candidate.start_posture),
    ):
        if wanted != found:
            return no(f"the recorded movement is not this movement: "
                      f"{Mismatch(field, wanted, found)}",
                      Mismatch(field, wanted, found))
        matched.append(field)

    # The board.
    if candidate.obstacles is None:
        if policy.require_obstacle_match:
            return no(
                "the trial did not record which objects were on the board, and "
                "a route certified over an unknown board is certified over "
                "nothing",
                Mismatch("obstacles", sorted(request.obstacles), None))
    elif candidate.obstacles != request.obstacles:
        return no(
            "the board is not the board it was certified against",
            Mismatch("obstacles", sorted(request.obstacles),
                     sorted(candidate.obstacles)))
    else:
        matched.append("obstacles")

    return ReuseDecision(
        True, candidate.trial_id, policy.policy_version,
        f"promoted trial {candidate.trial_id} matches this world and this "
        f"movement on every checked field",
        tuple(matched))


def select_reusable(rows: Sequence[Dict[str, Any]], identity: SimulatorIdentity,
                    request: ReuseRequest,
                    policy: Optional[ReusePolicy] = None
                    ) -> Tuple[Optional[ReuseCandidate], List[ReuseDecision]]:
    """The first reusable candidate, and the decision for every row examined.

    Every decision is returned, not just the winning one: "nothing was
    reusable" is a thing a caller has to be able to explain, and the reasons
    are the explanation.  Rows are judged in the order given, so a caller that
    wants newest-first should hand them over that way.
    """
    decisions: List[ReuseDecision] = []
    chosen: Optional[ReuseCandidate] = None
    for row in rows:
        candidate = ReuseCandidate.from_row(row)
        decision = check_reuse(candidate, identity, request, policy)
        decisions.append(decision)
        if decision.allowed and chosen is None:
            chosen = candidate
    return chosen, decisions


def _loads(blob: Optional[str]) -> Dict[str, Any]:
    if not blob:
        return {}
    try:
        loaded = json.loads(blob)
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
