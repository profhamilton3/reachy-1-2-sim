"""Deterministic command parser for the panel's planning-only mode (issue #49).

No Coral, no credentials, no cloud model, and no pretence of general language
understanding.  It recognises a small, documented set of phrasings, and when it
does not recognise something it says what it *does* accept rather than guessing.
A language-model adapter can be added later behind the same `PlannerOutcome`
contract; the point of starting here is that every proposal this produces can be
explained from the scene file and the text, with nothing inferred.

SUPPORTED FORMS
---------------
    put   <object> in|into|on|onto|to <destination>
    move  <object> ...
    place <object> ...
    drop  <object> ...

<object> is an object id (`soda_can`, or "soda can"), or a recyclable phrase
("the recycle item", "the recyclable", "recycling").
<destination> is a grid cell (`r2c2`), a tagged scene destination, or a bin
phrase ("the bin", "the trash").

WHY DESTINATION IS RESOLVED BEFORE TARGET
-----------------------------------------
For the worked example — "Put the recycle item in the bin" on the pool scene —
both halves have something to say, but the missing bin is the more useful thing
to hear first and the one that has to be settled before any target matters.  It
also matches the UX mock in docs/Reachy-Command-Panel-Design.md.

WHAT IT REFUSES OUTRIGHT
------------------------
Anything that reads as raw joint control.  The project's standing rule is that
joint angles never originate from a language layer, and a parser that quietly
ignored "set r_elbow_pitch to -75" would be one edit away from honouring it.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import panel_abilities as abilities
from panel_abilities import AbilityRefusal
from panel_scene import (NON_RECYCLABLE_TAG, RECYCLABLE_TAG, DestinationRef,
                         SceneView)
from tasks import (ConversationEvent, IntentState, PlannerOutcome,
                   PlannerRequest, Proposal)

#: How far a tracked object may drift before a proposal built on its position
#: stops describing the world.  Objects at rest in MuJoCo jitter by far less
#: than this; anything larger is a placement, a recall, a reset, or the arm.
POSE_TOLERANCE_M = 0.02

_VERBS = ("put", "move", "place", "drop", "sort")
_PREPS = ("into", "onto", "in", "on", "to")

_RECYCLE_WORDS = ("recycle", "recycling", "recyclable")
_BIN_WORDS = ("bin", "trash", "garbage", "waste", "rubbish")

# The recycling word itself, and the negation that may sit in front of it.
# `non-recyclable` normalises with the hyphen intact, so the negation is
# matched with an optional hyphen rather than as a separate word: "non" and
# "non-" are the same claim.
_RECYCLE_RE = re.compile(rf"\b({'|'.join(_RECYCLE_WORDS)})\b")
_NEGATION_RE = re.compile(r"(\bnon-?|\bnot\b|\bno\b|n't\b)\s*$")

_CELL_RE = re.compile(r"\br([1-9])c([1-9])\b", re.I)
# No trailing \b on the alternation: joint names carry a suffix
# (`r_elbow_pitch`), and an underscore is a word character, so a boundary there
# would never match the very names this is meant to catch.  Same for the plural
# in "degrees".  `rad`/`deg` keep their own boundary so "radius" and "degrade"
# do not trip it.
_JOINT_RE = re.compile(
    r"\b(joints?|radians?|rad\b|degrees?|deg\b|"
    r"[lr]_(shoulder|elbow|forearm|wrist|arm|gripper)|"
    r"neck_(roll|pitch|yaw))",
    re.I,
)

_HELP = ("I understand commands like \"put soda_can on r2c2\" or "
         "\"put the recycle item in the bin\".")


@dataclass
class _Intent:
    verb: str = ""
    target_phrase: str = ""
    dest_phrase: str = ""


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _strip_article(phrase: str) -> str:
    return re.sub(r"^(the|a|an|my)\s+", "", phrase).strip()


def parse_intent(text: str) -> Optional[_Intent]:
    """Split a command into verb / target / destination, or None if unrecognised."""
    norm = _normalise(text).rstrip(".!?")
    words = norm.split()
    if not words or words[0] not in _VERBS:
        return None
    intent = _Intent(verb=words[0])
    rest = " ".join(words[1:])

    # Longest preposition first so "into" is not split as "in" + "to".
    for prep in sorted(_PREPS, key=len, reverse=True):
        m = re.search(rf"\b{prep}\b", rest)
        if m:
            intent.target_phrase = _strip_article(rest[: m.start()].strip())
            intent.dest_phrase = _strip_article(rest[m.end():].strip())
            return intent
    intent.target_phrase = _strip_article(rest)
    return intent


def _match_object_id(phrase: str, scene: SceneView) -> Optional[str]:
    """Match an object id, tolerating "soda can" for `soda_can`."""
    if not phrase:
        return None
    key = re.sub(r"[\s-]+", "_", _strip_article(_normalise(phrase)))
    for oid in scene.objects:
        if oid.lower() == key:
            return oid
    return None


def _match_destination_id(phrase: str, scene: SceneView) -> Optional[str]:
    if not phrase:
        return None
    key = re.sub(r"[\s-]+", "_", _strip_article(_normalise(phrase)))
    for did in scene.destinations:
        if did.lower() == key:
            return did
    return None


def _match_cell(phrase: str, scene: SceneView) -> Optional[str]:
    m = _CELL_RE.search(phrase or "")
    if not m:
        return None
    name = f"r{m.group(1)}c{m.group(2)}".lower()
    for cell in scene.cells:
        if cell.lower() == name:
            return cell
    return name          # a well-formed cell name the scene does not have


def _category_of(phrase: str) -> Optional[str]:
    """Which recycling category a phrase names, or None if it names neither.

    Substring matching is what made this wrong.  `"recyclable" in
    "non-recyclable item"` is True, so the phrase that explicitly EXCLUDES the
    category selected it, and the panel proposed moving the one object the
    operator had ruled out.  The scene layer has always matched tags exactly
    for this reason; the test that undid it lived here, a layer earlier, where
    correct tag handling downstream could not save it.

    So: find the recycling word, then look at what sits immediately before it.
    A negation there means the complementary category, which is a tag of its
    own (`non-recyclable`) and is resolved exactly like the positive one.
    """
    norm = _normalise(phrase)
    m = _RECYCLE_RE.search(norm)
    if m is None:
        return None
    before = norm[: m.start()]
    if _NEGATION_RE.search(before):
        return NON_RECYCLABLE_TAG
    return RECYCLABLE_TAG


def _is_bin_phrase(phrase: str) -> bool:
    norm = _normalise(phrase)
    return any(re.search(rf"\b{w}\b", norm) for w in _BIN_WORDS)


#: Slot names, from the registry so a clarification here and an ability there
#: cannot drift into naming the same slot differently.
SLOT_CELL = abilities.SLOT_CELL
SLOT_OBJECT = abilities.SLOT_OBJECT
SLOT_ARM = abilities.SLOT_ARM
SLOT_POSTURE = abilities.SLOT_POSTURE


def _clarify(message: str, choices: Optional[List[str]] = None,
             slot: str = "") -> PlannerOutcome:
    return PlannerOutcome(kind="clarification", message=message,
                          choices=list(choices or []), slot=slot)


def _unsupported(message: str) -> PlannerOutcome:
    return PlannerOutcome(kind="unsupported", message=message)


def _reply(message: str) -> PlannerOutcome:
    """A successful answer with nothing to do.

    Distinct from `_unsupported` with friendly wording: the panel shows a
    failed task in red, and a greeting is not a failure.
    """
    return PlannerOutcome(kind="reply", message=message)


#: What "Hello" gets back.  It says what this panel can actually do rather
#: than making conversation about what it might: an opening line that implies
#: abilities the server does not have is the same false claim as a proposal
#: that cannot be executed, just earlier.
GREETING = ("Hello. I can move objects between the grid cells on the board — "
            "tell me which object and which cell, and I will plan it and show "
            "you the plan before anything moves.")


class DeterministicPlanner:
    """Callable planner over a scene supplied fresh for each request.

    `scene_provider` is called on the worker thread, so the scene it returns is
    read at planning time rather than at construction — placement and recall
    happen between one command and the next.
    """

    def __init__(self, scene_provider) -> None:
        self._scene_provider = scene_provider

    def __call__(self, request: PlannerRequest) -> PlannerOutcome:
        held = self._aside_during_a_question(request)
        if held is not None:
            return held
        state = _intent_for(request)
        outcome = self._plan(state.command, list(state.answers))
        return _record_intent(outcome, state)

    def _aside_during_a_question(self,
                                 request: PlannerRequest
                                 ) -> Optional[PlannerOutcome]:
        """Answer a greeting typed into an open question, and keep the question.

        `submit()` sends a greeting through `aside` and never disturbs the
        active task.  The browser does not use `submit()` here: anything typed
        while a task is awaiting an answer is POSTed to `reply()`, which has no
        aside hook.  So "Hello" arrived as the answer, `_is_a_command` agreed
        it was a request in its own right, the intent was replaced, and the
        task completed — destroying a question the operator was halfway
        through answering.

        THE QUESTION IS REPEATED FROM THE INTENT, NOT RE-DERIVED.  Planning
        the stored intent again would produce the same question in the normal
        case and something else entirely in three others: with the simulator
        down it reads `scene.error` and fails the whole task — the one thing
        #62 exists to prevent — and if the board has moved on it returns a
        PROPOSAL, so "Hello" silently produces a plan card with no greeting
        anywhere in it.  The question was already known when it was asked, so
        it is kept and repeated verbatim.

        Nothing here reads the scene.  That is the point: a greeting is
        answerable with the simulator switched off, whether or not there is a
        question open at the time.
        """
        prior = request.intent
        if prior is None or not prior.command or not prior.open_slot:
            return None
        if not prior.open_question:
            # A question was outstanding but its wording was not recorded — a
            # task in flight across this change.  Falling through treats the
            # greeting as the turn it looks like rather than inventing words
            # to put in Reachy's mouth.
            return None
        try:
            match = abilities.match(request.text)
        except AbilityRefusal:
            return None          # understood and refused: that is an answer
        if match is None or match.ability is None:
            return None
        # The same test `aside` applies, plus the name.  Both, because the
        # greeting wording below is `greet`'s: the next ability that needs
        # neither scene nor motion would otherwise be swallowed mid-question
        # and answered with somebody else's.
        if match.name != "greet":
            return None
        if match.ability.needs_scene or match.ability.needs_motion:
            return None          # a real request, and a real change of mind

        state = prior.copy()
        outcome = PlannerOutcome(
            kind="clarification",
            # Not the full GREETING: it recites what the panel can do, which
            # is a strange thing to say to someone already mid-way through
            # asking for one of those things.
            message=f"Hello. {state.open_question}",
            choices=list(state.open_choices),
            slot=state.open_slot,
        )
        outcome.intent = state
        return outcome

    def _plan(self, command: str,
              answers: List[Tuple[str, str]]) -> PlannerOutcome:
        if _JOINT_RE.search(command) or any(_JOINT_RE.search(a)
                                            for _, a in answers):
            return _unsupported(
                "I do not accept joint angles or joint names. Ask for a task — "
                "which object, and where it should go — and the motion layer "
                "works out the angles."
            )

        # Abilities are matched BEFORE the pick-and-place grammar, because two
        # of them open with a pick-and-place verb: "put your arm away" and
        # "place your forearm on the table".  Left to the old order they parse
        # as a move with "your arm" for a target, and the panel answers that it
        # does not know an object by that name.
        ability = self._ability_outcome(command, answers)
        if ability is not None:
            return ability

        scene = self._scene_provider()
        if scene.error:
            return _unsupported(f"I cannot read the scene right now: {scene.error}")

        intent = parse_intent(command)
        if intent is None:
            return _unsupported(f"I did not understand that. {_HELP}")

        # Answers to earlier clarifications fill whichever slot is still open.
        target_phrase, dest_phrase = _apply_answers(intent, answers, scene)

        dest_outcome, destination = self._resolve_destination(dest_phrase, scene)
        if dest_outcome is not None:
            return dest_outcome

        target_outcome, target_id = self._resolve_target(target_phrase, scene)
        if target_outcome is not None:
            return target_outcome

        return self._propose(scene, target_id, destination)

    def aside(self, text: str) -> Optional[PlannerOutcome]:
        """Answer a message that needs no task, or None.

        Called by the coordinator before it creates anything, so it must be
        cheap and must not touch the board — which is exactly the property
        that lets a greeting be answered while the simulator is down.  Only
        abilities the registry marks as needing neither scene nor motion
        qualify; everything else is a task.
        """
        try:
            match = abilities.match(text)
        except AbilityRefusal:
            return None          # a refusal is a task's answer, not an aside
        if match is None or not match.ability:
            return None
        if match.ability.needs_scene or match.ability.needs_motion:
            return None
        if match.name == "greet":
            return _reply(GREETING)
        return None

    # -- abilities ---------------------------------------------------------

    def _ability_outcome(self, command: str,
                         answers: List[Tuple[str, str]]) -> Optional[PlannerOutcome]:
        """Answer a non-pick-and-place request, or None to fall through.

        Everything an ability can answer WITHOUT the board happens here:
        greetings, refusals, the arm question and the rest/stow ambiguity.
        What needs the board — is that cell real, is that object on it —
        belongs to `_ability_proposal`, which reads the scene once and only
        for abilities whose registry entry says they need it.
        """
        try:
            match = abilities.match(command)
        except AbilityRefusal as exc:
            return _unsupported(exc.message)
        if match is None:
            return None

        if match.name == "greet":
            # Answered here as well as through `aside`, so a greeting still
            # works if the coordinator was built without that hook — and so
            # there is one definition of what a greeting says.
            return _reply(GREETING)

        given = {slot: text for slot, text in answers if slot}

        if match.ambiguous_between:
            resolved = _posture_answer(given.get(SLOT_POSTURE, ""))
            if resolved is None:
                return _clarify(
                    "\"Rest\" can mean two different places, and they are "
                    "eleven waypoints apart: my forearm supported on the "
                    "tabletop, or my arm stored back in the rail pocket. "
                    "Which did you mean?",
                    ["on the table", "in the rail pocket"],
                    SLOT_POSTURE,
                )
            match.name = resolved
            match.ability = abilities.REGISTRY[resolved]
            match.ambiguous_between = ()

        ability = match.ability
        if ability.arm_specific and match.arm != abilities.DEFAULT_ARM:
            # The corridor in tlh_motion-routine.ipynb was measured for the
            # right arm through a rig that is not symmetric.  Mirroring it is
            # an assumption about geometry, not a translation.
            return _unsupported(
                f"I can only {ability.summary} with my right arm. The route "
                "was measured for that arm through a rig that is not "
                "symmetric, so I will not mirror it without validating it."
            )

        if SLOT_CELL in ability.slots and SLOT_CELL not in match.slots:
            cell = given.get(SLOT_CELL, "")
            if not cell:
                scene = self._scene_provider()
                choices = [] if scene.error else scene.reachable_cells()
                return _clarify("Which cell should I point to?", choices,
                                SLOT_CELL)
            match.slots[SLOT_CELL] = cell

        if SLOT_OBJECT in ability.slots and SLOT_OBJECT not in match.slots:
            obj = given.get(SLOT_OBJECT, "")
            if not obj:
                scene = self._scene_provider()
                choices = [] if scene.error else sorted(scene.objects)
                return _clarify("Which object should I point to?", choices,
                                SLOT_OBJECT)
            match.slots[SLOT_OBJECT] = obj

        return self._ability_proposal(match)

    def _ability_proposal(self, match) -> PlannerOutcome:
        """A confirmable plan for an ability, with only the arguments it has.

        No placeholders.  A wave carries no target and no destination, and the
        card, the revalidator and the executor all read that absence as the
        answer it is.
        """
        ability = match.ability
        cell = match.slots.get(SLOT_CELL) or None
        obj = match.slots.get(SLOT_OBJECT) or None
        # An ability with a cell or an object to check needs the board to check
        # it against, whatever its registry entry says.  Deriving that here
        # rather than trusting `needs_scene` alone means a future ability that
        # sets one and forgets the other gets a scene rather than an
        # AttributeError on `scene.cells`.
        if ability.needs_scene or cell is not None or obj is not None:
            scene = self._scene_provider()
            if scene.error:
                return _unsupported(
                    f"I cannot read the scene right now: {scene.error}")
        else:
            scene = None

        summary = ability.summary
        if cell is not None:
            # Pointing is a hover, so an occupied cell is fine — the object is
            # what gets hovered over.  Reusing pick-and-place's empty-cell rule
            # here would refuse the most natural request in a populated scene.
            if cell not in scene.cells:
                return _clarify(
                    f"There is no cell called {cell} in this scene. "
                    "Pick one of these.", scene.reachable_cells(), SLOT_CELL)
            if not scene.cells[cell].reachable:
                return _clarify(
                    f"{cell} was measured out of the right arm's reach, so I "
                    "cannot point at it. Pick another cell.",
                    scene.reachable_cells(), SLOT_CELL)
            summary = f"{summary} {cell}"

        if obj is not None:
            oid = _match_object_id(obj, scene)
            if oid is None:
                return _clarify(
                    f"I do not know an object called \"{obj}\". "
                    "These are the objects in this scene.",
                    sorted(scene.objects), SLOT_OBJECT)
            if scene.objects[oid].on_board is False:
                # In the pool, not on the board.  Reaching for it would be a
                # tabletop motion aimed off the tabletop.
                return _unsupported(
                    f"{oid} is not on the board right now, so there is nothing "
                    "on the table for me to point at. Place it on a cell first."
                )
            # Write the resolution back so everything downstream — the
            # evidence, the card, the executor — sees the object ID and not
            # the words the operator typed.  "soda can" and `soda_can` are the
            # same object to a reader and different strings to a lookup.
            match.slots[SLOT_OBJECT] = oid
            obj = oid
            summary = f"{summary} {oid}"

        proposal = Proposal(
            plan_id=uuid.uuid4().hex,
            plan_version=0,
            task_type=match.name,
            arm=match.arm,
            cell=cell,
            object_id=obj,
            route=ability.route,
            route_version=ability.route_version,
            expected_start_posture=(ability.start_postures[0]
                                    if ability.start_postures else ""),
            end_posture=ability.end_posture,
            brief_reason=_ability_reason(match),
            requires_confirmation=ability.requires_confirmation,
            scene_name=(scene.name if scene is not None else ""),
            summary=summary,
            state_evidence=_ability_evidence(match, scene),
        )
        return PlannerOutcome(kind="proposal", proposal=proposal)

    # -- destination -------------------------------------------------------

    def _resolve_destination(self, phrase: str, scene: SceneView
                             ) -> Tuple[Optional[PlannerOutcome], Optional[DestinationRef]]:
        # Offer only cells that are reachable AND free.  Offering an occupied
        # cell and then refusing it on the next turn wastes the human's turn
        # and makes the panel look like it is not reading the board.
        cells = scene.available_cells()

        if not phrase:
            return _clarify("Where should it go?", cells, SLOT_CELL), None

        did = _match_destination_id(phrase, scene)
        if did:
            return None, _dest_ref(scene, did)

        if _is_bin_phrase(phrase):
            if not scene.has_destination:
                return _clarify(
                    "This scene has no bin or tray configured, so there is "
                    "nowhere I can call a bin. I can put it on a grid cell "
                    "instead — that is a spot on the board, not a container. "
                    "Which cell?",
                    cells, SLOT_CELL,
                ), None
            names = sorted(scene.destinations)
            if len(names) == 1:
                return None, _dest_ref(scene, names[0])
            return _clarify("Which one?", names, SLOT_CELL), None

        cell = _match_cell(phrase, scene)
        if cell:
            if cell not in scene.cells:
                return _clarify(
                    f"There is no cell called {cell} in this scene. "
                    "Pick one of these.", cells, SLOT_CELL
                ), None
            if not scene.cells[cell].reachable:
                return _clarify(
                    f"{cell} was measured out of the right arm's reach, so I "
                    "will not plan a move there. Pick another cell.", cells,
                    SLOT_CELL,
                ), None
            occupant = scene.cells[cell].occupant
            if occupant:
                return _clarify(
                    f"{cell} is already occupied by {occupant}. "
                    "Pick another cell.", cells, SLOT_CELL,
                ), None
            return None, DestinationRef(kind="cell", ref_id=cell, label=cell)

        return _clarify(
            f"I do not know a destination called \"{phrase}\". "
            "These are the destinations I can use.", cells, SLOT_CELL
        ), None

    # -- target ------------------------------------------------------------

    def _resolve_target(self, phrase: str, scene: SceneView
                        ) -> Tuple[Optional[PlannerOutcome], str]:
        if not phrase:
            return _clarify("Which object should I move?",
                            sorted(scene.objects), SLOT_OBJECT), ""

        oid = _match_object_id(phrase, scene)
        if oid:
            return None, oid

        category = _category_of(phrase)
        if category is not None:
            return self._resolve_category(scene, category)

        return _clarify(
            f"I do not know an object called \"{phrase}\". "
            "These are the objects in this scene.", sorted(scene.objects),
            SLOT_OBJECT
        ), ""

    def _resolve_category(self, scene: SceneView, tag: str
                          ) -> Tuple[Optional[PlannerOutcome], str]:
        """Resolve "the recyclable one" / "the non-recyclable one" to an object.

        One body for both categories.  The negated phrasing used to fall
        through to this method with `tag` fixed at `recyclable`, so it answered
        the opposite question with the same confidence; passing the tag in is
        what makes the negation reach a decision instead of being dropped.
        """
        label = "recyclable" if tag == RECYCLABLE_TAG else "non-recyclable"
        members = scene.tagged(tag)
        if not members:
            # Not a fallback to the other category.  A scene that classifies
            # nothing as `non-recyclable` has not thereby said everything else
            # is; saying so is how the substring bug produced wrong objects.
            return _unsupported(
                f"Nothing in this scene is tagged {label}."
            ), ""

        if not scene.live:
            # Honest stop: without live poses this cannot tell a can on the
            # board from one parked in the pool, and guessing would be exactly
            # the false claim the design brief warns about.  Issue #50 removes
            # this branch by supplying live object poses.
            names = sorted(o.object_id for o in members)
            return _clarify(
                "I cannot yet see which objects are actually on the board, so "
                "I will not guess. Name the one you mean.", names, SLOT_OBJECT
            ), ""

        on_board = scene.tabletop_tagged(tag)
        if not on_board:
            return _clarify(
                f"No {label} object is on the board right now. Place one on "
                "a cell first, then ask me again.",
                sorted(o.object_id for o in members), SLOT_OBJECT,
            ), ""
        if len(on_board) > 1:
            return _clarify(
                f"There is more than one {label} object on the board. "
                "Which one?",
                [f"{o.object_id} ({o.cell})" if o.cell else o.object_id
                 for o in on_board], SLOT_OBJECT,
            ), ""
        return None, on_board[0].object_id

    # -- proposal ----------------------------------------------------------

    def _propose(self, scene: SceneView, target_id: str,
                 destination: DestinationRef) -> PlannerOutcome:
        summary = f"move {target_id} to {destination.describe()}"

        obj = scene.objects.get(target_id)
        if obj is not None and obj.is_recyclable:
            reason = f"{target_id} is tagged recyclable in this scene."
        elif obj is not None and obj.semantic_class:
            reason = f"{target_id} is a {obj.semantic_class} in this scene."
        else:
            reason = f"{target_id} is a placeable object in this scene."
        if scene.live and obj is not None and obj.cell:
            reason += f" It is on {obj.cell} now."
        elif not scene.live:
            reason += " Its current position has not been verified."

        return PlannerOutcome(
            kind="proposal",
            message=f"Proposed action: {summary}.",
            proposal=Proposal(
                plan_id=uuid.uuid4().hex,
                plan_version=0,          # the coordinator stamps the real one
                task_type="pick_place",
                target_id=target_id,
                destination=destination.as_str(),
                destination_kind=destination.kind,
                destination_label=destination.describe(),
                legacy_destination=destination.legacy_destination(),
                brief_reason=reason,
                requires_confirmation=True,
                semantic_source="scene_data",
                scene_name=scene.name,
                summary=summary,
                state_evidence=_evidence(scene, target_id, destination),
            ),
        )


def _ability_reason(match) -> str:
    """Why this plan, said in terms the operator can check.

    Names the route, because the route IS the claim: "store your arm" means
    the eleven-waypoint rail-pocket corridor and not `P.go_home()`, which
    routes through `stow_from_side()` and is a different motion that is also
    fairly called going home.
    """
    ability = match.ability
    if not ability.route:
        return "no arm movement"
    where = f" to {match.slots[abilities.SLOT_CELL]}" if match.slots.get(
        abilities.SLOT_CELL) else ""
    return (f"{ability.route} with my {match.arm} arm{where}, ending "
            f"{ability.end_posture or 'where it started'}")


def _ability_evidence(match, scene) -> Dict[str, Any]:
    """What the revalidator will need at confirm time.

    Per ability, because what invalidates a plan differs: a wave does not care
    where the objects are, pointing at an object cares very much, and pointing
    at a cell cares only that the cell is still there and reachable.
    """
    ev: Dict[str, Any] = {"action": match.name, "arm": match.arm,
                          "route": match.ability.route,
                          "route_version": match.ability.route_version}
    if scene is None:
        return ev
    ev["scene_revision"] = scene.scene_revision
    ev["live"] = scene.live
    cell = match.slots.get(abilities.SLOT_CELL)
    if cell:
        ev["cell"] = cell
    oid = match.slots.get(abilities.SLOT_OBJECT)
    if oid and oid in scene.objects and scene.objects[oid].position is not None:
        ev["object_id"] = oid
        ev["object_position"] = list(scene.objects[oid].position)
    return ev


def _is_a_command(text: str) -> bool:
    """Whether a turn stands on its own as a request rather than an answer.

    An answer is a bare noun — "r2c2", "soda_can", "on the table".  None of
    those parse as a command, and every command carries a verb the grammars
    know, so the two do not overlap.
    """
    try:
        if abilities.match(text) is not None:
            return True
    except AbilityRefusal:
        # Understood, and refused.  Still a command: the operator should get
        # the refusal, not have "don't wave" filed as a cell name.
        return True
    return parse_intent(text) is not None


def _posture_answer(text: str) -> Optional[str]:
    """Which posture an answer to the rest/stow question names.

    None means it named neither, and the question gets asked again rather than
    resolved by preference.  The two postures are eleven waypoints apart
    through the rig; a coin-flip here is a long arm movement nobody asked for.
    """
    norm = _normalise(text)
    if not norm:
        return None
    if re.search(r"\b(table|tabletop|desk|board|forearm|down|surface)\b", norm):
        return "rest_forearm"
    if re.search(r"\b(pocket|rail|home|away|stow|store|default|back)\b", norm):
        return "stow_arm"
    return None


def _dest_ref(scene: SceneView, destination_id: str) -> DestinationRef:
    view = scene.destinations.get(destination_id)
    return DestinationRef(
        kind="destination", ref_id=destination_id,
        label=view.label if view else destination_id,
    )


def _evidence(scene: SceneView, target_id: str,
              destination: DestinationRef) -> dict:
    """What this plan assumed about the world, so it can be rechecked.

    Empty when the scene is not live: there is nothing to recheck against, and
    a plan built from the scene file alone never claimed a position in the
    first place.

    `sim_step` is recorded but deliberately NOT compared at confirm time.  The
    simulator advances it hundreds of times a second, so treating a changed
    step as a changed world would invalidate every proposal before anyone
    could read it.  What is compared is what actually matters: the scene
    revision, where the target is, and whether the destination is still free.
    """
    if not scene.live:
        return {}
    obj = scene.objects.get(target_id)
    occupant = None
    if destination.kind == "cell":
        cell = scene.cells.get(destination.ref_id)
        occupant = cell.occupant if cell else None
    return {
        "scene_revision": scene.scene_revision,
        "sim_step": scene.sim_step,
        "target_id": target_id,
        "target_pos": list(obj.position) if obj and obj.position else None,
        "target_cell": obj.cell if obj else None,
        "destination": destination.as_str(),
        "destination_occupant": occupant,
    }


class LiveProposalValidator:
    """Decides at confirm time whether a proposal still describes the world.

    Returns `(ok, why_not)`.  Called by the coordinator under its lock, just
    before a confirmation would be consumed, so a plan cannot be confirmed
    against a board that has changed underneath it.
    """

    def __init__(self, scene_provider) -> None:
        self._scene_provider = scene_provider

    def __call__(self, proposal: Proposal) -> Tuple[bool, str]:
        evidence = proposal.state_evidence or {}
        if not evidence:
            # A plan made without live state claimed nothing about positions,
            # so there is nothing here to contradict.  It is still confirmed
            # into planning-only mode, which moves nothing.
            return True, ""

        scene = self._scene_provider()
        if scene.error:
            return False, f"I cannot read the scene right now: {scene.error}"
        if not scene.live:
            return False, ("I have lost my link to the simulator, so I cannot "
                           "check that this plan still matches the board.")

        expected_rev = evidence.get("scene_revision")
        if expected_rev and scene.scene_revision != expected_rev:
            return False, ("The scene was reloaded since I made that plan. "
                           "Ask me again and I will look at the new one.")

        action = evidence.get("action")
        if action is not None and action != "pick_place":
            return self._check_ability(proposal, scene, evidence, action)

        target_id = evidence.get("target_id") or proposal.target_id
        obj = scene.objects.get(target_id)
        if obj is None:
            return False, f"{target_id} is no longer in this scene."

        expected_pos = evidence.get("target_pos")
        if expected_pos is not None:
            if obj.position is None:
                return False, (f"I can no longer see where {target_id} is, so "
                               "I will not confirm a plan that assumed it.")
            drift = _distance(obj.position, expected_pos)
            if drift > POSE_TOLERANCE_M:
                return False, (f"{target_id} has moved {drift * 100:.0f} cm "
                               "since I made that plan. Ask me again.")

        expected_cell = evidence.get("target_cell")
        if expected_cell is not None and obj.cell != expected_cell:
            return False, (f"{target_id} is no longer on {expected_cell}. "
                           "Ask me again.")

        if proposal.destination_kind == "cell":
            cell = scene.cells.get(_ref_id(proposal.destination))
            if cell is None:
                return False, "That destination is no longer in this scene."
            if cell.occupant != evidence.get("destination_occupant"):
                held = cell.occupant or "nothing"
                return False, (f"{cell.name} now holds {held}, which is not "
                               "what it held when I made that plan.")
        return True, ""

    def _check_ability(self, proposal: Proposal, scene: SceneView,
                       evidence: dict, action: str) -> Tuple[bool, str]:
        """What invalidates a plan differs by ability, so ask per ability.

        The pick-and-place rule — the target must not have moved, the
        destination must still be empty — is not a general one.  Applied to a
        wave it asserts things about objects the plan never mentioned; applied
        to pointing at a cell it refuses an occupied cell that pointing is
        perfectly happy to hover over.  One rule covering all of them would
        pass silently for every ability it was not written for.
        """
        cell_name = evidence.get("cell") or proposal.cell
        if cell_name is not None:
            cell = scene.cells.get(cell_name)
            if cell is None:
                return False, f"{cell_name} is no longer in this scene."
            if not cell.reachable:
                return False, f"{cell_name} is no longer within reach."
            # Deliberately NOT checking occupancy.  Pointing is a hover, and
            # a cell with something on it is a cell worth pointing at.

        object_id = evidence.get("object_id") or proposal.object_id
        expected_pos = evidence.get("object_position")
        if object_id is not None:
            obj = scene.objects.get(object_id)
            if obj is None:
                return False, f"{object_id} is no longer in this scene."
            if obj.on_board is False:
                return False, (f"{object_id} has been taken off the board "
                               "since I made that plan.")
            if expected_pos is not None:
                if obj.position is None:
                    return False, (f"I can no longer see where {object_id} "
                                   "is, so I will not point at where it was.")
                drift = _distance(obj.position, expected_pos)
                if drift > POSE_TOLERANCE_M:
                    # For pointing, the object moving is exactly what makes the
                    # plan wrong: the hover was computed over where it WAS.
                    return False, (f"{object_id} has moved "
                                   f"{drift * 100:.0f} cm since I made that "
                                   "plan. Ask me again.")

        if action == "rest_forearm":
            # The forearm footprint check belongs to the geometry #64 brings
            # in; there is nothing here that can measure it yet.  Saying so
            # rather than passing quietly matters because a rest lowers the
            # forearm onto the table, and an object under it is the one thing
            # that must stop it.  Nothing can execute this ability until #65
            # anyway, so the gate holds without this check pretending to be one.
            return True, ""

        return True, ""


def _ref_id(destination: str) -> str:
    return destination.split(":", 1)[1] if ":" in destination else destination


def _distance(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _intent_for(request: PlannerRequest) -> IntentState:
    """The canonical intent this turn is about.

    Two sources, and which one is used is decided by the caller, not guessed.
    A caller that keeps intent state — the coordinator does — hands back what
    the last turn settled, and this turn's text is folded in as an answer to
    whatever question was outstanding.  A caller that keeps none falls back to
    reconstructing from the transcript, which is what the direct-planner tests
    do and what any conversation stored before #87 has.

    The fallback is the path with the bug in it: `_split_history` can only see
    the turns still in the transcript, and the transcript is trimmed from the
    front.  Keep it for compatibility; do not route new callers through it.

    A NEW COMMAND is not an answer.  "Which cell?" followed by "stow your arm"
    is a change of mind, and folding it into the open slot would store "stow
    your arm" as a cell name and then complain it is not one.  Whichever the
    operator typed last wins, and — because the earlier answers were given to
    a request that no longer stands — they are dropped with it rather than
    carried onto the new intent.
    """
    prior = request.intent
    if prior is None or not prior.command:
        command, answers = _split_history(request.history, request.text)
        state = IntentState(command=command, answers=answers)
    else:
        state = prior.copy()
        # The slot comes from the question that was asked, recorded when it
        # was asked (#59).  An answer arriving with no question outstanding
        # carries the empty slot and is placed by shape, as it always was.
        # `remember` rather than `append`: re-asking a slot replaces its
        # answer, which is what both readers already meant by it, and keeps
        # this list from growing for as long as a client keeps replying.
        state.remember(state.open_slot, request.text)

    if state.answers and _is_a_command(state.answers[-1][1]):
        state = IntentState(command=state.answers[-1][1])
    return state


def _record_intent(outcome: PlannerOutcome, state: IntentState) -> PlannerOutcome:
    """Fold what this turn decided back into the intent, and attach it.

    Attached to every outcome, including the ones that end the task: the
    coordinator stores it before it branches, so a failed or completed task
    still records what it was about.
    """
    if outcome.kind == "clarification":
        state.open_slot = outcome.slot
        # Kept so it can be repeated without being re-derived.  See
        # `_aside_during_a_question`.
        state.open_question = outcome.message
        state.open_choices = list(outcome.choices)
    else:
        state.open_slot = ""
        state.open_question = ""
        state.open_choices = []

    proposal = outcome.proposal
    if proposal is not None:
        # The resolved answer, which is the one worth keeping: `task_type` is
        # the ability that will actually fly, after "rest" was settled and
        # after "soda can" became `soda_can`.
        state.kind = proposal.task_type
        state.selected_target = (proposal.object_id or proposal.target_id
                                 or state.selected_target)
        state.plan_version = proposal.plan_version
    elif not state.kind:
        state.kind = _provisional_kind(state.command)

    outcome.intent = state
    return outcome


def _provisional_kind(command: str) -> str:
    """What the command looks like before any clarification is answered.

    "" for a request that names two abilities equally well — `match` returns
    an empty name for those on purpose, and copying a guess in here would be
    the guess the rest/stow question exists to avoid.
    """
    try:
        match = abilities.match(command)
    except AbilityRefusal:
        return ""
    if match is not None and match.ability is not None:
        return match.name
    return "pick_place" if parse_intent(command) is not None else ""


def _split_history(history: List[ConversationEvent], latest: str
                   ) -> Tuple[str, List[Tuple[str, str]]]:
    """Return the original command and every answer since, each with its slot.

    The FALLBACK source of intent, for a caller that keeps no state — see
    `_intent_for`.  It can only see the turns still in the transcript, and the
    transcript is bounded, so a long enough clarification loses the command it
    is reconstructing.  That is #87, and it is why the coordinator stopped
    coming through here.

    `history` already contains `latest` as its last user turn, so the answers
    list is built from history alone and `latest` is only a fallback for the
    very first call.

    The slot comes from the question the answer follows, not from the answer's
    own wording.  Walking the transcript in order and remembering the last
    reachy question is the whole mechanism: at the moment the planner asked, it
    knew which slot it wanted, and that is the only reliable record of it.

    An answer with no question before it — the opening command's own turn is
    skipped, but a stray user turn is possible — carries an empty slot and is
    placed by the fallback in `_apply_answers`.
    """
    command = ""
    pending = ""
    answers: List[Tuple[str, str]] = []
    for ev in history:
        if ev.role == "user":
            if not command:
                command = ev.text
            else:
                answers.append((pending, ev.text))
            pending = ""
        elif ev.question_id:
            pending = ev.slot
    if not command:
        return latest, []
    return command, answers


def _apply_answers(intent: _Intent, answers: List[Tuple[str, str]],
                   scene: SceneView) -> Tuple[str, str]:
    """Fold clarification answers into the slots they were given for.

    Each answer arrives with the slot its question named, so an answer to
    "which cell?" fills the destination and an answer to "which object?" fills
    the target.  No inspection of the answer's wording decides that.

    That inspection is what used to go wrong.  The slot was re-derived by
    asking whether the CURRENT destination phrase looked resolvable:

        dest_open = ... _match_cell(dest, scene) not in scene.cells ...

    A cell that exists but was REJECTED — occupied, or out of the arm's reach —
    read as resolved, so the slot the planner had just asked about counted as
    filled.  "put soda_can on r2c2" with r2c2 occupied, answered "r1c1", stored
    r1c1 as the TARGET and replied that it did not know an object by that name.
    Both `cell_r3c1` and `cell_r3c2` are out of reach in FWDCenterLabMCC, so
    this was reachable in ordinary use.

    The docstring already stated the right rule — an answer fills a slot only
    while that slot is open — and "open" means unresolved.  The code read it as
    unrecognised.  Recording the slot at ask time is what makes the two agree.
    """
    target = intent.target_phrase
    dest = intent.dest_phrase

    for slot, answer in answers:
        answer = answer.strip()
        if not answer:
            continue
        if slot == SLOT_CELL:
            dest = answer
            continue
        if slot == SLOT_OBJECT:
            target = answer
            continue
        # No slot recorded: a transcript from before slots existed, or a user
        # turn that answered no question.  Fall back to placing it by shape,
        # which is what every answer used to get.
        if _match_cell(answer, scene) is not None or \
                _match_destination_id(answer, scene) is not None:
            dest = answer
        elif _match_object_id(answer, scene) is not None:
            target = answer
        elif not dest:
            dest = answer
        else:
            target = answer
    return target, dest
