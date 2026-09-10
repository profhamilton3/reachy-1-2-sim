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

import re
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

from panel_scene import SceneView
from tasks import ConversationEvent, PlannerOutcome, PlannerRequest, Proposal

_VERBS = ("put", "move", "place", "drop", "sort")
_PREPS = ("into", "onto", "in", "on", "to")

_RECYCLE_WORDS = ("recycle", "recycling", "recyclable")
_BIN_WORDS = ("bin", "trash", "garbage", "waste", "rubbish")

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


def _is_recycle_phrase(phrase: str) -> bool:
    return any(w in _normalise(phrase) for w in _RECYCLE_WORDS)


def _is_bin_phrase(phrase: str) -> bool:
    norm = _normalise(phrase)
    return any(re.search(rf"\b{w}\b", norm) for w in _BIN_WORDS)


def _clarify(message: str, choices: Optional[List[str]] = None) -> PlannerOutcome:
    return PlannerOutcome(kind="clarification", message=message,
                          choices=list(choices or []))


def _unsupported(message: str) -> PlannerOutcome:
    return PlannerOutcome(kind="unsupported", message=message)


class DeterministicPlanner:
    """Callable planner over a scene supplied fresh for each request.

    `scene_provider` is called on the worker thread, so the scene it returns is
    read at planning time rather than at construction — placement and recall
    happen between one command and the next.
    """

    def __init__(self, scene_provider) -> None:
        self._scene_provider = scene_provider

    def __call__(self, request: PlannerRequest) -> PlannerOutcome:
        scene = self._scene_provider()
        if scene.error:
            return _unsupported(f"I cannot read the scene right now: {scene.error}")

        command, answers = _split_history(request.history, request.text)
        if _JOINT_RE.search(command) or any(_JOINT_RE.search(a) for a in answers):
            return _unsupported(
                "I do not accept joint angles or joint names. Ask for a task — "
                "which object, and where it should go — and the motion layer "
                "works out the angles."
            )

        intent = parse_intent(command)
        if intent is None:
            return _unsupported(f"I did not understand that. {_HELP}")

        # Answers to earlier clarifications fill whichever slot is still open.
        target_phrase, dest_phrase = _apply_answers(intent, answers, scene)

        dest_outcome, destination, dest_kind = self._resolve_destination(dest_phrase, scene)
        if dest_outcome is not None:
            return dest_outcome

        target_outcome, target_id = self._resolve_target(target_phrase, scene)
        if target_outcome is not None:
            return target_outcome

        return self._propose(scene, target_id, destination, dest_kind)

    # -- destination -------------------------------------------------------

    def _resolve_destination(self, phrase: str, scene: SceneView
                             ) -> Tuple[Optional[PlannerOutcome], str, str]:
        cells = scene.reachable_cells()

        if not phrase:
            return _clarify("Where should it go?", cells), "", ""

        did = _match_destination_id(phrase, scene)
        if did:
            return None, did, "destination"

        if _is_bin_phrase(phrase):
            if not scene.has_destination:
                return _clarify(
                    "This scene has no bin or tray configured, so there is "
                    "nowhere I can call a bin. I can put it on a grid cell "
                    "instead — that is a spot on the board, not a container. "
                    "Which cell?",
                    cells,
                ), "", ""
            names = sorted(scene.destinations)
            if len(names) == 1:
                return None, names[0], "destination"
            return _clarify("Which one?", names), "", ""

        cell = _match_cell(phrase, scene)
        if cell:
            if cell not in scene.cells:
                return _clarify(
                    f"There is no cell called {cell} in this scene. "
                    "Pick one of these.", sorted(scene.cells)
                ), "", ""
            if not scene.cells[cell].reachable:
                return _clarify(
                    f"{cell} was measured out of the right arm's reach, so I "
                    "will not plan a move there. Pick another cell.", cells
                ), "", ""
            occupant = scene.cells[cell].occupant
            if occupant:
                return _clarify(
                    f"{cell} is already occupied by {occupant}. "
                    "Pick another cell.", cells
                ), "", ""
            return None, cell, "cell"

        return _clarify(
            f"I do not know a destination called \"{phrase}\". "
            "These are the destinations I can use.", cells
        ), "", ""

    # -- target ------------------------------------------------------------

    def _resolve_target(self, phrase: str, scene: SceneView
                        ) -> Tuple[Optional[PlannerOutcome], str]:
        if not phrase:
            return _clarify("Which object should I move?",
                            sorted(scene.objects)), ""

        oid = _match_object_id(phrase, scene)
        if oid:
            return None, oid

        if _is_recycle_phrase(phrase):
            return self._resolve_recyclable(scene)

        return _clarify(
            f"I do not know an object called \"{phrase}\". "
            "These are the objects in this scene.", sorted(scene.objects)
        ), ""

    def _resolve_recyclable(self, scene: SceneView
                            ) -> Tuple[Optional[PlannerOutcome], str]:
        recyclable = scene.recyclables()
        if not recyclable:
            return _unsupported(
                "Nothing in this scene is tagged recyclable."
            ), ""

        if not scene.live:
            # Honest stop: without live poses this cannot tell a can on the
            # board from one parked in the pool, and guessing would be exactly
            # the false claim the design brief warns about.  Issue #50 removes
            # this branch by supplying live object poses.
            names = sorted(o.object_id for o in recyclable)
            return _clarify(
                "I cannot yet see which objects are actually on the board, so "
                "I will not guess. Name the one you mean.", names
            ), ""

        on_board = scene.tabletop_recyclables()
        if not on_board:
            return _clarify(
                "No recyclable object is on the board right now. Place one on "
                "a cell first, then ask me again.",
                sorted(o.object_id for o in recyclable),
            ), ""
        if len(on_board) > 1:
            return _clarify(
                "There is more than one recyclable object on the board. "
                "Which one?",
                [f"{o.object_id} ({o.cell})" if o.cell else o.object_id
                 for o in on_board],
            ), ""
        return None, on_board[0].object_id

    # -- proposal ----------------------------------------------------------

    def _propose(self, scene: SceneView, target_id: str,
                 destination: str, dest_kind: str) -> PlannerOutcome:
        where = (f"grid cell {destination}" if dest_kind == "cell"
                 else str(destination))
        summary = f"move {target_id} to {where}"

        obj = scene.objects.get(target_id)
        if obj is not None and obj.is_recyclable:
            reason = f"{target_id} is tagged recyclable in this scene."
        elif obj is not None and obj.semantic_class:
            reason = f"{target_id} is a {obj.semantic_class} in this scene."
        else:
            reason = f"{target_id} is a placeable object in this scene."
        if not scene.live:
            reason += " Its current position has not been verified."

        return PlannerOutcome(
            kind="proposal",
            message=f"Proposed action: {summary}.",
            proposal=Proposal(
                plan_id=uuid.uuid4().hex,
                plan_version=0,          # the coordinator stamps the real one
                task_type="pick_place",
                target_id=target_id,
                destination=f"{dest_kind}:{destination}",
                brief_reason=reason,
                requires_confirmation=True,
                semantic_source="scene_data",
                scene_name=scene.name,
                summary=summary,
            ),
        )


def _split_history(history: List[ConversationEvent], latest: str
                   ) -> Tuple[str, List[str]]:
    """Return the original command and every answer given since.

    `history` already contains `latest` as its last user turn, so the answers
    list is built from history alone and `latest` is only a fallback for the
    very first call.
    """
    user_turns = [e.text for e in history if e.role == "user"]
    if not user_turns:
        return latest, []
    return user_turns[0], user_turns[1:]


def _apply_answers(intent: _Intent, answers: List[str], scene: SceneView
                   ) -> Tuple[str, str]:
    """Fold clarification answers into whichever slot is still unresolved.

    An answer is tried as a destination only while the destination is open, and
    as a target only while the target is open, so "r2c2" answering "which cell?"
    cannot later be mistaken for an object.
    """
    target = intent.target_phrase
    dest = intent.dest_phrase

    for answer in answers:
        answer = answer.strip()
        if not answer:
            continue
        dest_open = (not dest
                     or _is_bin_phrase(dest) and not scene.has_destination
                     or (_match_destination_id(dest, scene) is None
                         and _match_cell(dest, scene) not in scene.cells
                         and not _is_bin_phrase(dest)))
        if dest_open and (_match_cell(answer, scene) is not None
                          or _match_destination_id(answer, scene) is not None):
            dest = answer
            continue
        if not target or _match_object_id(target, scene) is None:
            if _match_object_id(answer, scene) is not None:
                target = answer
                continue
        if dest_open:
            dest = answer
        else:
            target = answer
    return target, dest
