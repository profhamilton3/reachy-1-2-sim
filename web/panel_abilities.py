"""What Reachy can be asked to do, and how a typed request maps onto it (#61).

Before this module, ability knowledge was spread across three literals and a
function: `_VERBS` in the parser, `task_type="pick_place"` hard-coded in
`_propose()`, and `SimulatorExecutor.available()` refusing anything else.
Nothing anywhere stated what the robot can do — the answer was implied by
whichever branch happened to run, which is why "wave" answered "I did not
understand that" rather than "I cannot do that yet".

This is the single place that answers "can you do X".  The browser must not
carry a second copy of the alias list: a page that offers an ability the server
has never heard of is worse than one that offers nothing.

MATCHING IS FULL-INTENT
-----------------------
Patterns are `fullmatch`ed against the normalised command, not searched for
inside it.  That is the difference between reading "wave" as a request and
reading it out of "don't wave" or "I waved earlier".  A substring test here
would repeat #58 one layer up, where the consequence is a moving arm rather
than a wrong object.

Negation is then checked explicitly on top, because a pattern permissive enough
to be useful is permissive enough to be fooled, and the failure is silent.

COMPOUNDS ARE REFUSED, NOT SPLIT
--------------------------------
"wave and then stow" is two requests.  Executing the first half of something
the operator confirmed as a pair is worse than refusing both: the arm ends up
somewhere they did not ask for and did not agree to.

RESET MEANS STOW
----------------
The operator has said so.  Bare "reset" is the stow route, and never the
simulator's `reset` — that would discard their objects and the simulation
clock.  "Reset the simulation" is a different request and is explained as one,
not caught by the stow alias because it contains the word.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: Slot names.  Shared with the planner so a clarification and the answer that
#: fills it agree on what was asked (#59).
SLOT_CELL = "which_cell"
SLOT_OBJECT = "which_object"
SLOT_ARM = "which_arm"
SLOT_POSTURE = "which_posture"

#: The arm the validated corridor was measured for.  Everything in
#: `notebooks/tlh_motion-routine.ipynb` is right-arm geometry through a rig
#: that is not symmetric, so the left arm is a question, not a mirror.
DEFAULT_ARM = "right"

#: `REST` is the forearm supported on the tabletop; `HOME` is stored in the
#: rail pocket.  The operator says "rest" for both, and the two are eleven
#: waypoints apart.
POSTURE_REST = "rest"
POSTURE_HOME = "home"


@dataclass(frozen=True)
class Ability:
    """One thing Reachy can be asked to do.

    `route` names the motion this ability flies, and is deliberately not the
    name of a function that already exists.  `P.go_home()` routes through
    `stow_from_side()` — a lateral abduction and lower, correct as the exit
    from the pick-and-place side hub and NOT the rail-pocket corridor the
    operator means by "store your arm".  Two motions, both fairly called
    "go home"; binding an ability to one by its name is how they get confused.
    """

    name: str
    summary: str
    patterns: Tuple[str, ...]
    #: Slots this ability accepts.  A slot listed here may still be unfilled,
    #: which is a question to ask rather than a value to invent.
    slots: Tuple[str, ...] = ()
    #: Whether planning it needs to read the board at all.  A greeting does
    #: not, which is what lets it work with the simulator disconnected (#62).
    needs_scene: bool = True
    needs_motion: bool = True
    arm_specific: bool = True
    route: str = ""
    route_version: int = 1
    start_postures: Tuple[str, ...] = ()
    end_posture: str = ""
    requires_confirmation: bool = True

    def compiled(self) -> List[re.Pattern]:
        return [re.compile(p) for p in self.patterns]


# Politeness and filler the operator types without meaning anything by it.
_POLITE = r"(?:please\s+|can you\s+|could you\s+|reachy[,\s]+)*"
_TRAIL = r"(?:\s+please)?"
_ARM = r"(?:(?P<arm>left|right)\s+)?"
_YOUR = r"(?:your\s+|the\s+)?"

REGISTRY: Dict[str, Ability] = {}


def _register(ability: Ability) -> Ability:
    REGISTRY[ability.name] = ability
    return ability


_register(Ability(
    name="greet",
    summary="say hello",
    patterns=(
        rf"{_POLITE}(?:hello|hi|hey|good morning|good afternoon|good evening)"
        rf"(?:\s+reachy)?{_TRAIL}",
    ),
    needs_scene=False,
    needs_motion=False,
    arm_specific=False,
    requires_confirmation=False,
))

_register(Ability(
    name="rest_forearm",
    summary="rest the forearm on the table",
    patterns=(
        # The posture is named explicitly, so there is nothing to ask about.
        rf"{_POLITE}(?:place|put|lay|rest|set)\s+{_YOUR}{_ARM}(?:forearm|arm|hand)"
        rf"\s+(?:down\s+)?(?:on|onto)\s+(?:the\s+)?(?:table|tabletop|board|desk)"
        rf"{_TRAIL}",
        rf"{_POLITE}rest\s+{_YOUR}{_ARM}(?:forearm)\s*{_TRAIL}",
    ),
    slots=(SLOT_ARM,),
    route="PLACE_ROUTE",
    start_postures=(POSTURE_HOME,),
    end_posture=POSTURE_REST,
))

_register(Ability(
    name="stow_arm",
    summary="stow the arm in the rail pocket",
    patterns=(
        rf"{_POLITE}(?:store|stow|put away)\s+{_YOUR}{_ARM}arm{_TRAIL}",
        rf"{_POLITE}put\s+{_YOUR}{_ARM}arm\s+(?:away|back){_TRAIL}",
        rf"{_POLITE}(?:stow|store)\s*{_TRAIL}",
        rf"{_POLITE}(?:go|return|move)\s+(?:back\s+)?(?:to\s+)?"
        rf"(?:the\s+)?(?:default|home|start(?:ing)?)\s*"
        rf"(?:position|pose|posture)?{_TRAIL}",
        rf"{_POLITE}reset\s+{_YOUR}{_ARM}arm{_TRAIL}",
        rf"{_POLITE}reset{_TRAIL}",
    ),
    slots=(SLOT_ARM,),
    route="STOW_ROUTE",
    end_posture=POSTURE_HOME,
))

_register(Ability(
    name="wave",
    summary="wave, and finish with the arm raised",
    patterns=(
        rf"{_POLITE}wave(?:\s+{_YOUR}{_ARM}(?:hand|arm))?"
        rf"(?:\s+(?:at|to)\s+me)?{_TRAIL}",
        rf"{_POLITE}say\s+(?:hello|hi)\s+with\s+{_YOUR}{_ARM}(?:hand|arm){_TRAIL}",
    ),
    slots=(SLOT_ARM,),
    route="WAVE",
    start_postures=("present",),
    end_posture="present",
))

_register(Ability(
    name="point_cell",
    summary="hover the gripper over a grid cell",
    patterns=(
        rf"{_POLITE}point\s+(?:to|at)\s+(?:the\s+)?(?P<cell>r[1-9]c[1-9])"
        rf"(?:\s+cell)?{_TRAIL}",
        rf"{_POLITE}point\s+(?:to|at)\s+(?:a|the|some)?\s*"
        rf"(?:grid\s+)?(?:cell|square)(?:\s+(?P<cell2>r[1-9]c[1-9]))?{_TRAIL}",
    ),
    slots=(SLOT_CELL, SLOT_ARM),
    route="POINT",
    start_postures=("present",),
    end_posture="hover",
))

_register(Ability(
    name="point_object",
    summary="hover the gripper above an object",
    patterns=(
        rf"{_POLITE}point\s+(?:to|at)\s+(?:the\s+)?(?P<object>.+?){_TRAIL}",
    ),
    slots=(SLOT_OBJECT, SLOT_ARM),
    route="POINT",
    start_postures=("present",),
    end_posture="hover",
))

#: Order matters: `point_cell` must be tried before `point_object`, whose
#: pattern deliberately accepts anything after "point to".  Registry order is
#: insertion order, which is the order above.
_ORDER: Tuple[str, ...] = tuple(REGISTRY)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

#: The one typo worth normalising by name.  The operator makes it; the brief
#: names it.  A general spell-corrector here would quietly turn near-misses
#: into confident matches, which is the opposite of what this module is for.
_TYPOS = {"foream": "forearm", "forarm": "forearm", "gripepr": "gripper"}

#: A command opening with one of these is not a request to do the thing it
#: names.  `fullmatch` already rejects most of them; this catches a pattern
#: permissive enough to be fooled, where the failure would be a moving arm.
_NEGATION_RE = re.compile(
    r"^(?:do\s+not|don't|dont|never|no|not|stop|cancel|abort|"
    r"you\s+(?:should|must)\s+not|avoid)\b"
)

#: Conjunctions that join two requests.  Ordered longest-first so "and then"
#: is not split as "and" leaving a dangling "then".
_JOINERS = (r"\s+and\s+then\s+", r"\s+and\s+also\s+", r"\s*;\s*",
            r"\s+then\s+", r"\s+after\s+that\s+", r"\s+and\s+")
_JOINER_RE = re.compile("|".join(_JOINERS))

#: "Reset the simulation" is a different operation from the operator's "reset".
#: Matched explicitly so the stow alias cannot catch it by containing the word.
_SIM_RESET_RE = re.compile(
    rf"{_POLITE}(?:reset|restart|reload|clear)\s+(?:the\s+)?"
    rf"(?:sim|simulation|simulator|world|scene|board|environment|everything)"
    rf"{_TRAIL}"
)

#: "Rest your arm" with no destination.  Colloquially either posture.
_AMBIGUOUS_REST_RE = re.compile(
    rf"{_POLITE}(?:"
    rf"(?:rest|relax|lower|settle)(?:\s+{_YOUR}{_ARM}(?:arm|forearm|hand))?"
    rf"|(?:go|return|move)\s+(?:back\s+)?to\s+(?:the\s+)?rest"
    rf"(?:\s+(?:position|pose|posture))?"
    rf")\s*{_TRAIL}"
)

#: Verbs the pick-and-place parser owns.  Used only to decide whether a
#: fragment of a compound is a command at all.
_PICK_PLACE_VERBS = ("put", "move", "place", "drop", "sort")


@dataclass
class AbilityMatch:
    """A recognised request, with whatever the phrasing already decided."""

    name: str
    ability: Optional[Ability]
    #: Slot values the text supplied.  A slot the ability accepts but the text
    #: did not fill is simply absent — never a placeholder.
    slots: Dict[str, str] = field(default_factory=dict)
    arm: str = DEFAULT_ARM
    #: True when the operator named an arm rather than leaving it to default.
    arm_was_named: bool = False
    #: Set when the phrasing names two abilities equally well.  `name` is then
    #: empty: choosing one would be inventing an answer the operator did not
    #: give, and the two differ by eleven waypoints.
    ambiguous_between: Tuple[str, ...] = ()


class AbilityRefusal(Exception):
    """A request that was understood well enough to be turned down."""

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason      # "negated" | "compound" | "sim_reset"


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, drop trailing punctuation, fix the typo."""
    norm = re.sub(r"\s+", " ", text.strip().lower()).rstrip(".!?")
    words = [_TYPOS.get(w, w) for w in norm.split(" ")]
    return " ".join(words)


def _looks_like_a_command(fragment: str) -> bool:
    if not fragment:
        return False
    if _match_one(fragment) is not None:
        return True
    return fragment.split(" ", 1)[0] in _PICK_PLACE_VERBS


def _match_one(norm: str) -> Optional[AbilityMatch]:
    """Try every ability against the whole command.  First win, in order."""
    for name in _ORDER:
        ability = REGISTRY[name]
        for pattern in ability.compiled():
            m = pattern.fullmatch(norm)
            if m is None:
                continue
            groups = {k: v for k, v in (m.groupdict() or {}).items()
                      if v is not None}
            slots: Dict[str, str] = {}
            cell = groups.get("cell") or groups.get("cell2")
            if cell:
                slots[SLOT_CELL] = cell
            if groups.get("object"):
                slots[SLOT_OBJECT] = groups["object"].strip()
            arm = groups.get("arm") or DEFAULT_ARM
            return AbilityMatch(name=name, ability=ability, slots=slots,
                                arm=arm, arm_was_named=bool(groups.get("arm")))
    return None


def match(text: str) -> Optional[AbilityMatch]:
    """Recognise a request, or None if no ability claims it.

    Raises AbilityRefusal for a request that IS understood and must not run:
    a negated one, a compound one, or a simulator reset wearing the stow
    alias's clothes.  Those are different from "not understood", and the
    operator gets a different answer for each.
    """
    norm = normalise(text)
    if not norm:
        return None

    if _NEGATION_RE.match(norm):
        raise AbilityRefusal(
            "That reads as telling me not to do something. Ask me for what "
            "you do want and I will do that instead.",
            "negated",
        )

    if _SIM_RESET_RE.fullmatch(norm):
        raise AbilityRefusal(
            "Resetting the simulation is a different operation from resetting "
            "my arm, and it would discard the objects on the board and the "
            "simulation clock. I will not do it from here. If you meant my "
            "arm, say \"reset your arm\" and I will stow it.",
            "sim_reset",
        )

    direct = _match_one(norm)
    if direct is not None:
        return direct

    if _AMBIGUOUS_REST_RE.fullmatch(norm):
        # "Rest" colloquially means either posture.  REST is the forearm
        # supported on the tabletop; HOME is stored in the rail pocket, eleven
        # waypoints away through the rig.  Picking one would be guessing at a
        # long arm movement.
        m = re.match(rf"^{_POLITE}\w+\s+{_YOUR}{_ARM}", norm)
        arm = (m.groupdict().get("arm") if m else None) or DEFAULT_ARM
        return AbilityMatch(name="", ability=None, arm=arm,
                            arm_was_named=bool(m and m.groupdict().get("arm")),
                            ambiguous_between=("rest_forearm", "stow_arm"))

    parts = [p.strip() for p in _JOINER_RE.split(norm) if p and p.strip()]
    if len(parts) > 1 and sum(_looks_like_a_command(p) for p in parts) > 1:
        raise AbilityRefusal(
            "That is more than one request. Ask me for one thing at a time — "
            "I will not do half of something you asked for as a pair.",
            "compound",
        )
    return None


def describe_all() -> List[Dict[str, object]]:
    """What this server can be asked to do.  The browser reads this, not a
    copy of the alias list kept in JavaScript."""
    return [
        {
            "name": a.name,
            "summary": a.summary,
            "slots": list(a.slots),
            "needs_scene": a.needs_scene,
            "needs_motion": a.needs_motion,
            "route": a.route,
            "end_posture": a.end_posture,
        }
        for a in REGISTRY.values()
    ]
