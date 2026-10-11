"""What an ability may touch: one rule table for every ability (#172, option C).

WHY A TABLE AND NOT A LIST PER ROUTE
------------------------------------
Until #172 each route carried its own list of bodies it was allowed to touch
(`PanelRouteTaskSpec.intended_contact_bodies`), and every list was empty —
because the offline world had no table and no rails, so nothing was ever
touched and nothing had to be decided.  With the table and rails in the
offline world (`SimulationCore._load_world` now resolves `extends:`), rest and
stow lay the forearm on the board, and the lift closes pads on a cube that is
standing on the table.  A per-route list cannot say "the pads may touch the
cube WHILE GRASPING and not while approaching it", which is the distinction
that matters.

So the question is asked in two terms that every ability shares:

  * ROLES — what each side of a contact IS: a gripper pad, another robot link,
    the target object, a support surface (the table, which is also the board),
    the rig (rails, pedestal, floor), or some other object;
  * PHASES — what the ability was DOING: approach, grasp, hold, carry,
    set-down, release, withdraw.  Each ability tags its own phases; the crane
    uses the segments it already names (`CRANE_PHASES`).

and one table (`RULES`) says which role pairs are allowed in which phases.
Everything the table does not allow is unintended.  #174 (placement) reuses the
table unchanged.

THE RULES
---------
  * pads touching the target are allowed from grasp to release;
  * the target touching its support is allowed while it is not held — standing
    there before the grasp, being lifted off it, being set down on it, and
    standing there again after the release.  WHILE HELD (hold, carry) it is
    not: the held object dragging on the table is exactly what a carry must
    not do;
  * an arm link on the board is allowed only at a declared rest pose, and the
    declaration is a per-route EXCEPTION, not a rule (`ROUTE_EXCEPTIONS`);
  * everything else involving the robot or the target is unintended.

WHAT IS NOT JUDGED HERE
-----------------------
A contact neither side of which is the robot or the target — a pool object
standing on the floor, the table on its frame — is the world at rest, not
something the ability did.  Whether the ability DISTURBED those things is the
drift clause's question (`rig_routes.OBJECT_DRIFT_TOL`), and it is asked
separately.  Self-contact (robot on robot) is not in the robot's collision
groups at all.

This module is offline-safe: no MuJoCo, no native imports.  Contacts arrive as
plain records (`ContactSample`) from whoever observed them — the offline route
runner, or the replay of a recorded run.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import (Any, Dict, FrozenSet, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)


class Role(str, enum.Enum):
    """What one side of a contact is, for the purposes of the rules."""

    PAD = "pad"                  #: the acting gripper's thumb or finger
    ARM_LINK = "arm_link"        #: any other robot link
    TARGET = "target"            #: the object the ability is acting on
    SUPPORT = "support"          #: the table (which is also the board)
    RIG = "rig"                  #: rails, pedestal, floor: the fixed world
    OTHER_OBJECT = "other_object"


class Phase(str, enum.Enum):
    """What the ability was doing.  Tagged by the ability, not inferred here."""

    APPROACH = "approach"
    GRASP = "grasp"
    HOLD = "hold"
    CARRY = "carry"
    SET_DOWN = "set_down"
    RELEASE = "release"
    WITHDRAW = "withdraw"


ALL_PHASES: FrozenSet[Phase] = frozenset(Phase)

#: The robot side of the rules.
ROBOT_ROLES: FrozenSet[Role] = frozenset({Role.PAD, Role.ARM_LINK})

#: The acting (right) gripper's collision bodies.  The thumb and finger carry
#: the pads; every other robot body is an arm link.
PAD_BODIES: FrozenSet[str] = frozenset({"r_gripper_thumb", "r_gripper_finger"})

#: Robot bodies by name.  The model's collision bodies are these, both arms.
ROBOT_BODY_PREFIXES: Tuple[str, ...] = ("r_", "l_")
ROBOT_BODIES: FrozenSet[str] = frozenset({"torso", "head"})

#: Support surfaces.  The board is drawn on the table top; in the compiled
#: scene they are one collidable body.
SUPPORT_BODIES: FrozenSet[str] = frozenset({"table_top", "board"})

#: The fixed world besides the support: rails, pedestal, floor.
RIG_BODIES: FrozenSet[str] = frozenset({"world", "floor", "pedestal"})
RIG_PREFIXES: Tuple[str, ...] = ("rig_",)


def _pair(a: Role, b: Role) -> FrozenSet[Role]:
    return frozenset((a, b))


@dataclasses.dataclass(frozen=True)
class Rule:
    """One row of the table: a role pair, the phases it is allowed in, and why."""

    roles: FrozenSet[Role]
    allowed_in: FrozenSet[Phase]
    reason: str


#: THE table.  A role pair that is not here is allowed in no phase.
RULES: Tuple[Rule, ...] = (
    Rule(_pair(Role.PAD, Role.TARGET),
         frozenset({Phase.GRASP, Phase.HOLD, Phase.CARRY, Phase.SET_DOWN,
                    Phase.RELEASE}),
         "pads on the target from grasp to release are the grasp"),
    Rule(_pair(Role.TARGET, Role.SUPPORT),
         frozenset({Phase.APPROACH, Phase.GRASP, Phase.SET_DOWN,
                    Phase.RELEASE, Phase.WITHDRAW}),
         "the target stands on its support until it is lifted, is set down "
         "on it, and stands there after the release; while HELD it must not "
         "touch it"),
    # An arm link on the board: allowed only at a declared rest pose, which a
    # route declares in ROUTE_EXCEPTIONS.  Listed so the table says so.
    Rule(_pair(Role.ARM_LINK, Role.SUPPORT), frozenset(),
         "an arm link on the board only at a declared rest pose (a route "
         "exception)"),
)

_RULES_BY_PAIR: Dict[FrozenSet[Role], Rule] = {r.roles: r for r in RULES}


@dataclasses.dataclass(frozen=True)
class RouteException:
    """One declared exception: on ``route``, ``link`` may touch a body of
    ``role`` while the arm stands at ``pose``.  Each one carries its reason."""

    route: str
    link: str
    role: Role
    pose: str
    reason: str


#: The per-route exceptions.  KEEP THIS SHORT: an exception is a contact the
#: rule table would call unintended and a route needs, and every entry is a
#: claim somebody has to be able to defend.
ROUTE_EXCEPTIONS: Tuple[RouteException, ...] = (
    RouteException(
        "PLACE_ROUTE", "r_forearm", Role.SUPPORT, "rest",
        "REST is the forearm laid on the board: rest_forearm ends with that "
        "contact, it is the posture (#91; offline, the forearm is on table_top "
        "only while the arm stands at REST)"),
    RouteException(
        "STOW_ROUTE", "r_forearm", Role.SUPPORT, "rest",
        "stow_arm starts at REST with the forearm on the board and lifts it "
        "off on the first waypoint (#91; offline, the contact ends as the arm "
        "leaves REST)"),
)


def exceptions_for(route: str) -> List[RouteException]:
    return [e for e in ROUTE_EXCEPTIONS if e.route == route]


def role_of(body: str, *, target_id: str = "") -> Role:
    """The role a body plays.  Unknown bodies are OTHER_OBJECT, which no rule
    allows anything with — an unrecognised body is never an intended contact."""
    if body in PAD_BODIES:
        return Role.PAD
    if body in ROBOT_BODIES or body.startswith(ROBOT_BODY_PREFIXES):
        return Role.ARM_LINK
    if target_id and body == target_id:
        return Role.TARGET
    if body in SUPPORT_BODIES:
        return Role.SUPPORT
    if body in RIG_BODIES or body.startswith(RIG_PREFIXES):
        return Role.RIG
    return Role.OTHER_OBJECT


@dataclasses.dataclass(frozen=True)
class ContactSample:
    """A contact between two bodies, observed during one phase.

    ``samples`` is how many observations it was seen in (steps offline, states
    in a replay); ``pose`` is the named posture the arm stood at, when it
    stood at one (`rig_routes.posture_of`).  ``route`` is the measured route
    being flown when that is not the ability's own — an ability's approach
    to its start posture (the lift's RAISE_TO_SIDE from HOME) is judged under
    that route's exceptions, not the ability's.  Aggregate by (bodies, phase,
    pose, route) — the rules do not care how often, only when.
    """

    body1: str
    body2: str
    phase: Phase
    pose: Optional[str] = None
    samples: int = 1
    deepest_m: float = 0.0
    route: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"body1": self.body1, "body2": self.body2,
                "phase": self.phase.value, "pose": self.pose,
                "samples": int(self.samples),
                "deepest_m": round(float(self.deepest_m), 6),
                "route": self.route}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ContactSample":
        return cls(body1=str(d.get("body1", "")), body2=str(d.get("body2", "")),
                   phase=Phase(d.get("phase", Phase.APPROACH.value)),
                   pose=d.get("pose") or None,
                   samples=int(d.get("samples", 1) or 1),
                   deepest_m=float(d.get("deepest_m", 0.0) or 0.0),
                   route=str(d.get("route") or ""))


@dataclasses.dataclass(frozen=True)
class JudgedContact:
    sample: ContactSample
    role1: Role
    role2: Role
    allowed: bool
    why: str

    def describe(self) -> str:
        s = self.sample
        at = f" at {s.pose.upper()}" if s.pose else ""
        on = f" on {s.route}" if s.route else ""
        return (f"{s.body1} ({self.role1.value}) on {s.body2} "
                f"({self.role2.value}) during {s.phase.value}{on}{at}, "
                f"{s.samples} sample(s), {s.deepest_m * 1000:.2f} mm deep")


def in_scope(r1: Role, r2: Role) -> bool:
    """Is this contact the ability's doing?  One side must be the robot or the
    target; anything else is the world at rest (see the module doc)."""
    judged = ROBOT_ROLES | {Role.TARGET}
    return r1 in judged or r2 in judged


def judge(sample: ContactSample, *, route: str = "",
          target_id: str = "") -> Optional[JudgedContact]:
    """Judge one contact against the table and the route's exceptions.

    Returns None for a contact that is out of scope (neither side is the robot
    or the target).  A sample tagged with its own ``route`` is judged under
    that route's exceptions.
    """
    route = sample.route or route
    r1 = role_of(sample.body1, target_id=target_id)
    r2 = role_of(sample.body2, target_id=target_id)
    if not in_scope(r1, r2):
        return None
    if r1 in ROBOT_ROLES and r2 in ROBOT_ROLES:
        return None          # self-contact: not this table's question

    rule = _RULES_BY_PAIR.get(_pair(r1, r2))
    if rule is not None and sample.phase in rule.allowed_in:
        return JudgedContact(sample, r1, r2, True, rule.reason)

    for exc in exceptions_for(route):
        robot_body = sample.body1 if r1 in ROBOT_ROLES else sample.body2
        other = r2 if r1 in ROBOT_ROLES else r1
        if (robot_body == exc.link and other == exc.role
                and (sample.pose or "") == exc.pose):
            return JudgedContact(sample, r1, r2, True,
                                 f"route exception: {exc.reason}")

    if rule is not None:
        why = (f"{r1.value} on {r2.value} is allowed only in "
               + (", ".join(sorted(p.value for p in rule.allowed_in))
                  or "a declared exception")
               + f", and this was during {sample.phase.value}")
    else:
        why = f"no rule allows {r1.value} on {r2.value} in any phase"
    return JudgedContact(sample, r1, r2, False, why)


def judge_all(samples: Sequence[ContactSample], *, route: str = "",
              target_id: str = "") -> List[JudgedContact]:
    """Every in-scope contact, judged.  Out-of-scope ones are dropped."""
    out = []
    for s in samples:
        j = judge(s, route=route, target_id=target_id)
        if j is not None:
            out.append(j)
    return out


def aggregate(raw: Iterable[Tuple]) -> List[ContactSample]:
    """Fold per-observation contacts — (body1, body2, phase, pose, depth_m)
    or (..., route) — into one ContactSample per (unordered body pair, phase,
    pose, route)."""
    acc: Dict[Tuple[str, str, Phase, Optional[str], str], List[float]] = {}
    for item in raw:
        b1, b2, phase, pose, depth = item[:5]
        route = str(item[5]) if len(item) > 5 and item[5] else ""
        a, b = sorted((str(b1), str(b2)))
        key = (a, b, Phase(phase), pose or None, route)
        slot = acc.setdefault(key, [0, 0.0])
        slot[0] += 1
        slot[1] = max(slot[1], max(0.0, -float(depth)))
    return [ContactSample(a, b, ph, pose, int(n), float(d), route)
            for (a, b, ph, pose, route), (n, d) in sorted(
                acc.items(), key=lambda kv: (kv[0][4], kv[0][0], kv[0][1],
                                             kv[0][2].value, kv[0][3] or ""))]


# ---------------------------------------------------------------------------
# Phases the abilities tag
# ---------------------------------------------------------------------------

#: The crane's own segment names (`tasks.crane_pick_live`, its `phase`
#: events), mapped onto the shared phases.  Matched by prefix, because the
#: crane writes durations into some of them ("hold (2 s)").
#:
#: The crane has no segment of its own for the release: it opens the hand at
#: the end of "replace", once the object is supported.  Its SUPPORTED and
#: RELEASED events bound that, and `crane_phase_windows` uses them.
CRANE_PHASES: Tuple[Tuple[str, Phase], ...] = (
    ("ATTEMPT_START", Phase.APPROACH),
    ("transit PRESENT -> via", Phase.APPROACH),
    ("via -> hover", Phase.APPROACH),
    ("HOVER pause", Phase.APPROACH),
    ("descent", Phase.APPROACH),
    ("close", Phase.GRASP),
    ("lift", Phase.GRASP),
    ("hold", Phase.HOLD),
    ("replace", Phase.SET_DOWN),
    ("withdrawal", Phase.WITHDRAW),
    ("back at PRESENT", Phase.WITHDRAW),
    ("stow_from_side", Phase.WITHDRAW),
)


def crane_phase(segment: str) -> Optional[Phase]:
    for prefix, phase in CRANE_PHASES:
        if segment.startswith(prefix):
            return phase
    return None


#: The phase a measured route's whole flight is tagged with.  None of these
#: routes handles an object, so no rule in the table turns on their phase; the
#: arm-on-board exception turns on the POSE, which is tagged per sample.
MEASURED_ROUTE_PHASES: Dict[str, Phase] = {
    "PLACE_ROUTE": Phase.APPROACH,
    "STOW_ROUTE": Phase.WITHDRAW,
    "WAVE": Phase.APPROACH,
    "POINT": Phase.APPROACH,
}


def rules_as_rows() -> List[Dict[str, str]]:
    """The table, for a report."""
    return [{"roles": " × ".join(sorted(r.value for r in rule.roles)),
             "allowed_in": ", ".join(p.value for p in Phase
                                     if p in rule.allowed_in) or "—",
             "reason": rule.reason} for rule in RULES]
