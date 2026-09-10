"""Issue #61: the ability registry and the intent contract.

Recognition only.  Nothing here moves anything, and every ability still
answers "I cannot do it yet" — the point of this stage is that it answers that
BY NAME, for the right phrases, and refuses the ones it must.

Two tiers like the sibling planner tests: the registry is pure and runs
anywhere; the planner tests below build a scene stand-in.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../web"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

import panel_abilities as abilities  # noqa: E402
from panel_abilities import AbilityRefusal, match, normalise  # noqa: E402
from panel_planner import DeterministicPlanner  # noqa: E402
from panel_scene import scene_view_from_doc  # noqa: E402
from tasks import ConversationEvent, PlannerRequest  # noqa: E402


class _Cell:
    def __init__(self, reachable=True):
        self.reachable = reachable
        self.shoulder_distance_m = 0.4


def make_scene(unreachable=("r3c1", "r3c2")):
    doc = {
        "name": "TestScene",
        "objects": [
            {"id": "soda_can", "semantic_class": "can",
             "tags": ["manipulable", "pickable", "recyclable"]},
            {"id": "foam_block", "semantic_class": "block",
             "tags": ["manipulable", "pickable", "non-recyclable"]},
        ],
    }
    cells = {f"r{r}c{c}": _Cell(f"r{r}c{c}" not in unreachable)
             for r in (1, 2, 3) for c in (1, 2, 3)}
    return scene_view_from_doc(doc, cells,
                               placeable=["soda_can", "foam_block"])


def plan(scene, text, answers=()):
    """Replay a conversation the way the coordinator does, slots included."""
    planner = DeterministicPlanner(lambda: scene)
    history = [ConversationEvent(role="user", text=text)]
    out = planner(PlannerRequest(text=text, history=history))
    for a in answers:
        history.append(ConversationEvent(
            role="reachy", text=out.message, question_id="q",
            choices=list(out.choices), slot=out.slot,
        ))
        history.append(ConversationEvent(role="user", text=a))
        out = planner(PlannerRequest(text=a, history=history))
    return out


# ---------------------------------------------------------------------------
# The phrase table, exactly as the operator states it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,name", [
    ("Place your forearm on the table", "rest_forearm"),
    ("place your foream on the table", "rest_forearm"),      # the named typo
    ("rest your arm on the table", "rest_forearm"),
    ("lay your forearm down on the tabletop", "rest_forearm"),
    ("Wave", "wave"),
    ("wave your hand", "wave"),
    ("wave at me", "wave"),
    ("Hello", "greet"),
    ("hi", "greet"),
    ("good morning", "greet"),
    ("Hey Reachy", "greet"),
    ("Store your arm", "stow_arm"),
    ("put your arm away", "stow_arm"),
    ("stow your arm", "stow_arm"),
    ("return to default position", "stow_arm"),
    ("reset", "stow_arm"),
    ("reset your arm", "stow_arm"),
    ("go home", "stow_arm"),
    ("point to a cell", "point_cell"),
    ("Point to r2c2", "point_cell"),
    ("point at the r1c3 cell", "point_cell"),
    ("point to the soda_can", "point_object"),
    ("point to the soda can", "point_object"),
])
def test_the_phrase_table_resolves(text, name):
    m = match(text)
    assert m is not None, f"{text!r} matched nothing"
    assert m.name == name


def test_the_named_typo_is_normalised_and_nothing_else_is():
    assert normalise("place your foream down") == "place your forearm down"
    # Not a general spell-corrector: a near-miss stays a near-miss rather than
    # becoming a confident match for something the operator did not type.
    assert normalise("place your forrarm down") == "place your forrarm down"
    assert match("place your forrarm on the table") is None


def test_a_named_cell_or_object_is_carried_on_the_match():
    assert match("point to r2c2").slots[abilities.SLOT_CELL] == "r2c2"
    assert match("point to a cell").slots == {}
    assert match("point to the soda can").slots[abilities.SLOT_OBJECT] == "soda can"


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "do not wave",
    "don't wave",
    "never point to r2c2",
    "do not store your arm",
    "don't reset",
    "no, do not rest your forearm on the table",
])
def test_a_negated_request_is_refused_not_matched(text):
    with pytest.raises(AbilityRefusal) as exc:
        match(text)
    assert exc.value.reason == "negated"


def test_matching_is_full_intent_not_substring():
    """The #58 defect one layer up, where the consequence is a moving arm."""
    assert match("I waved at it earlier") is None
    assert match("the reset button is stuck") is None
    assert match("tell me about pointing") is None


# ---------------------------------------------------------------------------
# Compounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "wave and then stow your arm",
    "wave and stow your arm",
    "store your arm then wave",
    "put soda_can on r2c2 and then wave",
    "wave; reset",
])
def test_a_compound_request_is_refused_whole(text):
    with pytest.raises(AbilityRefusal) as exc:
        match(text)
    assert exc.value.reason == "compound"


def test_a_conjunction_inside_one_request_is_not_a_compound():
    assert match("point to the soda can and cube") is not None


# ---------------------------------------------------------------------------
# Reset means stow.  It never means the simulator's reset.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "reset the simulation",
    "reset the sim",
    "reset the world",
    "reset the scene",
    "restart the simulator",
    "clear the board",
])
def test_resetting_the_simulation_is_a_different_request(text):
    with pytest.raises(AbilityRefusal) as exc:
        match(text)
    assert exc.value.reason == "sim_reset"
    assert "different operation" in exc.value.message


def test_bare_reset_is_the_stow_route():
    assert match("reset").ability.route == "STOW_ROUTE"


def test_no_ability_routes_to_a_simulator_message():
    """Bare "reset" must reach the stow route and nothing else.

    A source scan for the word is the wrong test here — the registry contains
    "reset" as a regex fragment, because the operator types it, and matching
    that literal would fail on the very thing it is meant to allow.  What
    matters is where an ability points, so check the routes.
    """
    forbidden = {"reset", "scene_load", "place_object", "joint_command"}
    for ability in abilities.REGISTRY.values():
        assert ability.route not in forbidden
        assert ability.name not in forbidden


def test_the_registry_cannot_talk_to_the_simulator_at_all():
    """Structural, not a promise: it holds no link and imports nothing that does.

    An ability that could send is an ability that could send `reset`, and the
    operator's "reset" would then be one careless line away from discarding
    their board.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(_HERE, "../../web/panel_abilities.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not (imported & {"panel_sim_link", "panel_executor", "websockets",
                            "socket", "asyncio", "reachy_sdk"})


# ---------------------------------------------------------------------------
# The arm, and the two postures
# ---------------------------------------------------------------------------

def test_the_right_arm_is_the_default_and_is_not_guessed_at():
    m = match("wave")
    assert m.arm == "right"
    assert m.arm_was_named is False


def test_a_left_arm_request_is_recorded_as_left_not_silently_mirrored():
    m = match("wave your left hand")
    assert m.arm == "left"
    assert m.arm_was_named is True


@pytest.mark.parametrize("text", ["rest", "rest your arm", "relax your arm",
                                  "return to rest position"])
def test_bare_rest_is_ambiguous_between_the_two_postures(text):
    m = match(text)
    assert m.name == ""
    assert m.ambiguous_between == ("rest_forearm", "stow_arm")


def test_naming_the_destination_removes_the_ambiguity():
    assert match("rest your arm on the table").name == "rest_forearm"
    assert match("put your arm away").name == "stow_arm"


# ---------------------------------------------------------------------------
# Pick-and-place still belongs to the old grammar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "put soda_can on r2c2",
    "move the foam block to r1c1",
    "Put the recycle item in the bin.",
    "sort the recyclable item",
])
def test_the_registry_declines_pick_and_place_phrases(text):
    assert match(text) is None


# ---------------------------------------------------------------------------
# Through the planner
# ---------------------------------------------------------------------------

def test_put_your_arm_away_is_an_ability_not_a_move():
    """It opens with a pick-and-place verb, so order of matching decides it.

    Left to the old order it parses as a move with "your arm" for a target and
    answers that it does not know an object by that name.
    """
    out = plan(make_scene(), "put your arm away")
    assert out.kind == "unsupported"
    assert "stow" in out.message or "rail pocket" in out.message
    assert "object called" not in out.message


def test_a_recognised_ability_is_refused_by_name():
    out = plan(make_scene(), "wave")
    assert out.kind == "unsupported"
    assert "I understand: wave" in out.message
    assert "I did not understand" not in out.message


def test_pointing_at_a_cell_asks_which_one_and_offers_reachable_cells():
    out = plan(make_scene(), "point to a cell")
    assert out.kind == "clarification"
    assert out.slot == abilities.SLOT_CELL
    assert "r2c2" in out.choices
    assert "r3c1" not in out.choices        # measured out of reach


def test_the_cell_answer_fills_the_cell_slot():
    out = plan(make_scene(), "point to a cell", answers=["r2c2"])
    assert out.kind == "unsupported"
    assert "r2c2" in out.message


def test_bare_rest_asks_which_posture_then_resolves():
    first = plan(make_scene(), "rest your arm")
    assert first.kind == "clarification"
    assert first.slot == abilities.SLOT_POSTURE
    assert "eleven waypoints apart" in first.message

    table = plan(make_scene(), "rest your arm", answers=["on the table"])
    assert "rest the forearm on the table" in table.message

    pocket = plan(make_scene(), "rest your arm", answers=["in the rail pocket"])
    assert "rail pocket" in pocket.message


def test_an_answer_naming_neither_posture_asks_again():
    out = plan(make_scene(), "rest your arm", answers=["whichever"])
    assert out.kind == "clarification"
    assert out.slot == abilities.SLOT_POSTURE


def test_a_left_arm_request_is_refused_with_the_reason():
    out = plan(make_scene(), "wave your left hand")
    assert out.kind == "unsupported"
    assert "right arm" in out.message
    assert "not symmetric" in out.message


def test_a_compound_request_plans_nothing():
    out = plan(make_scene(), "wave and then stow your arm")
    assert out.kind == "unsupported"
    assert out.proposal is None
    assert "one thing at a time" in out.message


def test_reset_the_simulation_is_explained_through_the_planner():
    out = plan(make_scene(), "reset the simulation")
    assert out.kind == "unsupported"
    assert "different operation" in out.message
    assert "reset your arm" in out.message


def test_an_ability_is_recognised_while_the_scene_is_unreadable():
    """Ability matching does not depend on the board.

    Not the greeting reply itself — that is #62 — but the matcher must not sit
    behind the scene read, or every ability inherits the simulator's health.
    """
    from panel_scene import SceneView
    out = DeterministicPlanner(lambda: SceneView(error="bad YAML"))(
        PlannerRequest(text="wave", history=[
            ConversationEvent(role="user", text="wave")])
    )
    assert "I understand: wave" in out.message
    assert "bad YAML" not in out.message


def test_capabilities_serves_the_ability_list():
    """One source of truth, read by the browser rather than copied into it."""
    from panel_routes import PanelRoutes

    routes = PanelRoutes(scene_provider=make_scene)
    status, payload = routes.handle_get("/capabilities", "session-1")
    assert status == 200
    served = {a["name"] for a in payload["abilities"]}
    assert served == set(abilities.REGISTRY)
    wave = next(a for a in payload["abilities"] if a["name"] == "wave")
    assert wave["needs_motion"] is True
    greet = next(a for a in payload["abilities"] if a["name"] == "greet")
    assert greet["needs_scene"] is False
    assert greet["needs_motion"] is False
