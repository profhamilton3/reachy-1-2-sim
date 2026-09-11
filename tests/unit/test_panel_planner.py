"""Issue #49: the panel's deterministic command parser.

Two tiers, like tests/unit/test_placement.py.  Everything above the divider is
pure and runs anywhere; the tests below it load the real FWDCenterLabSivaPool
document, because the claims that matter most — soda_can is recyclable,
foam_block is not, and this scene has no bin — are claims about that file and
are worth nothing if asserted against a fixture.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../web"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

from panel_planner import DeterministicPlanner, parse_intent  # noqa: E402
from panel_scene import SceneView, scene_view_from_doc  # noqa: E402
from tasks import ConversationEvent, PlannerRequest  # noqa: E402


class _Cell:
    def __init__(self, reachable=True, distance=0.4):
        self.reachable = reachable
        self.shoulder_distance_m = distance


def make_scene(*, destinations=False, live=False, unreachable=("r3c1", "r3c2"),
               occupant=None):
    """A stand-in scene with the pool scene's shape but no file on disk."""
    doc = {
        "name": "TestScene",
        "objects": [
            {"id": "soda_can", "semantic_class": "can",
             "tags": ["pool", "manipulable", "pickable", "recyclable", "sort-item"]},
            {"id": "foam_block", "semantic_class": "block",
             "tags": ["pool", "manipulable", "pickable", "non-recyclable", "sort-item"]},
            {"id": "red_cube", "semantic_class": "cube",
             "tags": ["pool", "manipulable", "pickable", "cube"]},
        ],
    }
    if destinations:
        doc["objects"].append(
            {"id": "left_tray", "semantic_class": "tray",
             "tags": ["tray", "destination", "placement-target", "sort-target"]}
        )
    cells = {}
    for r in (1, 2, 3):
        for c in (1, 2, 3):
            name = f"r{r}c{c}"
            cells[name] = _Cell(reachable=name not in unreachable)
    view = scene_view_from_doc(doc, cells,
                               placeable=["soda_can", "foam_block", "red_cube"])
    view.live = live
    if occupant:
        view.cells[occupant[0]].occupant = occupant[1]
    return view


def plan(scene, text, answers=()):
    """Replay a conversation the way the coordinator does.

    Each answer is preceded by the question the planner ACTUALLY asked — its
    slot included — rather than a stand-in event.  Inventing the question would
    leave every answer slotless, which exercises the fallback in
    `_apply_answers` and never touches the slot routing that #59 added.
    """
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
# Grammar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,target,dest", [
    ("put soda_can on r2c2", "soda_can", "r2c2"),
    ("Put the recycle item in the bin.", "recycle item", "bin"),
    ("move red_cube to r1c1", "red_cube", "r1c1"),
    ("place the soda can onto r2c3", "soda can", "r2c3"),
    ("drop foam_block into r1c2", "foam_block", "r1c2"),
])
def test_parse_intent_splits_target_and_destination(text, target, dest):
    intent = parse_intent(text)
    assert intent is not None
    assert intent.target_phrase == target
    assert intent.dest_phrase == dest


def test_into_is_not_split_as_in_plus_to():
    assert parse_intent("put soda_can into r2c2").dest_phrase == "r2c2"


@pytest.mark.parametrize("text", [
    "", "hello", "what can you see?", "make me a sandwich",
])
def test_unrecognised_commands_are_not_parsed(text):
    assert parse_intent(text) is None


def test_unsupported_input_explains_what_is_supported():
    out = plan(make_scene(), "make me a sandwich")
    assert out.kind == "unsupported"
    assert "put soda_can on r2c2" in out.message


# ---------------------------------------------------------------------------
# Raw joint control is refused, not ignored
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "move r_elbow_pitch to -75 degrees",
    "put joint r_shoulder_roll on 0.4 rad",
    "move neck_pitch to 12 deg",
])
def test_joint_level_commands_are_refused(text):
    out = plan(make_scene(), text)
    assert out.kind == "unsupported"
    assert "joint angles" in out.message


def test_joint_control_smuggled_into_an_answer_is_refused():
    out = plan(make_scene(), "put soda_can on r2c2",
               answers=["set r_wrist_pitch to 0.9 rad"])
    assert out.kind == "unsupported"
    assert "joint angles" in out.message


# ---------------------------------------------------------------------------
# Destinations
# ---------------------------------------------------------------------------

def test_missing_bin_asks_rather_than_substituting_a_tray():
    out = plan(make_scene(), "Put the recycle item in the bin.")
    assert out.kind == "clarification"
    assert "no bin or tray configured" in out.message
    # It must not have quietly resolved to a tray that is not in the scene.
    assert "tray" not in " ".join(out.choices)
    assert out.choices  # it offers the cells it can actually use


def test_a_grid_cell_is_never_called_a_bin():
    out = plan(make_scene(), "Put the recycle item in the bin.")
    assert "not a container" in out.message


def test_a_configured_destination_is_used_when_one_exists():
    scene = make_scene(destinations=True, live=True)
    scene.objects["soda_can"].on_board = True
    out = plan(scene, "Put the recycle item in the bin.")
    assert out.kind == "proposal"
    assert out.proposal.destination == "destination:left_tray"


def test_unknown_destination_is_rejected():
    out = plan(make_scene(), "put soda_can on the workbench")
    assert out.kind == "clarification"
    assert "do not know a destination" in out.message


def test_nonexistent_cell_is_rejected():
    out = plan(make_scene(), "put soda_can on r9c9")
    assert out.kind == "clarification"
    assert "no cell called r9c9" in out.message


def test_unreachable_cell_is_refused_without_an_override():
    out = plan(make_scene(), "put soda_can on r3c1")
    assert out.kind == "clarification"
    assert "out of the right arm's reach" in out.message
    assert "r3c1" not in out.choices


def test_occupied_cell_is_refused():
    scene = make_scene(occupant=("r2c2", "red_cube"))
    out = plan(scene, "put soda_can on r2c2")
    assert out.kind == "clarification"
    assert "already occupied by red_cube" in out.message


def test_missing_destination_asks_where():
    out = plan(make_scene(), "put soda_can")
    assert out.kind == "clarification"
    assert out.message == "Where should it go?"


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def test_object_id_with_spaces_resolves():
    out = plan(make_scene(), "put the soda can on r2c2")
    assert out.kind == "proposal"
    assert out.proposal.target_id == "soda_can"


def test_unknown_object_is_rejected_with_the_real_list():
    out = plan(make_scene(), "put the banana on r2c2")
    assert out.kind == "clarification"
    assert "do not know an object" in out.message
    assert set(out.choices) == {"soda_can", "foam_block", "red_cube"}


def test_without_live_poses_the_planner_refuses_to_guess_candidacy():
    out = plan(make_scene(live=False), "put the recycle item on r2c2")
    assert out.kind == "clarification"
    assert "cannot yet see which objects are actually on the board" in out.message
    assert out.choices == ["soda_can"]


def test_non_recyclable_is_never_matched_as_recyclable():
    scene = make_scene(live=True)
    scene.objects["soda_can"].on_board = True
    scene.objects["foam_block"].on_board = True
    out = plan(scene, "put the recycle item on r2c2")
    assert out.kind == "proposal"
    assert out.proposal.target_id == "soda_can"


def test_recyclable_still_in_the_pool_is_not_a_tabletop_candidate():
    scene = make_scene(live=True)
    scene.objects["soda_can"].on_board = False
    out = plan(scene, "put the recycle item on r2c2")
    assert out.kind == "clarification"
    assert "No recyclable object is on the board" in out.message


def test_several_tabletop_recyclables_ask_which_one():
    scene = make_scene(live=True)
    scene.objects["soda_can"].on_board = True
    scene.objects["soda_can"].cell = "r1c1"
    scene.objects["red_cube"].tags.append("recyclable")
    scene.objects["red_cube"].on_board = True
    scene.objects["red_cube"].cell = "r2c1"
    out = plan(scene, "put the recycle item on r2c2")
    assert out.kind == "clarification"
    assert "more than one recyclable" in out.message
    assert "soda_can (r1c1)" in out.choices


def test_scene_with_no_recyclable_says_so():
    scene = make_scene(live=True)
    scene.objects["soda_can"].tags.remove("recyclable")
    out = plan(scene, "put the recycle item on r2c2")
    assert out.kind == "unsupported"
    assert "tagged recyclable" in out.message


# ---------------------------------------------------------------------------
# Negated categories (issue #58)
#
# `"recyclable" in "non-recyclable item"` is True, so a substring test here
# answered the negated question with the positive category and proposed moving
# the one object the operator had ruled out.  The proposal card looked entirely
# reasonable; nothing in it showed the negation had been dropped.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrase", [
    "non-recyclable item",
    "non recyclable item",
    "not recyclable item",
    "the item that isn't recyclable",
])
def test_a_negated_phrase_never_resolves_to_the_recyclable_object(phrase):
    scene = make_scene(live=True)
    scene.objects["soda_can"].on_board = True
    scene.objects["foam_block"].on_board = True
    out = plan(scene, f"put the {phrase} on r2c2")
    assert out.kind == "proposal", out.message
    assert out.proposal.target_id == "foam_block"


@pytest.mark.parametrize("phrase", ["recycle item", "recycling item",
                                    "recyclable item"])
def test_the_positive_phrases_are_unchanged(phrase):
    scene = make_scene(live=True)
    scene.objects["soda_can"].on_board = True
    scene.objects["foam_block"].on_board = True
    out = plan(scene, f"put the {phrase} on r2c2")
    assert out.kind == "proposal", out.message
    assert out.proposal.target_id == "soda_can"


def test_a_scene_with_no_non_recyclable_tag_does_not_fall_back_to_recyclable():
    """The complementary category is a tag, not the absence of one.

    Answering "everything not tagged recyclable" would make red_cube — which
    carries neither tag — a non-recyclable item, and in a scene with trays it
    would make a tray one too.
    """
    scene = make_scene(live=True)
    scene.objects["foam_block"].tags.remove("non-recyclable")
    scene.objects["soda_can"].on_board = True
    scene.objects["foam_block"].on_board = True
    out = plan(scene, "put the non-recyclable item on r2c2")
    assert out.kind == "unsupported"
    assert "tagged non-recyclable" in out.message


def test_an_untagged_object_belongs_to_neither_category():
    scene = make_scene(live=True)
    for oid in ("soda_can", "foam_block", "red_cube"):
        scene.objects[oid].on_board = True
    assert "red_cube" not in [o.object_id for o in scene.tagged("recyclable")]
    assert "red_cube" not in [o.object_id
                              for o in scene.tagged("non-recyclable")]


def test_a_negated_request_asks_which_when_several_are_on_the_board():
    scene = make_scene(live=True)
    scene.objects["foam_block"].on_board = True
    scene.objects["foam_block"].cell = "r1c1"
    scene.objects["red_cube"].tags.append("non-recyclable")
    scene.objects["red_cube"].on_board = True
    scene.objects["red_cube"].cell = "r2c1"
    out = plan(scene, "put the non-recyclable item on r2c2")
    assert out.kind == "clarification"
    assert "more than one non-recyclable" in out.message
    assert "foam_block (r1c1)" in out.choices


def test_a_negated_request_with_none_on_the_board_says_so():
    scene = make_scene(live=True)
    scene.objects["foam_block"].on_board = False
    out = plan(scene, "put the non-recyclable item on r2c2")
    assert out.kind == "clarification"
    assert "No non-recyclable object is on the board" in out.message


def test_a_word_merely_containing_non_is_not_a_negation():
    """`\bnon-?` must not fire on the tail of another word."""
    from panel_planner import _category_of
    assert _category_of("canon recyclable") == "recyclable"
    assert _category_of("non-recyclable") == "non-recyclable"
    assert _category_of("a wooden block") is None


# ---------------------------------------------------------------------------
# Clarification answers
# ---------------------------------------------------------------------------

def test_an_answer_fills_the_missing_destination():
    out = plan(make_scene(), "Put the recycle item in the bin.", answers=["r2c2"])
    # target is still ambiguous without live poses, so it asks for that next
    assert out.kind == "clarification"
    assert "actually on the board" in out.message


def test_answers_can_complete_a_proposal():
    out = plan(make_scene(), "Put the recycle item in the bin.",
               answers=["r2c2", "soda_can"])
    assert out.kind == "proposal"
    assert out.proposal.target_id == "soda_can"
    assert out.proposal.destination == "cell:r2c2"


def test_an_answer_naming_a_cell_is_not_mistaken_for_an_object():
    out = plan(make_scene(), "put soda_can", answers=["r2c2"])
    assert out.kind == "proposal"
    assert out.proposal.target_id == "soda_can"
    assert out.proposal.destination == "cell:r2c2"


# ---------------------------------------------------------------------------
# The answer goes to the slot that was asked about (issue #59)
#
# A cell that EXISTS but was rejected — occupied, or out of reach — used to
# read as a resolved destination, so the slot the planner had just asked about
# counted as filled and the answer fell through to the target.  The reply was
# then about not knowing an object called "r1c1".
# ---------------------------------------------------------------------------

def test_a_new_cell_answering_an_occupied_cell_becomes_the_destination():
    scene = make_scene(occupant=("r2c2", "foam_block"))
    first = plan(scene, "put soda_can on r2c2")
    assert first.kind == "clarification"
    assert "already occupied" in first.message
    assert first.slot == "which_cell"

    out = plan(scene, "put soda_can on r2c2", answers=["r1c1"])
    assert out.kind == "proposal", out.message
    assert out.proposal.target_id == "soda_can"
    assert out.proposal.destination == "cell:r1c1"


def test_a_new_cell_answering_an_unreachable_cell_becomes_the_destination():
    scene = make_scene()
    first = plan(scene, "put soda_can on r3c1")
    assert first.kind == "clarification"
    assert "out of the right arm's reach" in first.message
    assert first.slot == "which_cell"

    out = plan(scene, "put soda_can on r3c1", answers=["r1c1"])
    assert out.kind == "proposal", out.message
    assert out.proposal.destination == "cell:r1c1"


def test_an_answer_to_which_object_still_fills_the_target():
    scene = make_scene()
    first = plan(scene, "put the widget on r2c2")
    assert first.kind == "clarification"
    assert first.slot == "which_object"

    out = plan(scene, "put the widget on r2c2", answers=["soda_can"])
    assert out.kind == "proposal", out.message
    assert out.proposal.target_id == "soda_can"
    assert out.proposal.destination == "cell:r2c2"


def test_every_clarification_names_a_slot():
    """A question with no slot sends its answer back to the guessing path."""
    scene = make_scene(live=True)
    asked = [
        plan(scene, "put soda_can"),                       # no destination
        plan(scene, "put soda_can on r9c9"),               # no such cell
        plan(scene, "put soda_can on r3c1"),               # unreachable
        plan(make_scene(occupant=("r2c2", "foam_block")),
             "put soda_can on r2c2"),                      # occupied
        plan(scene, "put soda_can on the shelf"),          # unknown destination
        plan(scene, "put soda_can in the bin"),            # no bin in scene
        plan(scene, "put the widget on r2c2"),             # unknown object
        plan(make_scene(), "put the recycle item on r2c2"),  # not live yet
    ]
    for out in asked:
        assert out.kind == "clarification", out.message
        assert out.slot in ("which_cell", "which_object"), out.message


def test_a_slotless_answer_is_still_placed_by_shape():
    """Transcripts written before slots existed must keep working.

    The coordinator replays whatever the task already holds, so an in-flight
    conversation across a restart can carry questions with no slot on them.
    """
    from panel_planner import DeterministicPlanner as _P
    scene = make_scene()
    history = [
        ConversationEvent(role="user", text="put soda_can"),
        ConversationEvent(role="reachy", text="Where should it go?",
                          question_id="q", choices=["r2c2"]),   # no slot
        ConversationEvent(role="user", text="r2c2"),
    ]
    out = _P(lambda: scene)(PlannerRequest(text="r2c2", history=history))
    assert out.kind == "proposal"
    assert out.proposal.destination == "cell:r2c2"


# ---------------------------------------------------------------------------
# Proposal contents
# ---------------------------------------------------------------------------

def test_proposal_labels_its_semantic_source_and_scene():
    out = plan(make_scene(), "put soda_can on r2c2")
    p = out.proposal
    assert p.semantic_source == "scene_data"      # not "camera perception"
    assert p.scene_name == "TestScene"
    assert p.requires_confirmation is True
    assert p.task_type == "pick_place"
    assert "grid cell r2c2" in p.summary


def test_proposal_without_live_poses_says_the_position_is_unverified():
    out = plan(make_scene(live=False), "put soda_can on r2c2")
    assert "has not been verified" in out.proposal.brief_reason


def test_a_scene_error_is_reported_not_swallowed():
    out = DeterministicPlanner(lambda: SceneView(error="bad YAML"))(
        PlannerRequest(text="put soda_can on r2c2", history=[])
    )
    assert out.kind == "unsupported"
    assert "bad YAML" in out.message


# ===========================================================================
# Against the real scene file
# ===========================================================================

def _real_pool_scene():
    from placement import cells_from_scene, pool_ids
    from scene_io import load_scene
    path = os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml")
    doc = load_scene(path)
    return scene_view_from_doc(doc, cells_from_scene(doc), placeable=pool_ids(doc))


def test_real_pool_scene_tags_soda_can_recyclable_and_foam_block_not():
    scene = _real_pool_scene()
    assert scene.objects["soda_can"].is_recyclable
    assert not scene.objects["foam_block"].is_recyclable
    assert scene.objects["foam_block"].is_non_recyclable
    assert [o.object_id for o in scene.recyclables()] == ["soda_can"]


def test_real_pool_scene_has_no_bin_or_tray():
    scene = _real_pool_scene()
    assert scene.destinations == {}
    assert scene.has_destination is False


def test_real_pool_scene_the_worked_example_explains_the_missing_bin():
    out = plan(_real_pool_scene(), "Put the recycle item in the bin.")
    assert out.kind == "clarification"
    assert "no bin or tray configured" in out.message
    assert out.choices          # only cells the arm can actually reach


def test_real_pool_scene_keeps_the_measured_unreachable_cells_out():
    scene = _real_pool_scene()
    reachable = scene.reachable_cells()
    # FWDCenterLabMCC measured r3c1 and r3c2 out of the right arm's reach.
    assert "r3c1" not in reachable
    assert "r3c2" not in reachable
    assert "r2c2" in reachable


def test_real_pool_scene_nothing_is_known_to_be_on_the_board():
    scene = _real_pool_scene()
    # Stage 1 has no live poses, so candidacy is unknown — not False.
    assert all(o.on_board is None for o in scene.objects.values())
    assert scene.tabletop_recyclables() == []


# ---------------------------------------------------------------------------
# Pointing at an object (#57)
# ---------------------------------------------------------------------------

def test_pointing_at_an_object_resolves_the_words_to_the_id():
    """"the soda can" and `soda_can` are the same object to a reader and two
    different strings to a lookup.  The resolution is written back onto the
    proposal, because everything downstream — the evidence, the card, the
    motion job — indexes the scene by id."""
    scene = make_scene()
    scene.objects["soda_can"].on_board = True
    out = plan(scene, "point to the soda can")
    assert out.kind == "proposal", out.message
    assert out.proposal.task_type == "point_object"
    assert out.proposal.object_id == "soda_can"
    # Pointing has no pick target and no destination, and writing a
    # placeholder into either would put a value meaning ABSENT into fields
    # whose readers all assume PRESENT.
    assert out.proposal.target_id is None
    assert out.proposal.destination is None


def test_pointing_at_an_object_in_the_pool_is_explained_not_attempted():
    """The brief is explicit: a can in the off-board pool should produce an
    explanation, not a tabletop reach aimed off the tabletop."""
    scene = make_scene()
    scene.objects["soda_can"].on_board = False
    out = plan(scene, "point to the soda_can")
    assert out.kind == "unsupported"
    assert "not on the board" in out.message
    assert "Place it on a cell first" in out.message


def test_pointing_at_an_unknown_object_asks_rather_than_guessing():
    out = plan(make_scene(), "point to the banana")
    assert out.kind == "clarification"
    assert "do not know an object" in out.message
    assert set(out.choices) == {"soda_can", "foam_block", "red_cube"}


def test_a_cell_is_still_read_as_a_cell_and_not_as_an_object():
    """`point_object`'s pattern deliberately accepts anything after "point
    to", so registry order is what keeps "point to r2c2" a cell.  A regression
    here would silently look for an object called "r2c2"."""
    out = plan(make_scene(), "point to r2c2")
    assert out.kind == "proposal", out.message
    assert out.proposal.task_type == "point_cell"
    assert out.proposal.cell == "r2c2"
    assert out.proposal.object_id is None


def test_pointing_at_an_object_survives_an_occupied_cell():
    """Pointing is a hover.  Pick-and-place's "the destination must be empty"
    rule refuses exactly the most natural request in a populated scene."""
    scene = make_scene(occupant=("r2c2", "foam_block"))
    scene.objects["foam_block"].on_board = True
    out = plan(scene, "point at the foam_block")
    assert out.kind == "proposal", out.message
    assert out.proposal.object_id == "foam_block"
