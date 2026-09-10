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
    assert out.kind == "proposal"
    assert out.proposal.task_type == "stow_arm"
    assert out.proposal.route == "STOW_ROUTE"
    assert "object called" not in (out.message or "")


def test_an_ability_proposal_carries_no_target_and_no_destination():
    """A wave has neither, and no placeholder stands in for one."""
    out = plan(make_scene(), "wave")
    assert out.kind == "proposal"
    p = out.proposal
    assert p.task_type == "wave"
    assert p.target_id is None
    assert p.destination is None
    assert p.cell is None and p.object_id is None
    assert p.arm == "right"
    assert p.route == "WAVE"
    assert p.expected_start_posture == "present"


def test_pointing_at_a_cell_asks_which_one_and_offers_reachable_cells():
    out = plan(make_scene(), "point to a cell")
    assert out.kind == "clarification"
    assert out.slot == abilities.SLOT_CELL
    assert "r2c2" in out.choices
    assert "r3c1" not in out.choices        # measured out of reach


def test_the_cell_answer_fills_the_cell_slot():
    out = plan(make_scene(), "point to a cell", answers=["r2c2"])
    assert out.kind == "proposal"
    assert out.proposal.task_type == "point_cell"
    assert out.proposal.cell == "r2c2"
    assert out.proposal.destination is None      # pointing places nothing


def test_bare_rest_asks_which_posture_then_resolves():
    first = plan(make_scene(), "rest your arm")
    assert first.kind == "clarification"
    assert first.slot == abilities.SLOT_POSTURE
    assert "eleven waypoints apart" in first.message

    table = plan(make_scene(), "rest your arm", answers=["on the table"])
    assert table.proposal.task_type == "rest_forearm"
    assert table.proposal.route == "PLACE_ROUTE"

    pocket = plan(make_scene(), "rest your arm", answers=["in the rail pocket"])
    assert pocket.proposal.task_type == "stow_arm"
    assert pocket.proposal.route == "STOW_ROUTE"


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


def test_the_scene_read_follows_the_registry_not_a_blanket_rule():
    """Each ability declares whether planning it needs the board.

    A wave sweeps the space above the board — `PRESENT` was chosen for
    whole-arm clearance against the objects on it — so an unreadable scene is
    a real obstacle and saying so is right.  A greeting is not, and must not
    inherit the simulator's health.
    """
    from panel_scene import SceneView

    planner = DeterministicPlanner(lambda: SceneView(error="bad YAML"))
    waved = planner(PlannerRequest(text="wave", history=[
        ConversationEvent(role="user", text="wave")]))
    assert waved.kind == "unsupported"
    assert "bad YAML" in waved.message

    greeted = planner(PlannerRequest(text="hello", history=[
        ConversationEvent(role="user", text="hello")]))
    assert greeted.kind == "reply"
    assert "bad YAML" not in greeted.message


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


# ---------------------------------------------------------------------------
# Conversation-only outcomes (issue #62)
#
# "Hello" could not work before, for two independent reasons: there was no
# successful non-action outcome — `_apply_locked` sent everything that was not
# a clarification or a proposal to the failure branch — and the planner read
# the scene before it parsed anything, so a greeting answered with a scene
# error when the simulator was down.
# ---------------------------------------------------------------------------

def test_a_greeting_is_a_reply_not_a_failure():
    out = plan(make_scene(), "hello")
    assert out.kind == "reply"
    assert out.proposal is None


def test_a_greeting_works_while_the_scene_is_unreadable():
    """The panel is at its least useful exactly when someone is checking
    whether anything is alive."""
    from panel_scene import SceneView

    out = DeterministicPlanner(lambda: SceneView(error="connection refused"))(
        PlannerRequest(text="hi", history=[ConversationEvent(role="user", text="hi")])
    )
    assert out.kind == "reply"
    assert "connection refused" not in out.message


def test_a_greeting_completes_the_task_with_no_confirmation():
    from tasks import TaskCoordinator, TaskState

    planner = DeterministicPlanner(make_scene)
    coord = TaskCoordinator(planner, aside=planner.aside)
    try:
        task = coord.submit("s1", "good morning")
        assert task.state is TaskState.completed
        assert task.proposal is None
        assert task.question_id == ""
        assert task.events[-1].role == "reachy"
    finally:
        coord.shutdown()


def test_a_greeting_never_reaches_the_executor():
    from tasks import TaskCoordinator

    class Recorder:
        def __init__(self):
            self.asked = 0

        def available(self, proposal=None):
            self.asked += 1
            return True, ""

        def execute(self, proposal, **kw):        # pragma: no cover
            raise AssertionError("a greeting must not execute anything")

    rec = Recorder()
    planner = DeterministicPlanner(make_scene)
    coord = TaskCoordinator(planner, executor=rec, aside=planner.aside)
    try:
        coord.submit("s1", "hello")
        assert rec.asked == 0
    finally:
        coord.shutdown()


def _await_state(coord, session, task_id, state, timeout=3.0):
    import time
    from tasks import TaskState

    deadline = time.time() + timeout
    while time.time() < deadline:
        task = coord.get(session, task_id)
        if task.state is TaskState(state):
            return task
        time.sleep(0.01)
    raise AssertionError(f"task stayed in {coord.get(session, task_id).state}")


def test_a_greeting_during_a_pending_clarification_leaves_it_alone():
    """The active task keeps its state, its version and its open question.

    The obvious implementation runs the greeting through `submit()`, which
    queues onto the session's one active task — replacing the plan, or being
    refused as a second task.  Neither is an answer to "Hello".
    """
    from tasks import TaskCoordinator

    planner = DeterministicPlanner(make_scene)
    coord = TaskCoordinator(planner, aside=planner.aside)
    try:
        active = coord.submit("s1", "put soda_can")
        active = _await_state(coord, "s1", active.task_id, "needs_clarification")
        before = (active.state, active.version, active.question_id,
                  len(active.events))

        greeting = coord.submit("s1", "hello")
        assert greeting.task_id != active.task_id
        assert greeting.state.value == "completed"

        after = coord.get("s1", active.task_id)
        assert (after.state, after.version, after.question_id,
                len(after.events)) == before

        # And the pending question is still answerable afterwards.
        resumed = coord.reply("s1", after.task_id, after.question_id, "r2c2")
        assert resumed.task_id == active.task_id
    finally:
        coord.shutdown()


def test_a_greeting_is_not_refused_as_a_second_task():
    """`submit()` normally answers 409 while a task is active.  A greeting is
    not a second task, so it must not collide with that rule."""
    from tasks import TaskCoordinator

    planner = DeterministicPlanner(make_scene)
    coord = TaskCoordinator(planner, aside=planner.aside)
    try:
        active = coord.submit("s1", "put soda_can")
        _await_state(coord, "s1", active.task_id, "needs_clarification")
        greeting = coord.submit("s1", "hey reachy")
        assert greeting.state.value == "completed"
    finally:
        coord.shutdown()


def test_a_greeting_task_never_becomes_the_session_active_one():
    from tasks import TaskCoordinator

    planner = DeterministicPlanner(make_scene)
    coord = TaskCoordinator(planner, aside=planner.aside)
    try:
        greeting = coord.submit("s1", "hello")
        # The session slot is still free, so real work starts immediately
        # rather than being told to finish or cancel the greeting.
        real = coord.submit("s1", "put soda_can on r2c2")
        assert real.task_id != greeting.task_id
    finally:
        coord.shutdown()


def test_the_greeting_does_not_promise_abilities_the_panel_lacks():
    """An opening line implying it can wave is the same false claim as a
    proposal that cannot be executed, just earlier."""
    from panel_planner import GREETING

    lowered = GREETING.lower()
    for absent in ("wave", "point", "rest", "stow"):
        assert absent not in lowered


def test_an_ability_that_needs_motion_is_never_an_aside():
    planner = DeterministicPlanner(make_scene)
    assert planner.aside("hello") is not None
    for text in ("wave", "reset", "point to r2c2",
                 "place your forearm on the table"):
        assert planner.aside(text) is None


def test_a_greeting_during_execution_does_not_disturb_the_arm():
    """No cancel, no lease, and not a phase in the running task."""
    import time

    from tasks import (Capabilities, PlannerOutcome, Proposal, TaskCoordinator,
                       TaskState)

    proposal = Proposal(plan_id="p1", plan_version=0, task_type="pick_place",
                        target_id="soda_can", destination="cell:r2c2",
                        destination_kind="cell", brief_reason="because",
                        summary="move soda_can to r2c2")

    class SlowExecutor:
        def __init__(self):
            self.cancel_seen = False
            self.phases = []

        def available(self, proposal=None):
            return True, ""

        def execute(self, proposal, *, should_cancel=None, on_phase=None):
            deadline = time.time() + 0.5
            while time.time() < deadline:
                if should_cancel and should_cancel():
                    self.cancel_seen = True
                    break
                time.sleep(0.01)
            from panel_executor import ExecutionResult
            return ExecutionResult(status="completed", detail="done")

    executor = SlowExecutor()
    real_planner = DeterministicPlanner(make_scene)
    coord = TaskCoordinator(
        lambda _req: PlannerOutcome(kind="proposal", proposal=proposal),
        capabilities=Capabilities(),
        executor=executor,
        aside=real_planner.aside,
    )
    try:
        task = coord.submit("s1", "put soda_can on r2c2")
        task = _await_state(coord, "s1", task.task_id, "awaiting_confirmation")
        task = coord.confirm("s1", task.task_id, task.proposal.plan_id,
                             task.proposal.plan_version)
        assert task.state is TaskState.executing
        version_while_executing = task.version

        greeting = coord.submit("s1", "hello")
        assert greeting.state is TaskState.completed

        moving = coord.get("s1", task.task_id)
        assert moving.state is TaskState.executing
        assert moving.cancel_requested is False
        assert moving.version == version_while_executing

        _await_state(coord, "s1", task.task_id, "completed")
        assert executor.cancel_seen is False
    finally:
        coord.shutdown()


# ---------------------------------------------------------------------------
# Per-ability proposal arguments (issue #63)
# ---------------------------------------------------------------------------

def test_a_pick_place_proposal_cannot_be_built_without_its_two_fields():
    """The invariant is enforced where it is cheap, not in the executor with
    the lease held."""
    import pytest as _pytest
    from tasks import Proposal

    with _pytest.raises(ValueError):
        Proposal(plan_id="p", plan_version=0, task_type="pick_place")
    with _pytest.raises(ValueError):
        Proposal(plan_id="p", plan_version=0, task_type="pick_place",
                 target_id="soda_can")


def test_an_ability_proposal_needs_neither():
    from tasks import Proposal

    p = Proposal(plan_id="p", plan_version=0, task_type="wave")
    assert p.target_id is None and p.destination is None


@pytest.mark.parametrize("text,action,route", [
    ("wave", "wave", "WAVE"),
    ("store your arm", "stow_arm", "STOW_ROUTE"),
    ("place your forearm on the table", "rest_forearm", "PLACE_ROUTE"),
    ("point to r2c2", "point_cell", "POINT"),
    ("point to the soda can", "point_object", "POINT"),
])
def test_each_ability_carries_its_route_identity(text, action, route):
    out = plan(make_scene(), text)
    assert out.kind == "proposal", out.message
    assert out.proposal.task_type == action
    assert out.proposal.route == route
    assert out.proposal.route_version == 1


def test_the_reason_names_the_route_because_the_route_is_the_claim():
    """"Store your arm" means the rail-pocket corridor, not `P.go_home()` —
    which routes through `stow_from_side()` and is a different motion also
    fairly called going home."""
    out = plan(make_scene(), "store your arm")
    assert "STOW_ROUTE" in out.proposal.brief_reason
    assert "right arm" in out.proposal.brief_reason


def test_pointing_at_an_occupied_cell_is_a_proposal():
    """Pointing is a hover.  Pick-and-place's empty-destination rule would
    refuse the most natural request in a populated scene."""
    scene = make_scene()
    scene.cells["r2c2"].occupant = "foam_block"
    out = plan(scene, "point to r2c2")
    assert out.kind == "proposal"
    assert out.proposal.cell == "r2c2"


def test_placing_into_an_occupied_cell_is_still_refused():
    scene = make_scene()
    scene.cells["r2c2"].occupant = "foam_block"
    out = plan(scene, "put soda_can on r2c2")
    assert out.kind == "clarification"
    assert "already occupied" in out.message


def test_pointing_at_an_unreachable_cell_is_refused():
    out = plan(make_scene(), "point to r3c1")
    assert out.kind == "clarification"
    assert "out of the right arm's reach" in out.message


def test_pointing_at_an_object_in_the_pool_explains_rather_than_reaches():
    scene = make_scene()
    scene.objects["soda_can"].on_board = False
    out = plan(scene, "point to the soda can")
    assert out.kind == "unsupported"
    assert "not on the board" in out.message


# ---------------------------------------------------------------------------
# Revalidation branches by ability
# ---------------------------------------------------------------------------

def _revalidate(scene, proposal):
    from panel_planner import LiveProposalValidator
    return LiveProposalValidator(lambda: scene)(proposal)


def _live(scene):
    from panel_scene import apply_snapshot
    from panel_sim_link import SimSnapshot
    import time as _t
    apply_snapshot(scene, SimSnapshot(scene_revision="rev-1", objects={},
                                      received_at=_t.monotonic()))
    return scene


def test_a_wave_is_not_invalidated_by_the_objects_moving():
    scene = _live(make_scene())
    out = plan(scene, "wave")
    scene.cells["r2c2"].occupant = "soda_can"
    ok, why = _revalidate(scene, out.proposal)
    assert ok, why


def test_pointing_at_a_cell_is_not_invalidated_by_it_becoming_occupied():
    scene = _live(make_scene())
    out = plan(scene, "point to r2c2")
    scene.cells["r2c2"].occupant = "foam_block"
    ok, why = _revalidate(scene, out.proposal)
    assert ok, why


def test_pointing_at_a_cell_is_invalidated_when_the_cell_goes_away():
    scene = _live(make_scene())
    out = plan(scene, "point to r2c2")
    del scene.cells["r2c2"]
    ok, why = _revalidate(scene, out.proposal)
    assert not ok
    assert "no longer in this scene" in why


def test_pointing_at_an_object_is_invalidated_when_it_moves():
    """The hover was computed over where it WAS."""
    scene = _live(make_scene())
    scene.objects["soda_can"].position = (0.30, 0.00, 0.78)
    scene.objects["soda_can"].on_board = True
    out = plan(scene, "point to the soda can")
    assert out.kind == "proposal", out.message
    scene.objects["soda_can"].position = (0.30, 0.20, 0.78)
    ok, why = _revalidate(scene, out.proposal)
    assert not ok
    assert "has moved" in why


def test_pointing_at_an_object_survives_it_sitting_still():
    scene = _live(make_scene())
    scene.objects["soda_can"].position = (0.30, 0.00, 0.78)
    scene.objects["soda_can"].on_board = True
    out = plan(scene, "point to the soda can")
    ok, why = _revalidate(scene, out.proposal)
    assert ok, why


# ---------------------------------------------------------------------------
# A new command during a clarification is a new command
# ---------------------------------------------------------------------------

def test_a_new_command_during_a_clarification_replaces_the_task():
    out = plan(make_scene(), "point to a cell", answers=["store your arm"])
    assert out.kind == "proposal"
    assert out.proposal.task_type == "stow_arm"
    assert out.proposal.cell is None


def test_a_pick_and_place_command_during_a_clarification_replaces_it_too():
    out = plan(make_scene(), "point to a cell",
               answers=["put soda_can on r1c1"])
    assert out.kind == "proposal"
    assert out.proposal.task_type == "pick_place"
    assert out.proposal.destination == "cell:r1c1"


def test_a_refused_command_during_a_clarification_is_still_a_command():
    """"Don't wave" must reach the operator as a refusal, not be filed as a
    cell name."""
    out = plan(make_scene(), "point to a cell", answers=["don't wave"])
    assert out.kind == "unsupported"
    assert "telling me not to" in out.message


def test_a_bare_answer_is_still_an_answer():
    """The two do not overlap: answers are bare nouns, commands carry verbs."""
    out = plan(make_scene(), "point to a cell", answers=["r2c2"])
    assert out.proposal.cell == "r2c2"
    out = plan(make_scene(), "put soda_can", answers=["r2c2"])
    assert out.proposal.destination == "cell:r2c2"


def test_a_retried_greeting_does_not_appear_twice():
    """Not being the session's active task and being idempotent are unrelated
    properties, and the aside path needs both — a double click must not put
    two greetings in the transcript."""
    from tasks import TaskCoordinator

    planner = DeterministicPlanner(make_scene)
    coord = TaskCoordinator(planner, aside=planner.aside)
    try:
        first = coord.submit("s1", "hello", "req-1")
        again = coord.submit("s1", "hello", "req-1")
        assert again.task_id == first.task_id
        different = coord.submit("s1", "hello", "req-2")
        assert different.task_id != first.task_id
    finally:
        coord.shutdown()
