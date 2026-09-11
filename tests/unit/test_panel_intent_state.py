"""Issue #87: the request survives the transcript that carried it.

The real parser wired to the real coordinator, which is the only combination
where the bug appears: `DeterministicPlanner` alone is handed a history by its
caller, and `TaskCoordinator` alone is given a stub planner that never reads
one.  The failure needed both — a planner that rebuilt the command from
`Task.events`, and a coordinator that trims `Task.events` from the front at
`MAX_EVENTS`.

Offline and stdlib-only.  Sets up its own sys.path rather than inheriting one
from another test module, so it means the same thing run alone as in the suite.
"""

import os
import sys
import time

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../web"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

from panel_planner import DeterministicPlanner  # noqa: E402
from panel_scene import scene_view_from_doc  # noqa: E402
from tasks import MAX_EVENTS, TaskCoordinator, TaskState  # noqa: E402


class _Cell:
    def __init__(self, reachable=True, distance=0.4):
        self.reachable = reachable
        self.shoulder_distance_m = distance


def make_scene(unreachable=("r3c1", "r3c2")):
    doc = {
        "name": "TestScene",
        "objects": [
            {"id": "soda_can", "semantic_class": "can",
             "tags": ["pool", "manipulable", "pickable", "recyclable"]},
            {"id": "red_cube", "semantic_class": "cube",
             "tags": ["pool", "manipulable", "pickable", "cube"]},
        ],
    }
    cells = {f"r{r}c{c}": _Cell(reachable=f"r{r}c{c}" not in unreachable)
             for r in (1, 2, 3) for c in (1, 2, 3)}
    return scene_view_from_doc(doc, cells, placeable=["soda_can", "red_cube"])


@pytest.fixture
def coord():
    scene = make_scene()
    planner = DeterministicPlanner(lambda: scene)
    c = TaskCoordinator(planner, aside=planner.aside)
    try:
        yield c
    finally:
        c.shutdown()


def settle(coord, task_id, session="s1", timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = coord.get(session, task_id)
        if task.state is not TaskState.planning:
            return task
        time.sleep(0.01)
    raise AssertionError("task stayed in planning")


def answer(coord, task, text, session="s1"):
    coord.reply(session, task.task_id, task.question_id, text)
    return settle(coord, task.task_id, session)


# ---------------------------------------------------------------------------
# The bug
# ---------------------------------------------------------------------------

def test_a_clarification_longer_than_the_transcript_still_knows_the_command(coord):
    """Answer wrongly for longer than the transcript is deep, then answer well.

    Each wrong answer costs two events — the question and the answer — so
    thirty rounds is well past `MAX_EVENTS` and the opening "point to a cell"
    has been trimmed off the front long before the end.  The planner used to
    take the first user turn still present, which by then is "r9c9", and reply
    that it did not understand it.
    """
    task = settle(coord, coord.submit("s1", "point to a cell").task_id)
    assert task.state is TaskState.needs_clarification

    for _ in range(30):
        task = answer(coord, task, "r9c9")
        assert task.state is TaskState.needs_clarification, task.detail

    assert len(task.events) == MAX_EVENTS          # the command is long gone
    assert all("point to a cell" not in e.text for e in task.events)

    task = answer(coord, task, "r2c2")
    assert task.state is TaskState.awaiting_confirmation, task.detail
    assert task.proposal.task_type == "point_cell"
    assert task.proposal.cell == "r2c2"


def test_trimming_the_transcript_does_not_change_what_is_planned(coord):
    """The same conversation, short enough not to trim, plans the same thing."""
    task = settle(coord, coord.submit("s1", "point to a cell").task_id)
    task = answer(coord, task, "r2c2")
    assert len(task.events) < MAX_EVENTS
    assert task.proposal.task_type == "point_cell"
    assert task.proposal.cell == "r2c2"


def test_the_intent_records_what_filled_which_slot(coord):
    task = settle(coord, coord.submit("s1", "point to a cell").task_id)
    assert task.intent.command == "point to a cell"
    assert task.intent.open_slot == "which_cell"
    assert task.intent.filled_slots() == {}

    task = answer(coord, task, "r2c2")
    assert task.intent.command == "point to a cell"
    assert task.intent.kind == "point_cell"
    assert task.intent.filled_slots() == {"which_cell": "r2c2"}
    assert task.intent.open_slot == ""


def test_a_rejected_answer_is_replaced_rather_than_accumulated(coord):
    """Re-asking a slot means the later answer wins, not that both are held."""
    task = settle(coord, coord.submit("s1", "point to a cell").task_id)
    task = answer(coord, task, "r9c9")
    task = answer(coord, task, "r1c1")
    assert task.intent.filled_slots() == {"which_cell": "r1c1"}
    assert task.proposal.cell == "r1c1"


# ---------------------------------------------------------------------------
# A new command is not an answer
# ---------------------------------------------------------------------------

def test_a_new_command_mid_clarification_drops_the_old_slots(coord):
    task = settle(coord, coord.submit("s1", "point to a cell").task_id)
    task = answer(coord, task, "r1c1")
    assert task.intent.filled_slots() == {"which_cell": "r1c1"}

    # Not a reply — the task is awaiting confirmation now, so the change of
    # mind arrives the way the operator makes one: cancel, then say the new
    # thing.  The point is that nothing from the first request survives.
    coord.cancel("s1", task.task_id)
    task = settle(coord, coord.submit("s1", "wave").task_id)
    assert task.proposal.task_type == "wave"
    assert task.intent.command == "wave"
    assert task.intent.filled_slots() == {}
    assert task.proposal.cell is None


def test_a_command_typed_into_an_open_question_replaces_the_request(coord):
    """"Which cell?" answered with "wave" is a change of mind, not a cell."""
    task = settle(coord, coord.submit("s1", "point to a cell").task_id)
    assert task.state is TaskState.needs_clarification

    task = answer(coord, task, "wave")
    assert task.state is TaskState.awaiting_confirmation, task.detail
    assert task.proposal.task_type == "wave"
    assert task.intent.command == "wave"
    assert task.intent.filled_slots() == {}


# ---------------------------------------------------------------------------
# A greeting is not a reset
# ---------------------------------------------------------------------------

def test_a_greeting_answered_into_an_open_question_keeps_the_question(coord):
    """THE PATH THE BROWSER ACTUALLY TAKES.

    `camera_server.py` POSTs anything typed while a task is awaiting an answer
    to /reply, not /tasks — and `reply()` has no aside hook.  So a greeting
    arrives here as the answer, and the earlier version of this file only ever
    tested `submit()`, which is the path the client never uses in this state.
    """
    task = settle(coord, coord.submit("s1", "point to a cell").task_id)
    assert task.state is TaskState.needs_clarification
    asked = [e.text for e in task.events if e.question_id]

    task = answer(coord, task, "Hello")

    assert task.state is TaskState.needs_clarification, task.detail
    assert task.intent.command == "point to a cell"
    assert task.intent.open_slot == "which_cell"
    # Greeted, and then asked the same thing again — a new question id,
    # because the browser answers the latest one.
    latest = [e for e in task.events if e.question_id][-1]
    assert latest.text.startswith("Hello. ")
    assert latest.text.endswith(asked[-1])
    assert latest.slot == "which_cell"

    # And the conversation carries on where it left off.
    task = answer(coord, task, "r2c2")
    assert task.proposal.task_type == "point_cell"
    assert task.proposal.cell == "r2c2"


def test_a_greeting_does_not_fill_the_slot_it_was_typed_into(coord):
    """It is not an answer, so it must not be remembered as one."""
    task = settle(coord, coord.submit("s1", "point to a cell").task_id)
    task = answer(coord, task, "Hello")
    assert task.intent.filled_slots() == {}
    assert task.intent.answers == []


def test_a_real_request_typed_into_an_open_question_is_still_a_change_of_mind(coord):
    """The greeting exception is exactly that.  "Wave" still replaces."""
    task = settle(coord, coord.submit("s1", "point to a cell").task_id)
    task = answer(coord, task, "wave")
    assert task.state is TaskState.awaiting_confirmation, task.detail
    assert task.intent.command == "wave"


def test_a_greeting_submitted_as_its_own_message_still_stands_aside(coord):
    task = settle(coord, coord.submit("s1", "point to a cell").task_id)
    question = task.question_id
    assert task.state is TaskState.needs_clarification

    aside = coord.submit("s1", "Hello")
    assert aside.state is TaskState.completed
    assert aside.task_id != task.task_id

    still = coord.get("s1", task.task_id)
    assert still.state is TaskState.needs_clarification
    assert still.question_id == question
    assert still.intent.command == "point to a cell"
    assert still.intent.open_slot == "which_cell"

    # And the question is still answerable afterwards.
    done = answer(coord, still, "r2c2")
    assert done.proposal.task_type == "point_cell"
    assert done.proposal.cell == "r2c2"


# ---------------------------------------------------------------------------
# What the intent is not
# ---------------------------------------------------------------------------

def test_the_intent_records_the_version_of_the_plan_it_produced(coord):
    """`plan_version` is stamped by the coordinator after the planner returns,
    so the copy the planner made of it was always the constructor's 0."""
    task = settle(coord, coord.submit("s1", "point to r2c2").task_id)
    assert task.proposal.plan_version > 0
    assert task.intent.plan_version == task.proposal.plan_version


def test_the_answers_list_is_bounded_by_replacement(coord):
    """A client answering the same question forever resets the task deadline
    every time, so nothing sweeps it.  The list it feeds must not grow."""
    task = settle(coord, coord.submit("s1", "point to a cell").task_id)
    for i in range(50):
        task = answer(coord, task, "r9c9")
    assert task.state is TaskState.needs_clarification
    assert len(task.intent.answers) == 1
    assert task.intent.filled_slots() == {"which_cell": "r9c9"}


def test_a_confirmation_is_not_retained_as_standing_authorization(coord):
    """One confirmation, one plan.  The intent state does not widen it."""
    from tasks import TaskError

    task = settle(coord, coord.submit("s1", "point to r2c2").task_id)
    assert task.state is TaskState.awaiting_confirmation
    coord.confirm("s1", task.task_id, task.proposal.plan_id,
                  task.proposal.plan_version)
    with pytest.raises(TaskError) as exc:
        coord.confirm("s1", task.task_id, task.proposal.plan_id,
                      task.proposal.plan_version)
    assert exc.value.code == "already_confirmed"


def test_the_planner_still_works_for_a_caller_that_keeps_no_state():
    """The direct contract — text plus history, no intent — is unchanged.

    The transcript reconstruction stays for conversations stored before this
    existed.  It carries the trimming bug with it, which is why the coordinator
    no longer uses it.
    """
    from tasks import ConversationEvent, PlannerRequest

    planner = DeterministicPlanner(make_scene)
    history = [ConversationEvent(role="user", text="point to a cell")]
    out = planner(PlannerRequest(text="point to a cell", history=history))
    assert out.kind == "clarification"
    assert out.slot == "which_cell"

    history.append(ConversationEvent(role="reachy", text=out.message,
                                     question_id="q", slot=out.slot))
    history.append(ConversationEvent(role="user", text="r2c2"))
    out = planner(PlannerRequest(text="r2c2", history=history))
    assert out.kind == "proposal"
    assert out.proposal.cell == "r2c2"


def test_an_abandoned_answer_no_longer_refuses_the_request_that_replaced_it():
    """A behavioural change to the no-state path, asserted rather than assumed.

    The joint-angle guard reads the command and its answers.  The new-command
    rule now runs first, so answers given to a request the operator has since
    replaced are dropped BEFORE the guard sees them — and a fresh "wave" is no
    longer refused because an abandoned answer three turns ago said
    "r_shoulder_pitch 30".  Nothing is honoured either way; the refusal moved.

    Joint angles in the turn the operator just typed are still refused, which
    is the case the guard exists for.
    """
    from tasks import ConversationEvent, PlannerRequest

    planner = DeterministicPlanner(make_scene)
    history = [
        ConversationEvent(role="user", text="point to a cell"),
        ConversationEvent(role="reachy", text="Which cell?", question_id="q1",
                          slot="which_cell"),
        ConversationEvent(role="user", text="r_shoulder_pitch 30"),
        ConversationEvent(role="reachy", text="Which cell?", question_id="q2",
                          slot="which_cell"),
        ConversationEvent(role="user", text="wave"),
    ]
    out = planner(PlannerRequest(text="wave", history=history))
    assert out.kind == "proposal"
    assert out.proposal.task_type == "wave"

    history[-1] = ConversationEvent(role="user", text="set r_wrist_pitch to 0.9 rad")
    out = planner(PlannerRequest(text="set r_wrist_pitch to 0.9 rad",
                                 history=history))
    assert out.kind == "unsupported"
    assert "joint angles" in out.message
