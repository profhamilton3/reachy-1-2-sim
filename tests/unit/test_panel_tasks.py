"""Issue #49: task coordinator lifecycle for the Talk to Reachy panel.

Offline and stdlib-only, like the module under test.  The planner is a stub
here so these tests exercise the state machine rather than the parser; the
parser has its own file.
"""

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../web"))

from tasks import (  # noqa: E402
    MAX_TEXT_CHARS,
    Capabilities,
    PlannerOutcome,
    Proposal,
    TaskCoordinator,
    TaskError,
    TaskState,
)


# --------------------------------------------------------------------------
# Stub planners
# --------------------------------------------------------------------------

def _proposal(summary="move soda_can to grid cell r2c2"):
    return Proposal(
        plan_id="plan-1", plan_version=0, task_type="pick_place",
        target_id="soda_can", destination="cell:r2c2",
        brief_reason="tagged recyclable", summary=summary,
    )


def always_propose(_request):
    return PlannerOutcome(kind="proposal", message="Proposed action.",
                          proposal=_proposal())


def always_clarify(_request):
    return PlannerOutcome(kind="clarification", message="Which one?",
                          choices=["soda_can", "red_cube"])


def always_unsupported(_request):
    return PlannerOutcome(kind="unsupported", message="I did not understand that.")


def always_raise(_request):
    raise RuntimeError("planner exploded")


class BlockingPlanner:
    """Planner that waits until released, to test timeout and cancellation."""

    def __init__(self):
        self.gate = threading.Event()
        self.entered = threading.Event()

    def __call__(self, _request):
        self.entered.set()
        self.gate.wait(timeout=5)
        return PlannerOutcome(kind="proposal", proposal=_proposal())


def settle(coord, session, task_id, timeout=3.0):
    """Wait for planning to finish; return the task."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = coord.get(session, task_id)
        if task.state is not TaskState.planning:
            return task
        time.sleep(0.01)
    raise AssertionError("task stayed in planning")


@pytest.fixture
def coord():
    c = TaskCoordinator(always_propose)
    yield c
    c.shutdown()


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def test_empty_input_creates_no_task(coord):
    for text in ("", "   ", "\n\t "):
        with pytest.raises(TaskError) as exc:
            coord.submit("s1", text)
        assert exc.value.status == 400
    # and nothing was recorded against the session
    assert coord.submit("s1", "put soda_can on r2c2").state is TaskState.planning


def test_oversized_input_is_refused_readably(coord):
    with pytest.raises(TaskError) as exc:
        coord.submit("s1", "x" * (MAX_TEXT_CHARS + 1))
    assert exc.value.status == 413
    assert str(MAX_TEXT_CHARS) in exc.value.message


def test_non_string_text_is_refused(coord):
    with pytest.raises(TaskError):
        coord.submit("s1", {"not": "text"})


def test_control_characters_are_stripped_but_newlines_survive(coord):
    task = coord.submit("s1", "put soda_can\x07 on\nr2c2")
    assert task.events[0].text == "put soda_can on\nr2c2"


# --------------------------------------------------------------------------
# Ownership and sessions
# --------------------------------------------------------------------------

def test_another_session_cannot_read_or_confirm(coord):
    task = coord.submit("owner", "put soda_can on r2c2")
    settle(coord, "owner", task.task_id)
    for call in (
        lambda: coord.get("intruder", task.task_id),
        lambda: coord.cancel("intruder", task.task_id),
        lambda: coord.confirm("intruder", task.task_id, "plan-1", 2),
    ):
        with pytest.raises(TaskError) as exc:
            call()
        # 404 not 403: a different status would confirm the task exists.
        assert exc.value.status == 404


def test_one_active_task_per_session(coord):
    coord.submit("s1", "put soda_can on r2c2")
    with pytest.raises(TaskError) as exc:
        coord.submit("s1", "put red_cube on r1c1")
    assert exc.value.status == 409
    # a different session is unaffected
    assert coord.submit("s2", "put red_cube on r1c1")


def test_duplicate_client_request_id_returns_the_same_task(coord):
    a = coord.submit("s1", "put soda_can on r2c2", client_request_id="abc")
    b = coord.submit("s1", "put soda_can on r2c2", client_request_id="abc")
    assert a.task_id == b.task_id


def test_cancel_frees_the_session_for_a_new_task(coord):
    task = coord.submit("s1", "put soda_can on r2c2")
    coord.cancel("s1", task.task_id)
    assert coord.submit("s1", "put red_cube on r1c1").task_id != task.task_id


# --------------------------------------------------------------------------
# Clarification
# --------------------------------------------------------------------------

def test_clarification_question_must_be_answered_by_id():
    coord = TaskCoordinator(always_clarify)
    try:
        task = coord.submit("s1", "put the recycle item in the bin")
        task = settle(coord, "s1", task.task_id)
        assert task.state is TaskState.needs_clarification
        assert task.question_id
        assert task.events[-1].choices == ["soda_can", "red_cube"]

        with pytest.raises(TaskError) as exc:
            coord.reply("s1", task.task_id, "some:other:question", "soda_can")
        assert exc.value.code == "stale_question"

        again = coord.reply("s1", task.task_id, task.question_id, "soda_can")
        assert again.state is TaskState.planning
    finally:
        coord.shutdown()


def test_reply_is_refused_when_nothing_was_asked(coord):
    task = coord.submit("s1", "put soda_can on r2c2")
    task = settle(coord, "s1", task.task_id)
    assert task.state is TaskState.awaiting_confirmation
    with pytest.raises(TaskError) as exc:
        coord.reply("s1", task.task_id, "anything", "soda_can")
    assert exc.value.code == "not_awaiting_reply"


def test_empty_reply_is_refused():
    coord = TaskCoordinator(always_clarify)
    try:
        task = settle(coord, "s1", coord.submit("s1", "put it there").task_id)
        with pytest.raises(TaskError):
            coord.reply("s1", task.task_id, task.question_id, "   ")
    finally:
        coord.shutdown()


# --------------------------------------------------------------------------
# Confirmation
# --------------------------------------------------------------------------

def test_confirm_in_planning_only_mode_performs_no_motion(coord):
    task = settle(coord, "s1", coord.submit("s1", "put soda_can on r2c2").task_id)
    p = task.proposal
    assert p is not None and p.execution_mode == "planning_only"

    done = coord.confirm("s1", task.task_id, p.plan_id, p.plan_version)
    assert done.state is TaskState.confirmed_no_motion
    assert done.detail == "Plan confirmed; no movement performed."


def test_confirmation_is_consumed_exactly_once(coord):
    task = settle(coord, "s1", coord.submit("s1", "put soda_can on r2c2").task_id)
    p = task.proposal
    coord.confirm("s1", task.task_id, p.plan_id, p.plan_version)
    with pytest.raises(TaskError) as exc:
        coord.confirm("s1", task.task_id, p.plan_id, p.plan_version)
    assert exc.value.code == "already_confirmed"


def test_stale_plan_version_cannot_confirm(coord):
    task = settle(coord, "s1", coord.submit("s1", "put soda_can on r2c2").task_id)
    p = task.proposal
    with pytest.raises(TaskError) as exc:
        coord.confirm("s1", task.task_id, p.plan_id, p.plan_version - 1)
    assert exc.value.code == "stale_plan_version"


def test_wrong_plan_id_cannot_confirm(coord):
    task = settle(coord, "s1", coord.submit("s1", "put soda_can on r2c2").task_id)
    p = task.proposal
    with pytest.raises(TaskError) as exc:
        coord.confirm("s1", task.task_id, "some-other-plan", p.plan_version)
    assert exc.value.code == "plan_mismatch"


def test_cancelled_task_cannot_be_confirmed(coord):
    task = settle(coord, "s1", coord.submit("s1", "put soda_can on r2c2").task_id)
    p = task.proposal
    coord.cancel("s1", task.task_id)
    with pytest.raises(TaskError) as exc:
        coord.confirm("s1", task.task_id, p.plan_id, p.plan_version)
    assert exc.value.code == "not_awaiting_confirmation"


def test_planner_confirmation_flag_cannot_bypass_policy(coord):
    """A planner asking for no confirmation still does not get motion."""
    def no_confirm(_request):
        p = _proposal()
        p.requires_confirmation = False
        return PlannerOutcome(kind="proposal", proposal=p)

    c = TaskCoordinator(no_confirm)
    try:
        task = settle(c, "s1", c.submit("s1", "put soda_can on r2c2").task_id)
        # The server parks it awaiting confirmation regardless of the flag.
        assert task.state is TaskState.awaiting_confirmation
        done = c.confirm("s1", task.task_id, task.proposal.plan_id,
                         task.proposal.plan_version)
        assert done.state is TaskState.confirmed_no_motion
    finally:
        c.shutdown()


def test_capabilities_drive_execution_mode():
    caps = Capabilities()
    assert caps.live_execution is False
    assert caps.execution_mode == "planning_only"


# --------------------------------------------------------------------------
# Failure, timeout, and late results
# --------------------------------------------------------------------------

def test_unsupported_input_fails_with_an_explanation():
    coord = TaskCoordinator(always_unsupported)
    try:
        task = settle(coord, "s1", coord.submit("s1", "make me a sandwich").task_id)
        assert task.state is TaskState.failed
        assert "did not understand" in task.detail
    finally:
        coord.shutdown()


def test_a_raising_planner_fails_the_task_not_the_pool():
    coord = TaskCoordinator(always_raise)
    try:
        task = settle(coord, "s1", coord.submit("s1", "put soda_can on r2c2").task_id)
        assert task.state is TaskState.failed
        assert "RuntimeError" in task.detail
        # the pool still works for the next task
        assert coord.submit("s2", "put red_cube on r1c1")
    finally:
        coord.shutdown()


def test_planning_timeout_fails_the_task():
    planner = BlockingPlanner()
    coord = TaskCoordinator(planner, planning_timeout_s=0.05)
    try:
        task = coord.submit("s1", "put soda_can on r2c2")
        assert planner.entered.wait(timeout=2)
        time.sleep(0.1)
        task = coord.get("s1", task.task_id)
        assert task.state is TaskState.failed
        assert "did not answer in time" in task.detail
    finally:
        planner.gate.set()
        coord.shutdown()


def test_a_late_planner_result_cannot_revive_a_cancelled_task():
    planner = BlockingPlanner()
    coord = TaskCoordinator(planner)
    try:
        task = coord.submit("s1", "put soda_can on r2c2")
        assert planner.entered.wait(timeout=2)
        coord.cancel("s1", task.task_id)
        planner.gate.set()
        time.sleep(0.15)
        assert coord.get("s1", task.task_id).state is TaskState.cancelled
    finally:
        coord.shutdown()


def test_a_late_planner_result_cannot_overwrite_a_timed_out_task():
    planner = BlockingPlanner()
    coord = TaskCoordinator(planner, planning_timeout_s=0.05)
    try:
        task = coord.submit("s1", "put soda_can on r2c2")
        assert planner.entered.wait(timeout=2)
        time.sleep(0.1)
        assert coord.get("s1", task.task_id).state is TaskState.failed
        planner.gate.set()
        time.sleep(0.15)
        assert coord.get("s1", task.task_id).state is TaskState.failed
    finally:
        coord.shutdown()


def test_idle_tasks_expire():
    coord = TaskCoordinator(always_clarify, task_ttl_s=0.05)
    try:
        task = settle(coord, "s1", coord.submit("s1", "put it there").task_id)
        assert task.state is TaskState.needs_clarification
        time.sleep(0.1)
        assert coord.get("s1", task.task_id).state is TaskState.expired
    finally:
        coord.shutdown()


def test_cancelling_a_terminal_task_is_a_no_op(coord):
    task = settle(coord, "s1", coord.submit("s1", "put soda_can on r2c2").task_id)
    p = task.proposal
    coord.confirm("s1", task.task_id, p.plan_id, p.plan_version)
    assert coord.cancel("s1", task.task_id).state is TaskState.confirmed_no_motion


def test_unknown_task_id_is_not_found(coord):
    with pytest.raises(TaskError) as exc:
        coord.get("s1", "deadbeef")
    assert exc.value.status == 404


def test_transcript_is_bounded():
    coord = TaskCoordinator(always_clarify)
    try:
        task = settle(coord, "s1", coord.submit("s1", "put it there").task_id)
        for _ in range(60):
            t = coord.get("s1", task.task_id)
            if t.state is not TaskState.needs_clarification:
                break
            coord.reply("s1", task.task_id, t.question_id, "soda_can")
            settle(coord, "s1", task.task_id)
        from tasks import MAX_EVENTS
        assert len(coord.get("s1", task.task_id).events) <= MAX_EVENTS
    finally:
        coord.shutdown()


def test_session_ids_are_unguessable_and_distinct():
    ids = {TaskCoordinator.new_session_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i) >= 24 for i in ids)
