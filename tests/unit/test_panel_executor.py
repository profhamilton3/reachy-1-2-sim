"""Issue #51: the execution adapter and the executing half of the lifecycle.

Nothing here moves a robot.  The executor is exercised through a stub for the
lifecycle tests, and the real `SimulatorExecutor` is exercised only as far as
its refusals — which is the part worth pinning down, because every one of them
is a case where the panel must decline rather than guess.
"""

import os
import sys
import time
import types

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../web"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

from panel_executor import (  # noqa: E402
    ExecutionResult,
    NullExecutor,
    SimulatorExecutor,
    build_executor,
)
from panel_scene import scene_view_from_doc  # noqa: E402
from panel_sim_link import SimSnapshot  # noqa: E402
from tasks import (  # noqa: E402
    Capabilities,
    PlannerOutcome,
    Proposal,
    TaskCoordinator,
    TaskState,
)


# ---------------------------------------------------------------------------
# A scene to plan against.  Built here rather than imported from a sibling test
# module: this repo has no conftest.py, so a cross-test import only resolves
# when the whole suite runs and leaves this file broken in isolation.
# ---------------------------------------------------------------------------

class _Cell:
    def __init__(self, name, x, y, reachable=True):
        self.name = name
        self.x = x
        self.y = y
        self.top_z = 0.80
        self.half_extent = 0.06
        self.reachable = reachable
        self.shoulder_distance_m = 0.4


def make_scene():
    cells = {}
    for r in (1, 2, 3):
        for c in (1, 2, 3):
            name = f"r{r}c{c}"
            cells[name] = _Cell(name, 0.20 + 0.15 * (r - 1), 0.15 * (c - 2),
                                reachable=name not in ("r3c1", "r3c2"))
    doc = {
        "name": "TestScene",
        "objects": [
            {"id": "soda_can", "semantic_class": "can",
             "tags": ["manipulable", "pickable", "recyclable"]},
            {"id": "red_cube", "semantic_class": "cube",
             "tags": ["manipulable", "pickable"]},
        ],
    }
    return scene_view_from_doc(doc, cells, placeable=["soda_can", "red_cube"])


def on_cell(scene, name, z_offset=0.03):
    c = scene.cells[name]
    return (c.x, c.y, c.top_z + z_offset)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubLink:
    def __init__(self, *, live=True, lease=True, grant=True):
        self._live = live
        self.supports_lease = lease
        self._grant = grant
        self.acquired = []
        self.released = 0

    def snapshot(self):
        return SimSnapshot(received_at=time.monotonic()) if self._live else None

    def acquire_control(self, motion_client_id, *, reason="", ttl_s=120.0):
        self.acquired.append((motion_client_id, reason, ttl_s))
        return (True, "") if self._grant else (False, "held by someone else")

    def release_control(self, timeout=5.0):
        self.released += 1


class StubExecutor:
    """Records what it was asked to do and returns a scripted result."""

    def __init__(self, result=None, available=(True, ""), delay=0.0):
        self.result = result or ExecutionResult(status="completed",
                                                detail="soda_can is on r2c2.")
        self._available = available
        self.delay = delay
        self.calls = []
        self.cancel_seen = False

    def available(self, proposal=None):
        return self._available

    def execute(self, proposal, *, should_cancel=None, on_phase=None):
        self.calls.append(proposal)
        if on_phase:
            on_phase("swing in over table")
        deadline = time.time() + self.delay
        while time.time() < deadline:
            if should_cancel and should_cancel():
                self.cancel_seen = True
                return ExecutionResult(status="cancelled",
                                       detail="Stopped, and the arm is back at rest.")
            time.sleep(0.01)
        if should_cancel and should_cancel():
            self.cancel_seen = True
            return ExecutionResult(status="cancelled",
                                   detail="Stopped, and the arm is back at rest.")
        return self.result


def a_proposal(**kw):
    base = dict(plan_id="plan-1", plan_version=0, task_type="pick_place",
                target_id="soda_can", destination="cell:r2c2",
                destination_kind="cell", destination_label="grid cell r2c2",
                brief_reason="tagged recyclable", summary="move soda_can to r2c2")
    base.update(kw)
    return Proposal(**base)


def planner_returning(proposal):
    def _planner(_request):
        return PlannerOutcome(kind="proposal", proposal=proposal)
    return _planner


def settle(coord, session, task_id, until, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = coord.get(session, task_id)
        if task.state in until:
            return task
        time.sleep(0.01)
    raise AssertionError(f"stuck in {coord.get(session, task_id).state}")


# ---------------------------------------------------------------------------
# NullExecutor and the build switch
# ---------------------------------------------------------------------------

def test_null_executor_always_refuses_with_its_reason():
    ex = NullExecutor("no arm here")
    assert ex.available() == (False, "no arm here")
    assert ex.available(a_proposal()) == (False, "no arm here")
    result = ex.execute(a_proposal())
    assert result.status == "failed"
    assert result.detail == "no arm here"


def test_execution_is_off_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("REACHY_PANEL_EXECUTOR", raising=False)
    ex = build_executor(StubLink(), lambda: make_scene(), "scene.yaml")
    ok, why = ex.available()
    assert ok is False
    assert "switched off" in why
    assert "REACHY_PANEL_EXECUTOR" in why


@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
def test_the_switch_accepts_the_usual_spellings(monkeypatch, value):
    monkeypatch.setenv("REACHY_PANEL_EXECUTOR", value)
    ex = build_executor(StubLink(), lambda: make_scene(), "scene.yaml")
    assert isinstance(ex, SimulatorExecutor)


@pytest.mark.parametrize("value", ["0", "false", "no", ""])
def test_the_switch_rejects_everything_else(monkeypatch, value):
    monkeypatch.setenv("REACHY_PANEL_EXECUTOR", value)
    assert isinstance(
        build_executor(StubLink(), lambda: make_scene(), "scene.yaml"),
        NullExecutor,
    )


# ---------------------------------------------------------------------------
# SimulatorExecutor refusals
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_sdk(monkeypatch):
    """Pretend the Reachy SDK is importable, without importing one."""
    monkeypatch.setitem(sys.modules, "reachy_sdk",
                        types.SimpleNamespace(ReachySDK=object))
    monkeypatch.setenv("REACHY_SIM_BACKEND", "mujoco-remote")
    yield


def live_scene(**kw):
    scene = make_scene(**kw)
    from panel_scene import apply_snapshot
    apply_snapshot(scene, SimSnapshot(
        scene_revision="rev-1",
        objects={"soda_can": on_cell(scene, "r1c1")},
        received_at=time.monotonic(),
    ))
    return scene


def test_no_sdk_is_reported_as_such(monkeypatch):
    monkeypatch.setitem(sys.modules, "reachy_sdk", None)
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ok, why = ex.available(a_proposal())
    assert ok is False
    assert "Reachy SDK is not installed" in why


def test_a_failed_safety_gate_refuses(fake_sdk, monkeypatch):
    monkeypatch.setenv("REACHY_SIM_BACKEND", "physical")
    monkeypatch.setenv("REACHY_ENABLE_MOTION", "false")
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ok, why = ex.available(a_proposal())
    assert ok is False
    assert "safety gate" in why


def test_no_live_link_refuses(fake_sdk):
    ex = SimulatorExecutor(StubLink(live=False), live_scene, "scene.yaml")
    ok, why = ex.available(a_proposal())
    assert ok is False
    assert "no live link" in why


def test_a_simulator_without_arbitration_refuses(fake_sdk):
    ex = SimulatorExecutor(StubLink(lease=False), live_scene, "scene.yaml")
    ok, why = ex.available(a_proposal())
    assert ok is False
    assert "arbitrate" in why


def test_a_non_cell_destination_refuses(fake_sdk):
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ok, why = ex.available(a_proposal(destination="destination:left_tray",
                                      destination_kind="destination",
                                      destination_label="tray"))
    assert ok is False
    assert "only place onto a grid cell" in why


def test_an_unsupported_task_type_refuses(fake_sdk):
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ok, why = ex.available(a_proposal(task_type="handover"))
    assert ok is False
    assert "no motion for a handover task" in why


def test_an_object_off_the_board_refuses(fake_sdk):
    def scene_provider():
        scene = make_scene()
        from panel_scene import apply_snapshot
        apply_snapshot(scene, SimSnapshot(
            objects={"soda_can": (0.55, 0.95, 0.0575)},   # parked in the pool
            received_at=time.monotonic(),
        ))
        return scene

    ex = SimulatorExecutor(StubLink(), scene_provider, "scene.yaml")
    ok, why = ex.available(a_proposal())
    assert ok is False
    assert "not on the board" in why


def test_an_unreachable_destination_refuses(fake_sdk):
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ok, why = ex.available(a_proposal(destination="cell:r3c1"))
    assert ok is False
    assert "out of the arm's reach" in why


def test_a_healthy_proposal_is_accepted(fake_sdk):
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    assert ex.available(a_proposal()) == (True, "")


def test_a_refused_lease_is_reported_and_nothing_moves(fake_sdk):
    link = StubLink(grant=False)
    ex = SimulatorExecutor(link, live_scene, "scene.yaml")
    result = ex.execute(a_proposal())
    assert result.status == "failed"
    assert "could not take control" in result.detail
    assert link.acquired            # it asked
    assert link.released == 0       # and never held it, so never released it


def test_the_lease_names_the_bridge_as_the_mover(fake_sdk):
    """The panel holds the lease; the SDK bridge is what actually moves."""
    from panel_executor import MOTION_CLIENT_ID
    link = StubLink(grant=False)
    SimulatorExecutor(link, live_scene, "scene.yaml").execute(a_proposal())
    assert link.acquired[0][0] == MOTION_CLIENT_ID


def test_the_executor_never_sends_place_object():
    """place_object teleports scene state; using it here would fake success.

    Checked against the parsed module rather than the raw text, so the prose
    explaining why it is absent does not count as using it — and so a string
    literal added later does.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(_HERE, "../../web/panel_executor.py").read_text())
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    forbidden = {"place_object", "joint_command", "reset", "scene_load"}
    assert not (forbidden & set(literals))


# ---------------------------------------------------------------------------
# The executing half of the lifecycle
# ---------------------------------------------------------------------------

def _coord(executor, proposal=None):
    proposal = proposal or a_proposal()
    return TaskCoordinator(
        planner_returning(proposal),
        capabilities=Capabilities(),
        executor=executor,
    )


def test_an_unavailable_executor_still_confirms_without_motion():
    coord = _coord(NullExecutor("the arm is not connected"))
    try:
        task = settle(coord, "s1", coord.submit("s1", "put soda_can on r2c2").task_id,
                      {TaskState.awaiting_confirmation})
        done = coord.confirm("s1", task.task_id, task.proposal.plan_id,
                             task.proposal.plan_version)
        assert done.state is TaskState.confirmed_no_motion
        # The exact sentence survives; the reason is a separate turn.
        assert done.detail == "Plan confirmed; no movement performed."
        assert "the arm is not connected" in done.events[-1].text
    finally:
        coord.shutdown()


def test_a_proposal_reports_the_mode_it_will_actually_be_confirmed_in():
    """The card must not promise motion Confirm then declines to perform.

    The proposal's execution_mode was taken from the configured default, so a
    server with a working executor still stamped every plan `planning_only`.
    """
    ex = StubExecutor()
    coord = _coord(ex)
    try:
        task = settle(coord, "s1", coord.submit("s1", "x").task_id,
                      {TaskState.awaiting_confirmation})
        assert task.proposal.execution_mode == "live_simulation"
    finally:
        coord.shutdown()

    coord = _coord(NullExecutor("nothing here"))
    try:
        task = settle(coord, "s1", coord.submit("s1", "x").task_id,
                      {TaskState.awaiting_confirmation})
        assert task.proposal.execution_mode == "planning_only"
    finally:
        coord.shutdown()


def test_a_confirmed_plan_executes_and_completes_from_the_result():
    ex = StubExecutor(ExecutionResult(status="completed",
                                      detail="soda_can is on r2c2.",
                                      evidence={"cell": "r2c2"}))
    coord = _coord(ex)
    try:
        task = settle(coord, "s1", coord.submit("s1", "put soda_can on r2c2").task_id,
                      {TaskState.awaiting_confirmation})
        running = coord.confirm("s1", task.task_id, task.proposal.plan_id,
                                task.proposal.plan_version)
        assert running.state is TaskState.executing
        done = settle(coord, "s1", task.task_id,
                      {TaskState.completed, TaskState.failed})
        assert done.state is TaskState.completed
        assert done.detail == "soda_can is on r2c2."
        assert done.execution_evidence == {"cell": "r2c2"}
        assert ex.calls and ex.calls[0].plan_id == task.proposal.plan_id
    finally:
        coord.shutdown()


def test_a_finished_trajectory_that_did_not_place_is_a_failure():
    """The arm can run every segment while the grasp slipped."""
    ex = StubExecutor(ExecutionResult(
        status="failed",
        detail="The motion finished but soda_can is not on r2c2.",
        evidence={"verified": "live_pose"},
    ))
    coord = _coord(ex)
    try:
        task = settle(coord, "s1", coord.submit("s1", "x").task_id,
                      {TaskState.awaiting_confirmation})
        coord.confirm("s1", task.task_id, task.proposal.plan_id,
                      task.proposal.plan_version)
        done = settle(coord, "s1", task.task_id,
                      {TaskState.completed, TaskState.failed})
        assert done.state is TaskState.failed
        assert "not on r2c2" in done.detail
    finally:
        coord.shutdown()


def test_executing_blocks_a_second_task_from_the_same_session():
    coord = _coord(StubExecutor(delay=0.5))
    try:
        task = settle(coord, "s1", coord.submit("s1", "x").task_id,
                      {TaskState.awaiting_confirmation})
        coord.confirm("s1", task.task_id, task.proposal.plan_id,
                      task.proposal.plan_version)
        from tasks import TaskError
        with pytest.raises(TaskError) as exc:
            coord.submit("s1", "another command")
        assert exc.value.code == "task_in_progress"
    finally:
        coord.shutdown()


def test_cancel_during_execution_waits_for_the_executor_to_acknowledge():
    ex = StubExecutor(delay=1.0)
    coord = _coord(ex)
    try:
        task = settle(coord, "s1", coord.submit("s1", "x").task_id,
                      {TaskState.awaiting_confirmation})
        coord.confirm("s1", task.task_id, task.proposal.plan_id,
                      task.proposal.plan_version)

        pending = coord.cancel("s1", task.task_id)
        # NOT cancelled yet: the arm is still moving, and saying otherwise
        # would claim a motion had stopped before the mover agreed.
        assert pending.state is TaskState.executing
        assert pending.cancel_requested is True
        assert "waiting for the arm" in pending.detail

        done = settle(coord, "s1", task.task_id,
                      {TaskState.cancelled, TaskState.completed,
                       TaskState.failed}, timeout=5)
        assert done.state is TaskState.cancelled
        assert "back at rest" in done.detail
        assert ex.cancel_seen
    finally:
        coord.shutdown()


def test_a_second_cancel_while_stopping_is_harmless():
    ex = StubExecutor(delay=1.0)
    coord = _coord(ex)
    try:
        task = settle(coord, "s1", coord.submit("s1", "x").task_id,
                      {TaskState.awaiting_confirmation})
        coord.confirm("s1", task.task_id, task.proposal.plan_id,
                      task.proposal.plan_version)
        coord.cancel("s1", task.task_id)
        again = coord.cancel("s1", task.task_id)
        assert again.state is TaskState.executing
        # One "stopping" turn, not one per click.
        assert sum(1 for e in again.events if "waiting for the arm" in e.text) == 1
        settle(coord, "s1", task.task_id, {TaskState.cancelled}, timeout=5)
    finally:
        coord.shutdown()


def test_an_executor_that_raises_fails_the_task_not_the_pool():
    class Boom:
        def available(self, proposal=None):
            return True, ""

        def execute(self, proposal, **kw):
            raise RuntimeError("gripper on fire")

    coord = _coord(Boom())
    try:
        task = settle(coord, "s1", coord.submit("s1", "x").task_id,
                      {TaskState.awaiting_confirmation})
        coord.confirm("s1", task.task_id, task.proposal.plan_id,
                      task.proposal.plan_version)
        done = settle(coord, "s1", task.task_id, {TaskState.failed})
        assert "RuntimeError" in done.detail
        assert coord.submit("s2", "another")     # pool still usable
    finally:
        coord.shutdown()


def test_progress_goes_to_detail_not_the_transcript():
    ex = StubExecutor(delay=0.3)
    coord = _coord(ex)
    try:
        task = settle(coord, "s1", coord.submit("s1", "x").task_id,
                      {TaskState.awaiting_confirmation})
        before = len(coord.get("s1", task.task_id).events)
        coord.confirm("s1", task.task_id, task.proposal.plan_id,
                      task.proposal.plan_version)
        settle(coord, "s1", task.task_id, {TaskState.completed, TaskState.failed},
               timeout=5)
        after = coord.get("s1", task.task_id).events
        # One turn for "Executing in simulation.", one for the outcome — the
        # dozen phase updates went to `detail` instead of burying the exchange.
        assert len(after) - before == 2
    finally:
        coord.shutdown()


def test_a_confirmation_is_still_consumed_once_when_executing():
    coord = _coord(StubExecutor(delay=0.4))
    try:
        task = settle(coord, "s1", coord.submit("s1", "x").task_id,
                      {TaskState.awaiting_confirmation})
        p = task.proposal
        coord.confirm("s1", task.task_id, p.plan_id, p.plan_version)
        from tasks import TaskError
        with pytest.raises(TaskError) as exc:
            coord.confirm("s1", task.task_id, p.plan_id, p.plan_version)
        assert exc.value.code == "already_confirmed"
        settle(coord, "s1", task.task_id,
               {TaskState.completed, TaskState.failed}, timeout=5)
    finally:
        coord.shutdown()
