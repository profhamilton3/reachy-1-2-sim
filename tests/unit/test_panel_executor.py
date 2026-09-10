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
    """An ability with no route is refused BY NAME, and says which of the
    three reasons it is: no motion at all, no validated route in this scene,
    or the wrong arm."""
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ok, why = ex.available(a_proposal(task_type="handover"))
    assert ok is False
    assert "handover" in why
    assert "not yet built" in why


def test_a_route_unvalidated_in_this_scene_names_the_scene(fake_sdk):
    """Not "the arm is unavailable" — that sends someone to the robot when the
    answer is in the compatibility record."""
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ok, why = ex.available(
        a_proposal(task_type="stow_arm", target_id=None, destination=None,
                   destination_kind="", route="STOW_ROUTE"))
    assert ok is False
    assert "TestScene" in why
    assert "arm is unavailable" not in why


def test_a_left_arm_ability_refuses_with_the_geometry_reason(fake_sdk):
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ok, why = ex.available(
        a_proposal(task_type="wave", target_id=None, destination=None,
                   destination_kind="", route="WAVE", arm="left"))
    assert ok is False
    assert "right arm" in why
    assert "not symmetric" in why


def test_pointing_is_refused_because_its_runs_moved_objects(fake_sdk):
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ok, why = ex.available(
        a_proposal(task_type="point_cell", target_id=None, destination=None,
                   destination_kind="", route="POINT", cell="r2c2"))
    assert ok is False
    assert "0.189 m" in why


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


# ---------------------------------------------------------------------------
# The lease comes first, then the snapshot (issue #60)
#
# `_execute_locked` used to read the board, pin a destination cell and a
# `started_step`, and only then ask for the lease.  Everything the motion was
# planned against was therefore sampled in the one window where other clients
# could still edit the scene.  A `place_object` landing there sent the arm to
# where the object had been; `_verify` caught it, so the panel did not claim
# success, but the failure read as a slipped grasp rather than as a race.
# ---------------------------------------------------------------------------

def test_the_lease_is_taken_before_the_board_is_read(fake_sdk):
    """The scene the MOTION is planned from is read under the lease.

    `execute()` reads the board before taking anything, to decide whether to
    try at all; that read pins nothing and costs no lease.  This is about
    `_execute_locked`, where a cell and a sim_step get fixed.
    """
    order = []
    link = StubLink()
    inner = link.acquire_control

    def acquire(*a, **kw):
        order.append("acquire")
        return inner(*a, **kw)

    link.acquire_control = acquire

    def provider():
        order.append("read")
        return live_scene()

    SimulatorExecutor(link, provider, "scene.yaml")._execute_locked(
        a_proposal(), None, None)
    assert order, "the executor never touched the link or the board"
    assert order[0] == "acquire"
    assert "read" in order


def test_a_board_change_under_the_lease_is_reported_as_one(fake_sdk):
    """Not as a motion failure.  The operator can act on the difference."""
    link = StubLink()
    reads = {"n": 0}

    def provider():
        reads["n"] += 1
        scene = live_scene()
        if reads["n"] > 1:           # recalled to the pool after the lease
            scene.objects["soda_can"].on_board = False
        return scene

    out = SimulatorExecutor(link, provider, "scene.yaml").execute(a_proposal())
    assert out.status == "failed"
    assert "the board changed while I was taking control" in out.detail
    assert "not on the board" in out.detail
    assert out.evidence["scene_changed"] is True
    assert link.released == 1


def test_a_snapshot_older_than_the_lease_is_not_planned_against(monkeypatch,
                                                                fake_sdk):
    """Moving the read below `acquire_control` is not on its own enough.

    The link hands back whatever the simulator last pushed, which may predate
    the grant — so a "post-lease" read can still describe the board as it was
    while it could be edited.
    """
    import panel_executor
    monkeypatch.setattr(panel_executor, "FRESH_SNAPSHOT_TIMEOUT_S", 0.2)

    class StaleLink(StubLink):
        def snapshot(self):
            return SimSnapshot(received_at=time.monotonic() - 10.0)

    link = StaleLink()
    out = SimulatorExecutor(link, live_scene, "scene.yaml").execute(a_proposal())
    assert out.status == "failed"
    assert "no fresh view" in out.detail
    assert link.released == 1


def test_a_scene_that_falls_back_to_the_file_is_not_fresh(monkeypatch, fake_sdk):
    """`live` false means the poses came from the YAML, not the simulator."""
    import panel_executor
    monkeypatch.setattr(panel_executor, "FRESH_SNAPSHOT_TIMEOUT_S", 0.2)

    ex = SimulatorExecutor(StubLink(), make_scene, "scene.yaml")
    assert ex._fresh_scene() is None


def test_the_lease_is_released_even_when_the_board_changed(fake_sdk):
    """The refusal paths added by #60 sit inside the try, not before it."""
    link = StubLink()
    reads = {"n": 0}

    def provider():
        reads["n"] += 1
        scene = live_scene()
        if reads["n"] > 1:
            scene.objects["soda_can"].on_board = False
        return scene

    SimulatorExecutor(link, provider, "scene.yaml").execute(a_proposal())
    assert len(link.acquired) == 1
    assert link.released == 1


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


# ---------------------------------------------------------------------------
# Ability execution: posture, cancellation, evidence (issue #65)
# ---------------------------------------------------------------------------

def _ability(**kw):
    base = dict(plan_id="plan-a", plan_version=0, task_type="stow_arm",
                target_id=None, destination=None, destination_kind="",
                brief_reason="STOW_ROUTE with my right arm", arm="right",
                route="STOW_ROUTE", route_version=1,
                expected_start_posture="rest", summary="stow the arm")
    base.update(kw)
    return Proposal(**base)


def _validated(monkeypatch):
    """Pretend the route passed validation in this scene, so the tests below
    reach the execution logic instead of the (correct) refusal."""
    from reachy_ai.motion import rig_routes as R
    monkeypatch.setattr(R, "check_route", lambda route, scene: (True, ""))


def test_an_ability_is_refused_in_a_scene_where_its_route_failed(fake_sdk):
    """The default, and the right answer today: FWDCenterLabSivaPool was flown
    and rejected, so nothing in this scene may fly."""
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ok, why = ex.available(_ability())
    assert ok is False
    assert "TestScene" in why


def test_an_arm_at_no_named_posture_reports_recovery_rather_than_guessing(
        monkeypatch, fake_sdk):
    """The nearest waypoint is the useful fact; flying to it is the one
    segment nobody measured."""
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    _validated(monkeypatch)
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.SWING_2))
    monkeypatch.setattr("panel_executor.ReachySDK" if False else
                        "reachy_sdk.ReachySDK", _FakeRobot, raising=False)

    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    out = ex.execute(_ability())
    assert out.status == "failed"
    assert out.evidence.get("recovery_needed") is True
    assert "SWING_2" in out.detail
    assert "will not guess" in out.detail


def test_the_arm_is_taken_to_the_start_rather_than_refused(monkeypatch, fake_sdk):
    """Getting there is part of doing it.

    An arm stored in the rail pocket cannot wave from where it is, and that is
    a fact about the rig rather than something the operator should have to
    know and type.
    """
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    _validated(monkeypatch)
    travelled = {}
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.HOME))
    monkeypatch.setattr(M, "travel",
                        lambda arm, to, **kw: travelled.setdefault("to", to)
                        and None or ["RAISE_TO_SIDE"])
    monkeypatch.setattr(M, "stow_to_home", lambda arm, **kw: [])
    monkeypatch.setattr("reachy_sdk.ReachySDK", _FakeRobot, raising=False)

    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ex.execute(_ability())                # stow starts at rest; arm is home
    assert travelled["to"] == "rest"


def test_an_unbridgeable_posture_says_so_rather_than_inventing_a_path(
        monkeypatch, fake_sdk):
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    _validated(monkeypatch)
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.PRESENT))
    monkeypatch.setattr("reachy_sdk.ReachySDK", _FakeRobot, raising=False)

    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    out = ex.execute(_ability(expected_start_posture="nowhere"))
    assert out.status == "failed"
    assert "no measured way" in out.detail
    assert "invent" in out.detail


class _FakeJoint:
    present_position = 0.0
    goal_position = 0.0
    compliant = False


class _FakeArm:
    """Enough of an arm that the liveness probe reads something real."""

    def __init__(self):
        for name in ("r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw",
                     "r_elbow_pitch", "r_forearm_yaw", "r_wrist_pitch",
                     "r_wrist_roll", "r_gripper"):
            setattr(self, name, _FakeJoint())


class _FakeRobot:
    def __init__(self, host=None, sdk_port=None):
        self.r_arm = _FakeArm()

    def turn_on(self, _part):
        pass


def test_arriving_with_the_board_disturbed_is_a_failure(monkeypatch, fake_sdk):
    """The two fail independently: the arm can reach HOME having swept
    something off the table on the way, and it can leave the board untouched
    while stopping three waypoints short."""
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    _validated(monkeypatch)
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.HOME))
    monkeypatch.setattr(M, "travel",
                        lambda arm, to, **kw: [w.name for w in R.STOW_ROUTE])
    monkeypatch.setattr("reachy_sdk.ReachySDK", _FakeRobot, raising=False)

    moved = {"n": 0}

    def provider():
        scene = live_scene()
        moved["n"] += 1
        if moved["n"] > 2:                       # after the motion
            x, y, z = on_cell(scene, "r1c1")
            scene.objects["soda_can"].position = (x, y + 0.25, z)
        return scene

    ex = SimulatorExecutor(StubLink(), provider, "scene.yaml")
    out = ex.execute(_ability(expected_start_posture="home",
                              route="STOW_ROUTE"))
    assert out.status == "failed"
    assert "moved something on the way" in out.detail
    assert out.evidence["object_drift"]["soda_can"] > 0.02


def test_arriving_cleanly_reports_the_posture_and_the_waypoints(
        monkeypatch, fake_sdk):
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    _validated(monkeypatch)
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.HOME))
    monkeypatch.setattr(M, "travel",
                        lambda arm, to, **kw: [w.name for w in R.STOW_ROUTE])
    monkeypatch.setattr("reachy_sdk.ReachySDK", _FakeRobot, raising=False)

    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    out = ex.execute(_ability(expected_start_posture="home"))
    assert out.status == "completed"
    assert out.evidence["final_posture"] == "home"
    assert out.evidence["waypoints_flown"] == [w.name for w in R.STOW_ROUTE]
    assert out.evidence["object_drift"] == {}
    assert "board is as it was" in out.detail


def test_a_route_stopped_part_way_is_not_reported_as_arrival(
        monkeypatch, fake_sdk):
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    _validated(monkeypatch)
    poses = {"at": dict(R.HOME)}
    monkeypatch.setattr(M, "present_pose", lambda _arm: poses["at"])

    def half(arm, to, **kw):
        poses["at"] = dict(R.SWING_2)          # stopped in the corridor
        return [w.name for w in R.STOW_ROUTE][:4]

    monkeypatch.setattr(M, "travel", half)
    monkeypatch.setattr("reachy_sdk.ReachySDK", _FakeRobot, raising=False)

    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    out = ex.execute(_ability(expected_start_posture="home"))
    assert out.status == "failed"
    assert "stopped before home" in out.detail
    assert out.evidence["final_posture"] is None


def test_the_cancel_check_reaches_the_route_runner(monkeypatch, fake_sdk):
    """A Stop has to be visible between waypoints, which is the only place a
    corridor can be left."""
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    _validated(monkeypatch)
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.HOME))
    seen = {}

    def capture(arm, to, **kw):
        seen["abort"] = kw.get("should_abort")
        return []

    monkeypatch.setattr(M, "travel", capture)
    monkeypatch.setattr("reachy_sdk.ReachySDK", _FakeRobot, raising=False)

    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ex.execute(_ability(expected_start_posture="home"),
               should_cancel=lambda: True)
    assert seen["abort"] is not None
    assert seen["abort"]() is True


def test_the_sdk_connection_is_made_once_and_reused(monkeypatch, fake_sdk):
    """`ReachySDK.__init__` opens a gRPC channel and starts sync threads, and
    nothing stopped them.  After a handful of tasks the bridge stopped
    answering new connections at all — the panel kept serving HTTP while no
    SDK client could connect and the task mid-motion never returned."""
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    _validated(monkeypatch)
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.HOME))
    monkeypatch.setattr(M, "travel", lambda arm, to, **kw: [])
    made = {"n": 0}

    class Counting(_FakeRobot):
        def __init__(self, host=None, sdk_port=None):
            made["n"] += 1
            super().__init__(host, sdk_port)

    monkeypatch.setattr("reachy_sdk.ReachySDK", Counting, raising=False)
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    for _ in range(4):
        ex.execute(_ability(expected_start_posture="home"))
    assert made["n"] == 1


def test_a_stale_connection_is_replaced_rather_than_used(monkeypatch, fake_sdk):
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    _validated(monkeypatch)
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.HOME))
    monkeypatch.setattr(M, "travel", lambda arm, to, **kw: [])
    monkeypatch.setattr("reachy_sdk.ReachySDK", _FakeRobot, raising=False)

    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ex.execute(_ability(expected_start_posture="home"))
    first = ex._robot

    class Dead:
        @property
        def r_arm(self):
            raise RuntimeError("channel closed")

        def _stop(self):
            pass

    ex._robot = Dead()
    ex.execute(_ability(expected_start_posture="home"))
    assert ex._robot is not first
    assert not isinstance(ex._robot, Dead)
