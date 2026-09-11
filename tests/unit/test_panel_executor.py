"""Issue #51: the execution adapter and the executing half of the lifecycle.

Nothing here moves a robot.  The executor is exercised through a stub for the
lifecycle tests, and the real `SimulatorExecutor` is exercised only as far as
its refusals — which is the part worth pinning down, because every one of them
is a case where the panel must decline rather than guess.
"""

import json
import os
import sys
import time
import types

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../web"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))

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
    # #82's footprint check reads a real scene FILE (object geometry lives
    # there, not on the SceneView these tests stub).  Every test in this
    # module points scene_file at a placeholder ("scene.yaml") that was never
    # meant to touch disk, so stand in with an object-free model here — the
    # dedicated footprint tests below build a real one deliberately.
    from reachy_ai.scene.awareness import SceneModel
    monkeypatch.setattr(
        SceneModel, "from_yaml",
        staticmethod(lambda path: SceneModel("pedestal", [], None)))
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


def test_pointing_is_refused_in_a_scene_it_was_not_flown_in(fake_sdk):
    """It has a row for FWDCenterLabSivaPool now, and that row is about that
    scene.  Endpoints matching does not mean the path between them is clear."""
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml")
    ok, why = ex.available(
        a_proposal(task_type="point_cell", target_id=None, destination=None,
                   destination_kind="", route="POINT", cell="r2c2"))
    assert ok is False
    assert "TestScene" in why


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


class StubWorker:
    """A motion process that answers without one.

    The arm logic it stands in for is tested in `test_motion_worker.py`,
    against the real thing.  What these tests are about is the half that stays
    here: what goes into a job, and what the panel does with an answer.
    """

    def __init__(self, *results):
        self.results = list(results) or [{"status": "moved", "flown": [],
                                          "final_posture": "home"}]
        self.jobs = []
        self.phases_to_emit = []
        self.cancelled = False

    def run(self, job, *, on_phase=None, should_cancel=None):
        self.jobs.append(job)
        for name in self.phases_to_emit:
            if on_phase is not None:
                on_phase(name)
        if should_cancel is not None and should_cancel():
            self.cancelled = True
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]

    def close(self):
        pass


def _with(worker, provider=live_scene):
    return SimulatorExecutor(StubLink(), provider, "scene.yaml", worker=worker)


def test_the_job_names_the_ability_the_scene_and_where_it_must_start(
        monkeypatch, fake_sdk):
    """Everything the motion process needs and nothing it does not: it has no
    link to the simulator, so the scene it validates routes against has to
    come across with the job."""
    _validated(monkeypatch)
    worker = StubWorker()
    _with(worker).execute(_ability(expected_start_posture="home"))

    job, = worker.jobs
    assert job["kind"] == "ability"
    assert job["task_type"] == "stow_arm"
    assert job["route"] == "STOW_ROUTE"
    assert job["expected_start_posture"] == "home"
    assert job["scene"] == "TestScene"
    assert job["sdk"]["port"] == 50051


def test_a_refusal_from_the_motion_process_is_passed_through_as_it_stands(
        monkeypatch, fake_sdk):
    """It knows things the panel does not — which posture the arm is in, which
    waypoint is nearest — so its wording and its evidence survive intact."""
    _validated(monkeypatch)
    worker = StubWorker({
        "status": "failed",
        "detail": ("my arm is not at a posture I have a measured route out "
                   "of — the nearest waypoint is SWING_2, 22 degrees away."),
        "evidence": {"recovery_needed": True, "nearest": "SWING_2"}})

    out = _with(worker).execute(_ability(expected_start_posture="home"))
    assert out.status == "failed"
    assert "SWING_2" in out.detail
    assert out.evidence["recovery_needed"] is True


def test_progress_from_the_motion_process_reaches_the_page(monkeypatch, fake_sdk):
    _validated(monkeypatch)
    worker = StubWorker()
    worker.phases_to_emit = ["connecting to the arm", "leaving home"]
    seen = []
    _with(worker).execute(_ability(expected_start_posture="home"),
                          on_phase=seen.append)
    assert seen == ["connecting to the arm", "leaving home"]


def test_a_stop_reaches_the_motion_process(monkeypatch, fake_sdk):
    """A Stop has to be visible between waypoints, which is the only place a
    corridor can be left."""
    _validated(monkeypatch)
    worker = StubWorker()
    _with(worker).execute(_ability(expected_start_posture="home"),
                          should_cancel=lambda: True)
    assert worker.cancelled is True


def test_the_board_is_read_before_the_approach_flies(monkeypatch, fake_sdk):
    """An arm that swept something off the table on its way to the route's
    start has disturbed the board.  Reading after the approach — which is what
    this used to do — was the one way for that to go unnoticed."""
    _validated(monkeypatch)
    where = {"soda_can": None}

    def provider():
        scene = live_scene()
        if where["soda_can"] is not None:
            scene.objects["soda_can"].position = where["soda_can"]
        return scene

    class Sweeps(StubWorker):
        """Knocks the can over on its way to the route's start."""

        def run(self, job, **kw):
            scene = provider()
            x, y, z = on_cell(scene, "r1c1")
            where["soda_can"] = (x, y + 0.25, z)
            return {"status": "moved", "flown": [], "final_posture": "home"}

    out = _with(Sweeps(), provider).execute(_ability(expected_start_posture="home"))
    assert out.status == "failed"
    assert "moved something on the way" in out.detail


def test_arriving_with_the_board_disturbed_is_a_failure(monkeypatch, fake_sdk):
    """The two fail independently: the arm can reach HOME having swept
    something off the table on the way, and it can leave the board untouched
    while stopping three waypoints short."""
    _validated(monkeypatch)
    moved = {"n": 0}

    def provider():
        scene = live_scene()
        moved["n"] += 1
        if moved["n"] > 2:                       # after the motion
            x, y, z = on_cell(scene, "r1c1")
            scene.objects["soda_can"].position = (x, y + 0.25, z)
        return scene

    worker = StubWorker({"status": "moved", "flown": ["HOME"],
                         "final_posture": "home"})
    out = _with(worker, provider).execute(
        _ability(expected_start_posture="home", route="STOW_ROUTE"))
    assert out.status == "failed"
    assert "moved something on the way" in out.detail
    assert out.evidence["object_drift"]["soda_can"] > 0.02


def test_arriving_cleanly_reports_the_posture_and_the_waypoints(
        monkeypatch, fake_sdk):
    from reachy_ai.motion import rig_routes as R
    _validated(monkeypatch)
    flown = [w.name for w in R.STOW_ROUTE]
    worker = StubWorker({"status": "moved", "flown": flown,
                         "final_posture": "home"})

    out = _with(worker).execute(_ability(expected_start_posture="home"))
    assert out.status == "completed"
    assert out.evidence["final_posture"] == "home"
    assert out.evidence["waypoints_flown"] == flown
    assert out.evidence["object_drift"] == {}
    assert "board is as it was" in out.detail


def test_a_route_stopped_part_way_is_not_reported_as_arrival(
        monkeypatch, fake_sdk):
    _validated(monkeypatch)
    worker = StubWorker({"status": "moved", "flown": ["TUCK"],
                         "final_posture": None})

    out = _with(worker).execute(_ability(expected_start_posture="home"))
    assert out.status == "failed"
    assert "stopped before home" in out.detail
    assert out.evidence["final_posture"] is None


def test_the_pick_place_job_carries_the_live_poses_and_the_cell(fake_sdk):
    """The motion process has no link to the simulator, so everything the arc
    used to read here has to travel with the job.

    This is the whole regression surface of moving pick-and-place out of
    process: the scene document supplies geometry, the live snapshot supplies
    where things actually are, and planning against the YAML's initial poses
    would aim the gripper at where an object started.
    """
    worker = StubWorker({"status": "moved"})
    scene = live_scene()
    ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml",
                           worker=worker)
    ex.execute(a_proposal())

    job, = worker.jobs
    assert job["kind"] == "pick_place"
    assert job["target_id"] == "soda_can"
    assert job["scene_file"] == "scene.yaml"
    assert job["cell_xy"] == [scene.cells["r2c2"].x, scene.cells["r2c2"].y]
    assert job["live"]["soda_can"] == list(scene.objects["soda_can"].position)


def test_a_cancelled_arc_is_reported_as_cancelled_not_as_a_failure(fake_sdk):
    """Only with the arm actually stopped and parked is it true to say the
    motion has stopped, and the motion process is what knows that."""
    worker = StubWorker({"status": "cancelled",
                         "detail": "Stopped, and the arm is back at rest.",
                         "evidence": {"cancelled_at_phase": True}})
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml", worker=worker)
    out = ex.execute(a_proposal())
    assert out.status == "cancelled"
    assert "back at rest" in out.detail


# ---------------------------------------------------------------------------
# Issue #79: the leash on the motion process.
#
# Against a real child, because the whole point is what happens when a child
# misbehaves, and a stub that misbehaves on request proves nothing about a
# pipe, a kill or a wait.
# ---------------------------------------------------------------------------

CHILD_ANSWERS = """
import json, sys
for line in sys.stdin:
    if "job" not in json.loads(line):
        continue
    print(json.dumps({"phase": "working"}), flush=True)
    print(json.dumps({"result": {"status": "moved", "flown": ["A"],
                                 "final_posture": "home"}}), flush=True)
"""

CHILD_WEDGES = """
import json, sys, time
sys.stdin.readline()
print(json.dumps({"phase": "wedging"}), flush=True)
time.sleep(600)
"""

CHILD_DIES = """
import sys
sys.stdin.readline()
sys.exit(3)
"""

CHILD_ECHOES_CANCEL = """
import json, sys
sys.stdin.readline()
line = sys.stdin.readline()
print(json.dumps({"result": {"status": "cancelled", "detail": line.strip()}}),
      flush=True)
"""


@pytest.fixture
def child(tmp_path, monkeypatch):
    """Build a stand-in motion process out of one of the scripts above."""
    import panel_executor
    monkeypatch.setattr(panel_executor, "WORKER_EXIT_GRACE_S", 0.2)

    made = []

    def build(source, **kw):
        path = tmp_path / f"child{len(made)}.py"
        path.write_text(source)
        from panel_executor import MotionWorker
        worker = MotionWorker(argv=[sys.executable, "-u", str(path)], **kw)
        made.append(worker)
        return worker

    yield build
    for worker in made:
        worker.close()


def test_a_job_gets_its_phases_and_its_result(child):
    worker = child(CHILD_ANSWERS)
    seen = []
    out = worker.run({"kind": "ability"}, on_phase=seen.append)
    assert seen == ["working"]
    assert out == {"status": "moved", "flown": ["A"], "final_posture": "home"}


def test_the_process_outlives_one_job(child):
    """Connecting costs the better part of a second and starts sync threads;
    paying that per request is what the reuse is for."""
    worker = child(CHILD_ANSWERS)
    worker.run({"kind": "ability"})
    first = worker._proc.pid
    worker.run({"kind": "ability"})
    assert worker._proc.pid == first


def test_a_move_that_never_returns_fails_the_task_instead_of_the_panel(child):
    """The failure this exists for: grpc.aio's poller dies, the blocking
    `goto` never returns, and the motion lock is held for the life of the
    process.  A deadline turns that into one failed command."""
    worker = child(CHILD_WEDGES, deadline_s=0.6)
    started = time.monotonic()
    out = worker.run({"kind": "ability"})
    assert time.monotonic() - started < 5.0
    assert out["status"] == "failed"
    assert out["evidence"]["timed_out"] is True
    assert "wherever it stopped" in out["detail"]


def test_a_wedged_process_is_killed_and_the_next_job_gets_a_new_one(child):
    """Leaving it alive would leave the SDK connection it wedged in place, and
    the next job would find the same dead poller."""
    worker = child(CHILD_WEDGES, deadline_s=0.4)
    worker.run({"kind": "ability"})
    assert worker._proc is None


def test_a_process_that_dies_mid_move_is_reported_rather_than_waited_out(child):
    """Its stdout closing is the answer arriving — waiting out the deadline
    for a process that has already gone would be three minutes of nothing."""
    worker = child(CHILD_DIES, deadline_s=30.0)
    started = time.monotonic()
    out = worker.run({"kind": "ability"})
    assert time.monotonic() - started < 5.0
    assert out["status"] == "failed"
    assert out["evidence"]["worker_died"] is True
    assert out["evidence"]["recovery_needed"] is True


def test_a_stop_is_sent_to_the_process_rather_than_killing_it(child):
    """Killing it mid-corridor would abandon the arm between rails at whatever
    waypoint it had reached, which is the one place it must not be left."""
    worker = child(CHILD_ECHOES_CANCEL, deadline_s=5.0)
    out = worker.run({"kind": "ability"}, should_cancel=lambda: True)
    assert json.loads(out["detail"]) == {"cancel": True}


def test_a_process_that_cannot_be_started_is_a_failed_task(child):
    worker = child("import sys; sys.exit(1)", deadline_s=5.0)
    out = worker.run({"kind": "ability"})
    assert out["status"] == "failed"
    assert out["evidence"].get("worker_died") is True


def test_pointing_is_refused_with_more_than_one_object_on_the_board(monkeypatch,
                                                                    fake_sdk):
    """Measured empty, and with each object type alone on the centre cell.
    What has NOT been flown is a reach threading past a second object, which
    is how a can moved 0.189 m in the runs this caution comes from."""
    _validated(monkeypatch)
    scene = live_scene()
    for obj in list(scene.objects.values())[:2]:
        obj.on_board = True
    ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml")
    ok, why = ex.available(_ability(task_type="point_cell", route="POINT",
                                    cell="r2c2", end_posture="present",
                                    expected_start_posture="present"))
    assert ok is False
    assert "one object at a time" in why
    assert "0.189 m" in why


def test_pointing_is_allowed_with_a_single_object(monkeypatch, fake_sdk):
    """Six object types, each alone on the centre cell, board undisturbed in
    every run — so one is measured and allowed."""
    _validated(monkeypatch)
    scene = live_scene()
    alone = sorted(scene.objects)[0]
    for oid, obj in scene.objects.items():
        obj.on_board = (oid == alone)
    ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml")
    ok, why = ex.available(_ability(task_type="point_cell", route="POINT",
                                    cell="r2c2", end_posture="present",
                                    expected_start_posture="present"))
    assert ok, why


def test_pointing_is_allowed_once_the_board_is_clear(monkeypatch, fake_sdk):
    _validated(monkeypatch)
    scene = live_scene()
    for obj in scene.objects.values():
        obj.on_board = False
    ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml")
    ok, why = ex.available(_ability(task_type="point_cell", route="POINT",
                                    cell="r2c2", end_posture="present",
                                    expected_start_posture="present"))
    assert ok, why


def test_the_point_job_names_the_cell_in_the_scenes_own_terms(monkeypatch,
                                                              fake_sdk):
    """The panel says r2c2; SceneModel calls it cell_r2c2, and the motion
    process has no link to ask."""
    _validated(monkeypatch)
    scene = live_scene()
    for obj in scene.objects.values():
        obj.on_board = False
    worker = StubWorker({"status": "moved", "flown": ["cell_r2c2 hover 12 cm"],
                         "final_posture": "present"})
    ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml",
                           worker=worker)
    ex.execute(_ability(task_type="point_cell", route="POINT", cell="r2c2",
                        end_posture="present",
                        expected_start_posture="present"))
    job, = worker.jobs
    assert job["cell"] == "cell_r2c2"
    assert job["scene_file"] == "scene.yaml"
    assert "soda_can" in job["live"]


def test_a_point_reports_what_it_achieved_not_that_it_finished(monkeypatch,
                                                               fake_sdk):
    """The notebook measured 21 cm of pad miss on a move that completed, and
    the brief requires this be labelled a hover pointer rather than a ray."""
    _validated(monkeypatch)
    scene = live_scene()
    for obj in scene.objects.values():
        obj.on_board = False
    worker = StubWorker({"status": "moved",
                         "flown": ["GRIP_SHUT", "BACK",      # the approach
                                   "point: cell_r2c2 at a 12 cm hover",
                                   "point: missed by 2.6 cm",
                                   "point: lifted 0 cm"],
                         "final_posture": "present"})
    ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml",
                           worker=worker)
    out = ex.execute(_ability(task_type="point_cell", route="POINT",
                              cell="r2c2", end_posture="present",
                              expected_start_posture="present"))
    assert out.status == "completed"
    assert "missed by 2.6 cm" in out.detail
    assert "not a calibrated ray" in out.detail
    assert "GRIP_SHUT" not in out.detail        # the approach is evidence,
    assert "GRIP_SHUT" in out.evidence["waypoints_flown"]   # not the answer


def test_the_point_job_carries_the_object_it_is_aimed_at(monkeypatch, fake_sdk):
    """Pointing at an object is planned against that object's own top, and the
    motion process has no link to the simulator — so the id travels with the
    job the same way the board does.  It is the scene's own id by then: the
    planner resolves "soda can" to `soda_can` and writes it back."""
    _validated(monkeypatch)
    scene = live_scene()
    for oid, obj in scene.objects.items():
        obj.on_board = (oid == "soda_can")
    worker = StubWorker({"status": "moved",
                         "flown": ["point: soda_can at a 18 cm hover",
                                   "point: missed by 1.4 cm",
                                   "point: lifted 0 cm"],
                         "final_posture": "present"})
    ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml",
                           worker=worker)
    out = ex.execute(_ability(task_type="point_object", route="POINT",
                              object_id="soda_can", end_posture="present",
                              expected_start_posture="present"))
    assert out.status == "completed", out.detail
    job, = worker.jobs
    assert job["object_id"] == "soda_can"
    assert job["cell"] == ""            # no cell, and no placeholder for one
    assert "missed by 1.4 cm" in out.detail
    assert "not a calibrated ray" in out.detail


def test_an_object_off_the_board_is_refused_under_the_lease(monkeypatch,
                                                            fake_sdk):
    """The planner refuses a pool object when the plan is made and again when
    it is confirmed.  This is the check that runs with the arm about to move:
    an object can be lifted off the board in between, and the motion it would
    cause is a tabletop reach aimed at the floor."""
    _validated(monkeypatch)
    scene = live_scene()
    for obj in scene.objects.values():
        obj.on_board = False
    ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml")
    ok, why = ex.available(_ability(task_type="point_object", route="POINT",
                                    object_id="soda_can",
                                    end_posture="present",
                                    expected_start_posture="present"))
    assert ok is False
    assert "soda_can" in why
    assert "not on the board" in why


def test_an_object_that_left_the_scene_is_named(monkeypatch, fake_sdk):
    _validated(monkeypatch)
    scene = live_scene()
    scene.objects.pop("soda_can", None)
    for obj in scene.objects.values():
        obj.on_board = False
    ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml")
    ok, why = ex.available(_ability(task_type="point_object", route="POINT",
                                    object_id="soda_can",
                                    end_posture="present",
                                    expected_start_posture="present"))
    assert ok is False
    assert "soda_can" in why


def test_pointing_at_the_one_object_on_the_board_is_allowed(monkeypatch,
                                                            fake_sdk):
    """The single-object gate counts what is standing there, not what is being
    aimed at — so the object you are pointing at does not refuse itself."""
    _validated(monkeypatch)
    scene = live_scene()
    for oid, obj in scene.objects.items():
        obj.on_board = (oid == "soda_can")
    ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml")
    ok, why = ex.available(_ability(task_type="point_object", route="POINT",
                                    object_id="soda_can",
                                    end_posture="present",
                                    expected_start_posture="present"))
    assert ok, why


# ---------------------------------------------------------------------------
# #82: refuse rest_forearm (and anything else that crosses REST) rather than
# command it, if a live object is where the swept path would land.
# ---------------------------------------------------------------------------

def _rest_hand_point():
    """A world point the REST pose's hand capsule actually occupies.

    Computed from the real kinematics rather than hand-typed, the same way
    test_arm_clearance.py pins its own reference points: if the link lengths
    or REST itself ever change, this point moves with them instead of quietly
    testing a footprint that no longer matches the real pose.
    """
    from reachy_ai.motion import rig_routes as RR
    from reachy_ai.motion.kinematics import link_capsules
    q = [RR.REST[j] for j in RR.ARM7]
    _name, p0, p1, _radius = next(
        c for c in link_capsules(q, side="right") if c[0] == "hand")
    return tuple((a + b) / 2.0 for a, b in zip(p0, p1))


def _scene_model_with(point, oid="soda_can"):
    from reachy_ai.scene.awareness import SceneModel, SceneObject
    obj = SceneObject(id=oid, kind="box", center=point, size=(0.06, 0.06, 0.06),
                      dynamic=True, tracked=True)
    return SceneModel("pedestal", [obj], None)


def _scene_with_object_at(point, oid="soda_can", cell="r2c3"):
    scene = make_scene()
    from panel_scene import apply_snapshot
    apply_snapshot(scene, SimSnapshot(
        scene_revision="rev-1", objects={oid: point},
        received_at=time.monotonic(),
    ))
    scene.objects[oid].cell = cell
    return scene


class TestFootprintCheck:
    """rest_forearm has no scene, so nothing in primitives.py can know an
    object drifted into the forearm footprint — the check has to sit above
    it, where `_ability_available` already has a fresh scene under the lease.
    """

    def test_an_object_in_the_rest_footprint_refuses(self, monkeypatch, fake_sdk):
        _validated(monkeypatch)
        point = _rest_hand_point()
        from reachy_ai.scene.awareness import SceneModel
        monkeypatch.setattr(SceneModel, "from_yaml",
                            staticmethod(lambda path: _scene_model_with(point)))
        scene = _scene_with_object_at(point)
        ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml")

        ok, why = ex.available(_ability(
            task_type="rest_forearm", route="PLACE_ROUTE",
            end_posture="rest", expected_start_posture="home"))

        assert ok is False
        assert "soda_can" in why and "r2c3" in why

    def test_an_object_one_cell_away_does_not_refuse(self, monkeypatch, fake_sdk):
        _validated(monkeypatch)
        far_away = (5.0, 5.0, 5.0)
        from reachy_ai.scene.awareness import SceneModel
        monkeypatch.setattr(SceneModel, "from_yaml",
                            staticmethod(lambda path: _scene_model_with(far_away)))
        scene = _scene_with_object_at(far_away, cell="r1c1")
        ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml")

        ok, why = ex.available(_ability(
            task_type="rest_forearm", route="PLACE_ROUTE",
            end_posture="rest", expected_start_posture="home"))

        assert ok is True, why

    def test_a_cleared_footprint_behaves_exactly_as_before(self, monkeypatch,
                                                            fake_sdk):
        """No object at all: unchanged behaviour, same as every existing
        rest_forearm test in this file (which use the empty fake_sdk model)."""
        _validated(monkeypatch)
        scene = live_scene()
        ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml")

        ok, why = ex.available(_ability(
            task_type="rest_forearm", route="PLACE_ROUTE",
            end_posture="rest", expected_start_posture="home"))

        assert ok is True, why

    def test_the_refusal_uses_live_poses_not_the_scene_files_own(
            self, monkeypatch, fake_sdk):
        """An object moved out of the footprint since the scene loaded must
        stop blocking it — the live position wins, not the YAML's."""
        _validated(monkeypatch)
        rest_point = _rest_hand_point()
        from reachy_ai.scene.awareness import SceneModel
        # The FILE says the object sits in the footprint; the LIVE snapshot
        # says it has since moved away.  update_poses() must make the live
        # position the one that is actually checked.
        monkeypatch.setattr(
            SceneModel, "from_yaml",
            staticmethod(lambda path: _scene_model_with(rest_point)))
        scene = _scene_with_object_at((5.0, 5.0, 5.0), cell="r1c1")
        ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml")

        ok, why = ex.available(_ability(
            task_type="rest_forearm", route="PLACE_ROUTE",
            end_posture="rest", expected_start_posture="home"))

        assert ok is True, why

    def test_it_also_gates_wave_from_the_pocket(self, monkeypatch, fake_sdk):
        """RAISE_TO_SIDE reaches PRESENT by way of REST, so a wave requested
        from the pocket crosses the identical footprint PLACE_ROUTE does —
        the decision #82 asks for, recorded in rig_routes.FOOTPRINT_LEGS."""
        _validated(monkeypatch)
        point = _rest_hand_point()
        from reachy_ai.scene.awareness import SceneModel
        monkeypatch.setattr(SceneModel, "from_yaml",
                            staticmethod(lambda path: _scene_model_with(point)))
        scene = _scene_with_object_at(point)
        ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml")

        ok, why = ex.available(_ability(
            task_type="wave", route="WAVE",
            end_posture="present", expected_start_posture="present"))

        assert ok is False
        assert "soda_can" in why

    def test_stow_arm_is_gated_the_same_way(self, monkeypatch, fake_sdk):
        _validated(monkeypatch)
        point = _rest_hand_point()
        from reachy_ai.scene.awareness import SceneModel
        monkeypatch.setattr(SceneModel, "from_yaml",
                            staticmethod(lambda path: _scene_model_with(point)))
        scene = _scene_with_object_at(point)
        ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml")

        ok, why = ex.available(_ability(
            task_type="stow_arm", route="STOW_ROUTE",
            end_posture="home", expected_start_posture="rest"))

        assert ok is False
        assert "soda_can" in why
