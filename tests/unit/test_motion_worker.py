"""Issue #79: the process that actually moves the arm.

Nothing here moves a robot.  `run_ability` is called directly, with a fake SDK
and the route runners stubbed, because that is where the arm logic now lives —
these are the tests that used to drive it through `SimulatorExecutor.execute`
and could not follow it across a process boundary.  The supervision half (the
deadline, a child that dies, forwarding phases) is in
`test_panel_executor.py`, against the real child.
"""

import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../web"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))

import motion_worker as W  # noqa: E402


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

    def _stop(self):
        pass


@pytest.fixture(autouse=True)
def _no_settle(monkeypatch):
    """The connect settle is real time on a real SDK and dead time here."""
    monkeypatch.setattr(W, "CONNECT_SETTLE_S", 0.0)


def _conn(sdk_class=_FakeRobot):
    return W.Connection("localhost", 50051, sdk_class=sdk_class)


def _job(**kw):
    base = dict(kind="ability", task_type="stow_arm", route="STOW_ROUTE",
                expected_start_posture="rest", scene="TestScene")
    base.update(kw)
    return base


@pytest.fixture
def validated(monkeypatch):
    """Pretend the route passed validation in this scene, so the tests below
    reach the motion logic instead of the (correct) refusal."""
    from reachy_ai.motion import rig_routes as R
    monkeypatch.setattr(R, "check_route", lambda route, scene: (True, ""))


# -- the connection -------------------------------------------------------

def test_the_sdk_connection_is_made_once_and_reused():
    """`ReachySDK.__init__` opens a gRPC channel and starts sync threads, and
    connecting costs the better part of a second.  Keeping it is safe in this
    process in a way it was not in the server's: when it goes wrong the whole
    child is replaced."""
    made = {"n": 0}

    class Counting(_FakeRobot):
        def __init__(self, host=None, sdk_port=None):
            made["n"] += 1
            super().__init__(host, sdk_port)

    conn = _conn(Counting)
    for _ in range(4):
        conn.robot()
    assert made["n"] == 1


def test_a_stale_connection_is_replaced_rather_than_used():
    conn = _conn()
    first = conn.robot()

    class Dead:
        @property
        def r_arm(self):
            raise RuntimeError("channel closed")

        def _stop(self):
            pass

    conn._robot = Dead()
    second = conn.robot()
    assert second is not first
    assert not isinstance(second, Dead)


def test_an_arm_without_that_joint_is_not_taken_for_a_dead_channel():
    """A joint that is simply absent is not evidence of anything, and the
    first version of this probe reconnected on every single call."""
    class Bare(_FakeRobot):
        def __init__(self, host=None, sdk_port=None):
            self.r_arm = object()

    conn = _conn(Bare)
    assert conn.robot() is conn.robot()


# -- flying an ability ----------------------------------------------------

def test_an_arm_at_no_named_posture_reports_recovery_rather_than_guessing(
        monkeypatch, validated):
    """The nearest waypoint is the useful fact; flying to it is the one
    segment nobody measured."""
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.SWING_2))

    out = W.run_ability(_job(), _conn())
    assert out["status"] == "failed"
    assert out["evidence"]["recovery_needed"] is True
    assert "SWING_2" in out["detail"]
    assert "will not guess" in out["detail"]


def test_the_arm_is_taken_to_the_start_rather_than_refused(monkeypatch, validated):
    """Getting there is part of doing it.

    An arm stored in the rail pocket cannot wave from where it is, and that is
    a fact about the rig rather than something the operator should have to
    know and type.
    """
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.HOME))
    went = []
    monkeypatch.setattr(M, "travel",
                        lambda arm, to, **kw: went.append(to) or [f"to {to}"])

    out = W.run_ability(_job(), _conn())          # stow starts at rest; arm is home
    assert went[0] == "rest"                      # the approach
    assert went[1] == R.POSTURE_HOME              # then the ability itself
    assert out["status"] == "moved"


def test_an_unbridgeable_posture_says_so_rather_than_inventing_a_path(
        monkeypatch, validated):
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.PRESENT))

    out = W.run_ability(_job(expected_start_posture="nowhere"), _conn())
    assert out["status"] == "failed"
    assert "no measured way" in out["detail"]
    assert "invent" in out["detail"]


def test_an_approach_leg_unvalidated_in_this_scene_names_the_leg(monkeypatch):
    """The approach is checked route by route, in the scene the panel is
    showing — a leg that was never flown here blocks the ability that needs
    it, and says which leg."""
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.HOME))
    monkeypatch.setattr(R, "check_route",
                        lambda route, scene: (False, "it has never been flown"))

    out = W.run_ability(_job(), _conn())
    assert out["status"] == "failed"
    assert "never been flown" in out["detail"]
    assert out["evidence"]["blocked_on"]


def test_arriving_reports_the_waypoints_and_the_posture(monkeypatch, validated):
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.HOME))
    monkeypatch.setattr(M, "travel",
                        lambda arm, to, **kw: [w.name for w in R.STOW_ROUTE])

    out = W.run_ability(_job(expected_start_posture="home"), _conn())
    assert out["status"] == "moved"
    assert out["final_posture"] == "home"
    assert out["flown"] == [w.name for w in R.STOW_ROUTE]


def test_a_route_stopped_part_way_reports_where_it_stopped(monkeypatch, validated):
    """It reports the posture and lets the panel judge it — the arm being
    three waypoints short of HOME is not something to correct from here."""
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    poses = {"at": dict(R.HOME)}
    monkeypatch.setattr(M, "present_pose", lambda _arm: poses["at"])

    def half(arm, to, **kw):
        poses["at"] = dict(R.SWING_2)             # stopped in the corridor
        return [w.name for w in R.STOW_ROUTE][:4]

    monkeypatch.setattr(M, "travel", half)

    out = W.run_ability(_job(expected_start_posture="home"), _conn())
    assert out["status"] == "moved"
    assert out["final_posture"] is None


def test_the_cancel_check_reaches_the_route_runner(monkeypatch, validated):
    """A Stop has to be visible between waypoints, which is the only place a
    corridor can be left."""
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.HOME))
    seen = {}

    def capture(arm, to, **kw):
        seen["abort"] = kw.get("should_abort")
        return []

    monkeypatch.setattr(M, "travel", capture)

    W.run_ability(_job(expected_start_posture="home"), _conn(),
                  should_abort=lambda: True)
    assert seen["abort"] is not None
    assert seen["abort"]() is True


def test_a_route_that_stops_mid_corridor_is_reported_not_corrected(
        monkeypatch, validated):
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.HOME))

    def stops(arm, to, **kw):
        raise M.RouteError("SWING_2 did not track")

    monkeypatch.setattr(M, "travel", stops)

    out = W.run_ability(_job(expected_start_posture="home"), _conn())
    assert out["status"] == "failed"
    assert out["evidence"]["stopped_mid_route"] is True
    assert "SWING_2" in out["detail"]


def test_motors_that_never_come_on_refuse_the_move(monkeypatch, validated):
    """The arm is parked with the motors off; asking for 40 degrees of
    shoulder before the controller has taken hold moved nothing at all."""
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.HOME))
    monkeypatch.setattr(W, "wait_for_motors", lambda arm, **kw: False)

    out = W.run_ability(_job(expected_start_posture="home"), _conn())
    assert out["status"] == "failed"
    assert out["evidence"]["motors_off"] is True


def test_an_interrupted_pocket_entry_is_finished_rather_than_refused(
        monkeypatch, validated):
    """Left at BACK — out of the pocket, roll home, nothing in front of it —
    every later request refused for want of a posture to start from.  That is
    correct and useless: BACK has two measured moves left in it."""
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    poses = {"at": dict(R.BACK)}
    monkeypatch.setattr(M, "present_pose", lambda _arm: poses["at"])
    monkeypatch.setattr(M, "stranded_at", lambda _arm: ("BACK", 3.0))

    def resume(arm, **kw):
        poses["at"] = dict(R.HOME)
        return ["GRIP_SHUT", "HOME"]

    monkeypatch.setattr(M, "resume_to_home", resume)
    monkeypatch.setattr(M, "travel", lambda arm, to, **kw: ["done"])

    out = W.run_ability(_job(expected_start_posture="home"), _conn())
    assert out["status"] == "moved"
    assert out["flown"][:2] == ["GRIP_SHUT", "HOME"]


# -- the gate and the job table -------------------------------------------

def test_a_failed_safety_gate_refuses_here_too(monkeypatch):
    """The parent checks it as well, and that is not a redundancy worth
    removing: this is the process that commands the joints."""
    from reachy_ai.motion import safety
    monkeypatch.setattr(safety, "gate_check", lambda *a, **kw: False)

    out = W.run_job(_job(), _conn())
    assert out["status"] == "failed"
    assert "safety gate" in out["detail"]


def test_an_unknown_job_is_named_rather_than_crashed(monkeypatch):
    from reachy_ai.motion import safety
    monkeypatch.setattr(safety, "gate_check", lambda *a, **kw: True)

    out = W.run_job({"kind": "juggle"}, _conn())
    assert out["status"] == "failed"
    assert "juggle" in out["detail"]


def test_an_unwired_ability_is_named_rather_than_crashed(monkeypatch, validated):
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M
    monkeypatch.setattr(M, "present_pose", lambda _arm: dict(R.HOME))

    out = W.run_ability(_job(task_type="point_cell",
                             expected_start_posture="home"), _conn())
    assert out["status"] == "failed"
    assert "point_cell" in out["detail"]


def test_a_raising_job_becomes_a_result_rather_than_a_traceback(monkeypatch):
    """The child must answer every job exactly once.  A job that raises and
    says nothing would leave the panel waiting out the whole deadline for an
    answer it already has."""
    from reachy_ai.motion import safety
    monkeypatch.setattr(safety, "gate_check", lambda *a, **kw: True)
    monkeypatch.setitem(W.JOBS, "boom",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no")))

    out = W.run_job({"kind": "boom"}, _conn())
    assert out["status"] == "failed"
    assert "RuntimeError" in out["detail"]


# -- the protocol ---------------------------------------------------------

def test_the_child_answers_a_job_and_exits_when_its_parent_goes():
    """End to end over the real pipes, with the SDK never reached: a job the
    gate refuses still gets exactly one result line, and closing stdin ends
    the process rather than leaving it behind."""
    env = dict(os.environ, REACHY_SIM_BACKEND="physical",
               REACHY_ENABLE_MOTION="false",
               PYTHONPATH=os.pathsep.join([
                   os.path.join(_HERE, "../../src"),
                   os.path.join(_HERE, "../../web")]))
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(_HERE, "../../web/motion_worker.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env)
    proc.stdin.write(json.dumps({"job": _job()}) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    proc.stdin.close()
    assert proc.wait(timeout=10) == 0
    assert json.loads(line)["result"]["status"] == "failed"
