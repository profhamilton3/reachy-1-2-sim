"""Issue #64: walking the rig routes, against a stub arm.

No robot, no simulator.  The stub tracks its goals exactly unless told to lag,
which is the interesting case: a waypoint the arm did not reach is a stop, not
something to fly on from.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../src"))

from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.tasks import rig_motion as M  # noqa: E402


class _Joint:
    def __init__(self, value=0.0):
        self.present_position = value
        self._goal = value
        self.lag = 0.0

    @property
    def goal_position(self):
        return self._goal

    @goal_position.setter
    def goal_position(self, v):
        self._goal = v
        self.present_position = v - self.lag


class StubArm:
    """Tracks perfectly by default; `lag_joint` makes one joint fall short."""

    def __init__(self, pose=None):
        for name in R.R_JOINTS:
            setattr(self, name, _Joint((pose or {}).get(name, 0.0)))

    def lag_joint(self, name, degrees):
        getattr(self, name).lag = degrees

    def at(self, pose):
        return all(abs(getattr(self, n).present_position - v) < 1e-6
                   for n, v in pose.items())


def exact_move(arm, pose, seconds):
    """A mover that tracks perfectly.  Injected, never patched in: the module's
    default is the SDK's `goto`, and a test that reached in and replaced
    `smooth_move` would keep passing after the default changed — which is
    exactly what happened, and why the routes could not be flown."""
    for name, value in pose.items():
        getattr(arm, name).goal_position = value


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    """The routes are ~35 s of real motion; the logic under test is not."""
    monkeypatch.setattr(M.time, "sleep", lambda _s: None)
    monkeypatch.setattr(M.P.time, "sleep", lambda _s: None)


@pytest.fixture(autouse=True)
def _inject_exact_mover(monkeypatch):
    """Default the module's mover to the exact stub for tests that do not
    pass one, so no test silently reaches for the real SDK."""
    monkeypatch.setattr(M, "sdk_move", exact_move)


# ---------------------------------------------------------------------------
# Flying a route
# ---------------------------------------------------------------------------

def test_the_stow_route_walks_every_waypoint_in_order():
    arm = StubArm(R.REST_SHUT)
    phases = []
    flown = M.stow_to_home(arm, on_phase=phases.append)
    assert flown == [w.name for w in R.STOW_ROUTE]
    assert phases == flown
    assert arm.at(R.HOME)


def test_the_placement_route_ends_with_the_forearm_rested():
    arm = StubArm(R.HOME)
    flown = M.deploy_to_rest(arm)
    assert flown[-1] == "REST"
    assert arm.at(R.REST)


def test_a_waypoint_the_arm_did_not_reach_stops_the_route():
    """The next waypoint's clearance was measured FROM this one."""
    arm = StubArm(R.REST_SHUT)
    arm.lag_joint("r_shoulder_pitch", 30.0)
    with pytest.raises(M.RouteError) as exc:
        M.stow_to_home(arm)
    assert "did not reach" in str(exc.value)
    assert "r_shoulder_pitch" in str(exc.value)
    assert "flying the next segment" in str(exc.value)


def test_the_stop_lands_between_waypoints_not_inside_one():
    """Mid-corridor the arm is between rails; it finishes the waypoint it is
    flying and holds."""
    arm = StubArm(R.REST_SHUT)
    seen = []

    def abort():
        return len(seen) >= 3

    def phase(name):
        seen.append(name)

    flown = M.stow_to_home(arm, should_abort=abort, on_phase=phase)
    assert flown == [w.name for w in R.STOW_ROUTE][:3]
    assert arm.at(R.STOW_ROUTE[2].pose)      # the last one completed


def test_an_unsupported_start_is_a_recovery_question_not_a_correction():
    """A direct move from over the board to HOME drives the upper arm through
    the board's near edge, so there is nothing to correct on the way."""
    arm = StubArm(R.SWING_1)
    with pytest.raises(M.RecoveryNeeded) as exc:
        M.stow_to_home(arm)
    assert "nearest waypoint" in str(exc.value)
    assert "SWING_1" in str(exc.value)


def test_recovery_needed_is_not_a_tracking_failure():
    """Different answers: one needs a human at the rig, the other a retry."""
    assert issubclass(M.RecoveryNeeded, M.RouteError)
    assert M.RecoveryNeeded is not M.RouteError


def test_already_home_is_a_verified_no_op():
    arm = StubArm(R.HOME)
    phases = []
    assert M.stow_to_home(arm, on_phase=phases.append) == []
    assert phases == ["already stowed"]


def test_already_rested_is_a_verified_no_op():
    arm = StubArm(R.REST)
    assert M.deploy_to_rest(arm) == []


def test_the_no_op_is_judged_on_the_gross_joints_only():
    """On a freshly reset sim the gripper falls open and wrist_roll drifts to
    ~40 deg while the motors are off.  Judging "is the arm home?" on those
    reports a clean reset as a fault."""
    arm = StubArm(dict(R.HOME, r_wrist_roll=40.0, r_gripper=R.OPEN))
    assert M.stow_to_home(arm) == []


# ---------------------------------------------------------------------------
# The wave stays bounded
# ---------------------------------------------------------------------------

def test_the_wave_runs_the_measured_number_of_cycles():
    arm = StubArm(R.PRESENT)
    phases = []
    assert M.wave(arm, on_phase=phases.append) == R.WAVE_CYCLES
    assert phases.count("wave 1a") == 1
    assert len([p for p in phases if p.startswith("wave ")]) == R.WAVE_CYCLES * 2


def test_the_wave_ends_where_it_started():
    """Returning to rest or the pocket is a separate validated transition, not
    something appended so the arm looks tidy."""
    arm = StubArm(R.PRESENT)
    M.wave(arm)
    assert arm.at(R.PRESENT)


def test_a_caller_cannot_ask_for_more_cycles_than_were_measured():
    arm = StubArm(R.PRESENT)
    assert M.wave(arm, cycles=50) == R.WAVE_CYCLES


def test_a_caller_can_ask_for_fewer():
    arm = StubArm(R.PRESENT)
    assert M.wave(arm, cycles=1) == 1


def test_waving_from_the_wrong_posture_is_refused():
    arm = StubArm(R.HOME)
    with pytest.raises(M.RecoveryNeeded):
        M.wave(arm)


def test_a_stop_during_the_wave_still_returns_to_present():
    arm = StubArm(R.PRESENT)
    done = M.wave(arm, should_abort=lambda: True)
    assert done == 0
    assert arm.at(R.PRESENT)


# ---------------------------------------------------------------------------
# Pointing reports what it achieved, not that it finished
# ---------------------------------------------------------------------------

class _Clear:
    def __init__(self, distance):
        self.distance = distance


class StubPlanner:
    """Enough of CartesianPlanner to exercise point_at's decisions.

    `room_at` decides the clearance for a target height, which is the knob the
    lift loop turns.
    """

    def __init__(self, room_at=None, unreachable_below=None):
        self.scene = None
        self._room_at = room_at or (lambda z: 0.10)
        self._unreachable_below = unreachable_below
        self.solved = []

    def solve(self, xyz, seed=None, maximise_clearance=False, from_joints=None,
              gripper_deg=None, ids=None, include_static=False, **kw):
        from reachy_ai.motion.kinematics import UnreachableError
        assert maximise_clearance, (
            "the point is scoring candidates by whole-arm clearance; without "
            "it the IK spends the arm's redundancy on pad error")
        assert gripper_deg == R.SHUT, "the hand travels shut"
        if (self._unreachable_below is not None
                and xyz[2] < self._unreachable_below):
            raise UnreachableError("nope")
        self.solved.append(xyz)
        return [xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0, 0.0]

    def clearance(self, joints, gripper_deg=None, **kw):
        return _Clear(self._room_at(joints[2]))

    def fk_world(self, joints):
        return (joints[0], joints[1], joints[2])


def _escorted(monkeypatch, *, completed=True, reason="", realised=0.09):
    from reachy_ai.motion import escort as escort_mod
    from reachy_ai.tasks import rig_motion as rm

    calls = {}

    def fake_escort(planner, q_to, send, read, **kw):
        calls.update(kw)
        calls["q_to"] = q_to
        return escort_mod.EscortResult(
            completed=completed, fraction=1.0 if completed else 0.4,
            legs=[escort_mod.Leg(index=1, fraction=1.0, commanded=list(q_to),
                                 reached=list(q_to), planned=_Clear(0.10),
                                 realised=_Clear(realised), flown=True)],
            reason=reason, reached=list(q_to),
        )

    monkeypatch.setattr(escort_mod, "escort", fake_escort)
    return calls


def _io(pose=(0.0, 0.0, 0.0)):
    state = {"q": [pose[0], pose[1], pose[2], 0.0, 0.0, 0.0, 0.0]}

    def send(joints, secs):
        state["q"] = list(joints)

    def read():
        return list(state["q"])

    return send, read


def test_pointing_lifts_the_target_until_the_whole_arm_fits(monkeypatch):
    _escorted(monkeypatch)
    send, read = _io()
    planner = StubPlanner(room_at=lambda z: 0.10 if z >= 0.90 else 0.01)
    out = M.point_at(planner, "r2c2", (0.35, 0.0), 0.80, send=send, read=read)
    assert out.reached
    assert out.lift_m == pytest.approx(0.10)
    assert out.target[2] == pytest.approx(0.90)


def test_pointing_gives_up_rather_than_flying_a_pose_that_does_not_fit(monkeypatch):
    _escorted(monkeypatch)
    send, read = _io()
    planner = StubPlanner(room_at=lambda z: 0.01)
    out = M.point_at(planner, "r2c2", (0.35, 0.0), 0.80, send=send, read=read)
    assert not out.reached
    assert "does not fit" in out.detail
    assert out.achieved is None


def test_the_target_keeps_a_margin_of_its_own_rather_than_being_excluded(monkeypatch):
    """Excluded from the guard, nothing watches the one object being aimed at:
    blue_cylinder was hovered to 1.6 cm with 6.1 cm reported and moved 0.123 m.
    """
    calls = _escorted(monkeypatch)
    send, read = _io()
    planner = StubPlanner(room_at=lambda z: 0.08)
    M.point_at(planner, "soda_can", (0.35, 0.0), 0.80, send=send, read=read,
               approaching="soda_can")
    assert "soda_can" in calls["margins"]
    # Derived from the pose being flown to, less the slack: as close as it has
    # to be and no closer.
    assert calls["margins"]["soda_can"] == pytest.approx(
        0.08 - R.POINT_APPROACH_SLACK)


def test_the_approach_is_flown_in_the_measured_number_of_legs(monkeypatch):
    calls = _escorted(monkeypatch)
    send, read = _io()
    M.point_at(StubPlanner(), "r2c2", (0.35, 0.0), 0.80, send=send, read=read)
    assert calls["legs"] == R.POINT_LEGS == 6
    assert calls["margin"] == R.POINT_MARGIN


def test_a_finished_trajectory_is_not_reported_as_an_accurate_point(monkeypatch):
    """The notebook measured 21 cm of pad miss on a move that completed."""
    _escorted(monkeypatch)
    send, read = _io()

    class Missing(StubPlanner):
        def fk_world(self, joints):
            return (joints[0] + 0.21, joints[1], joints[2])

    out = M.point_at(Missing(), "r1c2", (0.35, 0.0), 0.80, send=send, read=read)
    assert out.reached                      # the trajectory did finish
    assert out.miss_m == pytest.approx(0.21, abs=1e-6)


def test_object_drift_is_measured_and_reported(monkeypatch):
    """A can shifted 0.189 m by an arm reporting positive clearance."""
    _escorted(monkeypatch)
    send, read = _io()
    poses = {"n": 0}

    def object_positions():
        poses["n"] += 1
        if poses["n"] == 1:
            return {"soda_can": (0.30, 0.0, 0.78)}
        return {"soda_can": (0.30, 0.189, 0.78)}

    out = M.point_at(StubPlanner(), "r2c2", (0.35, 0.0), 0.80, send=send,
                     read=read, object_positions=object_positions)
    assert out.drift["soda_can"] == pytest.approx(0.189)
    assert out.disturbed_the_board


def test_an_undisturbed_board_reports_no_drift(monkeypatch):
    _escorted(monkeypatch)
    send, read = _io()
    fixed = {"soda_can": (0.30, 0.0, 0.78)}
    out = M.point_at(StubPlanner(), "r2c2", (0.35, 0.0), 0.80, send=send,
                     read=read, object_positions=lambda: dict(fixed))
    assert out.drift == {}
    assert not out.disturbed_the_board


def test_a_stop_before_the_approach_returns_without_flying(monkeypatch):
    _escorted(monkeypatch)
    send, read = _io()
    out = M.point_at(StubPlanner(), "r2c2", (0.35, 0.0), 0.80, send=send,
                     read=read, should_abort=lambda: True)
    assert not out.reached
    assert "stopped" in out.detail


def test_the_start_check_ignores_wrist_and_gripper_drift():
    """An arm sitting correctly at REST, with the wrist drift GROSS_JOINTS
    exists to ignore, was refused — with a message that contradicted itself:
    "the arm is not at REST_SHUT; the nearest waypoint is REST_SHUT"."""
    arm = StubArm(dict(R.REST, r_wrist_roll=R.REST["r_wrist_roll"] + 8.0))
    ok, why = M.check_start(arm, R.STOW_ROUTE)
    assert ok, why


def test_the_start_check_still_refuses_a_genuinely_wrong_posture():
    arm = StubArm(R.SWING_1)
    ok, why = M.check_start(arm, R.STOW_ROUTE)
    assert not ok
    assert "SWING_1" in why


def test_a_rested_arm_can_be_stowed_straight_afterwards():
    """The two routes have to compose: deploy ends at REST, and stow starts
    from there."""
    arm = StubArm(R.HOME)
    M.deploy_to_rest(arm)
    assert M.stow_to_home(arm) == [w.name for w in R.STOW_ROUTE]
    assert arm.at(R.HOME)


# ---------------------------------------------------------------------------
# Re-streaming (found by flying it, not by reading it)
#
# The first version of fly_route commanded each waypoint once.  Under
# mujoco-remote the arm only moves WHILE setpoints are streaming, so a fast
# segment finishes short and holding the goal does not close the gap.  A live
# run in FWDCenterLabSivaPool stopped at CURL with the elbow 77 deg off; a
# direct write to goal_position afterwards read back unchanged.
#
# The stub above tracks perfectly, which is exactly why the suite was green
# while the real arm was not.  This one does not.
# ---------------------------------------------------------------------------

class LaggingArm(StubArm):
    """Closes a fraction of the remaining error on each streamed command.

    One pass always falls short.  Repeated passes converge, which is the
    behaviour the notebook's `move_to` relies on and the property the route
    runner has to have.
    """

    def __init__(self, pose=None, closes=0.6):
        super().__init__(pose)
        self.closes = closes
        self.commands = 0
        for name in R.R_JOINTS:
            joint = getattr(self, name)
            joint._arm = self

    def _apply(self, name, goal):
        joint = getattr(self, name)
        joint.present_position += (goal - joint.present_position) * self.closes


def lagging_move(arm, pose, duration):
    arm.commands += 1
    for name, value in pose.items():
        arm._apply(name, value)


@pytest.fixture
def lagging(monkeypatch):
    monkeypatch.setattr(M, "sdk_move", lagging_move)


def test_one_pass_is_not_enough_and_the_runner_knows_it(lagging):
    """A single command leaves 40% of the error; TRACK_TOL is 6 degrees."""
    arm = LaggingArm(R.REST_SHUT)
    flown = M.stow_to_home(arm)
    assert flown == [w.name for w in R.STOW_ROUTE]
    # More commands than waypoints: every waypoint needed re-streaming.
    assert arm.commands > len(R.STOW_ROUTE)
    for name, value in R.HOME.items():
        if name == "r_gripper":
            continue
        assert abs(getattr(arm, name).present_position - value) <= R.TRACK_TOL


def test_a_joint_that_never_converges_still_stops_the_route(lagging):
    """Re-streaming is not a way to fly through a jam.  A joint that does not
    move is still a stop — that is what the waypoint tolerance is for."""
    arm = LaggingArm(R.REST_SHUT)
    arm.closes = 0.0                      # commanded, never moves
    with pytest.raises(M.RouteError) as exc:
        M.stow_to_home(arm)
    assert "re-streamed passes" in str(exc.value)


def test_re_streaming_stops_once_the_waypoint_is_reached(lagging):
    """Not a fixed number of passes: a waypoint the arm reaches first time
    costs one command, so a well-behaved arm is not slowed down."""
    arm = LaggingArm(R.REST_SHUT)
    arm.closes = 1.0                      # perfect tracking
    M.stow_to_home(arm)
    assert arm.commands == len(R.STOW_ROUTE)


def test_the_module_defaults_to_the_sdk_trajectory_generator():
    """The mover is part of what "validated" means.

    `primitives.smooth_move` is a 25 Hz interpolator; the notebook flies these
    routes with the SDK's `goto`/MINIMUM_JERK, and measured live at CURL the
    two behave differently in kind — smooth_move oscillates around 10 deg short
    and never converges, goto closes monotonically.  Substituting one for the
    other is not an implementation detail.

    Read from the module's source rather than through the attribute, because
    the fixture above replaces that attribute — which is the same reach-in the
    old tests did, and the reason a suite of 26 green tests said nothing about
    whether the routes could be flown.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(M.__file__).read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "sdk_move")
    names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "MINIMUM_JERK" in names
    assert "smooth_move" not in names

    import inspect
    for f in (M.fly_route, M.deploy_to_rest, M.stow_to_home, M.wave):
        assert "move" in inspect.signature(f).parameters


# ---------------------------------------------------------------------------
# Resuming from a stranded pose (bounded)
# ---------------------------------------------------------------------------

def test_an_arm_left_at_back_is_recognised_as_on_the_pocket_sequence():
    """The case this exists for: something stopped part way, left the arm at
    BACK — out of the pocket, roll home, nothing in front of it — and every
    later request refused for want of a posture. Correct, and useless."""
    arm = StubArm(R.BACK)
    where = M.stranded_at(arm)
    assert where is not None
    assert where[0] == "BACK"
    assert where[1] == pytest.approx(0.0, abs=0.5)


def test_resuming_finishes_the_pocket_entry():
    arm = StubArm(R.BACK)
    flown = M.resume_to_home(arm, move=exact_move)
    assert flown == ["BACK", "GRIP_SHUT", "HOME"]
    assert R.posture_of(M.present_pose(arm)) == R.POSTURE_HOME


def test_an_arm_out_over_the_board_is_not_on_the_pocket_sequence():
    """Same shoulder pitch, roll swung out: a different situation with a
    different answer, and guessing between them is what this must not do."""
    arm = StubArm(dict(R.BACK, r_shoulder_roll=-45.0))
    assert M.stranded_at(arm) is None


def test_a_pose_far_from_every_waypoint_is_not_resumable():
    """The notebook warns that the connecting move is the one unverified
    segment.  The answer is to keep it small, not to allow any gap."""
    arm = StubArm(dict(R.HOME, r_elbow_pitch=-60.0))
    assert M.stranded_at(arm) is None
    with pytest.raises(M.RecoveryNeeded):
        M.resume_to_home(arm, move=exact_move)


def test_travel_resumes_rather_than_refusing_from_the_pocket_sequence():
    arm = StubArm(R.BACK)
    assert R.posture_of(M.present_pose(arm)) is None          # not a named posture
    flown = M.travel(arm, R.POSTURE_HOME, robot=None)
    assert R.posture_of(M.present_pose(arm)) == R.POSTURE_HOME
    assert "HOME" in flown


def test_travel_still_refuses_from_somewhere_it_does_not_recognise():
    arm = StubArm(dict(R.SWING_2))
    with pytest.raises(M.RecoveryNeeded) as exc:
        M.travel(arm, R.POSTURE_HOME, robot=None)
    assert "not guess" in str(exc.value)
