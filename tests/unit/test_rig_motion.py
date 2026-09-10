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


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    """The routes are ~35 s of real motion; the logic under test is not."""
    monkeypatch.setattr(M.time, "sleep", lambda _s: None)
    monkeypatch.setattr(M.P.time, "sleep", lambda _s: None)


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
