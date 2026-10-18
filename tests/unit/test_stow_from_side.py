"""The raise and the stow fly the measured route, and only it.

The test that matters here is geometric, and it is the one that was missing.
`raise_to_side` and `stow_from_side` each had a careful comment saying what
they did and why, were reviewed, and drove the gripper through the board and
the elbow through `rig_rail_outer_right` anyway (August, out of the rail
pocket) — because they BUILT their own waypoints out of live joint readings
and nothing checked the path they commanded against the scene they commanded
it in.

Since the October layout (#192) they fly REST -> PRESENT (LIFT_TO_PRESENT)
and PRESENT -> REST (LOWER_TO_REST).  So: record every goal each function
writes, and assert two things.  That the poses are the route's, waypoint for
waypoint.  And that the path they trace is no worse, replayed through the real
scene document, than the route they claim to be flying.

Offline.  No robot, no simulator: `link_capsules` is pure kinematics and
`SceneModel` is a YAML document.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../src"))

from reachy_ai.motion import primitives as P  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.kinematics import R_ARM_JOINTS, link_capsules  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

SCENES = os.path.join(_HERE, "../../scenes")


class _Joint:
    def __init__(self, owner, name, value):
        self._owner = owner
        self._name = name
        self.present_position = value
        self._goal = value

    @property
    def goal_position(self):
        return self._goal

    @goal_position.setter
    def goal_position(self, v):
        self._goal = v
        # Track exactly.  A real arm lags; tracking perfectly here is the
        # HARSHER test, because it puts the arm precisely where the function
        # asked for it, which is the path being checked.
        self.present_position = v
        self._owner.trace.append(self._owner.pose())


class RecordingArm:
    """Records the pose after every commanded joint write."""

    def __init__(self, start):
        self.trace = []
        for name in R.R_JOINTS:
            setattr(self, name, _Joint(self, name, start.get(name, 0.0)))
        self.trace.append(self.pose())

    def pose(self):
        return {n: getattr(self, n).present_position for n in R.R_JOINTS}


class _Robot:
    def __init__(self):
        self.off = []

    def turn_off(self, part):
        self.off.append(part)


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    monkeypatch.setattr(P.time, "sleep", lambda _s: None)


@pytest.fixture(scope="module")
def pool_scene():
    return SceneModel.from_yaml(os.path.join(SCENES, "FWDCenterLabSivaPool.yaml"))


def _rig_ids(model):
    return [i for i in model.obstacle_ids(include_static=True)
            if i.startswith("rig_")]


def _clearance(model, pose, ids=None):
    q = [pose[j] for j in R_ARM_JOINTS]
    # The BOARD is in, not just the rig.  Leaving it out is how a sweep that
    # dragged the gripper across the board's near edge measured clean here.
    cs = model.clearances(link_capsules(q, "right", pose["r_gripper"]),
                          ids=ids, include_static=True, include_table=True)
    if not cs:
        return None, None
    worst = min(cs, key=lambda k: cs[k].distance)
    return worst, cs[worst].distance


def _run_stow(start=None):
    arm = RecordingArm(start or dict(R.PRESENT))
    robot = _Robot()
    P.stow_from_side(robot, arm)
    return arm, robot


def _run_raise(start=None):
    arm = RecordingArm(start or dict(R.REST))
    P.raise_to_side(arm)
    return arm


def _worst(model, trace, ids=None):
    return min(_clearance(model, p, ids)[1] for p in trace)


def _route_poses(route):
    return [dict(w.pose) for w in route]


def _same(a, b):
    return all(abs(a[n] - b[n]) < 0.01 for n in R.R_JOINTS)


def _in_order(trace, want):
    """The trace entries that match `want`, in the order `want` gives.

    Asserted rather than filtered: a route flown out of order is exactly the
    failure this file exists for.
    """
    out, i = [], 0
    for pose in trace:
        if i < len(want) and _same(pose, want[i]):
            out.append(pose)
            i += 1
    assert i == len(want), f"reached {i} of {len(want)} waypoints"
    return out


# ---------------------------------------------------------------------------
# It flies the route, waypoint for waypoint
# ---------------------------------------------------------------------------

def test_the_raise_is_the_measured_lift():
    arm = _run_raise()
    _in_order(arm.trace, _route_poses(R.LIFT_TO_PRESENT))
    assert _same(arm.trace[-1], R.PRESENT)


def test_the_stow_is_the_measured_lowering():
    arm, _ = _run_stow()
    _in_order(arm.trace, _route_poses(R.LOWER_TO_REST))


def test_every_pose_it_commands_is_a_whole_route_pose():
    """THE REGRESSION THIS FILE IS NAMED FOR.

    The old versions built targets from live readings, so the pose flown was
    one nobody had verified, carrying whatever the last route left in the
    wrist and forearm.  Every SETTLED pose either function reaches must be a
    pose that appears in a route.
    """
    known = {tuple(round(w.pose[n], 3) for n in R.R_JOINTS)
             for route in (R.LIFT_TO_PRESENT, R.LOWER_TO_REST) for w in route}
    for arm in (_run_raise(), _run_stow()[0]):
        settled = [p for p in arm.trace
                   if tuple(round(p[n], 3) for n in R.R_JOINTS) in known]
        assert len(settled) >= len(R.LOWER_TO_REST)
        assert tuple(round(arm.trace[-1][n], 3) for n in R.R_JOINTS) in known


# ---------------------------------------------------------------------------
# The path it traces, against the scene
# ---------------------------------------------------------------------------

#: The tightest point of each route against the rig, interpolated between
#: consecutive waypoints, in both lab scenes (they share the rig).  Not a test
#: for a comfortable margin: a test that the route is the one that was
#: measured, so a change to it is visible.  August, out of the rail pocket:
#: 2.57 cm (the upper arm on SWING_3 -> HOVER).  October (#192): the rails are
#: behind the arm.
_ROUTE_RIG_CM = 9.45


def _rig_ids(model):
    return [i for i in model.obstacle_ids(include_static=True)
            if i.startswith("rig_")]


def _interpolated(poses, legs=12):
    out = []
    for a, b in zip(poses, poses[1:]):
        out += [{k: a[k] * (1 - i / legs) + b[k] * (i / legs) for k in a}
                for i in range(legs + 1)]
    return out or list(poses)


def test_the_route_it_flies_is_the_measured_one(pool_scene):
    mcc = SceneModel.from_yaml(os.path.join(SCENES, "FWDCenterLabMCC.yaml"))
    for model in (pool_scene, mcc):
        for start, route in ((R.REST, R.LIFT_TO_PRESENT), (R.PRESENT, R.LOWER_TO_REST)):
            path = _interpolated([dict(start)] + _route_poses(route))
            assert _worst(model, path, _rig_ids(model)) == pytest.approx(
                _ROUTE_RIG_CM / 100.0, abs=0.002)


def test_what_it_commands_never_leaves_the_route(pool_scene):
    """Replayed through the scene, the waypoints the functions actually reach
    are the route's waypoints — so their clearance is the route's clearance,
    not something new."""
    for arm, route in ((_run_raise(), R.LIFT_TO_PRESENT),
                       (_run_stow()[0], R.LOWER_TO_REST)):
        want = _route_poses(route)
        reached = _in_order(arm.trace, want)
        assert _worst(pool_scene, reached) == pytest.approx(
            _worst(pool_scene, want), abs=1e-9)


# ---------------------------------------------------------------------------
# Where it starts, and where it stops
# ---------------------------------------------------------------------------

def test_it_ends_at_rest_with_the_motors_off():
    """HOME IS REST (#192): the stow parks on the board's right margin."""
    arm, robot = _run_stow()
    end = arm.trace[-1]
    for name in R.GROSS_JOINTS:
        assert abs(end[name] - R.REST[name]) < 1.0
    assert R.posture_of(end) == R.POSTURE_HOME == R.POSTURE_REST
    assert robot.off == ["r_arm"]


def test_the_raise_refuses_to_start_anywhere_but_rest():
    """Getting onto the route from somewhere else is the one segment nobody
    measured."""
    arm = RecordingArm(dict(R.REST_OUT))
    with pytest.raises(RuntimeError) as exc:
        P.raise_to_side(arm)
    assert "starts at REST" in str(exc.value)
    assert len(arm.trace) == 1                    # nothing was commanded


def test_the_stow_refuses_to_start_anywhere_but_the_raised_pose():
    arm = RecordingArm(dict(R.REST_OUT))
    with pytest.raises(RuntimeError) as exc:
        P.stow_from_side(_Robot(), arm)
    assert "raised pose" in str(exc.value)
    assert len(arm.trace) == 1


def test_a_waypoint_that_falls_short_stops_the_route():
    """A waypoint reached ten degrees short is no longer the pose that was
    verified, and the next segment's clearance was measured from it."""
    arm = RecordingArm(dict(R.REST))

    def short(a, pose, duration):
        for name, value in pose.items():
            getattr(a, name).goal_position = (
                max(value, -40.0) if name == "r_shoulder_pitch" else value)

    original = P._stream
    try:
        P._stream = short
        with pytest.raises(RuntimeError) as exc:
            P.raise_to_side(arm)
    finally:
        P._stream = original
    assert "stopped at PRESENT_SHUT" in str(exc.value)
    assert "r_shoulder_pitch" in str(exc.value)


# ---------------------------------------------------------------------------
# Streaming, not holding
# ---------------------------------------------------------------------------

def test_converge_re_streams_rather_than_holding_a_goal():
    """Under mujoco-remote the arm only moves while setpoints are streaming;
    holding a goal does nothing.  Verified live: after `goal_position = 0.0`
    the value read back unchanged for six seconds, 107 degrees from target."""
    arm = RecordingArm(dict(R.REST))
    calls = {"n": 0}
    real = P.smooth_move

    def counting(a, pose, duration=2.0):
        calls["n"] += 1
        real(a, pose, duration)

    try:
        P.smooth_move = counting
        assert P.converge(arm, dict(R.REST, r_elbow_pitch=-40.0), 1.0)
        assert calls["n"] >= 1
    finally:
        P.smooth_move = real


def test_the_mover_prefers_the_sdk_generator():
    """`smooth_move` reaches the target and sags off it; goto closes
    monotonically.  Live (August), that difference was a fold that stopped
    at -55."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(P._stream))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "MINIMUM_JERK" in names
    assert "smooth_move" in names        # the offline fallback
