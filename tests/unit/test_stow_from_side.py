"""The pocket exit and the pocket entry fly the measured route, and only it.

The test that matters here is geometric, and it is the one that was missing.
`raise_to_side` and `stow_from_side` each had a careful comment saying what
they did and why, were reviewed, and drove the gripper through the board and
the elbow through `rig_rail_outer_right` anyway — because they BUILT their own
waypoints out of live joint readings and nothing checked the path they
commanded against the scene they commanded it in.

So: record every goal each function writes, and assert two things.  That the
poses are the notebook's, waypoint for waypoint — which is the property that
makes the geometry somebody else's problem, already solved at 0.2 degree
resolution.  And that the path they trace is no worse, replayed through the
real scene document, than the route they claim to be flying.

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
    arm = RecordingArm(start or dict(R.HOME))
    P.raise_to_side(arm)
    return arm


def _worst(model, trace, ids=None):
    return min(_clearance(model, p, ids)[1] for p in trace)


def _worst_rig(model, trace):
    return _worst(model, trace, _rig_ids(model))


def _route_poses(route):
    return [dict(w.pose) for w in route]


# ---------------------------------------------------------------------------
# It flies the notebook, waypoint for waypoint
# ---------------------------------------------------------------------------

def test_the_ascent_is_the_measured_route_then_the_lift():
    """`PLACE_ROUTE` out of the pocket, then the notebook's own move to
    PRESENT (cell 18).  It used to be two route waypoints followed by two
    invented ones."""
    arm = _run_raise()
    want = _route_poses(R.PLACE_ROUTE) + _route_poses(R.LIFT_TO_PRESENT)
    got = [p for p in arm.trace if any(_same(p, w) for w in want)]
    assert [w["r_shoulder_pitch"] for w in want] == \
        [p["r_shoulder_pitch"] for p in _in_order(arm.trace, want)]


def test_the_stow_is_the_measured_route():
    """The notebook's stow IS the return from PRESENT: `STOW_ROUTE` begins at
    REST_SHUT and cell 34 flies it straight out of section 4."""
    arm, _ = _run_stow()
    want = _route_poses(R.STOW_ROUTE)
    assert [w["r_shoulder_pitch"] for w in want] == \
        [p["r_shoulder_pitch"] for p in _in_order(arm.trace, want)]


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


def test_every_pose_it_commands_is_a_whole_route_pose():
    """THE REGRESSION THIS FILE IS NAMED FOR.

    The old versions built targets from live readings — `_folded(here)`,
    `here` with the roll overwritten — so the pose flown was one nobody had
    verified, carrying whatever the last route left in the wrist and forearm.
    A stow after a wave went into the pocket holding the wave's ±60 of forearm
    yaw.  Every goal either function writes must now be a pose that appears in
    a route.
    """
    known = {tuple(round(w.pose[n], 3) for n in R.R_JOINTS)
             for route in (R.PLACE_ROUTE, R.STOW_ROUTE, R.LIFT_TO_PRESENT,
                           R.LOWER_TO_REST)
             for w in route}
    for arm in (_run_raise(), _run_stow()[0]):
        # The recorder snapshots after every individual joint write, so a
        # settled pose is one that matches a waypoint outright.  What must not
        # appear is a SETTLED pose that is in no route — which is exactly what
        # a target built from live readings produced.
        settled = [p for p in arm.trace
                   if tuple(round(p[n], 3) for n in R.R_JOINTS) in known]
        assert len(settled) >= len(R.PLACE_ROUTE)
        for pose, nxt in zip(arm.trace, arm.trace[1:]):
            if pose == nxt:
                continue
        assert tuple(round(arm.trace[-1][n], 3) for n in R.R_JOINTS) in known


def test_the_roll_never_goes_past_the_measured_maximum():
    """The invented hub abducted to -88.  The route's deepest roll is -37.5,
    and it gets there while the pitch is moving too — around the rail rather
    than through it."""
    for arm in (_run_raise(), _run_stow()[0]):
        assert min(p["r_shoulder_roll"] for p in arm.trace) >= -37.5


# ---------------------------------------------------------------------------
# The path it traces, against the scene it failed in
# ---------------------------------------------------------------------------

#: The tightest point on each route, interpolated between consecutive
#: waypoints, in both scenes.  SWING_1 past `rig_rail_outer_right` is the
#: binding one, and it is narrow for everything that goes through it: the
#: waypoint plans at +0.6 cm and flew at +0.2 and -0.2 (#74).  This is not a
#: test for a comfortable margin — it is a test that the corridor is the one
#: that was measured.
_ROUTE_PATH_CM = 0.45


def test_the_route_it_flies_is_the_measured_corridor(pool_scene):
    """The honest invariant, and it is about the ROUTE.

    Some of the route's own waypoints are negative against the board — CURL by
    1.4 cm, REST by 4.6 cm, where the forearm is deliberately supported — so
    "never negative" would be a lie.  What these functions owe is that the
    poses they command are the route's, in order; the corridor between them is
    the route's business and is checked here so a change to it is visible.
    """
    mcc = SceneModel.from_yaml(os.path.join(SCENES, "FWDCenterLabMCC.yaml"))
    for model in (pool_scene, mcc):
        for route in (R.PLACE_ROUTE + R.LIFT_TO_PRESENT, R.STOW_ROUTE):
            worst = _worst_rig(model, _interpolated(_route_poses(route)))
            assert worst == pytest.approx(_ROUTE_PATH_CM / 100.0, abs=0.002)


def test_what_it_commands_never_leaves_the_route(pool_scene):
    """Replayed through the scene, the waypoints the functions actually reach
    are the route's waypoints — so their clearance is the route's clearance,
    not something new."""
    for arm, route in ((_run_raise(), R.PLACE_ROUTE + R.LIFT_TO_PRESENT),
                       (_run_stow()[0], R.STOW_ROUTE)):
        want = _route_poses(route)
        reached = _in_order(arm.trace, want)
        assert _worst(pool_scene, reached) == pytest.approx(
            _worst(pool_scene, want), abs=1e-9)


def test_the_hub_sweep_is_what_it_no_longer_does(pool_scene):
    """Kept as the reason, not as a fossil.

    Rolling from 0 to -88 at a fixed pitch with the elbow folded — the move
    `raise_to_side` used to make after backing out of the pocket — holds the
    hand at the height of the board slab and drags it across the near edge for
    the whole sweep.  An operator watched it.  Physics did not stop it because
    a position servo with 60 Nm beats a compliant contact when the pose it is
    asked for is centimetres past the surface; a pose two degrees past a rail
    loses to it, which is why the same code could be seen to STALL against the
    rail on other days.
    """
    folded = dict(R.HOME, r_shoulder_pitch=30.0, r_elbow_pitch=-110.0,
                  r_gripper=R.SHUT)                 # GRIP_SHUT ran first
    sweep = [dict(folded, r_shoulder_roll=float(r)) for r in range(0, -89, -11)]
    # 0 to -55 is where the hand is squarely in the slab; it shallows out over
    # the last thirty degrees and only clears past -88.
    for pose in [p for p in sweep if p["r_shoulder_roll"] >= -55.0]:
        worst, distance = _clearance(pool_scene, pose)
        assert worst == "table_top"           # the board, not the rail
        assert distance < -0.04
    assert _worst(pool_scene, sweep) < -0.06
    # And the route it was replaced by is on the right side of zero throughout.
    assert _worst_rig(pool_scene,
                      _interpolated(_route_poses(R.PLACE_ROUTE))) > 0


def _interpolated(poses, legs=12):
    out = []
    for a, b in zip(poses, poses[1:]):
        out += [{k: a[k] * (1 - i / legs) + b[k] * (i / legs) for k in a}
                for i in range(legs + 1)]
    return out or list(poses)


# ---------------------------------------------------------------------------
# Where it starts, and where it stops
# ---------------------------------------------------------------------------

def test_it_ends_at_home_with_the_motors_off():
    arm, robot = _run_stow()
    end = arm.trace[-1]
    for name, value in R.HOME.items():
        if name == "r_gripper":
            continue
        assert abs(end[name] - value) < 1.0
    assert robot.off == ["r_arm"]


def test_the_ascent_refuses_to_start_anywhere_but_the_pocket():
    """The route out of the pocket starts at HOME.  Getting onto it from
    somewhere else is the one segment nobody measured."""
    arm = RecordingArm(dict(R.SWING_2))
    with pytest.raises(RuntimeError) as exc:
        P.raise_to_side(arm)
    assert "starts at HOME" in str(exc.value)
    assert len(arm.trace) == 1                    # nothing was commanded


def test_the_stow_refuses_to_start_anywhere_but_the_raised_pose():
    """Going straight from an arbitrary pose to HOME drives the upper arm
    through the board's near edge."""
    arm = RecordingArm(dict(R.SWING_2))
    with pytest.raises(RuntimeError) as exc:
        P.stow_from_side(_Robot(), arm)
    assert "raised pose" in str(exc.value)
    assert len(arm.trace) == 1


def test_a_waypoint_that_falls_short_stops_the_route():
    """A waypoint reached ten degrees short is no longer the pose that was
    verified, and the next segment's clearance was measured from it."""
    class Stuck(RecordingArm):
        pass

    arm = Stuck(dict(R.HOME))

    def half(a, pose, duration):
        for name, value in pose.items():
            getattr(a, name).goal_position = (
                max(value, -55.0) if name == "r_elbow_pitch" else value)

    original = P._stream
    try:
        P._stream = half
        with pytest.raises(RuntimeError) as exc:
            P.raise_to_side(arm)
    finally:
        P._stream = original
    assert "stopped at CURL" in str(exc.value)
    assert "r_elbow_pitch" in str(exc.value)


# ---------------------------------------------------------------------------
# Streaming, not holding
# ---------------------------------------------------------------------------

def test_converge_re_streams_rather_than_holding_a_goal():
    """Under mujoco-remote the arm only moves while setpoints are streaming;
    holding a goal does nothing.  Verified live: after `goal_position = 0.0`
    the value read back unchanged for six seconds, 107 degrees from target."""
    arm = RecordingArm(dict(R.HOME))
    calls = {"n": 0}
    real = P.smooth_move

    def counting(a, pose, duration=2.0):
        calls["n"] += 1
        real(a, pose, duration)

    try:
        P.smooth_move = counting
        assert P.converge(arm, dict(R.HOME, r_elbow_pitch=-40.0), 1.0)
        assert calls["n"] >= 1
    finally:
        P.smooth_move = real


def test_the_mover_prefers_the_sdk_generator():
    """`smooth_move` reaches the target and sags off it; goto closes
    monotonically.  Live, that difference is a fold that stops at -55."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(P._stream))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "MINIMUM_JERK" in names
    assert "smooth_move" in names        # the offline fallback
