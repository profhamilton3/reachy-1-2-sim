"""Issue #73: the side-hub exit must not fly through the rig.

The test that matters here is geometric.  `stow_from_side()` had a comment
saying exactly what it did and why, was reviewed, and drove the arm through
`rig_rail_outer_right` anyway — because nothing checked the path it commanded
against the scene it commanded it in.  So: record every goal the function
writes, replay them through the real scene document, and assert the whole-arm
clearance never goes negative.

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
    cs = model.clearances(link_capsules(q, "right", pose["r_gripper"]),
                          ids=ids, include_static=True)
    if not cs:
        return None, None
    worst = min(cs, key=lambda k: cs[k].distance)
    return worst, cs[worst].distance


def _run_stow(start=None):
    arm = RecordingArm(start or dict(P.SIDE_HIGH))
    robot = _Robot()
    P.stow_from_side(robot, arm, duration=3.0)
    return arm, robot


# ---------------------------------------------------------------------------
# The path it commands, against the scene it failed in
# ---------------------------------------------------------------------------

#: What the tucked sweep measures at its tightest point, and the shallowest
#: the old straight-armed one got.  The corridor past `rig_rail_outer_right` is
#: narrow for everything that goes through it — SWING_1 on the measured route
#: plans at +0.6 cm and flew at +0.2 and -0.2 (#74) — so this is not a test for
#: a comfortable margin.  It is a test that the tuck is worth doing.
#: What the tucked paths measure at their tightest point, with an arm that
#: tracks exactly.  They differ because they carry different shoulder pitches:
#: from the hub the arm is already folded and keeps its own (-25, which models
#: best), while from HOME it has to back out of the pocket to +30 first.
_TUCKED_FROM_HUB_CM = 1.5
_TUCKED_FROM_HOME_CM = -0.4
_TUCKED_PARTWAY_CM = -2.2
_STRAIGHT_CM = -3.2


def _worst_rig(model, trace):
    ids = _rig_ids(model)
    return min(_clearance(model, p, ids)[1] for p in trace)


def test_the_tuck_is_an_order_of_magnitude_better_than_the_straight_sweep(
        pool_scene):
    """The model calls both marginal; they are not the same kind of marginal.

    Straight, the arm is 3.2 cm inside the rail and stops dead there — flown,
    it jammed at roll -33.4 and refused every further command.  Tucked, the
    model reads a few millimetres and the arm sweeps the full range.
    """
    arm, _ = _run_stow()
    tucked = _worst_rig(pool_scene, arm.trace)
    assert tucked > 0, "{:+.1f} cm".format(tucked * 100)
    assert tucked == pytest.approx(_TUCKED_FROM_HUB_CM / 100.0, abs=0.01)


def test_the_same_path_behaves_the_same_in_the_other_scene(pool_scene):
    """The rig is inherited, so this should hold in both — and if it ever
    stops holding, the rails moved and everything else needs re-measuring."""
    mcc = SceneModel.from_yaml(os.path.join(SCENES, "FWDCenterLabMCC.yaml"))
    arm, _ = _run_stow()
    assert _worst_rig(mcc, arm.trace) == pytest.approx(
        _worst_rig(pool_scene, arm.trace), abs=0.001)


def test_the_old_order_is_the_one_that_failed(pool_scene):
    """Straighten-then-lower, to show the test can tell the difference.

    Without this, "the path is clear" says nothing about whether the check
    would have caught the defect it was written for.
    """
    ids = _rig_ids(pool_scene)
    straight_first = []
    pose = dict(P.SIDE_HIGH)
    pose["r_elbow_pitch"] = 0.0
    pose["r_shoulder_pitch"] = 0.0
    for i in range(41):                       # roll swept home, arm straight
        p = dict(pose)
        p["r_shoulder_roll"] = P.SIDE_HIGH["r_shoulder_roll"] * (1 - i / 40)
        straight_first.append(p)
    worst = min(_clearance(pool_scene, p, ids)[1] for p in straight_first)
    assert worst < _STRAIGHT_CM / 100.0


def test_the_object_exposure_is_real_and_stated(pool_scene):
    """What the fix does NOT do, pinned so it cannot be quietly forgotten.

    `primitives` has no scene.  The straightening at the end sweeps the
    forearm across the near-right grid cell, and against FWDCenterLabMCC's
    initial placements that is inside `red_cube`.  The docstring says so; this
    asserts the docstring is still true, and will fail if someone "fixes" the
    number without fixing the sweep.
    """
    mcc = SceneModel.from_yaml(os.path.join(SCENES, "FWDCenterLabMCC.yaml"))
    arm, _ = _run_stow()
    worst = min(_clearance(mcc, p)[1] for p in arm.trace)
    assert worst < 0, ("the board is now clear too — good, but the docstring "
                       "still warns about it and should be updated")
    assert "escort" in P.stow_from_side.__doc__
    assert "CANNOT SEE" in P.stow_from_side.__doc__


# ---------------------------------------------------------------------------
# The ordering invariant, stated directly
# ---------------------------------------------------------------------------

def test_the_roll_comes_home_before_the_elbow_straightens():
    """The whole fix in one assertion: lower first, straighten last."""
    arm, _ = _run_stow()
    roll_home_at = next(i for i, p in enumerate(arm.trace)
                        if abs(p["r_shoulder_roll"]) < 1.0)
    elbow_straight_at = next(i for i, p in enumerate(arm.trace)
                             if abs(p["r_elbow_pitch"]) < 10.0)
    assert roll_home_at < elbow_straight_at


def test_the_elbow_stays_folded_through_the_rail_band():
    """Roll -35 to -13 is where a straight arm is inside the outer rail."""
    arm, _ = _run_stow()
    inside_band = [p for p in arm.trace if -35.0 <= p["r_shoulder_roll"] <= -13.0]
    assert inside_band, "the trace never passes through the band"
    for p in inside_band:
        assert p["r_elbow_pitch"] < -60.0


def test_the_arm_straightens_backed_out_of_the_pocket():
    """HOME is inside the rail pocket and the only approach to it is from
    behind — the measured route's first waypoint is BACK at +40 for the same
    reason.  Probed live from HOME, a forward shoulder pitch does not move at
    all: commanded -60, -40 and -25, it held at -0.9, +4.5 and -0.5.
    """
    arm, _ = _run_stow()
    unfolding = [p for p in arm.trace if -60.0 < p["r_elbow_pitch"] < -10.0]
    assert unfolding
    for p in unfolding:
        assert p["r_shoulder_pitch"] > 15.0


def test_it_ends_at_home_with_the_motors_off():
    arm, robot = _run_stow()
    end = arm.trace[-1]
    for name, value in R.HOME.items():
        if name == "r_gripper":
            continue
        assert abs(end[name] - value) < 1.0
    assert robot.off == ["r_arm"]


def test_it_works_from_a_part_way_return(pool_scene):
    """The return trajectory can leave the arm flexed and forward.

    This is the worst of the three starts — an arm caught half folded at
    roll -55 is already inside the band, so the tuck happens there rather
    than before it, and the first part of the sweep carries whatever it
    started with.  Better than the straight sweep, and not by much: a start
    inside the band is a start the tuck cannot fully rescue.
    """
    arm, _ = _run_stow(start={"r_shoulder_pitch": -40.0, "r_shoulder_roll": -55.0,
                              "r_arm_yaw": 10.0, "r_elbow_pitch": -70.0,
                              "r_forearm_yaw": 0.0, "r_wrist_pitch": 0.0,
                              "r_wrist_roll": 0.0, "r_gripper": R.OPEN})
    worst = _worst_rig(pool_scene, arm.trace)
    assert worst > _STRAIGHT_CM / 100.0
    assert worst == pytest.approx(_TUCKED_PARTWAY_CM / 100.0, abs=0.01)


# ---------------------------------------------------------------------------
# Streaming, not holding
# ---------------------------------------------------------------------------

def test_converge_re_streams_rather_than_holding_a_goal():
    """Under mujoco-remote the arm only moves while setpoints are streaming;
    holding a goal does nothing.  Verified live: after `goal_position = 0.0`
    the value read back unchanged for six seconds, 107 degrees from target."""
    class Lagging(RecordingArm):
        def __init__(self, start, closes=0.5):
            self.closes = closes
            super().__init__(start)
            self.passes = 0

    arm = Lagging(dict(R.HOME))
    calls = {"n": 0}
    real = P.smooth_move

    def counting(a, pose, duration=2.0):
        calls["n"] += 1
        real(a, pose, duration)

    P_smooth = P.smooth_move
    try:
        P.smooth_move = counting
        assert P.converge(arm, dict(R.HOME, r_elbow_pitch=-40.0), 1.0)
        assert calls["n"] >= 1
    finally:
        P.smooth_move = P_smooth


# ---------------------------------------------------------------------------
# The trip out has the same constraint as the trip back
# ---------------------------------------------------------------------------

def _run_raise(start=None):
    arm = RecordingArm(start or dict(R.HOME))
    P.raise_to_side(arm, duration=3.0)
    return arm


def test_raising_to_the_side_never_enters_the_rig(pool_scene):
    """It used to abduct the STRAIGHT arm out of HOME, straight into the band.

    Measured live: commanded to -88, the roll reached -14.1 and held there
    against `rig_rail_outer_right`, and the SIDE_HIGH goals that follow never
    moved it.  The old docstring claimed it held at its range limit.
    """
    arm = _run_raise()
    worst = _worst_rig(pool_scene, arm.trace)
    assert worst == pytest.approx(_TUCKED_FROM_HOME_CM / 100.0, abs=0.01)
    assert worst > _STRAIGHT_CM / 100.0 + 0.02


def test_the_old_ascent_is_the_one_that_jammed(pool_scene):
    """Isolated abduction with a straight arm, to show the check discriminates."""
    ids = _rig_ids(pool_scene)
    poses = []
    for i in range(45):
        p = dict(R.HOME)
        p["r_shoulder_roll"] = P.SIDE_HIGH["r_shoulder_roll"] * i / 44
        poses.append(p)
    assert min(_clearance(pool_scene, p, ids)[1]
               for p in poses) < _STRAIGHT_CM / 100.0


def test_the_elbow_is_folded_before_the_roll_moves_on_the_way_out():
    arm = _run_raise()
    first_roll_move = next(i for i, p in enumerate(arm.trace)
                           if p["r_shoulder_roll"] < -13.0)
    assert arm.trace[first_roll_move]["r_elbow_pitch"] < -60.0


def test_the_ascent_reaches_the_hub():
    arm = _run_raise()
    end = arm.trace[-1]
    for name, value in P.SIDE_HIGH.items():
        if name == "r_gripper":
            continue
        assert abs(end[name] - value) < 1.0


def test_out_and_back_is_clear_in_both_scenes(pool_scene):
    """The round trip the pick-and-place arc actually makes."""
    mcc = SceneModel.from_yaml(os.path.join(SCENES, "FWDCenterLabMCC.yaml"))
    up = _run_raise()
    down = RecordingArm(dict(P.SIDE_HIGH))
    P.stow_from_side(_Robot(), down, duration=3.0)
    for model in (pool_scene, mcc):
        worst = _worst_rig(model, up.trace + down.trace)
        assert worst > _STRAIGHT_CM / 100.0 + 0.02, "{:+.1f} cm".format(
            worst * 100)
        assert worst == pytest.approx(_TUCKED_FROM_HOME_CM / 100.0, abs=0.01)


def test_a_half_fold_stops_rather_than_sweeping():
    """The dangerous failure is not a slow fold, it is a partial one.

    Measured live: `converge` left the elbow at -55 of -100, the arm was still
    straight enough to catch the rail, and the sweep that followed reached
    -0.28 cm.  A fold that did not take must stop the move, not slow it.
    """
    class Stuck(RecordingArm):
        pass

    arm = Stuck(dict(R.HOME))

    def half(a, pose, duration):
        for name, value in pose.items():
            joint = getattr(a, name)
            if name == "r_elbow_pitch":
                joint.goal_position = max(value, -55.0)   # never fully folds
            else:
                joint.goal_position = value

    original = P._stream
    try:
        P._stream = half
        with pytest.raises(RuntimeError) as exc:
            P.stow_from_side(_Robot(), arm, duration=3.0)
        assert "did not take" in str(exc.value)
        assert "elbow is at -55" in str(exc.value)
        assert "rig_rail_outer_right" in str(exc.value)
        # And it stopped BEFORE moving the roll.
        assert all(abs(p["r_shoulder_roll"]) < 1.0 for p in arm.trace)
    finally:
        P._stream = original


def test_raising_also_stops_on_a_half_fold():
    class Stuck(RecordingArm):
        pass

    arm = Stuck(dict(R.HOME))

    def half(a, pose, duration):
        for name, value in pose.items():
            joint = getattr(a, name)
            joint.goal_position = (max(value, -55.0)
                                   if name == "r_elbow_pitch" else value)

    original = P._stream
    try:
        P._stream = half
        with pytest.raises(RuntimeError) as exc:
            P.raise_to_side(arm, duration=3.0)
        assert "did not take" in str(exc.value)
        assert "elbow is at -55" in str(exc.value)
        assert all(abs(p["r_shoulder_roll"]) < 1.0 for p in arm.trace)
    finally:
        P._stream = original


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
