"""Unit tests for the closed-loop clearance guard (motion.escort).

The plan-time guard in ``test_arm_clearance.py`` answers "would this pose be
safe".  These answer the question that one cannot: "was the pose the arm
ACTUALLY reached safe, and what happens when it wasn't".

Everything here is offline.  ``escort`` takes ``send`` and ``read`` as
callables, so a fake arm that mis-tracks on purpose stands in for the physics.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from reachy_ai.motion.escort import EscortResult, Leg, escort  # noqa: E402
from reachy_ai.motion.kinematics import (  # noqa: E402
    CartesianPlanner,
    R_ARM_JOINTS,
)
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_SCENE_PATH = os.path.join(_ROOT, "scenes", "FWDCenterLabMCC.yaml")

SHUT = 20.0
STATIC = dict(include_static=True)


def q(**kw):
    d = dict.fromkeys(R_ARM_JOINTS, 0.0)
    d.update(kw)
    return [d[j] for j in R_ARM_JOINTS]


PRESENT = q(r_shoulder_pitch=-70.0, r_shoulder_roll=-25.0, r_elbow_pitch=-80.0)


@pytest.fixture
def scene():
    return SceneModel.from_yaml(_SCENE_PATH)


@pytest.fixture
def planner(scene):
    # arm=None: every clearance method is pure geometry and never touches the
    # SDK, which is what makes the guard testable off the robot.
    return CartesianPlanner(arm=None, scene=scene, side="right")


class TracedArm:
    """An arm that flies its own curve rather than the line it is commanded.

    ``send`` works out how far along the commanded move the given pose is and
    reports what the arm's REAL trajectory is doing at that fraction.  With no
    ``bow`` the two coincide and this is a perfect tracker; with one, the arm
    still passes through both endpoints and departs in between — which is the
    behaviour that defeated the plan-time guard on the real robot.
    """

    def __init__(self, q_start, q_goal, bow=None):
        self.q_start = list(q_start)
        self.q_goal = list(q_goal)
        self.bow = bow
        self.t = 0.0
        self.sent = []

    def _fraction(self, pose):
        num = den = 0.0
        for a, b, c in zip(self.q_start, self.q_goal, pose):
            d = b - a
            num += d * (c - a)
            den += d * d
        return 0.0 if den == 0.0 else num / den

    def send(self, pose, secs):
        self.sent.append((list(pose), secs))
        self.t = self._fraction(pose)

    def read(self):
        t = self.t
        pose = [a + t * (b - a) for a, b in zip(self.q_start, self.q_goal)]
        if self.bow is not None:
            pose = [x + d for x, d in zip(pose, self.bow(t))]
        return pose


def bow_on(joint, amplitude):
    """A half-sine excursion on one joint, zero at both ends of the move.

    The shape heterogeneous joint tracking produces: the arm passes through the
    poses the plan-time guard cleared at the endpoints and leaves the line in
    between.
    """
    idx = R_ARM_JOINTS.index(joint)

    def f(t):
        out = [0.0] * 7
        out[idx] = amplitude * math.sin(math.pi * t)
        return out

    return f


def slip_on(joint, amplitude, after):
    """A joint that jumps ``amplitude`` degrees once the move passes ``after``.

    Unlike a bow this is invisible to any re-plan: it appears entirely inside
    one leg, so the check made before that leg was made against a pose the arm
    had not yet departed from.  This is what only the after-the-fact
    measurement can catch, and the reason the loop measures at all rather than
    just re-planning more often.
    """
    idx = R_ARM_JOINTS.index(joint)

    def f(t):
        out = [0.0] * 7
        if t > after:
            out[idx] = amplitude
        return out

    return f


# A move that the plan-time guard passes comfortably: both endpoints and every
# point on the joint-space line between them stay ~10 cm clear of everything.
# Found by sweeping the reachable poses for the widest gap between what the
# straight line promises and what a bowed flight delivers.
CLEAR_GOAL = q(r_shoulder_pitch=-50.0, r_shoulder_roll=-30.0,
               r_elbow_pitch=-90.0, r_arm_yaw=-20.0)

# +30 deg of roll at mid-move takes the UPPER ARM to within 1.7 cm of red_cube —
# not the hand, and not at either end of the move.  This is the measured shape
# of the failure that made the module necessary.
BOW_JOINT, BOW_DEG = "r_shoulder_roll", 30.0


class TestTheFixtureItself:
    """If the fixture move is not actually clear, nothing below means anything."""

    def test_the_clear_move_passes_the_plan_time_guard(self, planner):
        c = planner.path_clearance(PRESENT, CLEAR_GOAL, steps=21,
                                   gripper_deg=SHUT, **STATIC)
        assert c.distance > 0.08

    def test_a_perfect_tracker_ends_where_it_was_sent(self):
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        arm.send(CLEAR_GOAL, 1.0)
        assert arm.read() == pytest.approx(CLEAR_GOAL)


class TestACleanMove:
    def test_it_completes(self, planner):
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=6, gripper_deg=SHUT, **STATIC)
        assert r.completed
        assert r.fraction == 1.0
        assert r.stopped_by == ""
        assert r.reached == pytest.approx(CLEAR_GOAL)

    def test_it_flies_the_number_of_legs_it_was_asked_for(self, planner):
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=4, gripper_deg=SHUT, **STATIC)
        assert len(r.legs) == 4
        assert len(arm.sent) == 4
        assert all(leg.flown for leg in r.legs)

    def test_the_legs_are_evenly_spaced_along_the_commanded_move(self, planner):
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        escort(planner, CLEAR_GOAL, arm.send, arm.read,
               margin=0.05, legs=4, gripper_deg=SHUT, **STATIC)
        for i, (pose, _) in enumerate(arm.sent, start=1):
            want = [a + (i / 4) * (b - a) for a, b in zip(PRESENT, CLEAR_GOAL)]
            assert pose == pytest.approx(want)

    def test_a_perfect_tracker_shows_no_degradation(self, planner):
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=6, gripper_deg=SHUT, **STATIC)
        assert r.worst_degraded == pytest.approx(0.0, abs=1e-9)

    def test_one_leg_reproduces_the_old_unguarded_shape(self, planner):
        """legs=1 is the pre-existing behaviour: check, fly, check once."""
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=1, gripper_deg=SHUT, **STATIC)
        assert r.completed and len(arm.sent) == 1

    def test_legs_must_be_positive(self, planner):
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        with pytest.raises(ValueError):
            escort(planner, CLEAR_GOAL, arm.send, arm.read, margin=0.05, legs=0)


class TestLegDuration:
    def test_the_move_is_split_across_the_legs(self, planner):
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        escort(planner, CLEAR_GOAL, arm.send, arm.read, margin=0.05, legs=4,
               duration=4.0, min_leg_secs=0.1, gripper_deg=SHUT, **STATIC)
        assert [secs for _, secs in arm.sent] == [1.0, 1.0, 1.0, 1.0]

    def test_a_leg_is_never_shorter_than_the_floor(self, planner):
        """Six legs of a 1.2 s move is 0.2 s each, which the weak wrist joints
        cannot track.  The floor buys the time back rather than pretending."""
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        escort(planner, CLEAR_GOAL, arm.send, arm.read, margin=0.05, legs=6,
               duration=1.2, min_leg_secs=0.4, gripper_deg=SHUT, **STATIC)
        assert [secs for _, secs in arm.sent] == [0.4] * 6


class TestThePlanTimeRefusal:
    """The old guard, still in place — it just no longer works alone."""

    def test_an_impossible_margin_refuses_the_first_leg(self, planner):
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=5.0, legs=6, gripper_deg=SHUT, **STATIC)
        assert not r.completed
        assert r.stopped_by == "plan"
        assert r.fraction == 0.0
        assert arm.sent == []              # nothing was flown at all
        assert r.legs[-1].flown is False

    def test_a_refused_leg_reports_what_it_was_refused_for(self, planner):
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=5.0, legs=6, gripper_deg=SHUT, **STATIC)
        assert r.legs[-1].planned is not None
        assert r.legs[-1].planned.object_id in r.reason
        assert "margin" in r.reason


class TestTheDefectThisModuleExistsFor:
    """One move, one mis-tracking arm, two ways of guarding it.

    This is the whole argument in four tests: the plan-time guard says the move
    is clear, the arm really does foul, the old one-shot check passes it, and
    the loop stops it.
    """

    def _arm(self):
        return TracedArm(PRESENT, CLEAR_GOAL, bow=bow_on(BOW_JOINT, BOW_DEG))

    def test_the_plan_time_guard_says_the_move_is_clear(self, planner):
        c = planner.path_clearance(PRESENT, CLEAR_GOAL, steps=41,
                                   gripper_deg=SHUT, **STATIC)
        assert c.distance > 0.09

    def test_but_the_curve_the_arm_flies_reaches_into_red_cube(self, planner):
        bow = bow_on(BOW_JOINT, BOW_DEG)
        worst = min(
            (planner.clearance(
                [a + t * (b - a) + d
                 for a, b, d in zip(PRESENT, CLEAR_GOAL, bow(t))],
                SHUT, **STATIC)
             for t in (i / 40 for i in range(41))),
            key=lambda c: c.distance,
        )
        assert worst.distance < 0.025
        assert worst.object_id == "red_cube"
        assert worst.link == "upper_arm"   # not the hand, and not the pad

    def test_checked_once_at_the_ends_the_move_goes_through(self, planner):
        """legs=1 is the behaviour that shipped, and this is what it does."""
        arm = self._arm()
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=1, gripper_deg=SHUT, **STATIC)
        assert r.completed

    def test_flown_in_legs_it_is_stopped_partway(self, planner):
        arm = self._arm()
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=6, gripper_deg=SHUT, **STATIC)
        assert not r.completed
        assert 0.0 < r.fraction < 1.0


class TestTheReplanCatch:
    """The usual catch: the next leg is planned FROM WHERE THE ARM IS.

    A one-shot guard plans the whole move from the pose the arm was supposed to
    be in.  Re-planning each leg from the measured pose is already a closed
    loop, and because the plan margin is twice the realised floor and covers a
    whole path rather than one point, this is normally what fires first.
    """

    def _arm(self):
        return TracedArm(PRESENT, CLEAR_GOAL, bow=bow_on(BOW_JOINT, BOW_DEG))

    def test_it_stops_at_plan_time_on_a_later_leg(self, planner):
        arm = self._arm()
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=6, gripper_deg=SHUT, **STATIC)
        assert r.stopped_by == "plan"
        assert r.legs[-1].index > 1        # leg 1 was planned from the truth
        assert not r.legs[-1].flown

    def test_the_stop_names_what_it_was_stopped_for(self, planner):
        arm = self._arm()
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=6, gripper_deg=SHUT, **STATIC)
        assert r.legs[-1].planned.object_id in r.reason
        assert "margin" in r.reason

    def test_the_arm_is_left_where_it_last_measured_clear(self, planner):
        """Nothing to retreat from: every leg flown ended above the floor, or
        the loop would have stopped on that leg instead."""
        arm = self._arm()
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=6, gripper_deg=SHUT, **STATIC)
        assert not r.backed_off
        assert planner.clearance(r.reached, SHUT, **STATIC).distance >= 0.025

    def test_more_legs_stop_the_move_sooner(self, planner):
        """Not a guarantee, a tendency — and the reason to prefer short legs."""
        coarse, fine = self._arm(), self._arm()
        rc = escort(planner, CLEAR_GOAL, coarse.send, coarse.read,
                    margin=0.05, legs=4, gripper_deg=SHUT, **STATIC)
        rf = escort(planner, CLEAR_GOAL, fine.send, fine.read,
                    margin=0.05, legs=12, gripper_deg=SHUT, **STATIC)
        assert not rc.completed and not rf.completed
        assert rf.fraction <= rc.fraction


class TestTheRealisedCatch:
    """The catch no amount of re-planning gives you.

    ``slip_on`` puts the whole deviation inside one leg: the pose the leg was
    planned from was accurate, so the plan check had nothing to object to, and
    the arm still arrived somewhere it should not be.  Only measuring the pose
    reached finds this.

    Measured on this fixture, six legs, the slip appearing during leg 4:

        leg   planned   realised
         3     12.5 cm   12.5 cm    tracking cleanly
         4     12.5 cm    1.4 cm    <- planned clear, flown into the cube

    1.4 cm is under the 2.5 cm floor, so leg 4 is the last one flown, and legs 5
    and 6 — which the same slip would have taken to 0.6 cm — never happen.  That
    is the guarantee on offer: not that nothing is ever touched, but that one
    disturbance does not become three.
    """

    SLIP = BOW_DEG

    def _arm(self):
        return TracedArm(PRESENT, CLEAR_GOAL,
                         bow=slip_on(BOW_JOINT, self.SLIP, after=0.55))

    def test_it_stops_on_the_realised_measurement(self, planner):
        arm = self._arm()
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=6, gripper_deg=SHUT, **STATIC)
        assert not r.completed
        assert r.stopped_by == "realised"

    def test_the_leg_it_stopped_on_was_planned_clear(self, planner):
        """The distinguishing fact: the plan check passed and the flight did not."""
        arm = self._arm()
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=6, gripper_deg=SHUT, **STATIC)
        last = r.legs[-1]
        assert last.flown
        assert last.planned.distance >= 0.05
        assert last.realised.distance < 0.025

    def test_the_reason_quotes_both_numbers(self, planner):
        arm = self._arm()
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=6, gripper_deg=SHUT, **STATIC)
        assert "planned" in r.reason
        assert "floor" in r.reason

    def test_the_degradation_is_reported(self, planner):
        arm = self._arm()
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=6, gripper_deg=SHUT, **STATIC)
        assert r.worst_degraded > 0.025

    def test_it_backs_off_to_the_last_measured_safe_pose(self, planner):
        arm = self._arm()
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=6, gripper_deg=SHUT, **STATIC)
        assert r.backed_off
        retreat = arm.sent[-1][0]
        good = [l for l in r.legs if l.flown][-2]
        assert retreat == pytest.approx(good.reached)

    def test_the_retreat_can_be_turned_off(self, planner):
        arm = self._arm()
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=6, gripper_deg=SHUT, back_off=False,
                   **STATIC)
        assert not r.backed_off
        # One command per leg flown, and no retreat on the end.
        assert len(arm.sent) == len([l for l in r.legs if l.flown])


class TestTheAbortMargin:
    def test_it_defaults_to_half_the_margin(self, planner):
        """Documented behaviour, and the number a caller is most likely to get
        wrong by leaving it out."""
        a, b = (TracedArm(PRESENT, CLEAR_GOAL,
                          bow=slip_on(BOW_JOINT, BOW_DEG, after=0.55))
                for _ in range(2))
        implicit = escort(planner, CLEAR_GOAL, a.send, a.read,
                          margin=0.05, legs=6, gripper_deg=SHUT, **STATIC)
        explicit = escort(planner, CLEAR_GOAL, b.send, b.read, margin=0.05,
                          abort_margin=0.025, legs=6, gripper_deg=SHUT,
                          **STATIC)
        assert implicit.fraction == explicit.fraction
        assert implicit.stopped_by == explicit.stopped_by

    def test_a_generous_floor_lets_the_slip_through(self, planner):
        """The knob does what it says — which is also the way to misuse it."""
        arm = TracedArm(PRESENT, CLEAR_GOAL,
                        bow=slip_on(BOW_JOINT, BOW_DEG, after=0.55))
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=-1.0, abort_margin=-1.0, legs=6,
                   gripper_deg=SHUT, **STATIC)
        assert r.completed


class TestTheSceneIsRefreshed:
    def test_refresh_runs_before_every_measurement(self, planner):
        calls = []
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        escort(planner, CLEAR_GOAL, arm.send, arm.read, margin=0.05, legs=3,
               gripper_deg=SHUT, refresh=lambda: calls.append(1), **STATIC)
        # One before each leg's plan check and one after each leg's flight.
        assert len(calls) == 6

    def test_an_object_that_moves_mid_move_is_noticed(self, planner, scene):
        """The guard reads the scene it is given.  Move the can under the arm
        and the realised check has to see it, or the model is checking a board
        that no longer exists."""
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        state = {"n": 0}

        def refresh():
            state["n"] += 1
            if state["n"] == 5:
                # Put the can directly under the elbow's path.
                from reachy_ai.motion.kinematics import link_frames
                mid = [a + 0.5 * (b - a) for a, b in zip(PRESENT, CLEAR_GOAL)]
                elbow = tuple(float(v) for v in link_frames(mid)[1])
                scene.update_poses({"soda_can": elbow})

        r = escort(planner, CLEAR_GOAL, arm.send, arm.read, margin=0.05,
                   legs=6, gripper_deg=SHUT, refresh=refresh, **STATIC)
        assert not r.completed
        assert "soda_can" in r.reason


class TestReporting:
    def test_on_leg_sees_every_leg_as_it_happens(self, planner):
        seen = []
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read, margin=0.05,
                   legs=5, gripper_deg=SHUT, on_leg=seen.append, **STATIC)
        assert [l.index for l in seen] == [1, 2, 3, 4, 5]
        assert seen == r.legs

    def test_drift_is_the_gap_between_commanded_and_reached(self):
        leg = Leg(1, 0.5, q(r_elbow_pitch=-60.0), q(r_elbow_pitch=-52.0),
                  None, None, True)
        assert leg.drift == pytest.approx(8.0)

    def test_a_completed_result_says_so(self, planner):
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read,
                   margin=0.05, legs=3, gripper_deg=SHUT, **STATIC)
        assert "arrived in 3 legs" in str(r)

    def test_a_stopped_result_says_where(self, planner):
        arm = TracedArm(PRESENT, CLEAR_GOAL,
                        bow=slip_on(BOW_JOINT, BOW_DEG, after=0.55))
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read, margin=0.05,
                   legs=6, gripper_deg=SHUT, **STATIC)
        assert str(r).startswith("STOPPED at")
        assert "backed off" in str(r)


class TestNoScene:
    """A planner with no scene has nothing to check, and must not pretend to."""

    def test_it_flies_the_whole_move(self):
        planner = CartesianPlanner(arm=None, scene=None, side="right")
        arm = TracedArm(PRESENT, CLEAR_GOAL)
        r = escort(planner, CLEAR_GOAL, arm.send, arm.read, margin=0.05,
                   legs=3, gripper_deg=SHUT)
        assert r.completed
        assert r.worst_realised is None
        assert r.worst_degraded == 0.0
