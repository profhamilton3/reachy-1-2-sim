"""Unit tests for whole-arm clearance against scene objects.

Covers the two failures that made the notebook's old pad-only check useless:

  * the arm is not its pad — the forearm and upper arm sweep a much larger
    volume, and they are what actually reach an object on the near half of the
    board;
  * a collision happens in TRANSIT — checking the two endpoints of a move says
    nothing about the poses between them.

All offline: link_frames reproduces the arm's kinematic chain in-process, so
none of this needs a simulator, ROS, or reachy_sdk.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from reachy_ai.motion.kinematics import (  # noqa: E402
    CartesianPlanner,
    R_ARM_JOINTS,
    _FOREARM_LEN,
    _FOREARM_RADIUS,
    _HAND_LEN,
    _UPPER_ARM_LEN,
    _UPPER_ARM_RADIUS,
    hand_radius,
    joint_path,
    link_capsules,
    link_frames,
)
from reachy_ai.scene.awareness import (  # noqa: E402
    SceneModel,
    SceneObject,
    object_sdf,
    segment_object_distance,
)

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_SCENE_PATH = os.path.join(_ROOT, "scenes", "FWDCenterLabMCC.yaml")


def q(**kw):
    """A 7-element right-arm pose in R_ARM_JOINTS order, degrees."""
    d = dict.fromkeys(R_ARM_JOINTS, 0.0)
    d.update(kw)
    return [d[j] for j in R_ARM_JOINTS]


# The pose Routine 2 used to sweep from, and the one it sweeps from now.
OLD_PRESENT = q(r_shoulder_pitch=-37.5, r_shoulder_roll=-2.0, r_elbow_pitch=-80.0)
PRESENT = q(r_shoulder_pitch=-70.0, r_shoulder_roll=-25.0, r_elbow_pitch=-80.0)


@pytest.fixture
def scene():
    return SceneModel.from_yaml(_SCENE_PATH)


@pytest.fixture
def planner(scene):
    # None for the arm: every clearance method is pure geometry and never
    # touches the SDK, which is what makes this testable off the robot.
    return CartesianPlanner(arm=None, scene=scene, side="right")


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class TestLinkFrames:
    """link_frames must reproduce the SDK's own FK, not approximate it.

    The reference values were read off the running simulator by calling
    reachy.r_arm.forward_kinematics() and comparing element by element; the
    agreement was exact to float precision.  If a link length or an axis order
    ever drifts, these numbers catch it.
    """

    def test_zero_pose_hangs_straight_down(self):
        shoulder, elbow, wrist, _ = link_frames(q())
        assert shoulder == pytest.approx((0.0, -0.19, 1.0))
        assert elbow == pytest.approx((0.0, -0.19, 1.0 - _UPPER_ARM_LEN))
        assert wrist == pytest.approx(
            (0.0, -0.19, 1.0 - _UPPER_ARM_LEN - _FOREARM_LEN))

    def test_matches_sdk_forward_kinematics_at_present(self):
        _, elbow, wrist, _ = link_frames(OLD_PRESENT)
        assert wrist == pytest.approx((0.3921, -0.2013, 0.8935), abs=5e-4)
        assert elbow == pytest.approx((0.1703, -0.1998, 0.7780), abs=5e-4)

    def test_link_lengths_are_invariant_under_joint_angles(self):
        for pose in (q(), OLD_PRESENT, PRESENT,
                     q(r_shoulder_pitch=-20.0, r_arm_yaw=35.0,
                       r_elbow_pitch=-100.0, r_forearm_yaw=60.0)):
            shoulder, elbow, wrist, _ = link_frames(pose)
            assert _dist(shoulder, elbow) == pytest.approx(_UPPER_ARM_LEN)
            assert _dist(elbow, wrist) == pytest.approx(_FOREARM_LEN)

    def test_wrist_and_forearm_joints_do_not_move_the_elbow(self):
        _, base_elbow, _, _ = link_frames(OLD_PRESENT)
        moved = dict(zip(R_ARM_JOINTS, OLD_PRESENT))
        moved.update(r_forearm_yaw=90.0, r_wrist_pitch=45.0, r_wrist_roll=-45.0)
        _, elbow, _, _ = link_frames([moved[j] for j in R_ARM_JOINTS])
        assert elbow == pytest.approx(base_elbow)

    def test_left_arm_mirrors_the_right(self):
        pose = q(r_shoulder_pitch=-40.0, r_shoulder_roll=-25.0, r_arm_yaw=15.0)
        _, r_elbow, r_wrist, _ = link_frames(pose, side="right")
        mirrored = list(pose)
        mirrored[2] = -mirrored[2]                       # arm_yaw flips with y
        _, l_elbow, l_wrist, _ = link_frames(mirrored, side="left")
        assert l_elbow == pytest.approx((r_elbow[0], -r_elbow[1], r_elbow[2]))
        assert l_wrist == pytest.approx((r_wrist[0], -r_wrist[1], r_wrist[2]))


class TestLinkCapsules:
    def test_three_links_with_the_mjcf_collision_radii(self):
        caps = link_capsules(PRESENT)
        assert [c[0] for c in caps] == ["upper_arm", "forearm", "hand"]
        assert [c[3] for c in caps] == [
            _UPPER_ARM_RADIUS, _FOREARM_RADIUS, hand_radius(None)]

    def test_capsules_are_chained_end_to_end(self):
        caps = link_capsules(PRESENT)
        assert caps[0][2] == pytest.approx(caps[1][1])   # elbow
        assert caps[1][2] == pytest.approx(caps[2][1])   # wrist

    def test_hand_capsule_spans_the_gripper(self):
        _, _, hand = link_capsules(PRESENT)
        assert _dist(hand[1], hand[2]) == pytest.approx(_HAND_LEN)

    def test_the_elbow_sits_far_below_the_pad_at_the_old_present(self, planner):
        """The reason the pad-height check was worthless."""
        _, elbow, wrist, _ = link_frames(OLD_PRESENT)
        assert elbow[2] < 0.79            # 5 cm above a 0.740 tabletop
        assert wrist[2] > elbow[2]        # while the hand points up and away


class TestHandRadius:
    """The gripper's own width, which a fixed radius got wrong by 3.2 cm.

    The moving finger hangs off the thumb and swings outward as the hand opens,
    so a capsule sized for a closed hand under-models an open one — and the
    notebook hovers with the hand OPEN.
    """

    def test_opening_the_hand_widens_it(self):
        # Shut and neutral are both bounded by the fixed thumb shell, so they
        # tie; past that the swinging finger is what sets the width.
        assert hand_radius(20.0) <= hand_radius(0.0) < hand_radius(-45.0)
        assert hand_radius(-45.0) < hand_radius(-68.8)

    def test_the_working_open_position_exceeds_the_old_fixed_radius(self):
        """The bug, as an assertion: 5 cm did not bound an open hand."""
        assert hand_radius(-45.0) > 0.05
        assert hand_radius(-45.0) == pytest.approx(0.075, abs=0.002)

    def test_a_closed_hand_is_bounded_by_the_fixed_thumb_shell(self):
        assert hand_radius(20.0) == pytest.approx(hand_radius(0.0))

    def test_an_unknown_aperture_assumes_the_widest(self):
        """A guard that has to guess must guess wide."""
        assert hand_radius(None) == max(hand_radius(g)
                                        for g in (20.0, 0.0, -45.0, -68.8))

    def test_capsules_use_the_aperture_when_given(self):
        wide = link_capsules(PRESENT)[2][3]
        shut = link_capsules(PRESENT, gripper_deg=20.0)[2][3]
        assert shut < wide
        assert wide == pytest.approx(hand_radius(None))

    def test_a_wider_hand_can_only_reduce_clearance(self, scene):
        shut = scene.clearance(link_capsules(PRESENT, gripper_deg=20.0))
        wide = scene.clearance(link_capsules(PRESENT))
        assert wide.distance <= shut.distance


class TestObjectSdf:
    def _box(self, **kw):
        base = dict(id="b", kind="box", center=(0.0, 0.0, 0.0),
                    size=(0.2, 0.1, 0.4), dynamic=True, tracked=True)
        base.update(kw)
        return SceneObject(**base)

    def test_box_face_distance(self):
        o = self._box()
        assert object_sdf(o, (0.3, 0.0, 0.0)) == pytest.approx(0.2)
        assert object_sdf(o, (0.0, 0.0, 0.5)) == pytest.approx(0.3)

    def test_box_surface_is_zero(self):
        o = self._box()
        assert object_sdf(o, (0.1, 0.0, 0.0)) == pytest.approx(0.0)

    def test_box_interior_is_negative_and_measures_penetration(self):
        o = self._box()
        # 1 cm inside the +y face, which is the nearest one.
        assert object_sdf(o, (0.0, 0.04, 0.0)) == pytest.approx(-0.01)

    def test_box_corner_distance_is_euclidean_not_axis_wise(self):
        o = self._box()
        d = object_sdf(o, (0.1 + 0.03, 0.05 + 0.04, 0.0))
        assert d == pytest.approx(0.05)

    def test_yawed_box_is_measured_in_its_own_frame(self):
        upright = self._box(size=(0.2, 0.2, 0.2))
        yaw45 = (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))
        yawed = self._box(size=(0.2, 0.2, 0.2), quat=yaw45)
        # Turned 45 deg the cube points a corner at +x, so it reaches further.
        assert object_sdf(upright, (0.3, 0.0, 0.0)) == pytest.approx(0.2)
        assert object_sdf(yawed, (0.3, 0.0, 0.0)) == pytest.approx(
            0.3 - 0.1 * math.sqrt(2))

    def test_a_square_box_is_unchanged_by_a_quarter_turn(self):
        quarter = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
        turned = self._box(size=(0.2, 0.2, 0.2), quat=quarter)
        assert object_sdf(turned, (0.3, 0.0, 0.0)) == pytest.approx(0.2)

    def test_cylinder_uses_flat_ends_not_a_dome(self):
        """A capsule model would invent up to `radius` of extra height.

        On the scene's 6 cm cube that is 3 cm of imaginary object, which is the
        difference between an honest clearance report and a false collision.
        """
        o = SceneObject(id="c", kind="cylinder", center=(0.0, 0.0, 0.0),
                        size=(0.1, 0.1, 0.2), dynamic=True, tracked=True)
        assert object_sdf(o, (0.0, 0.0, 0.15)) == pytest.approx(0.05)
        assert object_sdf(o, (0.2, 0.0, 0.0)) == pytest.approx(0.15)


class TestSegmentObjectDistance:
    def _cube(self):
        return SceneObject(id="cube", kind="box", center=(0.0, 0.0, 0.0),
                           size=(0.1, 0.1, 0.1), dynamic=True, tracked=True)

    def test_parallel_segment_above_the_object(self):
        d, at = segment_object_distance(self._cube(), (-1.0, 0, 0.2), (1.0, 0, 0.2))
        assert d == pytest.approx(0.15)
        assert at[2] == pytest.approx(0.2)

    def test_the_minimum_can_lie_between_the_endpoints(self):
        """Endpoint sampling is exactly what the old check got wrong."""
        obj = self._cube()
        a, b = (-1.0, 0.0, 0.06), (1.0, 0.0, 0.06)
        interior, _ = segment_object_distance(obj, a, b)
        endpoints = min(object_sdf(obj, a), object_sdf(obj, b))
        assert interior == pytest.approx(0.01)
        assert endpoints > 0.9
        assert interior < endpoints

    def test_radius_is_subtracted(self):
        obj = self._cube()
        bare, _ = segment_object_distance(obj, (-1, 0, 0.2), (1, 0, 0.2))
        fat, _ = segment_object_distance(obj, (-1, 0, 0.2), (1, 0, 0.2), radius=0.05)
        assert fat == pytest.approx(bare - 0.05)

    def test_a_link_that_overlaps_reports_negative(self):
        d, _ = segment_object_distance(self._cube(), (-1, 0, 0.0), (1, 0, 0.0),
                                       radius=0.02)
        assert d < 0

    def test_degenerate_segment_reduces_to_a_point(self):
        obj = self._cube()
        d, at = segment_object_distance(obj, (0.3, 0.0, 0.0), (0.3, 0.0, 0.0))
        assert d == pytest.approx(object_sdf(obj, (0.3, 0.0, 0.0)))
        assert at == pytest.approx((0.3, 0.0, 0.0))


class TestJointPath:
    def test_includes_both_endpoints(self):
        path = joint_path(q(), PRESENT, steps=5)
        assert len(path) == 5
        assert path[0] == pytest.approx(q())
        assert path[-1] == pytest.approx(PRESENT)

    def test_interpolates_linearly(self):
        path = joint_path(q(r_elbow_pitch=0.0), q(r_elbow_pitch=-100.0), steps=3)
        assert path[1][R_ARM_JOINTS.index("r_elbow_pitch")] == pytest.approx(-50.0)


class TestSceneClearance:
    def test_manipulable_objects_are_obstacles_here_but_not_in_static(self, scene):
        """The gap that let the sweeps knock things over."""
        assert "red_cube" in scene.obstacle_ids()
        assert "red_cube" not in [o.id for o in scene.static_obstacles()]

    def test_old_present_pose_overlapped_the_red_cube(self, scene):
        c = scene.clearance(link_capsules(OLD_PRESENT))
        assert c.object_id == "red_cube"
        assert c.link == "forearm"
        assert c.distance < 0

    def test_new_present_pose_clears_every_object(self, scene):
        c = scene.clearance(link_capsules(PRESENT))
        assert c.distance > 0.10

    def test_per_object_report_covers_every_manipulable(self, scene):
        per = scene.clearances(link_capsules(PRESENT))
        assert set(per) == set(scene.manipulable_ids())

    def test_worst_matches_the_per_object_minimum(self, scene):
        per = scene.clearances(link_capsules(OLD_PRESENT))
        assert scene.clearance(link_capsules(OLD_PRESENT)).distance == pytest.approx(
            min(c.distance for c in per.values()))

    def test_including_statics_can_only_lower_the_clearance(self, scene):
        caps = link_capsules(PRESENT)
        assert (scene.clearance(caps, include_static=True).distance
                <= scene.clearance(caps).distance)

    def test_statics_bring_in_the_rails_but_not_the_tabletop(self, scene):
        statics = scene.obstacle_ids(include_static=True)
        assert "rig_rail_outer_right" in statics
        assert "table_top" not in statics
        assert "table_top" in scene.obstacle_ids(include_static=True,
                                                 include_table=True)

    def test_the_tabletop_is_a_support_surface_not_an_obstacle(self, scene):
        """Why it takes a separate opt-in.

        REST is a waypoint from the notebook's verified placement route, and
        the arm is resting ON the board there — so a guard that counts the
        tabletop rejects the route the rig was measured against.  It rejected
        all nine grid cells in section 4.7 before this was separated out.
        """
        rest = q(r_shoulder_pitch=-40.0, r_shoulder_roll=-10.0,
                 r_elbow_pitch=-45.0, r_wrist_pitch=-10.0, r_wrist_roll=30.0)
        caps = link_capsules(rest)
        assert scene.clearance(caps, ids=["table_top"]).distance < 0
        assert scene.clearance(caps, include_static=True).distance > 0.02

    # The placement route, with the gripper angle each waypoint actually
    # commands.  PLACE_ROUTE holds the hand SHUT for the whole trip through the
    # rig and only opens it at REST, which turns out to be load-bearing — see
    # the test below.
    ROUTE = [
        (q(), 20.0),                                                # HOME
        (q(r_shoulder_pitch=40.0), 20.0),                           # BACK
        (q(r_shoulder_pitch=40.0, r_elbow_pitch=-125.0,
           r_wrist_pitch=45.0), 20.0),                              # CURL
        (q(r_shoulder_pitch=70.0, r_elbow_pitch=-120.0,
           r_wrist_pitch=-45.0), 20.0),                             # TUCK
        (q(r_shoulder_pitch=37.5, r_shoulder_roll=-32.5,
           r_elbow_pitch=-120.0, r_wrist_pitch=-45.0), 20.0),       # SWING_1
        (q(r_shoulder_pitch=-17.5, r_shoulder_roll=-37.5,
           r_elbow_pitch=-120.0, r_wrist_pitch=-45.0), 20.0),       # SWING_3
        (q(r_shoulder_pitch=-40.0, r_shoulder_roll=-10.0,
           r_elbow_pitch=-60.0, r_wrist_pitch=-15.0), 20.0),        # HOVER
    ]

    def test_every_verified_waypoint_clears_the_rig_rails(self, scene):
        """The other half: the rails ARE a real no-go volume.

        Nothing on the placement route touches a rail, so unlike the tabletop
        they can be guarded without rejecting known-good poses.
        """
        rails = [o.id for o in scene.static_obstacles() if "rig-frame" in o.tags]
        for pose, grip in self.ROUTE:
            c = scene.clearance(link_capsules(pose, gripper_deg=grip), ids=rails)
            assert c.distance > 0.005, str(c)

    def test_the_swing_only_clears_the_rail_because_the_hand_is_shut(self, scene):
        """Why PLACE_ROUTE carries r_gripper=SHUT through the whole rig.

        It reads like tidiness and is not: at SWING_1 the shut hand passes the
        outer-right rail with 6 mm to spare, and an open one fouls it by 16 mm.
        The margin is the gripper's own width.
        """
        rails = [o.id for o in scene.static_obstacles() if "rig-frame" in o.tags]
        swing1 = q(r_shoulder_pitch=37.5, r_shoulder_roll=-32.5,
                   r_elbow_pitch=-120.0, r_wrist_pitch=-45.0)
        shut = scene.clearance(link_capsules(swing1, gripper_deg=20.0), ids=rails)
        opened = scene.clearance(link_capsules(swing1, gripper_deg=-45.0), ids=rails)
        assert shut.distance > 0
        assert opened.distance < 0
        assert shut.link == opened.link == "hand"

    def test_no_obstacles_reports_nothing(self, scene):
        assert scene.clearance(link_capsules(PRESENT), ids=[]) is None


class TestUpdatePoses:
    """The guard has to check where objects ARE, not where they were placed."""

    def test_moving_an_object_changes_the_clearance(self, scene):
        caps = link_capsules(PRESENT)
        before = scene.clearance(caps, ids=["red_cube"]).distance
        scene.update_poses({"red_cube": (0.279, -0.152, 0.90)})
        after = scene.clearance(caps, ids=["red_cube"]).distance
        assert after != pytest.approx(before)

    def test_reports_only_the_ids_it_actually_moved(self, scene):
        here = scene.get("red_cube").center
        assert scene.update_poses({"red_cube": here}) == []
        assert scene.update_poses({"red_cube": (0.3, -0.15, 0.77)}) == ["red_cube"]

    def test_unknown_ids_are_ignored(self, scene):
        """The live feed carries whatever the simulator tracks, not this scene."""
        assert scene.update_poses({"not_in_this_scene": (0, 0, 0)}) == []

    def test_derived_geometry_follows_the_new_centre(self, scene):
        scene.update_poses({"red_cube": (0.4, 0.1, 0.90)})
        cube = scene.get("red_cube")
        assert cube.top_z == pytest.approx(0.90 + cube.half_height)
        assert cube.center == pytest.approx((0.4, 0.1, 0.90))

    def test_a_stale_model_can_call_a_collision_clear(self, scene):
        """The failure this exists to prevent, on the real scene.

        Observed live: red_cube was dragged 7.9 cm off its cell by a routine
        that is not guarded, and the guard then went on measuring against the
        cell it had left.
        """
        caps = link_capsules(PRESENT)
        assert scene.clearance(caps, ids=["red_cube"]).distance > 0.10
        scene.update_poses({"red_cube": (0.29, -0.24, 0.90)})
        assert scene.clearance(caps, ids=["red_cube"]).distance < 0.10


class TestPathClearance:
    def test_endpoints_can_both_clear_while_the_move_does_not(self, planner):
        """The transit failure, on the real scene.

        Both ends of this move keep the arm off the cube; the middle does not.
        """
        a = q(r_shoulder_pitch=-70.0, r_shoulder_roll=-25.0, r_elbow_pitch=-80.0)
        b = q(r_shoulder_pitch=-70.0, r_shoulder_roll=-25.0, r_elbow_pitch=-80.0,
              r_arm_yaw=40.0)
        ends = min(planner.clearance(a).distance, planner.clearance(b).distance)
        assert planner.path_clearance(a, b).distance <= ends

    def test_a_move_into_an_object_is_caught(self, planner):
        stationary = planner.clearance(PRESENT).distance
        into = planner.path_clearance(PRESENT, OLD_PRESENT).distance
        assert stationary > 0.10
        assert into < 0

    def test_more_steps_never_report_a_larger_clearance(self, planner):
        coarse = planner.path_clearance(PRESENT, OLD_PRESENT, steps=3).distance
        fine = planner.path_clearance(PRESENT, OLD_PRESENT, steps=41).distance
        assert fine <= coarse + 1e-9


class TestSafeFraction:
    def test_a_clear_move_is_not_clipped(self, planner):
        target = dict(zip(R_ARM_JOINTS, PRESENT))
        target["r_wrist_roll"] = 45.0
        frac, _ = planner.safe_fraction(
            PRESENT, [target[j] for j in R_ARM_JOINTS], margin=0.03)
        assert frac == 1.0

    def test_a_move_into_the_cube_is_clipped_short(self, planner):
        frac, c = planner.safe_fraction(PRESENT, OLD_PRESENT, margin=0.03)
        assert 0.0 < frac < 1.0
        assert c.object_id == "red_cube"

    def test_the_clipped_pose_actually_clears_the_margin(self, planner):
        clipped, frac, _ = planner.clip(PRESENT, OLD_PRESENT, margin=0.03)
        assert planner.path_clearance(PRESENT, clipped).distance >= 0.03 - 1e-3
        assert frac < 1.0

    def test_a_tighter_margin_allows_more_of_the_move(self, planner):
        loose, _ = planner.safe_fraction(PRESENT, OLD_PRESENT, margin=0.01)
        tight, _ = planner.safe_fraction(PRESENT, OLD_PRESENT, margin=0.08)
        assert loose > tight

    def test_starting_inside_the_margin_yields_nothing_safe(self, planner):
        target = dict(zip(R_ARM_JOINTS, OLD_PRESENT))
        target["r_wrist_roll"] = 45.0
        frac, _ = planner.safe_fraction(
            OLD_PRESENT, [target[j] for j in R_ARM_JOINTS], margin=0.03)
        assert frac == 0.0

    def test_clip_of_a_clear_move_returns_the_target_unchanged(self, planner):
        target = dict(zip(R_ARM_JOINTS, PRESENT))
        target["r_wrist_pitch"] = -45.0
        want = [target[j] for j in R_ARM_JOINTS]
        clipped, frac, _ = planner.clip(PRESENT, want, margin=0.03)
        assert frac == 1.0
        assert clipped == pytest.approx(want)

    def test_without_a_scene_there_is_nothing_to_clip(self):
        bare = CartesianPlanner(arm=None, scene=None, side="right")
        assert bare.clearance(PRESENT) is None
        assert bare.safe_fraction(PRESENT, OLD_PRESENT)[0] == 1.0


class TestRoutineTwoSweepsAreSafe:
    """The whole point: every sweep Routine 2 performs must clear the table.

    Sweep ranges mirror notebooks/tlh_motion-routine.ipynb section 4.  The big
    joints sweep RELATIVE to the base pose, which is what keeps them safe when
    the base pose moves.
    """

    ABSOLUTE = [("r_wrist_pitch", (45.0, -45.0)),
                ("r_wrist_roll", (45.0, -45.0)),
                ("r_forearm_yaw", (90.0, -90.0)),
                ("r_arm_yaw", (-40.0, 40.0))]
    RELATIVE = [("r_shoulder_roll", (-18.0, 7.0)),
                ("r_elbow_pitch", (-15.0, 15.0)),
                ("r_shoulder_pitch", (-12.5, 12.5))]
    WAVES = [dict(r_forearm_yaw=-60.0, r_wrist_pitch=25.0, r_wrist_roll=-30.0),
             dict(r_forearm_yaw=60.0, r_wrist_pitch=-25.0, r_wrist_roll=30.0)]

    def _targets(self, base):
        bd = dict(zip(R_ARM_JOINTS, base))
        for joint, values in self.ABSOLUTE:
            for v in values:
                t = dict(bd, **{joint: v})
                yield f"{joint}={v:+.0f}", [t[j] for j in R_ARM_JOINTS]
        for joint, deltas in self.RELATIVE:
            for d in deltas:
                t = dict(bd, **{joint: bd[joint] + d})
                yield f"{joint}{d:+.1f}", [t[j] for j in R_ARM_JOINTS]
        for i, wave in enumerate(self.WAVES):
            t = dict(bd, **wave)
            yield f"wave{'AB'[i]}", [t[j] for j in R_ARM_JOINTS]

    def test_every_sweep_clears_the_objects_from_the_new_present(self, planner):
        for label, target in self._targets(PRESENT):
            c = planner.path_clearance(PRESENT, target)
            assert c.distance > 0.05, f"{label}: {c}"

    def test_every_sweep_also_clears_the_rig_and_the_table(self, planner):
        for label, target in self._targets(PRESENT):
            c = planner.path_clearance(PRESENT, target, include_static=True)
            assert c.distance > 0.05, f"{label}: {c}"

    def test_the_old_base_pose_could_not_have_been_saved_by_clipping(self, planner):
        """Why PRESENT had to move rather than the sweeps merely being clipped.

        red_cube's nearest surface is 0.320 m from the shoulder and the upper
        arm's own surface reaches 0.315 m, so the near-right cell sits inside
        the elbow's arc.  From the old pose the arm is already inside the
        margin, and no fraction of any sweep is safe.
        """
        for _, target in self._targets(OLD_PRESENT):
            frac, _ = planner.safe_fraction(OLD_PRESENT, target, margin=0.03)
            assert frac == 0.0


class TestGaze:
    """Head aiming — see motion/gaze.py for the two measured facts behind it."""

    def test_straight_ahead_is_level_but_for_the_two_offsets(self):
        from reachy_ai.motion.gaze import (BUILT_IN_PITCH_DEG, PITCH_BIAS_DEG,
                                           NECK_PIVOT_IN_WORLD, neck_angles_for)
        ahead = (NECK_PIVOT_IN_WORLD[0] + 1.0,
                 NECK_PIVOT_IN_WORLD[1], NECK_PIVOT_IN_WORLD[2])
        pitch, yaw = neck_angles_for(ahead)
        assert yaw == pytest.approx(0.0)
        assert pitch == pytest.approx(PITCH_BIAS_DEG - BUILT_IN_PITCH_DEG)

    def test_looking_at_the_table_pitches_down(self, scene):
        """Positive pitch is DOWN — the sign that had to be measured."""
        from reachy_ai.motion.gaze import neck_angles_for
        pitch, _ = neck_angles_for((0.43, 0.0, scene.table_surface_z))
        assert pitch > 25.0

    def test_lower_targets_need_more_pitch(self):
        from reachy_ai.motion.gaze import neck_angles_for
        high, _ = neck_angles_for((0.45, 0.0, 1.20))
        low, _ = neck_angles_for((0.45, 0.0, 0.74))
        assert low > high

    def test_yaw_follows_the_target_across_the_midline(self):
        from reachy_ai.motion.gaze import neck_angles_for
        _, right = neck_angles_for((0.40, -0.30, 0.90))   # robot's right
        _, centre = neck_angles_for((0.40, 0.0, 0.90))
        _, left = neck_angles_for((0.40, +0.30, 0.90))
        assert right < centre < left
        assert centre == pytest.approx(0.0)

    def test_the_raised_arm_pose_is_something_the_head_can_watch(self, planner):
        """The whole reason this module exists: PRESENT is off to the right."""
        from reachy_ai.motion.gaze import can_look_at, neck_angles_for
        _, _, wrist, _ = link_frames(PRESENT)
        target = (float(wrist[0]), float(wrist[1]), float(wrist[2]))
        assert can_look_at(target)
        pitch, yaw = neck_angles_for(target)
        assert yaw < -20.0            # it really is off to the right
        assert -45.0 <= pitch <= 64.0

    def test_clamping_keeps_an_impossible_target_inside_the_travel(self):
        from reachy_ai.motion.gaze import (PITCH_LIMITS, can_look_at,
                                           neck_angles_for)
        underneath = (0.02, 0.0, 0.0)          # essentially straight down
        assert not can_look_at(underneath)
        pitch, _ = neck_angles_for(underneath)
        assert pitch == pytest.approx(PITCH_LIMITS[1])


class TestClearanceMaximisingIK:
    """solve(maximise_clearance=True) spends the arm's redundancy on the table.

    These use a stub arm rather than the SDK: the point under test is which
    candidate gets chosen, not the IK itself.
    """

    class _StubArm:
        """Two 'orientations' that reach the same pad point very differently."""
        def __init__(self, solutions):
            self._solutions = solutions
            self._i = 0

        def inverse_kinematics(self, M, q0=None):
            sol = self._solutions[self._i % len(self._solutions)]
            self._i += 1
            return list(sol)

        def forward_kinematics(self, q):
            import numpy as np
            from reachy_ai.motion.kinematics import _BASE, _TOOL, link_frames
            _, _, wrist, R = link_frames(q)
            F = np.eye(4)
            F[:3, :3] = R
            F[:3, 3] = np.asarray(wrist) - _BASE
            return F

    def _planner_returning(self, scene, solutions):
        arm = self._StubArm(solutions)
        p = CartesianPlanner(arm, scene, side="right",
                             yaw_samples=len(solutions), yaw_range=1.0,
                             pitch_samples=(0.0,), tol=10.0)
        return p, arm

    def test_it_picks_the_roomier_of_two_reaching_solutions(self, scene):
        low = q(r_shoulder_pitch=-37.5, r_shoulder_roll=-2.0, r_elbow_pitch=-80.0)
        high = q(r_shoulder_pitch=-70.0, r_shoulder_roll=-25.0, r_elbow_pitch=-80.0)
        assert scene.clearance(link_capsules(low)).distance < 0
        assert scene.clearance(link_capsules(high)).distance > 0.10

        p, _ = self._planner_returning(scene, [low, high])
        chosen = p.solve((0.4, 0.0, 0.95), maximise_clearance=True)
        assert chosen == pytest.approx(high)

    def test_order_does_not_matter(self, scene):
        low = q(r_shoulder_pitch=-37.5, r_shoulder_roll=-2.0, r_elbow_pitch=-80.0)
        high = q(r_shoulder_pitch=-70.0, r_shoulder_roll=-25.0, r_elbow_pitch=-80.0)
        p, _ = self._planner_returning(scene, [high, low])
        assert p.solve((0.4, 0.0, 0.95), maximise_clearance=True) == pytest.approx(high)

    def test_it_tries_every_orientation_rather_than_stopping_at_the_first(self, scene):
        """The default solver breaks out at 1 mm of pad error; this must not."""
        low = q(r_shoulder_pitch=-37.5, r_shoulder_roll=-2.0, r_elbow_pitch=-80.0)
        high = q(r_shoulder_pitch=-70.0, r_shoulder_roll=-25.0, r_elbow_pitch=-80.0)
        p, arm = self._planner_returning(scene, [low, high])
        p.solve((0.4, 0.0, 0.95), maximise_clearance=True)
        assert arm._i == 2

    def test_scoring_the_path_can_beat_scoring_the_endpoint(self, scene):
        """from_joints scores the whole move, which is the honest question."""
        low = q(r_shoulder_pitch=-37.5, r_shoulder_roll=-2.0, r_elbow_pitch=-80.0)
        high = q(r_shoulder_pitch=-70.0, r_shoulder_roll=-25.0, r_elbow_pitch=-80.0)
        p, _ = self._planner_returning(scene, [low, high])
        chosen = p.solve((0.4, 0.0, 0.95), maximise_clearance=True,
                         from_joints=high)
        assert chosen == pytest.approx(high)

    def test_without_a_scene_it_falls_back_to_the_pad_error_rule(self):
        """Nothing to score against, so the flag is inert rather than fatal."""
        low = q(r_shoulder_pitch=-37.5, r_shoulder_roll=-2.0, r_elbow_pitch=-80.0)
        high = q(r_shoulder_pitch=-70.0, r_shoulder_roll=-25.0, r_elbow_pitch=-80.0)
        arm = self._StubArm([low, high])
        p = CartesianPlanner(arm, scene=None, side="right", yaw_samples=2,
                             pitch_samples=(0.0,), tol=10.0)
        target = (0.4, 0.0, 0.95)
        chosen = p.solve(target, maximise_clearance=True)
        # Scored by pad error, which is the default rule — and here that picks
        # the pose with the WORSE clearance, which is the whole problem.
        nearest = min((low, high),
                      key=lambda c: _dist(CartesianPlanner(
                          arm, None, "right").fk_world(c), target))
        assert chosen == pytest.approx(nearest)


class TestJointLimits:
    """The IK returned poses the arm cannot hold, and nothing checked.

    Measured on the running simulator before this was enforced — solved against
    reached, after the arm had settled:

        cell_r2c3   r_arm_yaw      +119.7  ->   +90.0   (stop is +90)
                    r_forearm_yaw  -136.0  ->  -101.5   (stop is -100)
        cell_r1c1   r_arm_yaw      +101.9  ->   +90.2
                    r_forearm_yaw  -137.6  ->  -102.1

    The arm clamps, sits 12-36 deg from the pose it was given, and the pad lands
    20 cm from the target — while every clearance check passes, because the
    guard is asked about the pose that was COMMANDED and the arm never went
    there.  A whole class of "it tracked badly" that was never tracking at all.
    """

    def test_the_limits_match_the_mjcf(self):
        import os
        import re
        from reachy_ai.motion.kinematics import JOINT_LIMITS_DEG
        path = os.path.join(_ROOT, "native_mujoco", "model", "reachy_1_2.xml")
        xml = open(path).read()
        for name, (lo, hi) in JOINT_LIMITS_DEG.items():
            m = re.search(r'<joint[^>]*name="%s"[^>]*>' % name, xml)
            assert m, f"{name} not in the MJCF"
            r = re.search(r'range="([-\d.]+)\s+([-\d.]+)"', m.group(0))
            assert r, f"{name} has no range"
            assert math.degrees(float(r.group(1))) == pytest.approx(lo, abs=0.5)
            assert math.degrees(float(r.group(2))) == pytest.approx(hi, abs=0.5)

    def test_a_pose_inside_the_travel_passes(self):
        from reachy_ai.motion.kinematics import within_limits
        assert within_limits(PRESENT)

    def test_the_arm_yaw_overshoot_is_rejected(self):
        """The exact value solve() returned for cell_r2c3."""
        from reachy_ai.motion.kinematics import within_limits
        assert not within_limits(q(r_arm_yaw=119.7))

    def test_the_forearm_yaw_overshoot_is_rejected(self):
        from reachy_ai.motion.kinematics import within_limits
        assert not within_limits(q(r_forearm_yaw=-136.0))

    def test_a_pose_exactly_on_a_stop_is_allowed(self):
        """Reachable in principle; rejecting it would refuse poses the arm can
        actually hold."""
        from reachy_ai.motion.kinematics import within_limits
        assert within_limits(q(r_arm_yaw=90.0))
        assert within_limits(q(r_forearm_yaw=-100.0))

    def test_the_left_arm_mirrors_roll_and_yaw(self):
        from reachy_ai.motion.kinematics import joint_limits
        right = dict(zip(R_ARM_JOINTS, joint_limits("right")))
        left = dict(zip(R_ARM_JOINTS, joint_limits("left")))
        # shoulder_roll is -180..+10 on the right, so +(-10)..+180 on the left
        assert left["r_shoulder_roll"] == (-10.0, 180.0)
        assert right["r_shoulder_roll"] == (-180.0, 10.0)
        # symmetric joints are unchanged by mirroring
        assert left["r_arm_yaw"] == right["r_arm_yaw"] == (-90.0, 90.0)
        # pitch axes are shared
        assert left["r_elbow_pitch"] == right["r_elbow_pitch"]

    def test_every_verified_waypoint_is_inside_the_travel(self, planner):
        """If a verified route needed an out-of-range pose, the limits would be
        wrong — this pins that they are not."""
        from reachy_ai.motion.kinematics import within_limits
        for pose in (PRESENT, OLD_PRESENT):
            assert within_limits(pose)
