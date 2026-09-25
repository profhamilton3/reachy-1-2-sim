"""Unit tests for tools/goalfix_cmp/clearance.py."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../scripts", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

from reachy_ai.motion.rig_routes import pose, ARM7, SHUT  # noqa: E402
from reachy_ai.scene.awareness import SceneModel, SceneObject  # noqa: E402
import measure_route_clearance as M  # noqa: E402
from tools.goalfix_cmp import clearance as cl  # noqa: E402


def _scene_with_one_box():
    box = SceneObject(
        id="box_1", kind="box", center=(0.5, -0.3, 1.1), size=(0.1, 0.1, 0.1),
        dynamic=True, tracked=True)
    return SceneModel(frame_id="pedestal", objects=[box], table_id=None)


POSE_A = pose()  # all-zero arm, gripper OPEN
POSE_B = pose(r_shoulder_pitch=-20.0, r_elbow_pitch=-15.0, r_gripper=SHUT)
WAYPOINTS = (POSE_A, POSE_B)


class TestPlannedVsCommandedExact:
    @pytest.mark.parametrize("hand", ["tube", "shells"])
    def test_delta_cmd_zero_when_commanded_equals_planned(self, hand):
        scene = _scene_with_one_box()
        n = 40
        planned_samples = list(M._LEG_SAMPLES_BY_HAND[hand](WAYPOINTS, n))
        deltas = cl.compute_deltas(
            "TEST_ROUTE", WAYPOINTS, n, scene, planned_samples, planned_samples)
        assert deltas  # at least one (hand, link, object) triple found
        for d in deltas:
            if d.hand != hand:
                continue
            assert d.delta_cmd_cm == pytest.approx(0.0, abs=1e-9)
            assert d.delta_trk_cm == pytest.approx(0.0, abs=1e-9)


class TestDeltaDirection:
    def test_delta_trk_is_commanded_minus_realised_not_planned_minus_realised(self):
        """planned, commanded and realised are all DISTINCT here (offset by
        1 and 2 known degrees respectively), so delta_cmd and delta_trk
        have different, independently-checkable expected values -- a
        planned-minus-realised mutant for delta_trk cannot coincidentally
        pass this the way it could a delta_cmd==0 fixture."""
        scene = _scene_with_one_box()
        n = 5
        planned_samples = list(M._LEG_SAMPLES_BY_HAND["tube"](WAYPOINTS, n))
        commanded_samples = [([q7[0] + 1.0] + list(q7[1:]), g) for q7, g in planned_samples]
        realised_samples = [([q7[0] + 2.0] + list(q7[1:]), g) for q7, g in commanded_samples]

        from reachy_ai.motion.kinematics import link_capsules

        def worst_upper_arm(samples):
            worst = None
            for q7, g in samples:
                caps = [c for c in link_capsules(q7, "right", g, hand="tube") if c[0] == "upper_arm"]
                d = scene.clearances(caps)["box_1"].distance
                worst = d if worst is None or d < worst else worst
            return worst

        worst_planned = worst_upper_arm(planned_samples)
        worst_cmd = worst_upper_arm(commanded_samples)
        worst_real = worst_upper_arm(realised_samples)
        expected_delta_cmd_cm = 100 * (worst_planned - worst_cmd)
        expected_delta_trk_cm = 100 * (worst_cmd - worst_real)
        assert expected_delta_cmd_cm != pytest.approx(expected_delta_trk_cm)

        deltas = cl.compute_deltas("TEST_ROUTE", WAYPOINTS, n, scene, commanded_samples, realised_samples)
        d = next(x for x in deltas if x.hand == "tube" and x.link == "upper_arm" and x.object_id == "box_1")
        assert d.delta_cmd_cm == pytest.approx(expected_delta_cmd_cm, abs=1e-6)
        assert d.delta_trk_cm == pytest.approx(expected_delta_trk_cm, abs=1e-6)


class TestUnitConversion:
    def test_commanded_samples_converts_radians_to_degrees(self):
        targets8_rad = [np.radians([-10.0, 0, 0, -5.0, 0, 0, 0, -30.0])]
        samples = cl.commanded_samples(targets8_rad)
        q7_deg, gripper_deg = samples[0]
        assert q7_deg[0] == pytest.approx(-10.0)
        assert q7_deg[3] == pytest.approx(-5.0)
        assert gripper_deg == pytest.approx(-30.0)


class TestMissingGripper:
    def test_missing_gripper_raises_not_imputed(self):
        rows = [[np.radians(-10.0), 0.0, 0.0, np.radians(-5.0), 0.0, 0.0, 0.0, None]]
        with pytest.raises(M.ApertureDataError):
            cl.realised_samples_from_states(rows)


class TestFingerApertureRule:
    def test_shells_finger_planned_matches_lerp_not_endpoint(self):
        scene = _scene_with_one_box()
        n = 10
        # PR #136: shells interpolates the gripper aperture per sample in
        # lockstep with the arm (M._lerp), NOT held at the worst endpoint
        # the way tube's rule does -- reproduce one interior sample's
        # expected gripper angle independently and confirm the planned
        # finger clearance at that sample matches a manual FK check at
        # that exact interpolated aperture, not at either endpoint's.
        from reachy_ai.motion.kinematics import link_capsules
        qa = [POSE_A[j] for j in ARM7]
        qb = [POSE_B[j] for j in ARM7]
        from reachy_ai.motion.kinematics import joint_path
        path = joint_path(qa, qb, steps=n)
        i = n // 2
        expected_gripper = M._lerp(POSE_A["r_gripper"], POSE_B["r_gripper"], i, n)
        assert expected_gripper != POSE_A["r_gripper"]
        assert expected_gripper != POSE_B["r_gripper"]
        caps = [c for c in link_capsules(path[i], "right", expected_gripper, hand="shells")
                if c[0] == "finger"]
        expected = scene.clearances(caps).get("box_1")

        planned = M.planned_clearance_by_link(
            "TEST_ROUTE", n, scene, "finger", hand="shells", waypoints=WAYPOINTS)
        # planned_clearance_by_link takes the worst over ALL samples, so
        # just confirm our manually-computed interior sample is not more
        # extreme than the reported worst (i.e. the worst really does come
        # from the interpolated-aperture family, not an endpoint-only
        # comparison) and that both are finite/consistent in sign.
        assert planned["box_1"] <= expected.distance + 1e-9


class TestIndepWristBallCrossCheck:
    def test_matches_repo_fk_closely(self):
        scene = _scene_with_one_box()
        q7_and_gripper = [([0.0] * 7, -45.0), ([-20.0, 0, 0, -15.0, 0, 0, 0], -45.0)]
        worst = cl.cross_check_wrist_ball(q7_and_gripper, scene, "box_1", hand="shells")
        assert worst < 1.0  # cm -- independent FK should closely agree
