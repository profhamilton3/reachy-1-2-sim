"""R12-504: control-panel scene-awareness, schema, and left-arm mirror tests."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from reachy_ai.motion import primitives as P
from reachy_ai.motion.kinematics import (
    L_ARM_JOINTS,
    R_ARM_JOINTS,
    _R0,
    _R0_LEFT,
)
from reachy_ai.scene.awareness import SceneModel

_REPO = os.path.join(os.path.dirname(__file__), "../..")
_PANEL = os.path.join(_REPO, "scenes", "control_panel.yaml")


@pytest.fixture(scope="module")
def panel():
    return SceneModel.from_yaml(_PANEL)


class TestPanelSchema:
    def test_scene_validates(self):
        """The panel scene (articulation + interactive fields) passes the loader."""
        import sys
        sys.path.insert(0, _REPO)
        from scene_loader import load_scene
        doc = load_scene(_PANEL)
        assert len(doc.objects) == 13


class TestControlQueries:
    def test_controls_discovered(self, panel):
        ids = [c.id for c in panel.controls()]
        assert "btn_red" in ids and "lever_r" in ids and "switch_l" in ids
        assert len(ids) == 10

    def test_split_by_arm(self, panel):
        right = panel.controls_for_arm("right")
        left = panel.controls_for_arm("left")
        assert all(panel.get(c).center[1] <= 0.03 for c in right)
        assert all(panel.get(c).center[1] >= -0.03 for c in left)
        # every control is claimed by at least one arm
        assert set(right) | set(left) == {c.id for c in panel.controls()}

    def test_panel_obstacles_are_fixtures_not_controls(self, panel):
        obs = {o.id for o in panel.panel_obstacles()}
        assert "console_base" in obs
        assert not (obs & {c.id for c in panel.controls()})

    def test_button_press_direction(self, panel):
        t = panel.control_target("btn_red")     # flat-top button
        assert t.control_type == "button"
        # presses straight down (-z)
        assert t.actuate_dir[2] < -0.9

    def test_hinge_handle_tip_above_pivot(self, panel):
        t = panel.control_target("lever_r")
        pivot = panel.get("lever_r").center
        assert t.point[2] > pivot[2]            # tip is offset up from the pivot
        assert t.preferred_arm == "right"


class TestLeftArmMirror:
    def test_mirror_pose_flips_roll_and_gripper(self):
        r = {"r_shoulder_pitch": -25, "r_shoulder_roll": -88, "r_gripper": -45}
        l = P.mirror_pose(r)
        assert l["l_shoulder_pitch"] == -25       # pitch unchanged
        assert l["l_shoulder_roll"] == 88         # roll flipped
        assert l["l_gripper"] == 45               # gripper sign inverted

    def test_left_joint_names(self):
        assert L_ARM_JOINTS[0] == "l_shoulder_pitch"
        assert len(L_ARM_JOINTS) == len(R_ARM_JOINTS) == 7

    def test_mirror_orientation_is_proper_rotation(self):
        import numpy as np
        # S R S has the same determinant as R (mirror of a rotation).
        assert np.isclose(np.linalg.det(_R0_LEFT), np.linalg.det(_R0), atol=1e-6)
