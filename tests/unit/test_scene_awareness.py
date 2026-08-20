"""Unit tests for src/reachy_ai/scene/awareness.py (scene collision model).

All offline — no ROS, MuJoCo, or reachy_sdk required.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from reachy_ai.scene.awareness import SceneModel, CollisionViolation

_DEMO_SCENE = os.path.join(
    os.path.dirname(__file__), "..", "..", "scenes", "tabletop_demo.yaml"
)


@pytest.fixture
def scene():
    return SceneModel.from_yaml(_DEMO_SCENE)


class TestLoading:
    def test_frame_id(self, scene):
        assert scene.frame_id == "pedestal"

    def test_manipulable_objects(self, scene):
        assert scene.manipulable_ids() == ["red_cube", "blue_cylinder"]

    def test_table_detected(self, scene):
        assert scene.table is not None
        assert scene.table.id == "table_top"

    def test_table_surface_z(self, scene):
        # table_top centre z=0.72, box half-height 0.02 → surface 0.74
        assert scene.table_surface_z == pytest.approx(0.74, abs=1e-6)


class TestObjectGeometry:
    def test_cube_center_and_top(self, scene):
        cube = scene.get("red_cube")
        assert cube.center == pytest.approx((0.42, -0.18, 0.77))
        assert cube.top_z == pytest.approx(0.80)
        assert cube.bottom_z == pytest.approx(0.74)

    def test_cylinder_full_extents(self, scene):
        cyl = scene.get("blue_cylinder")
        # radius 0.035 → 0.07 diameter; length 0.10
        assert cyl.size == pytest.approx((0.07, 0.07, 0.10))
        assert cyl.top_z == pytest.approx(0.84)

    def test_hover_above_top(self, scene):
        hp = scene.hover_point("red_cube")
        assert hp[2] > scene.get("red_cube").top_z

    def test_grasp_is_center(self, scene):
        assert scene.grasp_point("red_cube") == pytest.approx((0.42, -0.18, 0.77))

    def test_carry_above_surface(self, scene):
        assert scene.carry_z() > scene.table_surface_z

    def test_rest_point_on_surface(self, scene):
        # cube half-height 0.03 → rest centre at 0.74 + 0.03 = 0.77
        rp = scene.rest_point((0.42, -0.02), "red_cube")
        assert rp == pytest.approx((0.42, -0.02, 0.77))


class TestCollisionAwareness:
    def test_point_below_table_flagged(self, scene):
        v = scene.check_point((0.42, -0.10, 0.70))
        assert v is not None
        assert v.kind == "below_table"
        assert v.obstacle_id == "table_top"

    def test_point_above_table_ok(self, scene):
        assert scene.check_point((0.42, -0.10, 0.86)) is None

    def test_point_beside_table_ok(self, scene):
        # Far outside table XY footprint, low z — not under the table
        assert scene.check_point((2.0, 2.0, 0.10)) is None

    def test_grasp_height_is_safe(self, scene):
        # Grasp/place descents go to object centre (above the surface)
        for oid in scene.manipulable_ids():
            assert scene.check_point(scene.grasp_point(oid)) is None

    def test_surface_margin(self, scene):
        # Just below the surface, within margin, is allowed
        assert scene.check_point((0.42, -0.10, 0.738), surface_margin=0.005) is None
        # Clearly below is flagged
        assert scene.check_point((0.42, -0.10, 0.70), surface_margin=0.005) is not None

    def test_validate_path_reports_all(self, scene):
        # Two points inside the table XY footprint but below the surface (z<0.74),
        # one point safely above — expects exactly 2 violations.
        path = [(0.42, -0.19, 0.47), (0.50, -0.15, 0.60), (0.42, -0.10, 0.86)]
        violations = scene.validate_path(path)
        assert len(violations) == 2
        assert all(isinstance(v, CollisionViolation) for v in violations)

    def test_validate_clean_path(self, scene):
        path = [(0.42, -0.18, 0.86), (0.42, -0.10, 0.87), (0.42, -0.02, 0.77)]
        assert scene.validate_path(path) == []
