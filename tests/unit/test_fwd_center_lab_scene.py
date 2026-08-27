"""Unit tests for scenes/FWDCenterLabMCC.yaml — the measured FWD Center lab setup.

Locks the geometry to the physical measurements in docs/ActualLabSetupNotes.txt
so a later edit cannot silently drift away from the real table.

All offline — no ROS, MuJoCo, or reachy_sdk required.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from reachy_ai.scene.awareness import SceneModel

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_SCENE_PATH = os.path.join(_ROOT, "scenes", "FWDCenterLabMCC.yaml")

IN = 0.0254  # metres per inch


@pytest.fixture
def scene():
    return SceneModel.from_yaml(_SCENE_PATH)


@pytest.fixture
def doc():
    with open(_SCENE_PATH) as f:
        return yaml.safe_load(f)


class TestSchema:
    def test_validates_against_scene_schema(self, doc):
        jsonschema = pytest.importorskip("jsonschema")
        import json
        with open(os.path.join(_ROOT, "scenes", "scene.schema.json")) as f:
            jsonschema.validate(doc, json.load(f))

    def test_object_ids_unique(self, doc):
        ids = [o["id"] for o in doc["objects"]]
        assert len(ids) == len(set(ids))

    def test_frame_and_name(self, doc):
        assert doc["name"] == "FWDCenterLabMCC"
        assert doc["frame_id"] == "pedestal"


class TestMeasuredTable:
    """~24.8 x 27.5 in board, surface at z = 0.74.

    Sizes come from rectifying Siva's phone photos through the known 5 in
    cells, cross-checked against the head-camera calibration; both put the far
    edge at 12.4-12.5 in from grid centre.
    """

    def test_board_far_edge_is_the_measured_one(self, scene):
        """Grid centre + 12.4 in, agreed by head-camera and phone rectification."""
        far = scene.table.center[0] + scene.table.size[0] / 2.0
        assert far == pytest.approx(0.7468, abs=1e-3)

    def test_board_near_edge_clears_the_frame_and_carries_the_pattern(self, scene):
        """Not measurable — bounded between the rig frame and the tape."""
        near = scene.table.center[0] - scene.table.size[0] / 2.0
        frame_front = (scene.get("rig_rail_front").center[0]
                       + scene.get("rig_rail_front").size[0] / 2.0)
        pattern_near = (scene.get("grid_border_near").center[0]
                        - scene.get("grid_border_near").size[0] / 2.0)
        assert frame_front < near < pattern_near

    def test_board_width_matches_the_measured_edges(self, scene):
        # +/-13.75 in from grid centre along y
        assert scene.table.size[1] == pytest.approx(27.5 * IN, abs=2e-4)

    def test_board_is_bigger_than_the_pattern(self, scene):
        """A 19 in pattern cannot sit on the 18 in table the notes describe."""
        assert scene.table.size[0] > 19 * IN
        assert scene.table.size[1] > 19 * IN

    def test_table_is_centred_on_the_midline(self, scene):
        assert scene.table.center[1] == pytest.approx(0.0, abs=1e-9)

    def test_surface_height(self, scene):
        assert scene.table_surface_z == pytest.approx(0.74, abs=1e-9)

    def test_pattern_near_edge_is_8_inches_from_pedestal_axis(self, scene):
        """The notes' '8 inches' is the robot to the taped pattern, not the board."""
        border = scene.get("grid_border_near")
        outer = border.center[0] - border.size[0] / 2.0
        assert outer == pytest.approx(7.5 * IN, abs=1e-6)


class TestGrid:
    def test_nine_addressable_cells(self, scene):
        assert scene.grid_cells() == [
            "cell_r1c1", "cell_r1c2", "cell_r1c3",
            "cell_r2c1", "cell_r2c2", "cell_r2c3",
            "cell_r3c1", "cell_r3c2", "cell_r3c3",
        ]

    def test_cells_are_5_inch_squares(self, scene):
        for cid in scene.grid_cells():
            sx, sy, _ = scene.get(cid).size
            assert sx == pytest.approx(5 * IN, abs=1e-6)
            assert sy == pytest.approx(5 * IN, abs=1e-6)

    def test_cell_pitch_is_6_inches(self, scene):
        """5 in cell + 1 in interior tape border."""
        xs = sorted({round(scene.cell_center(c)[0], 6) for c in scene.grid_cells()})
        ys = sorted({round(scene.cell_center(c)[1], 6) for c in scene.grid_cells()})
        assert len(xs) == 3 and len(ys) == 3
        for axis in (xs, ys):
            assert axis[1] - axis[0] == pytest.approx(6 * IN, abs=1e-6)
            assert axis[2] - axis[1] == pytest.approx(6 * IN, abs=1e-6)

    def test_pattern_spans_19_inches_and_fits_the_board(self, scene):
        """3 x 5 in cells + 4 x 1 in borders = 19 in, measured at 18.90 in."""
        for axis, half in ((0, 9.5 * IN), (1, 9.5 * IN)):
            lo = min(scene.get(b).center[axis] - scene.get(b).size[axis] / 2.0
                     for b in ("grid_border_near", "grid_border_far",
                               "grid_border_left", "grid_border_right"))
            hi = max(scene.get(b).center[axis] + scene.get(b).size[axis] / 2.0
                     for b in ("grid_border_near", "grid_border_far",
                               "grid_border_left", "grid_border_right"))
            assert hi - lo == pytest.approx(19 * IN, abs=1e-6)
            assert hi - lo < scene.table.size[axis]

    def test_outer_border_exists_on_all_four_sides(self, scene):
        for b in ("grid_border_near", "grid_border_far",
                  "grid_border_left", "grid_border_right"):
            o = scene.get(b)
            assert "marking" in o.tags
            assert not o.collides
            assert min(o.size[0], o.size[1]) == pytest.approx(1 * IN, abs=1e-6)

    def test_outer_border_abuts_the_edge_cells(self, scene):
        """Border centre sits 3 in out: 2.5 in half-cell + 0.5 in half-border."""
        near_cell = scene.cell_center("cell_r1c2")[0]
        border = scene.get("grid_border_near").center[0]
        assert near_cell - border == pytest.approx(3 * IN, abs=1e-6)

    def test_grid_is_centred_laterally_but_sits_toward_the_robot(self, scene):
        """Centred in y; offset toward the robot in x because the board's near
        edge is limited by the rig frame."""
        xs = [scene.cell_center(c)[0] for c in scene.grid_cells()]
        ys = [scene.cell_center(c)[1] for c in scene.grid_cells()]
        assert (max(ys) + min(ys)) / 2 == pytest.approx(scene.table.center[1], abs=1e-6)
        grid_x = (max(xs) + min(xs)) / 2
        assert grid_x < scene.table.center[0]
        assert scene.table.center[0] - grid_x == pytest.approx(0.0216, abs=1e-3)

    def test_row_1_is_nearest_the_robot(self, scene):
        assert scene.cell_center("cell_r1c2")[0] < scene.cell_center("cell_r3c2")[0]

    def test_col_1_is_the_robots_left(self, scene):
        """+y is the robot's left."""
        assert scene.cell_center("cell_r2c1")[1] > scene.cell_center("cell_r2c3")[1]

    def test_cell_center_returns_the_table_surface(self, scene):
        for cid in scene.grid_cells():
            assert scene.cell_center(cid)[2] == pytest.approx(scene.table_surface_z)

    def test_cell_center_rejects_non_cells(self, scene):
        with pytest.raises(KeyError):
            scene.cell_center("table_top")


class TestMarkingsAreNotObstacles:
    """Tape and cell decals must never block a descend or place motion."""

    def test_static_obstacles_are_the_table_and_the_rig_frame_only(self, scene):
        """No tape strip or grid cell may register as an obstacle."""
        ids = sorted(o.id for o in scene.static_obstacles())
        assert ids == sorted(
            ["table_top"] + [o.id for o in scene.objects.values()
                             if "rig-frame" in o.tags]
        )
        assert not any("marking" in o.tags or "grid-cell" in o.tags
                       for o in scene.static_obstacles())

    def test_markings_are_non_colliding(self, scene):
        for o in scene.objects.values():
            if "marking" in o.tags or "grid-cell" in o.tags:
                assert not o.collides

    def test_descend_into_a_cell_is_legal(self, scene):
        for cid in scene.grid_cells():
            x, y, z = scene.cell_center(cid)
            assert scene.check_point((x, y, z + 0.001)) is None

    def test_below_the_table_is_still_rejected(self, scene):
        x, y, _ = scene.cell_center("cell_r2c2")
        v = scene.check_point((x, y, 0.70))
        assert v is not None and v.kind == "below_table"


class TestRigFrame:
    """The 80/20 ladder Reachy's mast is bolted to, with the arm openings."""

    RAILS = ("rig_rail_inner_right", "rig_rail_inner_left",
             "rig_rail_outer_right", "rig_rail_outer_left",
             "rig_rail_front", "rig_rail_back")

    def _span(self, scene, oid, axis):
        o = scene.get(oid)
        return o.center[axis] - o.size[axis] / 2, o.center[axis] + o.size[axis] / 2

    def test_all_six_rails_present(self, scene):
        for r in self.RAILS:
            assert "rig-frame" in scene.get(r).tags

    def test_opening_is_9_lateral_by_19_fore_aft(self, scene):
        """The short (9 in) edge faces the table; the long edge runs back."""
        fore_aft = (self._span(scene, "rig_rail_front", 0)[0]
                    - self._span(scene, "rig_rail_back", 0)[1])
        lateral = (self._span(scene, "rig_rail_inner_right", 1)[0]
                   - self._span(scene, "rig_rail_outer_right", 1)[1])
        assert lateral == pytest.approx(9 * IN, abs=2e-4)
        assert fore_aft == pytest.approx(19 * IN, abs=2e-4)

    def test_frame_does_not_overhang_the_board_sideways(self, scene):
        """The rig photos show the rails inside the board's width, not past it.

        This is what fixes the opening's orientation: two 19 in lateral
        openings would make the frame 47 in wide against a 27.5 in board.
        """
        width = (max(self._span(scene, r, 1)[1] for r in self.RAILS)
                 - min(self._span(scene, r, 1)[0] for r in self.RAILS))
        assert width <= scene.table.size[1]

    def test_frame_extends_backward_not_forward(self, scene):
        """It is not centred on the mast — it runs back behind the robot."""
        front = max(self._span(scene, r, 0)[1] for r in self.RAILS)
        back = min(self._span(scene, r, 0)[0] for r in self.RAILS)
        assert front < 0.2
        assert back < -0.35
        assert abs(back) > abs(front)

    def test_arm_rests_at_the_centre_of_the_opening(self, scene):
        lo, hi = (self._span(scene, "rig_rail_outer_right", 1)[1],
                  self._span(scene, "rig_rail_inner_right", 1)[0])
        assert (lo + hi) / 2 == pytest.approx(-0.19, abs=1e-3)

    def test_openings_are_symmetric(self, scene):
        right = self._span(scene, "rig_rail_outer_right", 1)[1]
        left = self._span(scene, "rig_rail_outer_left", 1)[0]
        assert right == pytest.approx(-left, abs=1e-6)

    def test_rail_tops_are_flush_with_the_tabletop(self, scene):
        for r in self.RAILS:
            assert self._span(scene, r, 2)[1] == pytest.approx(
                scene.table_surface_z, abs=1e-6)

    def test_frame_does_not_intersect_the_board(self, scene):
        frame_front = max(self._span(scene, r, 0)[1] for r in self.RAILS)
        board_near = self._span(scene, "table_top", 0)[0]
        assert frame_front < board_near

    def test_rest_pose_arm_passes_through_the_right_opening(self, scene):
        """The right arm hangs at (0, -0.19); it must clear the rails."""
        x, y = 0.0, -0.19
        ox = (self._span(scene, "rig_rail_back", 0)[1],
              self._span(scene, "rig_rail_front", 0)[0])
        oy = (self._span(scene, "rig_rail_outer_right", 1)[1],
              self._span(scene, "rig_rail_inner_right", 1)[0])
        assert ox[0] < x < ox[1]
        assert oy[0] < y < oy[1]
        # "arms easily fit" — want real clearance, not a squeak-through
        assert min(y - oy[0], oy[1] - y) > 0.05

    def test_frame_is_on_the_fixture_collision_channel(self, doc):
        """Fixture collides with robot links but not the pedestal or tabletop."""
        for o in doc["objects"]:
            if "rig-frame" in (o.get("tags") or []):
                assert o["physics"]["collision"] == "fixture"

    def test_frame_is_a_planning_obstacle(self, scene):
        ids = [o.id for o in scene.static_obstacles()]
        for r in self.RAILS:
            assert r in ids


class TestEmptyTable:
    """The scene ships with no manipulation objects (added in a later slice)."""

    def test_no_manipulable_objects(self, scene):
        assert scene.manipulable_ids() == []

    def test_no_dynamic_objects(self, doc):
        assert not [
            o for o in doc["objects"] if (o.get("physics") or {}).get("dynamic")
        ]
