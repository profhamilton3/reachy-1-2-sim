"""Unit tests for the synthetic dataset generator (native_mujoco/dataset.py).

Offline: exercises the label derivation and YOLO conversion, which is where the
correctness risk lives.  Rendering itself needs a real GL context and is
verified by running native_mujoco/cli/generate_dataset.py under mjpython.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "native_mujoco"))

from dataset import (
    _ZOOM_SAMPLE_LEVELS,
    _ZOOM_SAMPLE_WEIGHTS,
    BoxLabel,
    boxes_from_segmentation,
)
from zoom import ZoomLevel

BODIES = {"red_cube": 48, "blue_cylinder": 49}
CLASSES = {"red_cube": "block.red", "blue_cylinder": "cylinder.blue"}
INDEX = {"block.red": 0, "cylinder.blue": 1}


def _seg(h=48, w=64):
    return np.zeros((h, w), dtype=np.uint16)


class TestBoxDerivation:
    def test_box_is_the_exact_pixel_extent(self):
        """Boxes are exact by construction — that is the whole reason to render
        rather than annotate, so an off-by-one here defeats the purpose."""
        seg = _seg()
        seg[10:21, 30:41] = 48          # rows 10..20, cols 30..40
        b = boxes_from_segmentation(seg, BODIES, CLASSES, INDEX)[0]
        assert (b.x_min, b.y_min, b.x_max, b.y_max) == (30, 10, 40, 20)
        assert b.width == 11 and b.height == 11
        assert b.pixel_count == 121

    def test_absent_object_yields_no_box(self):
        """An empty label file is a valid negative; inventing a box for an
        off-screen object would be worse than having none."""
        assert boxes_from_segmentation(_seg(), BODIES, CLASSES, INDEX) == []

    def test_two_objects_yield_two_boxes(self):
        seg = _seg()
        seg[5:15, 5:15] = 48
        seg[20:30, 40:50] = 49
        got = boxes_from_segmentation(seg, BODIES, CLASSES, INDEX)
        assert {b.object_id for b in got} == {"red_cube", "blue_cylinder"}

    def test_tiny_sliver_is_dropped(self):
        """A handful of pixels is a mostly-occluded object, not a sample."""
        seg = _seg()
        seg[10:12, 10:12] = 48          # 4 px, under the floor
        assert boxes_from_segmentation(seg, BODIES, CLASSES, INDEX) == []

    def test_thin_strip_is_dropped_even_when_large(self):
        """An object clipped to a 2px strip at the frame edge has enough pixels
        but no usable shape."""
        seg = _seg()
        seg[0:2, 0:60] = 48             # 120 px but only 2 rows tall
        assert boxes_from_segmentation(seg, BODIES, CLASSES, INDEX) == []

    def test_disjoint_regions_share_one_box(self):
        """A partly-occluded object still gets one box spanning both parts,
        which is what a detector is trained to predict."""
        seg = _seg()
        seg[10:20, 5:15] = 48
        seg[10:20, 40:50] = 48
        b = boxes_from_segmentation(seg, BODIES, CLASSES, INDEX)[0]
        assert b.x_min == 5 and b.x_max == 49

    def test_class_index_comes_from_the_mapping(self):
        seg = _seg()
        seg[10:25, 10:25] = 49
        b = boxes_from_segmentation(seg, BODIES, CLASSES, INDEX)[0]
        assert b.semantic_class == "cylinder.blue"
        assert b.class_index == 1

    def test_all_background_frame_labels_nothing(self):
        """Body 0 is world/background, and also where MuJoCo's -1 lands.  A
        frame of pure background must produce no labels at all."""
        assert boxes_from_segmentation(_seg(), BODIES, CLASSES, INDEX) == []


class TestYoloConversion:
    def test_centre_and_size_are_normalised(self):
        # abs tolerance, not relative: to_yolo formats to 6 decimals, so an
        # exact ratio like 15.5/48 cannot round-trip more precisely than that.
        b = BoxLabel("o", "c", 3, x_min=30, y_min=10, x_max=40, y_max=20,
                     pixel_count=121)
        parts = b.to_yolo(64, 48).split()
        assert parts[0] == "3"
        assert float(parts[1]) == pytest.approx(35.5 / 64, abs=1e-6)
        assert float(parts[2]) == pytest.approx(15.5 / 48, abs=1e-6)
        assert float(parts[3]) == pytest.approx(11 / 64, abs=1e-6)
        assert float(parts[4]) == pytest.approx(11 / 48, abs=1e-6)

    def test_all_values_stay_in_unit_range(self):
        b = BoxLabel("o", "c", 0, 0, 0, 63, 47, 3072)
        vals = [float(v) for v in b.to_yolo(64, 48).split()[1:]]
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_full_frame_box_is_centred_and_full_size(self):
        b = BoxLabel("o", "c", 0, 0, 0, 63, 47, 3072)
        _, cx, cy, w, h = b.to_yolo(64, 48).split()
        assert float(cx) == pytest.approx(0.5)
        assert float(cy) == pytest.approx(0.5)
        assert float(w) == pytest.approx(1.0)
        assert float(h) == pytest.approx(1.0)


class TestZoomSampling:
    def test_zero_is_excluded_from_sampling(self):
        """ZoomLevel.ZERO renders identically to INTER, so including it put
        half of every dataset at one field of view by accident."""
        assert ZoomLevel.ZERO not in _ZOOM_SAMPLE_LEVELS

    def test_every_other_level_is_sampled(self):
        assert set(_ZOOM_SAMPLE_LEVELS) == {
            ZoomLevel.IN, ZoomLevel.INTER, ZoomLevel.OUT}

    def test_weights_are_a_distribution(self):
        assert len(_ZOOM_SAMPLE_WEIGHTS) == len(_ZOOM_SAMPLE_LEVELS)
        assert sum(_ZOOM_SAMPLE_WEIGHTS) == pytest.approx(1.0)

    def test_calibrated_level_is_weighted_highest(self):
        """INTER is the only level with a measured barrel profile."""
        w = dict(zip(_ZOOM_SAMPLE_LEVELS, _ZOOM_SAMPLE_WEIGHTS))
        assert w[ZoomLevel.INTER] > w[ZoomLevel.IN]
        assert w[ZoomLevel.INTER] > w[ZoomLevel.OUT]


class TestSplitMasks:
    """A label is min/max over an object's pixels, which is only a box when
    those pixels all belong to the object.

    The wide zoom's barrel warp folds the source render's extreme corners back
    into frame, so a sliver of an object reappears far from the object.  On the
    300-frame run that produced the first handoff set this hit 10 of 1177
    boxes — all at OUT, 3.5% of that zoom — the worst spanning 198x217 px for
    1317 px of object.  A detector trained on those learns that the class looks
    like empty board.
    """

    def test_a_distant_fold_does_not_stretch_the_box(self):
        seg = _seg()
        seg[10:24, 10:24] = 48          # the object
        seg[44:47, 60:63] = 48          # its fold, far corner, same body id
        b = boxes_from_segmentation(seg, BODIES, CLASSES, INDEX)[0]
        assert (b.x_min, b.y_min, b.x_max, b.y_max) == (10, 10, 23, 23)
        assert b.pixel_count == 14 * 14

    def test_a_solid_object_is_untouched(self):
        seg = _seg()
        seg[10:30, 10:30] = 48
        b = boxes_from_segmentation(seg, BODIES, CLASSES, INDEX)[0]
        assert (b.x_min, b.y_min, b.x_max, b.y_max) == (10, 10, 29, 29)
        assert b.pixel_count == 400

    def test_a_concave_but_connected_shape_survives_whole(self):
        """Fill ratio alone would condemn this; connectivity is the real test.
        An L is one object and its box legitimately covers both arms."""
        seg = _seg()
        seg[10:30, 10:16] = 48
        seg[24:30, 10:30] = 48
        b = boxes_from_segmentation(seg, BODIES, CLASSES, INDEX)[0]
        assert (b.x_min, b.y_min, b.x_max, b.y_max) == (10, 10, 29, 29)

    def test_comparable_halves_are_both_kept(self):
        """The rule is relative size, not connectivity.  An arm across the
        object leaves two halves of similar size, and one box spanning both is
        what a detector should predict — see
        TestBoxDerivation.test_disjoint_regions_share_one_box, which this must
        not break."""
        seg = _seg()
        seg[10:20, 5:15] = 48           # 100 px
        seg[10:20, 40:50] = 48          # 100 px
        b = boxes_from_segmentation(seg, BODIES, CLASSES, INDEX)[0]
        assert (b.x_min, b.x_max) == (5, 49)
        assert b.pixel_count == 200

    def test_the_line_sits_between_the_two_populations(self):
        """Measured folds ran 3-18% of the object's pixels; halves of a split
        object are comparable.  A blob at 20% goes, one at 30% stays."""
        for share, kept in ((0.20, False), (0.30, True)):
            seg = _seg(h=120, w=120)   # room for the satellite to fit whole;
            seg[10:30, 10:30] = 48     # a clipped one would test the clipping
            side = int(round((400 * share) ** 0.5))
            seg[80:80 + side, 90:90 + side] = 48
            b = boxes_from_segmentation(seg, BODIES, CLASSES, INDEX)[0]
            reaches = b.y_max >= 80
            assert reaches is kept, f"share {share}: expected kept={kept}"

    def test_a_fold_that_leaves_too_little_behind_is_dropped(self):
        """Trimming to the main blob can take an object under the minimum.
        It must then be dropped, not emitted as a sliver."""
        seg = _seg()
        seg[10:17, 10:17] = 48          # 49 px, under _MIN_BOX_PIXELS (60)
        seg[40:43, 55:58] = 48          # 9 px fold (18%); 58 px total is over
                                        # the minimum only because of the fold
        assert boxes_from_segmentation(seg, BODIES, CLASSES, INDEX) == []
