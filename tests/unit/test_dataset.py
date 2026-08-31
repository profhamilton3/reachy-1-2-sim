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
