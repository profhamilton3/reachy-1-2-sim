"""Unit tests for the motorised-zoom model (native_mujoco/zoom.py)."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "native_mujoco"))

from calibration import load_calibration
from zoom import (
    DIAGONAL_FOV_TELE_DEG,
    DIAGONAL_FOV_WIDE_DEG,
    ZoomLevel,
    all_levels,
    diagonal_to_fov_y_deg,
    fov_y_for_level,
    parse_level,
)

W, H = 640, 480
CAL_FOV_Y = 61.1          # the measured lab profile


class TestDiagonalConversion:
    def test_reproduces_the_lab_measurement(self):
        """The whole diagonal reading rests on this: our calibrated frames
        solve to an 88.9 deg diagonal, and that must map back to the 61.1 deg
        vertical we actually measured.  If this drifts, the premise is wrong."""
        assert diagonal_to_fov_y_deg(88.9, W, H) == pytest.approx(61.1, abs=0.2)

    def test_wider_diagonal_gives_wider_vertical(self):
        assert (diagonal_to_fov_y_deg(125.0, W, H)
                > diagonal_to_fov_y_deg(88.9, W, H)
                > diagonal_to_fov_y_deg(65.0, W, H))

    def test_square_frame_diagonal_exceeds_vertical(self):
        """Sanity on the geometry itself, independent of Reachy's numbers."""
        assert diagonal_to_fov_y_deg(90.0, 480, 480) < 90.0


class TestLevels:
    def test_inter_and_zero_return_the_calibrated_value_untouched(self):
        """The simulator must not drift off a real measurement because a model
        says otherwise."""
        for lv in (ZoomLevel.INTER, ZoomLevel.ZERO):
            assert fov_y_for_level(lv, W, H, CAL_FOV_Y) == CAL_FOV_Y

    def test_out_is_wider_and_in_is_narrower_than_calibrated(self):
        assert fov_y_for_level(ZoomLevel.OUT, W, H, CAL_FOV_Y) > CAL_FOV_Y
        assert fov_y_for_level(ZoomLevel.IN, W, H, CAL_FOV_Y) < CAL_FOV_Y

    def test_extremes_match_the_published_range(self):
        assert fov_y_for_level(ZoomLevel.OUT, W, H, CAL_FOV_Y) == pytest.approx(
            diagonal_to_fov_y_deg(DIAGONAL_FOV_WIDE_DEG, W, H))
        assert fov_y_for_level(ZoomLevel.IN, W, H, CAL_FOV_Y) == pytest.approx(
            diagonal_to_fov_y_deg(DIAGONAL_FOV_TELE_DEG, W, H))

    def test_calibrated_level_sits_inside_the_range(self):
        """Corroborates reading 65-125 as diagonal: the lab captures land
        mid-range, consistent with 'shot at an intermediate zoom'."""
        lv = all_levels(W, H, CAL_FOV_Y)
        assert lv[ZoomLevel.IN] < CAL_FOV_Y < lv[ZoomLevel.OUT]

    def test_all_levels_covers_every_enum_member(self):
        assert set(all_levels(W, H, CAL_FOV_Y)) == set(ZoomLevel)

    def test_unknown_level_rejected(self):
        with pytest.raises(ValueError):
            fov_y_for_level("telephoto", W, H, CAL_FOV_Y)


class TestParsing:
    @pytest.mark.parametrize("raw,expected", [
        ("in", ZoomLevel.IN), ("OUT", ZoomLevel.OUT),
        ("  inter ", ZoomLevel.INTER), ("Zero", ZoomLevel.ZERO),
    ])
    def test_accepts_sdk_names_case_insensitively(self, raw, expected):
        assert parse_level(raw) == expected

    def test_rejects_nonsense_with_a_helpful_message(self):
        with pytest.raises(ValueError, match="expected one of"):
            parse_level("macro")


class TestAgainstTheMeasuredProfile:
    def test_measured_profile_drives_the_levels(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "scenes", "calibration_measured_2026_08_27.yaml")
        p = load_calibration(path)
        cal = p.left_camera.fov_y_deg
        lv = all_levels(W, H, cal)
        assert lv[ZoomLevel.INTER] == pytest.approx(cal)
        # A real zoom range, not a rounding artefact.
        assert lv[ZoomLevel.OUT] - lv[ZoomLevel.IN] > 30.0

    def test_focal_length_scales_inversely_with_field(self):
        """Zoom must change field of view, not crop: a narrower field implies a
        longer focal length at the same sensor size."""
        def f(fov):
            return (H / 2.0) / math.tan(math.radians(fov) / 2.0)
        lv = all_levels(W, H, CAL_FOV_Y)
        assert f(lv[ZoomLevel.IN]) > f(lv[ZoomLevel.INTER]) > f(lv[ZoomLevel.OUT])
