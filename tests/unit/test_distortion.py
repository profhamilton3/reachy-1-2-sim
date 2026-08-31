"""Unit tests for the post-render lens-distortion pass (native_mujoco/distortion.py)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "native_mujoco"))

from calibration import CameraIntrinsics, load_calibration
from distortion import LensDistorter, _undistort_normalised

W, H = 64, 48
_MEASURED_K = (-0.3163, 0.1027, 0.0, 0.0, 0.0)


def _intr(distortion=_MEASURED_K, width=W, height=H, f=40.0):
    return CameraIntrinsics(
        resolution=(width, height), fx=f, fy=f,
        cx=width / 2.0, cy=height / 2.0, distortion=distortion,
    )


class TestInverseModel:
    def test_zero_coefficients_is_identity(self):
        x = np.linspace(-0.5, 0.5, 17)
        y = np.linspace(-0.4, 0.4, 17)
        xu, yu = _undistort_normalised(x, y, 0.0, 0.0, 0.0)
        assert np.allclose(xu, x)
        assert np.allclose(yu, y)

    def test_inverts_the_forward_model(self):
        """undistort(distort(p)) must return p — this is the whole contract."""
        k1, k2, k3 = -0.3163, 0.1027, 0.0
        x = np.linspace(-0.6, 0.6, 25)
        y = np.linspace(-0.45, 0.45, 25)
        xx, yy = np.meshgrid(x, y)
        r2 = xx * xx + yy * yy
        d = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        x_d, y_d = xx * d, yy * d           # forward: undistorted -> distorted
        xu, yu = _undistort_normalised(x_d, y_d, k1, k2, k3)
        assert np.max(np.abs(xu - xx)) < 1e-6
        assert np.max(np.abs(yu - yy)) < 1e-6

    def test_centre_pixel_is_a_fixed_point(self):
        xu, yu = _undistort_normalised(
            np.array([0.0]), np.array([0.0]), -0.3163, 0.1027, 0.0)
        assert xu[0] == pytest.approx(0.0)
        assert yu[0] == pytest.approx(0.0)


class TestIdentityProfile:
    def test_zero_distortion_reports_identity(self):
        assert LensDistorter(_intr(distortion=(0.0,) * 5), W, H).is_identity

    def test_measured_profile_is_not_identity(self):
        assert not LensDistorter(_intr(), W, H).is_identity

    def test_identity_returns_input_unchanged(self):
        d = LensDistorter(_intr(distortion=(0.0,) * 5), W, H)
        rgb = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)
        assert d.apply_rgb(rgb) is rgb

    def test_tangential_coefficients_warn(self, caplog):
        LensDistorter(_intr(distortion=(-0.3, 0.1, 0.01, 0.02, 0.0)), W, H)
        assert "Tangential" in caplog.text


class TestRgbWarp:
    def test_shape_and_dtype_preserved(self):
        d = LensDistorter(_intr(), W, H)
        rgb = np.random.default_rng(1).integers(0, 256, (H, W, 3), dtype=np.uint8)
        out = d.apply_rgb(rgb)
        assert out.shape == rgb.shape
        assert out.dtype == np.uint8

    def test_uniform_image_survives_the_warp(self):
        """A flat field has no geometry to bend, so every defined pixel keeps
        its value.  Only the undefined corners go black."""
        d = LensDistorter(_intr(), W, H)
        rgb = np.full((H, W, 3), 137, dtype=np.uint8)
        out = d.apply_rgb(rgb)
        assert out[H // 2, W // 2, 0] == 137
        assert set(np.unique(out)) <= {0, 137}

    def test_barrel_leaves_undefined_corners_black(self):
        """Barrel distortion sees wider than the pinhole render contains, so the
        corners have no source data.  Documented limitation, asserted here so it
        cannot regress silently into garbage."""
        d = LensDistorter(_intr(), W, H)
        rgb = np.full((H, W, 3), 255, dtype=np.uint8)
        out = d.apply_rgb(rgb)
        assert out[0, 0].sum() == 0
        assert out[H // 2, W // 2].sum() > 0

    def test_centre_is_unmoved(self):
        """Radial distortion is zero on axis, so the centre pixel must not shift."""
        d = LensDistorter(_intr(), W, H)
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[H // 2, W // 2] = (255, 255, 255)
        out = d.apply_rgb(rgb)
        assert out[H // 2, W // 2].sum() > 0


class TestLabelWarp:
    def test_segmentation_invents_no_new_ids(self):
        """Nearest-neighbour only: the output must be a subset of the input IDs
        plus the background fill.  Bilinear here would fabricate body IDs."""
        d = LensDistorter(_intr(), W, H)
        rng = np.random.default_rng(2)
        seg = rng.choice(np.array([0, 5, 9, 41], dtype=np.uint16), size=(H, W))
        out = d.apply_label(seg, fill=0)
        assert set(np.unique(out)) <= {0, 5, 9, 41}
        assert out.dtype == seg.dtype

    def test_depth_fill_marks_undefined_pixels(self):
        d = LensDistorter(_intr(), W, H)
        depth = np.full((H, W), 1.25, dtype=np.float32)
        out = d.apply_label(depth, fill=0.0)
        assert out[0, 0] == 0.0                      # undefined corner
        assert out[H // 2, W // 2] == pytest.approx(1.25)

    def test_rgb_and_label_use_the_same_map(self):
        """Labels must stay aligned with the pixels they describe — that is the
        entire reason the warp is applied to every buffer."""
        d = LensDistorter(_intr(), W, H)
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        seg = np.zeros((H, W), dtype=np.uint16)
        rgb[10:20, 12:30] = (200, 200, 200)
        seg[10:20, 12:30] = 7
        out_rgb = d.apply_rgb(rgb)
        out_seg = d.apply_label(seg, fill=0)
        # Where the label says "body 7", the RGB must be the bright patch.
        assert np.all(out_rgb[out_seg == 7].sum(axis=-1) > 0)


class TestResolutionScaling:
    def test_intrinsics_scale_to_a_different_render_size(self):
        """A profile calibrated at 640x480 must drive a 320x240 render."""
        intr = _intr(width=640, height=480, f=408.0)
        d = LensDistorter(intr, 320, 240)
        rgb = np.full((240, 320, 3), 90, dtype=np.uint8)
        out = d.apply_rgb(rgb)
        assert out.shape == (240, 320, 3)
        assert out[120, 160, 0] == 90


class TestMeasuredProfileFile:
    def test_committed_measured_profile_drives_the_distorter(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "scenes", "calibration_measured_2026_08_27.yaml")
        p = load_calibration(path)
        assert not p.is_synthetic()
        for intr in (p.left_camera, p.right_camera):
            d = LensDistorter(intr, intr.width, intr.height)
            assert not d.is_identity, "measured profile must carry real barrel k1"
