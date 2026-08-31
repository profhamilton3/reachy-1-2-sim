"""Unit tests for R12-600: camera calibration ingestion.

All tests run offline — no MuJoCo or gRPC required.
"""

import math
import os
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "native_mujoco"))

from calibration import (
    CameraIntrinsics,
    StereoCalibrationProfile,
    StereoGeometry,
    load_calibration,
    save_calibration,
    synthetic_defaults,
)


# ── CameraIntrinsics ──────────────────────────────────────────────────────────

class TestCameraIntrinsics:

    def _make(self, **kw):
        defaults = dict(resolution=(640, 480), fx=480.0, fy=480.0,
                        cx=320.0, cy=240.0, distortion=(0.0,)*5)
        defaults.update(kw)
        return CameraIntrinsics(**defaults)

    def test_width_height(self):
        c = self._make(resolution=(640, 480))
        assert c.width == 640
        assert c.height == 480

    def test_fov_y_identity(self):
        # fy = height → fov_y = 2*atan(0.5) ≈ 53.13°
        c = self._make(fy=480.0)
        expected = math.degrees(2.0 * math.atan(480 / (2 * 480)))
        assert c.fov_y_deg == pytest.approx(expected, abs=0.01)

    def test_fov_y_wide_angle(self):
        # smaller fy → wider FOV
        c_wide  = self._make(fy=240.0)
        c_normal = self._make(fy=480.0)
        assert c_wide.fov_y_deg > c_normal.fov_y_deg

    def test_fov_y_narrow(self):
        # fy very large → very narrow FOV
        c = self._make(fy=9600.0)
        assert c.fov_y_deg < 3.0

    def test_as_dict_round_trip(self):
        c = self._make(fx=500.0, distortion=(0.1, -0.05, 0.0, 0.0, 0.0))
        d = c.as_dict()
        assert d["fx"] == 500.0
        assert d["distortion"] == [0.1, -0.05, 0.0, 0.0, 0.0]
        assert d["resolution"] == [640, 480]

    def test_frozen(self):
        c = self._make()
        with pytest.raises(AttributeError):
            c.fx = 999.0  # type: ignore[misc]


# ── SyntheticDefaults ─────────────────────────────────────────────────────────

class TestSyntheticDefaults:

    def test_returns_profile(self):
        p = synthetic_defaults()
        assert isinstance(p, StereoCalibrationProfile)

    def test_is_synthetic(self):
        assert synthetic_defaults().is_synthetic()

    def test_resolution_matches_args(self):
        p = synthetic_defaults(width=320, height=240)
        assert p.left_camera.width == 320
        assert p.left_camera.height == 240
        assert p.right_camera.width == 320

    def test_zero_distortion(self):
        p = synthetic_defaults()
        assert all(v == 0.0 for v in p.left_camera.distortion)

    def test_baseline(self):
        p = synthetic_defaults(baseline_m=0.070)
        assert p.stereo.baseline_m == pytest.approx(0.070)

    def test_cx_cy_centered(self):
        p = synthetic_defaults(width=640, height=480)
        assert p.left_camera.cx == pytest.approx(320.0)
        assert p.left_camera.cy == pytest.approx(240.0)

    def test_fov_reasonable(self):
        p = synthetic_defaults()
        fov = p.left_camera.fov_y_deg
        assert 30.0 < fov < 90.0, f"Unexpected fov_y: {fov}"


# ── YAML round-trip (save + load) ─────────────────────────────────────────────

class TestYAMLRoundTrip:

    def test_save_and_reload(self, tmp_path):
        original = synthetic_defaults(width=640, height=480, baseline_m=0.065)
        path = tmp_path / "cal.yaml"
        save_calibration(original, path)
        loaded = load_calibration(path)

        assert loaded.format_version == original.format_version
        assert loaded.provenance == original.provenance
        assert loaded.left_camera.fx == pytest.approx(original.left_camera.fx)
        assert loaded.left_camera.fov_y_deg == pytest.approx(
            original.left_camera.fov_y_deg, abs=0.01
        )
        assert loaded.stereo.baseline_m == pytest.approx(0.065)
        assert all(v == 0.0 for v in loaded.left_camera.distortion)

    def test_full_calibration_yaml(self, tmp_path):
        cal_data = {
            "format_version": 1,
            "provenance": "physical_20260811",
            "left_camera": {
                "resolution": [640, 480],
                "fx": 512.3, "fy": 511.8,
                "cx": 322.1, "cy": 241.4,
                "distortion": [-0.12, 0.05, 0.001, -0.001, 0.0],
            },
            "right_camera": {
                "resolution": [640, 480],
                "fx": 513.0, "fy": 512.2,
                "cx": 318.9, "cy": 240.1,
                "distortion": [-0.11, 0.04, 0.0, 0.0, 0.0],
            },
            "stereo": {
                "baseline_m": 0.0643,
                "rectified": True,
                "R": None, "T": None,
            },
        }
        path = tmp_path / "physical.yaml"
        path.write_text(yaml.dump(cal_data))
        p = load_calibration(path)
        assert p.provenance == "physical_20260811"
        assert not p.is_synthetic()
        assert p.left_camera.fx == pytest.approx(512.3)
        assert p.right_camera.cy == pytest.approx(240.1)
        assert p.stereo.rectified is True
        assert p.stereo.baseline_m == pytest.approx(0.0643)
        # distortion coefficients preserved
        assert p.left_camera.distortion[0] == pytest.approx(-0.12)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_calibration(tmp_path / "nonexistent.yaml")

    def test_wrong_version_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("format_version: 99\nleft_camera: {}\nright_camera: {}\n")
        with pytest.raises(ValueError, match="format_version"):
            load_calibration(path)

    def test_bad_distortion_length_raises(self, tmp_path):
        cal_data = {
            "format_version": 1,
            "provenance": "test",
            "left_camera": {
                "resolution": [640, 480],
                "fx": 480.0, "fy": 480.0, "cx": 320.0, "cy": 240.0,
                "distortion": [0.0, 0.0],   # only 2 coeffs, needs 5
            },
            "right_camera": {
                "resolution": [640, 480],
                "fx": 480.0, "fy": 480.0, "cx": 320.0, "cy": 240.0,
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
        }
        path = tmp_path / "bad_dist.yaml"
        path.write_text(yaml.dump(cal_data))
        with pytest.raises(ValueError, match="distortion"):
            load_calibration(path)

    def test_defaults_yaml_file_valid(self):
        """The committed calibration_defaults.yaml must parse without errors."""
        defaults_path = (
            os.path.join(os.path.dirname(__file__), "..", "..", "scenes", "calibration_defaults.yaml")
        )
        p = load_calibration(defaults_path)
        assert p.is_synthetic()
        assert p.left_camera.width == 640
        assert p.stereo.baseline_m > 0


class TestMeasuredProfile:
    """The committed measured lab calibration (scenes/calibration_measured_2026_08_27.yaml).

    This is the runtime default that scripts/start_sim.sh passes to the server,
    so a break here silently changes every render's field of view.
    """

    def _load(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "scenes", "calibration_measured_2026_08_27.yaml")
        return load_calibration(path)

    def test_parses_and_is_not_synthetic(self):
        p = self._load()
        assert p.provenance == "measured_2026_08_27"
        assert not p.is_synthetic()

    def test_landscape_resolution_unchanged(self):
        """640x480 is the orientation the sim renders (fixed in 31fe0d5).  The
        portrait-vs-landscape question is still open on the hardware side; until
        it is settled this file must not quietly flip the render aspect."""
        p = self._load()
        assert p.left_camera.resolution == (640, 480)
        assert p.right_camera.resolution == (640, 480)

    def test_field_of_view_matches_the_measurement(self):
        p = self._load()
        assert p.left_camera.fov_y_deg == pytest.approx(61.1, abs=0.2)
        assert p.right_camera.fov_y_deg == pytest.approx(62.1, abs=0.2)

    def test_wider_than_the_synthetic_default(self):
        """The whole point of this profile: the synthetic 53.1 deg camera was
        clipping the outer edge of the scene."""
        p = self._load()
        synth = synthetic_defaults(640, 480)
        assert p.left_camera.fov_y_deg > synth.left_camera.fov_y_deg

    def test_carries_real_barrel_distortion(self):
        p = self._load()
        assert p.left_camera.distortion[0] == pytest.approx(-0.3163)
        assert p.right_camera.distortion[0] == pytest.approx(-0.3895)

    def test_measured_baseline(self):
        p = self._load()
        assert p.stereo.baseline_m == pytest.approx(0.080)

    def test_principal_point_is_centred_pending_the_transpose_question(self):
        """cx/cy are deliberately the image centre, not the measured principal
        point: the 480x640 -> 640x480 transpose direction differs per eye and is
        unresolved.  If someone fills in real values, this test should be
        updated in the same commit as the evidence that settled it."""
        p = self._load()
        for intr in (p.left_camera, p.right_camera):
            assert intr.cx == pytest.approx(intr.width / 2.0)
            assert intr.cy == pytest.approx(intr.height / 2.0)
