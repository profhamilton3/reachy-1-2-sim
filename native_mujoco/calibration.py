"""
R12-600: Camera calibration ingestion.

Defines the calibration file format (YAML), a loader, synthetic defaults,
and a helper to apply intrinsics to a live MuJoCo model.

File format (YAML) — see scenes/calibration_defaults.yaml for a template:
  format_version: 1
  provenance: "synthetic_defaults"
  left_camera:
    resolution: [640, 480]
    fx: 480.0     # focal length x in pixels
    fy: 480.0     # focal length y in pixels
    cx: 320.0     # principal point x
    cy: 240.0     # principal point y
    distortion: [0.0, 0.0, 0.0, 0.0, 0.0]   # k1 k2 p1 p2 k3 (OpenCV)
  right_camera:   # identical structure
    ...
  stereo:
    baseline_m: 0.065
    rectified: false
    R: null       # 3×3 rotation matrix (null → identity)
    T: null       # 3-element translation (null → [baseline_m, 0, 0])

MuJoCo limitation: the simulator uses a pinhole model with zero distortion.
Distortion coefficients are stored for external post-processing only.
fov_y is computed from fy and height, then written to model.cam_fovy[id].
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

_FORMAT_VERSION = 1
_SYNTHETIC_PROVENANCE = "synthetic_defaults"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraIntrinsics:
    """Per-camera pinhole intrinsics + OpenCV 5-coefficient distortion model."""
    resolution: Tuple[int, int]      # (width, height) in pixels
    fx: float                        # focal length x (pixels)
    fy: float                        # focal length y (pixels)
    cx: float                        # principal point x (pixels)
    cy: float                        # principal point y (pixels)
    distortion: Tuple[float, ...]    # k1 k2 p1 p2 k3

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    @property
    def fov_y_deg(self) -> float:
        """Vertical FOV in degrees derived from fy and image height."""
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.fy)))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "resolution": list(self.resolution),
            "fx": self.fx, "fy": self.fy,
            "cx": self.cx, "cy": self.cy,
            "distortion": list(self.distortion),
        }


@dataclass(frozen=True)
class StereoGeometry:
    baseline_m: float = 0.065
    rectified: bool = False
    R: Optional[Tuple[Tuple[float, ...], ...]] = None   # 3×3 rotation
    T: Optional[Tuple[float, ...]] = None               # 3-vector translation


@dataclass(frozen=True)
class StereoCalibrationProfile:
    format_version: int
    provenance: str
    left_camera: CameraIntrinsics
    right_camera: CameraIntrinsics
    stereo: StereoGeometry = field(default_factory=StereoGeometry)

    def is_synthetic(self) -> bool:
        return self.provenance == _SYNTHETIC_PROVENANCE

    def as_dict(self) -> Dict[str, Any]:
        geo = self.stereo
        return {
            "format_version": self.format_version,
            "provenance": self.provenance,
            "left_camera": self.left_camera.as_dict(),
            "right_camera": self.right_camera.as_dict(),
            "stereo": {
                "baseline_m": geo.baseline_m,
                "rectified": geo.rectified,
                "R": [list(row) for row in geo.R] if geo.R is not None else None,
                "T": list(geo.T) if geo.T is not None else None,
            },
        }


# ---------------------------------------------------------------------------
# Synthetic defaults
# ---------------------------------------------------------------------------

def synthetic_defaults(
    width: int = 640,
    height: int = 480,
    baseline_m: float = 0.065,
) -> StereoCalibrationProfile:
    """Return a synthetic calibration profile matching Reachy's nominal stereo rig.

    Uses a ~53° vertical FOV (fy = height) and zero distortion.  The baseline
    matches the mechanical inter-camera spacing in the Reachy 1.2 head URDF.
    """
    fx = fy = float(height)      # fy=height → fov_y ≈ 53.1°
    cx = width / 2.0
    cy = height / 2.0
    dist: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    left  = CameraIntrinsics((width, height), fx, fy, cx, cy, dist)
    right = CameraIntrinsics((width, height), fx, fy, cx, cy, dist)
    return StereoCalibrationProfile(
        format_version=_FORMAT_VERSION,
        provenance=_SYNTHETIC_PROVENANCE,
        left_camera=left,
        right_camera=right,
        stereo=StereoGeometry(baseline_m=baseline_m),
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _parse_intrinsics(raw: Dict[str, Any]) -> CameraIntrinsics:
    res = raw["resolution"]
    dist_raw: List[float] = raw.get("distortion", [0.0, 0.0, 0.0, 0.0, 0.0])
    dist = tuple(float(x) for x in dist_raw)
    if len(dist) != 5:
        raise ValueError(
            f"distortion must have exactly 5 coefficients, got {len(dist)}"
        )
    return CameraIntrinsics(
        resolution=(int(res[0]), int(res[1])),
        fx=float(raw["fx"]),
        fy=float(raw["fy"]),
        cx=float(raw["cx"]),
        cy=float(raw["cy"]),
        distortion=dist,
    )


def _parse_stereo(raw: Optional[Dict[str, Any]]) -> StereoGeometry:
    if raw is None:
        return StereoGeometry()
    r_raw = raw.get("R")
    t_raw = raw.get("T")
    r = tuple(tuple(float(v) for v in row) for row in r_raw) if r_raw else None
    t = tuple(float(v) for v in t_raw) if t_raw else None
    return StereoGeometry(
        baseline_m=float(raw.get("baseline_m", 0.065)),
        rectified=bool(raw.get("rectified", False)),
        R=r,
        T=t,
    )


def load_calibration(path: str | pathlib.Path) -> StereoCalibrationProfile:
    """Load a calibration YAML file and return a StereoCalibrationProfile."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    ver = int(raw.get("format_version", 0))
    if ver != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported calibration format_version {ver}; expected {_FORMAT_VERSION}"
        )
    return StereoCalibrationProfile(
        format_version=ver,
        provenance=str(raw.get("provenance", "unknown")),
        left_camera=_parse_intrinsics(raw["left_camera"]),
        right_camera=_parse_intrinsics(raw["right_camera"]),
        stereo=_parse_stereo(raw.get("stereo")),
    )


def save_calibration(profile: StereoCalibrationProfile, path: str | pathlib.Path) -> None:
    """Write a calibration profile to YAML."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(profile.as_dict(), f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# MuJoCo integration
# ---------------------------------------------------------------------------

def apply_to_model(profile: StereoCalibrationProfile, model: Any) -> None:
    """Write calibration fov_y to a live MuJoCo model's camera slots.

    Only fov_y is adjusted; principal point and distortion cannot be expressed
    directly in MuJoCo's camera model (distortion is handled in post-processing).
    """
    import mujoco  # local import — module must not require mujoco at import time
    for cam_name, intrinsics in (
        ("left_camera", profile.left_camera),
        ("right_camera", profile.right_camera),
    ):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cid < 0:
            continue
        model.cam_fovy[cid] = intrinsics.fov_y_deg
