"""Motorised-zoom model for the Reachy 1.2 head cameras.

The physical cameras are Kurokesu C1 Pro bodies with a motorised zoom lens that
Pollen fitted themselves; the Reachy 2021 docs give the range as **65 deg to
125 deg** and expose three levels plus a default (`in`, `out`, `inter`, `zero`).
Until now the simulator acknowledged zoom commands and did nothing with them --
`fake_reachy_server.py` stored `_zoom_level` and the rendered field of view
never moved -- so any behaviour that depends on changing the field of view
could not be developed or tested against the sim at all.

WHICH AXIS IS "65 to 125 deg"?
    Pollen do not say.  We can infer it, because our own calibration pins one
    point on the curve: the lab captures solve to fx/fy ~ 408 px, i.e. 76.2 deg
    horizontal x 61.1 deg vertical on a 4:3 frame -- a **diagonal** of 88.9 deg.
    Reading the published range as diagonal puts that measurement almost exactly
    mid-range, which is consistent with the frames having been shot at an
    intermediate zoom.  Reading it as horizontal or vertical would put our
    measurement outside or at the very edge of the range, which it plainly is
    not.  So: the range is treated as DIAGONAL, and `INTER` is anchored on the
    measured profile rather than on a guess.

    This is an inference from one data point.  Confirm it by commanding `in`
    and `out` on the physical robot and re-solving the intrinsics at each --
    which is issue #4's outstanding "test zoom control" task, and the same
    exercise that would give us a real per-zoom calibration.

CALIBRATION IS PER-ZOOM
    `scenes/calibration_measured_2026_08_27.yaml` is only valid at the zoom the
    photos were taken at.  Commanding a different level in the simulator changes
    the field of view but NOT the distortion coefficients, because we have no
    measurement of how k1/k2 move with zoom -- on a real varifocal lens they do.
    Frames rendered away from INTER therefore have a trustworthy field of view
    and an approximate barrel profile.  Treat them as such.
"""

from __future__ import annotations

import math
from enum import Enum

# Published range for the fitted lens, read as diagonal (see module docstring).
DIAGONAL_FOV_WIDE_DEG = 125.0
DIAGONAL_FOV_TELE_DEG = 65.0


class ZoomLevel(str, Enum):
    """Mirrors reachy-sdk-api's ZoomLevelPossibilities."""

    IN = "in"          # telephoto — narrowest field, for inspecting a target
    OUT = "out"        # wide — widest field, for surveying the scene
    INTER = "inter"    # intermediate — the level the lab calibration was shot at
    ZERO = "zero"      # SDK's "unset"; treated as INTER, the calibrated level


def diagonal_to_fov_y_deg(diagonal_deg: float, width: int, height: int) -> float:
    """Convert a diagonal field of view to the vertical one MuJoCo wants.

    Assumes a rectilinear pinhole projection, which is what MuJoCo renders.
    """
    diag_px = math.hypot(width, height)
    f = (diag_px / 2.0) / math.tan(math.radians(diagonal_deg) / 2.0)
    return math.degrees(2.0 * math.atan((height / 2.0) / f))


def fov_y_for_level(
    level: ZoomLevel,
    width: int,
    height: int,
    calibrated_fov_y_deg: float,
) -> float:
    """Vertical FOV in degrees for one zoom level.

    INTER and ZERO return the calibrated value unchanged -- the simulator must
    not drift away from a real measurement just because a model says otherwise.
    IN and OUT are derived from the published diagonal range.
    """
    if level in (ZoomLevel.INTER, ZoomLevel.ZERO):
        return calibrated_fov_y_deg
    if level is ZoomLevel.OUT:
        return diagonal_to_fov_y_deg(DIAGONAL_FOV_WIDE_DEG, width, height)
    if level is ZoomLevel.IN:
        return diagonal_to_fov_y_deg(DIAGONAL_FOV_TELE_DEG, width, height)
    raise ValueError(f"unknown zoom level: {level!r}")


def all_levels(
    width: int, height: int, calibrated_fov_y_deg: float
) -> dict[ZoomLevel, float]:
    """Every level's vertical FOV — handy for sweeps and for logging."""
    return {
        lv: fov_y_for_level(lv, width, height, calibrated_fov_y_deg)
        for lv in ZoomLevel
    }


def parse_level(value: str) -> ZoomLevel:
    """Accept the SDK's lower-case names, case-insensitively."""
    try:
        return ZoomLevel(str(value).strip().lower())
    except ValueError as exc:
        valid = ", ".join(lv.value for lv in ZoomLevel)
        raise ValueError(f"unknown zoom level {value!r}; expected one of: {valid}") from exc
