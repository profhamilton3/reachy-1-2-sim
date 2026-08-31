"""Post-render lens-distortion pass.

MuJoCo renders a perfect pinhole camera.  The real Reachy 1.2 head cameras have
strong barrel distortion (k1 ~ -0.32 left / -0.39 right, measured 2026-08-27),
which bows straight lines visibly.  A detector trained on undistorted simulator
renders and pointed at the real feed therefore sees a systematically different
image near the frame edges.

This module warps a rendered frame to look like the real lens produced it.

WHICH DIRECTION TO PREFER
    For geometric reasoning, undistorting the REAL feed is cheaper and lossless
    where it matters, and that remains the recommended convention.  This pass
    exists for the other direction: generating training data that matches the
    raw camera, and eyeballing sim-vs-real frames side by side.  It is therefore
    OPT-IN (server --distortion), not on by default: renders are ground truth
    for the collision and evaluation paths, and silently warping them would
    corrupt consumers that never asked for it.

LABEL ALIGNMENT
    When enabled, the identical warp is applied to the depth and segmentation
    buffers as well, so auto-generated bounding boxes still line up with the RGB
    they came from.  RGB is sampled bilinearly; depth and segmentation are
    sampled NEAREST-NEIGHBOUR -- interpolating a body-ID map would invent IDs
    that belong to no body, and interpolating depth across a silhouette edge
    would invent surfaces that are not there.

KNOWN LIMITATION -- dark corners
    Barrel distortion compresses the periphery, so a real barrel lens sees a
    WIDER field than a pinhole of the same focal length.  A pinhole render
    simply does not contain the extra field the real camera would have caught,
    so the distorted output has undefined corners, filled with black.  To get
    edge-to-edge coverage, render with a margin (a larger fov_y than the
    calibrated one) and distort down into the target frame.  Not done here --
    it changes the renderer's field of view, which is a separate decision.

MODEL
    OpenCV's radial model, k1/k2/k3 only.  Tangential p1/p2 are read but are
    zero in every profile we have, and are ignored with a warning if not.

        r^2 = x^2 + y^2                       (normalised, undistorted)
        D   = 1 + k1*r^2 + k2*r^4 + k3*r^6
        x_d = x*D,  y_d = y*D

    The output frame is indexed by DISTORTED pixel coordinates, so filling it
    requires the inverse (undistort) at each output pixel.  That has no closed
    form; it is solved by fixed-point iteration, which converges quickly for
    the coefficient magnitudes involved here.  The map is built once per camera
    at construction and reused for every frame.
"""

from __future__ import annotations

import logging

import numpy as np
from calibration import CameraIntrinsics

log = logging.getLogger(__name__)

_INVERSE_ITERATIONS = 12


def _undistort_normalised(
    x_d: np.ndarray,
    y_d: np.ndarray,
    k1: float,
    k2: float,
    k3: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the radial model by fixed-point iteration.

    Solves for (x, y) such that (x, y) * D(r) == (x_d, y_d).
    """
    x = x_d.copy()
    y = y_d.copy()
    for _ in range(_INVERSE_ITERATIONS):
        r2 = x * x + y * y
        d = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        # A pole (d -> 0) means the model has folded; clamp rather than divide
        # by ~0 and scatter samples across the whole image.
        d = np.where(np.abs(d) < 1e-6, 1e-6, d)
        x = x_d / d
        y = y_d / d
    return x, y


class LensDistorter:
    """Applies one camera's barrel distortion to rendered buffers.

    The sampling map is computed once, from the intrinsics and the render size,
    and reused.  Construct one per camera; they are not interchangeable.

    Args:
        intrinsics:  the camera's calibrated intrinsics (fx, fy, cx, cy, k).
        width, height: the RENDER size in pixels.  If it differs from the
            intrinsics' own resolution, fx/fy/cx/cy are scaled to match, so a
            profile calibrated at one resolution can drive a render at another.
    """

    def __init__(self, intrinsics: CameraIntrinsics, width: int, height: int) -> None:
        self._width = width
        self._height = height

        k = list(intrinsics.distortion) + [0.0] * 5
        k1, k2, p1, p2, k3 = k[0], k[1], k[2], k[3], k[4]
        if p1 != 0.0 or p2 != 0.0:
            log.warning(
                "Tangential distortion p1=%.4g p2=%.4g is not modelled; ignoring",
                p1, p2,
            )
        self._identity = (k1 == 0.0 and k2 == 0.0 and k3 == 0.0)

        sx = width / float(intrinsics.width)
        sy = height / float(intrinsics.height)
        fx, fy = intrinsics.fx * sx, intrinsics.fy * sy
        cx, cy = intrinsics.cx * sx, intrinsics.cy * sy

        # Output pixel grid -> normalised distorted coords -> undistorted
        # normalised -> source pixel coords in the pinhole render.
        v, u = np.meshgrid(
            np.arange(height, dtype=np.float64),
            np.arange(width, dtype=np.float64),
            indexing="ij",
        )
        x_d = (u - cx) / fx
        y_d = (v - cy) / fy
        x_u, y_u = _undistort_normalised(x_d, y_d, k1, k2, k3)
        src_x = x_u * fx + cx
        src_y = y_u * fy + cy

        # Samples outside the render have no data (see "dark corners" above).
        self._valid = (
            (src_x >= 0) & (src_x <= width - 1) & (src_y >= 0) & (src_y <= height - 1)
        )
        self._src_x = np.clip(src_x, 0.0, width - 1.0)
        self._src_y = np.clip(src_y, 0.0, height - 1.0)

        # Bilinear weights, precomputed.
        x0 = np.floor(self._src_x).astype(np.int32)
        y0 = np.floor(self._src_y).astype(np.int32)
        self._x0 = x0
        self._y0 = y0
        self._x1 = np.minimum(x0 + 1, width - 1)
        self._y1 = np.minimum(y0 + 1, height - 1)
        self._wx = (self._src_x - x0)[..., None]
        self._wy = (self._src_y - y0)[..., None]

        # Nearest-neighbour indices for label buffers.
        self._nx = np.rint(self._src_x).astype(np.int32)
        self._ny = np.rint(self._src_y).astype(np.int32)

    @property
    def is_identity(self) -> bool:
        """True when the profile has no radial distortion — skip the warp."""
        return self._identity

    def apply_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Warp an H×W×3 uint8 RGB render.  Bilinear; undefined pixels black."""
        if self._identity:
            return rgb
        src = rgb.astype(np.float32)
        top = src[self._y0, self._x0] * (1 - self._wx) + src[self._y0, self._x1] * self._wx
        bot = src[self._y1, self._x0] * (1 - self._wx) + src[self._y1, self._x1] * self._wx
        out = top * (1 - self._wy) + bot * self._wy
        out = np.where(self._valid[..., None], out, 0.0)
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_label(self, buf: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Warp an H×W depth or segmentation buffer.

        Nearest-neighbour, so no value appears in the output that was not
        already in the input.  Undefined pixels take ``fill`` — 0 for
        segmentation is MuJoCo's world/background body ID, which is what an
        unobserved pixel should read as.
        """
        if self._identity:
            return buf
        out = buf[self._ny, self._nx]
        return np.where(self._valid, out, np.asarray(fill, dtype=buf.dtype))
