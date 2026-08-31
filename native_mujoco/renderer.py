"""
R12-404 / R12-601: Offscreen stereo RGB + optional depth/segmentation renderer.

Renders left_camera and right_camera from reachy_1_2.xml using MuJoCo's
Renderer (EGL/osmesa on Linux, Metal on macOS arm64).

Depth output (R12-601):
  float16 numpy array, shape H×W, values in metres.
  Encoded as base64(raw float16 bytes) for transmission.

Segmentation output (R12-601):
  uint16 numpy array, shape H×W, values are MuJoCo BODY IDs.
  Encoded as base64(raw uint16 bytes).
  Body ID 0 = world/background, and also what MuJoCo's -1 "nothing here"
  is mapped to.

  MuJoCo's own segmentation pixel is [object_id, object_TYPE], and for a
  scene render the type is uniformly mjOBJ_GEOM — so the raw buffer holds
  GEOM ids.  _encode_seg maps them through model.geom_bodyid, so one object
  gets one label even when it is built from several geoms.  See _encode_seg.

Design constraints (from ADR-0001 and CLAUDE.md):
  * Do not hold the state mutation lock while encoding images.
  * Camera rendering must not advance sim state (read-only).
  * Produce coherent stereo pairs: both cameras from the same sim_step.
  * Thread-safe: one Renderer instance per OS thread (MuJoCo requirement).
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import mujoco
import numpy as np
from distortion import LensDistorter
from PIL import Image

_DEFAULT_WIDTH   = 640
_DEFAULT_HEIGHT  = 480
_DEFAULT_QUALITY = 85


@dataclass(frozen=True)
class RenderedFrame:
    camera: str
    jpeg_bytes: bytes
    width: int
    height: int
    render_us: int          # render + encode wall time in microseconds
    depth_b64: str = ""     # base64 float16 H×W depth map (R12-601); "" = disabled
    seg_b64: str = ""       # base64 uint16 H×W body-ID segmentation (R12-601); "" = disabled


class StereoRenderer:
    """Offscreen stereo renderer.  Create one per thread; do not share.

    Args:
        model:            MuJoCo model.
        width, height:    Render resolution in pixels.
        jpeg_quality:     JPEG compression quality (1–95).
        enable_depth:     If True, also produce a depth map per frame (R12-601).
        enable_seg:       If True, also produce a body-ID segmentation per frame (R12-601).
        distorters:       Optional {camera_name: LensDistorter}.  When supplied, each
                          camera's frame is warped to match the real lens's barrel
                          distortion before encoding, and the SAME warp is applied to
                          the depth and segmentation buffers so auto-generated labels
                          stay aligned with the RGB they came from.  Absent unless
                          explicitly passed: renders are ground truth for the
                          collision and evaluation paths.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        jpeg_quality: int = _DEFAULT_QUALITY,
        enable_depth: bool = False,
        enable_seg: bool = False,
        distorters: Optional[Dict[str, LensDistorter]] = None,
    ) -> None:
        self._model = model
        self._width = width
        self._height = height
        self._quality = jpeg_quality
        self._enable_depth = enable_depth
        self._enable_seg = enable_seg
        self._distorters: Dict[str, LensDistorter] = distorters or {}

        # Resolve camera IDs once
        self._cam_ids: Dict[str, int] = {}
        for name in ("left_camera", "right_camera"):
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                raise RuntimeError(f"Camera '{name}' not found in model")
            self._cam_ids[name] = cid

        # A distorter with a margin samples a LARGER source render than it
        # emits, so the internal renderer is sized to that source and each
        # camera's cam_fovy is widened to make the extra pixels real field
        # rather than a zoom.  One mujoco.Renderer serves both cameras, so all
        # active distorters must agree on the source size — the server builds
        # them with a single shared margin for exactly this reason.
        self._src_w, self._src_h = width, height
        active = [d for d in self._distorters.values() if not d.is_identity]
        if active:
            sizes = {d.source_size for d in active}
            if len(sizes) != 1:
                raise ValueError(
                    f"distorters disagree on source size: {sorted(sizes)}; "
                    "build them with one shared margin"
                )
            self._src_w, self._src_h = sizes.pop()
            for cam_name, d in self._distorters.items():
                if not d.is_identity:
                    model.cam_fovy[self._cam_ids[cam_name]] = d.source_fov_y_deg

        self._renderer = mujoco.Renderer(model, height=self._src_h, width=self._src_w)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def source_size(self) -> tuple:
        """(width, height) actually rendered before distortion maps it down.

        Equals (width, height) when no distorter has a margin.
        """
        return (self._src_w, self._src_h)

    def render_stereo(
        self,
        data: mujoco.MjData,
        cameras: Optional[tuple] = None,
    ) -> Dict[str, RenderedFrame]:
        """Render the requested cameras from the current data state.

        data must be a snapshot copy: caller owns it and must not mutate it
        while this method runs.

        Returns a dict keyed by camera name.
        """
        targets = cameras if cameras is not None else ("left_camera", "right_camera")
        result: Dict[str, RenderedFrame] = {}
        for cam_name in targets:
            frame = self._render_one(data, cam_name)
            result[cam_name] = frame
        return result

    def _render_one(self, data: mujoco.MjData, cam_name: str) -> RenderedFrame:
        t0 = time.monotonic_ns()

        self._renderer.update_scene(data, camera=cam_name)

        # Lens distortion, when configured for this camera.  Applied to every
        # buffer with the same map so labels keep matching the pixels.
        distorter = self._distorters.get(cam_name)
        if distorter is not None and distorter.is_identity:
            distorter = None

        # RGB (always)
        rgb = self._renderer.render()
        if distorter is not None:
            rgb = distorter.apply_rgb(rgb)
        jpeg_bytes = _encode_jpeg(rgb, self._quality)

        # Depth (R12-601) — float16 base64
        depth_b64 = ""
        if self._enable_depth:
            self._renderer.enable_depth_rendering()
            depth_f32 = self._renderer.render()       # H×W float32, metres
            self._renderer.disable_depth_rendering()
            if distorter is not None:
                # 0 m marks "no data" in the undefined corners; a real depth
                # sample is never 0, so consumers can mask on it.
                depth_f32 = distorter.apply_label(depth_f32, fill=0.0)
            depth_b64 = _encode_depth(depth_f32)

        # Segmentation (R12-601) — uint16 body-ID base64
        seg_b64 = ""
        if self._enable_seg:
            self._renderer.enable_segmentation_rendering()
            seg_raw = self._renderer.render()   # H×W×2 int32 [object_id, object_TYPE]
            self._renderer.disable_segmentation_rendering()
            # Map geom -> body BEFORE any warp, so the warp moves final labels
            # rather than intermediate ids.
            seg_b64 = _encode_seg(seg_raw, self._model)
            if distorter is not None:
                body = decode_seg(seg_b64, self._src_h, self._src_w)
                body = distorter.apply_label(body, fill=0)
                seg_b64 = base64.b64encode(
                    body.astype(np.uint16).tobytes()).decode("ascii")

        elapsed_us = (time.monotonic_ns() - t0) // 1_000
        return RenderedFrame(
            camera=cam_name,
            jpeg_bytes=jpeg_bytes,
            width=self._width,
            height=self._height,
            render_us=elapsed_us,
            depth_b64=depth_b64,
            seg_b64=seg_b64,
        )

    def set_zoom(self, fov_y_deg: float) -> None:
        """Retune both cameras to a new vertical field of view (R12-605).

        Zoom changes the focal length, which invalidates any distortion map
        built for the old one, so the maps are rebuilt here rather than left to
        warp the frame by the wrong amount.  The distortion COEFFICIENTS are
        kept as-is: we have no measurement of how k1/k2 move with zoom on this
        lens, so a frame away from the calibrated level has a trustworthy field
        of view and an approximate barrel profile.  See native_mujoco/zoom.py.

        Raises if the new zoom would need a different source buffer than the
        internal renderer was built with -- rebuild the StereoRenderer for that
        rather than silently rendering at the wrong size.
        """
        import math

        from calibration import CameraIntrinsics

        for cam_name, cid in self._cam_ids.items():
            d = self._distorters.get(cam_name)
            if d is None or d.is_identity:
                self._model.cam_fovy[cid] = fov_y_deg
                continue

            # Focal length implied by the requested output field of view.
            fy = (self._height / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)
            fx = fy  # square pixels; the measured fx/fy differ by 0.3%
            rebuilt = LensDistorter(
                CameraIntrinsics(
                    resolution=(self._width, self._height),
                    fx=fx, fy=fy,
                    cx=self._width / 2.0, cy=self._height / 2.0,
                    distortion=d.distortion,
                ),
                self._width, self._height, margin=d.margin,
            )
            if rebuilt.source_size != (self._src_w, self._src_h):
                raise ValueError(
                    f"zoom to {fov_y_deg:.1f}° needs source "
                    f"{rebuilt.source_size}, renderer built for "
                    f"{(self._src_w, self._src_h)}; rebuild the StereoRenderer"
                )
            self._distorters[cam_name] = rebuilt
            self._model.cam_fovy[cid] = rebuilt.source_fov_y_deg

    def close(self) -> None:
        self._renderer.close()


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def _encode_depth(depth_f32: np.ndarray) -> str:
    """Encode a float32 H×W depth map as base64(float16 raw bytes)."""
    depth_f16 = depth_f32.astype(np.float16)
    return base64.b64encode(depth_f16.tobytes()).decode("ascii")


def _encode_seg(seg_raw: np.ndarray, model: "mujoco.MjModel" = None) -> str:
    """Encode MuJoCo's segmentation render as a base64 uint16 BODY-ID map.

    MuJoCo's segmentation pixel is [object_id, object_TYPE] — not
    [body_id, geom_id], as this function previously assumed.  For a normal
    scene render object_type is uniformly mjOBJ_GEOM (5), so channel 0 holds
    GEOM ids.  Emitting those raw was wrong in two ways:

      * a body built from several geoms produced several different labels for
        one object.  Every robot link is such a body, and so is any scene
        object compiled with more than one geom — so "one object, one label"
        silently did not hold;
      * the ids happened to coincide with body ids for the late-declared scene
        objects, which is why it looked correct when spot-checked on a cube.

    Mapping geom -> body via model.geom_bodyid fixes both.  Background is -1
    from MuJoCo and is emitted as 0 (the world body), so an unobserved pixel
    and the world read the same, as callers already assume.

    model may be None only for pre-mapped input (tests); then channel 0 is
    passed through unchanged.
    """
    obj_ids = seg_raw[:, :, 0]
    if model is not None:
        # -1 (background) must not index geom_bodyid; send it to world (0).
        safe = np.where(obj_ids >= 0, obj_ids, 0)
        body_ids = np.asarray(model.geom_bodyid)[safe]
        body_ids = np.where(obj_ids >= 0, body_ids, 0)
    else:
        body_ids = np.where(obj_ids >= 0, obj_ids, 0)
    return base64.b64encode(body_ids.astype(np.uint16).tobytes()).decode("ascii")


def decode_depth(depth_b64: str, height: int, width: int) -> np.ndarray:
    """Decode a depth_b64 string back to a float32 H×W numpy array."""
    raw = base64.b64decode(depth_b64)
    return np.frombuffer(raw, dtype=np.float16).reshape(height, width).astype(np.float32)


def decode_seg(seg_b64: str, height: int, width: int) -> np.ndarray:
    """Decode a seg_b64 string back to a uint16 H×W numpy array of body IDs."""
    raw = base64.b64decode(seg_b64)
    return np.frombuffer(raw, dtype=np.uint16).reshape(height, width)


def jpeg_to_b64(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("ascii")


def b64_to_jpeg(b64: str) -> bytes:
    return base64.b64decode(b64)
