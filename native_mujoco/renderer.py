"""
R12-404 / R12-601: Offscreen stereo RGB + optional depth/segmentation renderer.

Renders left_camera and right_camera from reachy_1_2.xml using MuJoCo's
Renderer (EGL/osmesa on Linux, Metal on macOS arm64).

Depth output (R12-601):
  float16 numpy array, shape H×W, values in metres.
  Encoded as base64(raw float16 bytes) for transmission.

Segmentation output (R12-601):
  uint16 numpy array, shape H×W, values are MuJoCo body IDs.
  Encoded as base64(raw uint16 bytes).
  Body ID 0 = world/background.

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

        self._renderer = mujoco.Renderer(model, height=height, width=width)

        # Resolve camera IDs once
        self._cam_ids: Dict[str, int] = {}
        for name in ("left_camera", "right_camera"):
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                raise RuntimeError(f"Camera '{name}' not found in model")
            self._cam_ids[name] = cid

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

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
            seg_raw = self._renderer.render()          # H×W×2 int32 [body, geom]
            self._renderer.disable_segmentation_rendering()
            if distorter is not None:
                # Warp the body-ID channel only; fill 0 = world/background.
                body = distorter.apply_label(seg_raw[:, :, 0], fill=0)
                seg_raw = np.stack([body, np.zeros_like(body)], axis=-1)
            seg_b64 = _encode_seg(seg_raw)

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


def _encode_seg(seg_raw: np.ndarray) -> str:
    """Encode an H×W×2 int32 segmentation array as base64(uint16 body-ID map)."""
    # MuJoCo segmentation pixel: [body_id, geom_id]; take body_id channel.
    body_ids = seg_raw[:, :, 0].astype(np.uint16)
    return base64.b64encode(body_ids.tobytes()).decode("ascii")


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
