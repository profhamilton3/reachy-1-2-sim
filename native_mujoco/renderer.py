"""
R12-404: Offscreen stereo RGB renderer for the native MuJoCo server.

Renders left_camera and right_camera from reachy_1_2.xml into JPEG bytes
using MuJoCo's Renderer (EGL/osmesa on Linux, Metal on macOS arm64).

Design constraints (from ADR-0001 and CLAUDE.md)
-------------------------------------------------
* Do not hold the state mutation lock while encoding images.
* Camera rendering must not advance sim state (read-only).
* Produce coherent stereo pairs: both cameras from the same sim_step.
* Thread-safe: one Renderer instance per OS thread (MuJoCo requirement).
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Dict, Optional

import mujoco
import numpy as np
from PIL import Image
import io

_DEFAULT_WIDTH  = 640
_DEFAULT_HEIGHT = 480
_DEFAULT_QUALITY = 85


@dataclass(frozen=True)
class RenderedFrame:
    camera: str
    jpeg_bytes: bytes
    width: int
    height: int
    render_us: int   # render + encode wall time in microseconds


class StereoRenderer:
    """Offscreen stereo renderer.  Create one per thread; do not share."""

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        jpeg_quality: int = _DEFAULT_QUALITY,
    ) -> None:
        self._model = model
        self._width = width
        self._height = height
        self._quality = jpeg_quality

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
        cameras: Optional[tuple[str, ...]] = None,
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
        rgb = self._renderer.render()   # uint8 H×W×3 ndarray

        jpeg_bytes = _encode_jpeg(rgb, self._quality)

        elapsed_us = (time.monotonic_ns() - t0) // 1_000
        return RenderedFrame(
            camera=cam_name,
            jpeg_bytes=jpeg_bytes,
            width=self._width,
            height=self._height,
            render_us=elapsed_us,
        )

    def close(self) -> None:
        self._renderer.close()


def _encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def jpeg_to_b64(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("ascii")


def b64_to_jpeg(b64: str) -> bytes:
    return base64.b64decode(b64)
