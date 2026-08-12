"""
R12-602: Sensor-effect pipeline for simulated camera degradation.

Each effect is independently disableable.  A deterministic seed produces
byte-identical output for the same simulation run, enabling reproducible
research comparisons.

Effects applied per frame (in order):
  1. blur          — Gaussian blur via PIL (sigma in pixels; 0 = disabled)
  2. noise         — additive Gaussian noise (std in pixel units; 0 = disabled)
  3. exposure      — multiplicative brightness scale + optional gamma (1.0 = disabled)
  4. jpeg_quality  — JPEG re-compression quality (None = use renderer setting)
  5. drop          — drop this frame entirely (probability in [0, 1]; 0 = disabled)

Latency and rate jitter affect frame timing in the server dispatch loop, not
pixel content, so they are configured here but applied in server.py.

Configuration (YAML):
  blur_sigma: 0.0           # pixels; 0 disables
  noise_std: 0.0            # pixel units [0-255]; 0 disables
  exposure_scale: 1.0       # multiplicative scale; 1.0 disables
  gamma: 1.0                # gamma exponent; 1.0 disables
  jpeg_quality: null        # override renderer JPEG quality; null = no change
  drop_probability: 0.0     # fraction of frames dropped; 0 disables
  latency_ms: 0.0           # frame hold time before release (ms); 0 disables
  rate_jitter_fraction: 0.0 # fractional ± jitter on frame period; 0 disables
  seed: null                # integer RNG seed; null = non-deterministic
"""

from __future__ import annotations

import io
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
from PIL import Image as _PIL
from PIL.ImageFilter import GaussianBlur as _GaussianBlur


@dataclass
class EffectConfig:
    blur_sigma: float = 0.0
    noise_std: float = 0.0
    exposure_scale: float = 1.0
    gamma: float = 1.0
    jpeg_quality: Optional[int] = None
    drop_probability: float = 0.0
    latency_ms: float = 0.0
    rate_jitter_fraction: float = 0.0
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EffectConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str) -> "EffectConfig":
        import yaml
        import pathlib
        with pathlib.Path(path).open() as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def any_pixel_effect(self) -> bool:
        return (
            self.blur_sigma > 0
            or self.noise_std > 0
            or self.exposure_scale != 1.0
            or self.gamma != 1.0
        )

    def disabled(self) -> bool:
        return (
            self.blur_sigma == 0.0
            and self.noise_std == 0.0
            and self.exposure_scale == 1.0
            and self.gamma == 1.0
            and self.jpeg_quality is None
            and self.drop_probability == 0.0
            and self.latency_ms == 0.0
            and self.rate_jitter_fraction == 0.0
        )


@dataclass
class _PendingFrame:
    jpeg_bytes: bytes
    release_at: float   # monotonic time when frame may be released


class SensorEffectPipeline:
    """Applies configurable sensor degradation effects to camera frames.

    Thread-safety: one instance per camera stream; do not share across threads.
    """

    def __init__(self, config: EffectConfig) -> None:
        self._cfg = config
        self._rng = (
            np.random.default_rng(config.seed)
            if config.seed is not None
            else np.random.default_rng()
        )
        # Latency buffer: holds frames until their release_at time.
        self._latency_buf: Deque[_PendingFrame] = deque()

    @property
    def config(self) -> EffectConfig:
        return self._cfg

    # ── Per-frame pixel pipeline ──────────────────────────────────────────────

    def apply_pixels(self, jpeg_bytes: bytes) -> Optional[bytes]:
        """Apply all pixel-level effects to a JPEG frame.

        Returns the processed JPEG bytes, or None if the frame is dropped.

        The returned bytes use the configured jpeg_quality if set, otherwise
        the same quality level is preserved via re-encoding.
        """
        cfg = self._cfg

        # Drop check (before any processing)
        if cfg.drop_probability > 0.0:
            if self._rng.random() < cfg.drop_probability:
                return None

        if not cfg.any_pixel_effect() and cfg.jpeg_quality is None:
            return jpeg_bytes  # pass-through

        img = _PIL.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32)

        # 1. Blur
        if cfg.blur_sigma > 0.0:
            img = img.filter(_GaussianBlur(radius=cfg.blur_sigma))
            arr = np.array(img, dtype=np.float32)

        # 2. Exposure (multiplicative scale)
        if cfg.exposure_scale != 1.0:
            arr = arr * cfg.exposure_scale

        # 3. Gamma
        if cfg.gamma != 1.0:
            arr = np.clip(arr, 0.0, 255.0) / 255.0
            arr = np.power(arr, cfg.gamma)
            arr = arr * 255.0

        # 4. Noise (additive Gaussian)
        if cfg.noise_std > 0.0:
            noise = self._rng.normal(0.0, cfg.noise_std, arr.shape)
            arr = arr + noise

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        out_img = _PIL.fromarray(arr, mode="RGB")

        quality = cfg.jpeg_quality if cfg.jpeg_quality is not None else 85
        buf = io.BytesIO()
        out_img.save(buf, format="JPEG", quality=quality, subsampling=0)
        return buf.getvalue()

    # ── Timing effects (applied in server dispatch loop) ─────────────────────

    def push_latency(self, jpeg_bytes: bytes) -> None:
        """Buffer a frame for latency simulation."""
        release_at = time.monotonic() + self._cfg.latency_ms / 1000.0
        self._latency_buf.append(_PendingFrame(jpeg_bytes, release_at))

    def pop_ready(self) -> Optional[bytes]:
        """Return the oldest buffered frame if its latency hold has elapsed."""
        if self._latency_buf and time.monotonic() >= self._latency_buf[0].release_at:
            return self._latency_buf.popleft().jpeg_bytes
        return None

    def jitter_sleep(self, base_period_s: float) -> float:
        """Return a jittered sleep duration around base_period_s."""
        frac = self._cfg.rate_jitter_fraction
        if frac <= 0.0:
            return base_period_s
        jitter = self._rng.uniform(-frac, frac)
        return max(0.0, base_period_s * (1.0 + jitter))
