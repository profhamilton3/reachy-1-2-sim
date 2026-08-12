"""Unit tests for R12-602: sensor-effect pipeline.

All tests run offline — no MuJoCo required.
"""

import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "native_mujoco"))

from sensor_effects import EffectConfig, SensorEffectPipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jpeg(width: int = 64, height: int = 48, color=(128, 128, 128)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _decode_jpeg(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


# ── EffectConfig ──────────────────────────────────────────────────────────────

class TestEffectConfig:

    def test_defaults_all_disabled(self):
        cfg = EffectConfig()
        assert cfg.disabled()

    def test_from_dict(self):
        cfg = EffectConfig.from_dict({"blur_sigma": 1.5, "noise_std": 5.0})
        assert cfg.blur_sigma == pytest.approx(1.5)
        assert cfg.noise_std == pytest.approx(5.0)
        assert cfg.drop_probability == 0.0

    def test_from_dict_ignores_unknown_keys(self):
        cfg = EffectConfig.from_dict({"blur_sigma": 1.0, "nonexistent": 99})
        assert cfg.blur_sigma == pytest.approx(1.0)

    def test_from_yaml(self, tmp_path):
        path = tmp_path / "effects.yaml"
        path.write_text("blur_sigma: 2.0\nnoise_std: 3.0\n")
        cfg = EffectConfig.from_yaml(str(path))
        assert cfg.blur_sigma == pytest.approx(2.0)
        assert cfg.noise_std == pytest.approx(3.0)

    def test_any_pixel_effect_false_when_clean(self):
        assert not EffectConfig().any_pixel_effect()

    def test_any_pixel_effect_true_with_blur(self):
        assert EffectConfig(blur_sigma=1.0).any_pixel_effect()

    def test_any_pixel_effect_true_with_noise(self):
        assert EffectConfig(noise_std=2.0).any_pixel_effect()

    def test_any_pixel_effect_true_with_exposure(self):
        assert EffectConfig(exposure_scale=1.5).any_pixel_effect()


# ── Pass-through (no effects) ─────────────────────────────────────────────────

class TestPassThrough:

    def test_disabled_returns_same_bytes(self):
        cfg = EffectConfig()
        pipe = SensorEffectPipeline(cfg)
        jpeg = _make_jpeg()
        result = pipe.apply_pixels(jpeg)
        assert result == jpeg

    def test_none_quality_no_recompression(self):
        cfg = EffectConfig(jpeg_quality=None)
        pipe = SensorEffectPipeline(cfg)
        jpeg = _make_jpeg()
        result = pipe.apply_pixels(jpeg)
        assert result == jpeg


# ── Blur ──────────────────────────────────────────────────────────────────────

class TestBlur:

    def test_blur_changes_output(self):
        jpeg = _make_jpeg(64, 48, color=(200, 50, 50))
        # Image with a sharp edge — blur should make it softer
        arr = np.zeros((48, 64, 3), dtype=np.uint8)
        arr[:, :32] = [255, 0, 0]
        arr[:, 32:] = [0, 255, 0]
        img = Image.fromarray(arr, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        sharp_jpeg = buf.getvalue()

        pipe = SensorEffectPipeline(EffectConfig(blur_sigma=3.0, seed=0))
        result = pipe.apply_pixels(sharp_jpeg)
        assert result is not None
        out = _decode_jpeg(result)
        # At the boundary column, the R channel should be between 0 and 255
        mid_col = out[:, 31, 0].mean()
        assert 50 < mid_col < 230, f"Expected blurred boundary, got {mid_col}"

    def test_blur_returns_bytes(self):
        pipe = SensorEffectPipeline(EffectConfig(blur_sigma=1.0))
        result = pipe.apply_pixels(_make_jpeg())
        assert isinstance(result, bytes)


# ── Noise ─────────────────────────────────────────────────────────────────────

class TestNoise:

    def test_noise_changes_pixel_values(self):
        jpeg = _make_jpeg(64, 48, color=(128, 128, 128))
        pipe = SensorEffectPipeline(EffectConfig(noise_std=20.0, seed=42))
        result = pipe.apply_pixels(jpeg)
        assert result is not None
        out = _decode_jpeg(result)
        # With std=20, the output should not be uniform 128
        assert out.std() > 2.0

    def test_noise_deterministic_with_seed(self):
        jpeg = _make_jpeg(64, 48)
        pipe1 = SensorEffectPipeline(EffectConfig(noise_std=10.0, seed=7))
        pipe2 = SensorEffectPipeline(EffectConfig(noise_std=10.0, seed=7))
        r1 = pipe1.apply_pixels(jpeg)
        r2 = pipe2.apply_pixels(jpeg)
        assert r1 == r2

    def test_noise_different_seeds_differ(self):
        jpeg = _make_jpeg(64, 48)
        r1 = SensorEffectPipeline(EffectConfig(noise_std=10.0, seed=1)).apply_pixels(jpeg)
        r2 = SensorEffectPipeline(EffectConfig(noise_std=10.0, seed=2)).apply_pixels(jpeg)
        assert r1 != r2

    def test_noise_clamps_to_valid_range(self):
        jpeg = _make_jpeg(64, 48, color=(250, 250, 250))
        pipe = SensorEffectPipeline(EffectConfig(noise_std=30.0, seed=0))
        result = pipe.apply_pixels(jpeg)
        assert result is not None
        out = _decode_jpeg(result)
        assert out.max() <= 255
        assert out.min() >= 0


# ── Exposure ──────────────────────────────────────────────────────────────────

class TestExposure:

    def test_overexposure(self):
        jpeg = _make_jpeg(64, 48, color=(100, 100, 100))
        pipe = SensorEffectPipeline(EffectConfig(exposure_scale=2.5, seed=0))
        result = pipe.apply_pixels(jpeg)
        assert result is not None
        out = _decode_jpeg(result)
        assert out.mean() > 150  # brighter than original

    def test_underexposure(self):
        jpeg = _make_jpeg(64, 48, color=(200, 200, 200))
        pipe = SensorEffectPipeline(EffectConfig(exposure_scale=0.3, seed=0))
        result = pipe.apply_pixels(jpeg)
        assert result is not None
        out = _decode_jpeg(result)
        assert out.mean() < 100  # darker than original


# ── Drop frames ───────────────────────────────────────────────────────────────

class TestDropFrames:

    def test_zero_probability_never_drops(self):
        jpeg = _make_jpeg()
        pipe = SensorEffectPipeline(EffectConfig(drop_probability=0.0, seed=0))
        for _ in range(20):
            assert pipe.apply_pixels(jpeg) is not None

    def test_probability_one_always_drops(self):
        jpeg = _make_jpeg()
        pipe = SensorEffectPipeline(EffectConfig(drop_probability=1.0, seed=0))
        for _ in range(10):
            assert pipe.apply_pixels(jpeg) is None

    def test_partial_probability_drops_some(self):
        jpeg = _make_jpeg()
        pipe = SensorEffectPipeline(EffectConfig(drop_probability=0.5, seed=123))
        results = [pipe.apply_pixels(jpeg) for _ in range(100)]
        dropped = sum(1 for r in results if r is None)
        # Expect ~50% dropped; accept 20–80 for statistical safety
        assert 20 <= dropped <= 80, f"Unexpected drop count: {dropped}"

    def test_drop_deterministic_with_seed(self):
        jpeg = _make_jpeg()
        def run(seed):
            pipe = SensorEffectPipeline(EffectConfig(drop_probability=0.3, seed=seed))
            return [pipe.apply_pixels(jpeg) is None for _ in range(20)]
        assert run(42) == run(42)
        assert run(1) != run(2)


# ── Latency buffer ────────────────────────────────────────────────────────────

class TestLatencyBuffer:

    def test_zero_latency_pop_immediately(self):
        pipe = SensorEffectPipeline(EffectConfig(latency_ms=0.0))
        # With no latency push/pop logic involved, just check pop returns None
        assert pipe.pop_ready() is None

    def test_push_then_pop_after_delay(self):
        import time
        pipe = SensorEffectPipeline(EffectConfig(latency_ms=50.0))
        jpeg = _make_jpeg()
        pipe.push_latency(jpeg)
        # Should not be ready immediately
        assert pipe.pop_ready() is None
        # After 60 ms it should be ready
        time.sleep(0.06)
        result = pipe.pop_ready()
        assert result == jpeg

    def test_pop_empty_buffer_returns_none(self):
        pipe = SensorEffectPipeline(EffectConfig(latency_ms=10.0))
        assert pipe.pop_ready() is None


# ── JPEG quality override ─────────────────────────────────────────────────────

class TestJpegQuality:

    def test_low_quality_produces_smaller_output(self):
        jpeg = _make_jpeg(640, 480)
        hi = SensorEffectPipeline(EffectConfig(jpeg_quality=95, noise_std=1.0, seed=0)).apply_pixels(jpeg)
        lo = SensorEffectPipeline(EffectConfig(jpeg_quality=20, noise_std=1.0, seed=0)).apply_pixels(jpeg)
        assert hi is not None and lo is not None
        assert len(lo) < len(hi)
