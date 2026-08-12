"""Unit tests for R12-601: depth/segmentation encode-decode helpers.

All tests run offline — no MuJoCo model or GPU required.
Only the pure numpy/base64 helpers (_encode_depth, _encode_seg,
decode_depth, decode_seg) are tested here.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "native_mujoco"))

from renderer import _encode_depth, _encode_seg, decode_depth, decode_seg


# ── Depth encode / decode ─────────────────────────────────────────────────────

class TestDepthRoundTrip:

    def test_encode_returns_nonempty_string(self):
        arr = np.zeros((48, 64), dtype=np.float32)
        result = _encode_depth(arr)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_decode_shape(self):
        h, w = 48, 64
        arr = np.ones((h, w), dtype=np.float32)
        b64 = _encode_depth(arr)
        out = decode_depth(b64, h, w)
        assert out.shape == (h, w)

    def test_decode_dtype_is_float32(self):
        arr = np.zeros((10, 10), dtype=np.float32)
        b64 = _encode_depth(arr)
        out = decode_depth(b64, 10, 10)
        assert out.dtype == np.float32

    def test_roundtrip_zeros(self):
        arr = np.zeros((48, 64), dtype=np.float32)
        out = decode_depth(_encode_depth(arr), 48, 64)
        np.testing.assert_array_equal(out, 0.0)

    def test_roundtrip_values_preserved(self):
        rng = np.random.default_rng(42)
        arr = rng.uniform(0.1, 3.0, size=(32, 32)).astype(np.float32)
        out = decode_depth(_encode_depth(arr), 32, 32)
        # float32 → float16 → float32 loses ~3 decimal places of precision
        np.testing.assert_allclose(out, arr, rtol=1e-2)

    def test_roundtrip_large_array(self):
        arr = np.full((480, 640), 1.5, dtype=np.float32)
        out = decode_depth(_encode_depth(arr), 480, 640)
        assert out.shape == (480, 640)
        np.testing.assert_allclose(out, 1.5, rtol=1e-2)

    def test_encode_is_base64_ascii(self):
        arr = np.zeros((8, 8), dtype=np.float32)
        b64 = _encode_depth(arr)
        # Should decode cleanly
        import base64
        raw = base64.b64decode(b64)
        assert len(raw) == 8 * 8 * 2   # float16 = 2 bytes each

    def test_roundtrip_float16_precision(self):
        # Values representable exactly in float16
        arr = np.array([[0.5, 1.0, 2.0], [0.25, 0.125, 4.0]], dtype=np.float32)
        out = decode_depth(_encode_depth(arr), 2, 3)
        np.testing.assert_allclose(out, arr, rtol=0)

    def test_encode_different_shapes_differ(self):
        arr_a = np.ones((10, 20), dtype=np.float32)
        arr_b = np.zeros((10, 20), dtype=np.float32)
        assert _encode_depth(arr_a) != _encode_depth(arr_b)


# ── Segmentation encode / decode ──────────────────────────────────────────────

class TestSegRoundTrip:

    def _make_seg_raw(self, h: int, w: int, body_ids=None, geom_ids=None) -> np.ndarray:
        """Build a fake H×W×2 int32 segmentation array."""
        arr = np.zeros((h, w, 2), dtype=np.int32)
        if body_ids is not None:
            arr[:, :, 0] = body_ids
        if geom_ids is not None:
            arr[:, :, 1] = geom_ids
        return arr

    def test_encode_returns_nonempty_string(self):
        raw = self._make_seg_raw(48, 64)
        result = _encode_seg(raw)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_decode_shape(self):
        h, w = 48, 64
        raw = self._make_seg_raw(h, w)
        b64 = _encode_seg(raw)
        out = decode_seg(b64, h, w)
        assert out.shape == (h, w)

    def test_decode_dtype_is_uint16(self):
        raw = self._make_seg_raw(10, 10)
        b64 = _encode_seg(raw)
        out = decode_seg(b64, 10, 10)
        assert out.dtype == np.uint16

    def test_roundtrip_zeros(self):
        raw = self._make_seg_raw(32, 32)
        out = decode_seg(_encode_seg(raw), 32, 32)
        np.testing.assert_array_equal(out, 0)

    def test_roundtrip_body_ids_preserved(self):
        h, w = 24, 32
        body_ids = np.arange(h * w, dtype=np.int32).reshape(h, w) % 255
        raw = self._make_seg_raw(h, w, body_ids=body_ids)
        out = decode_seg(_encode_seg(raw), h, w)
        np.testing.assert_array_equal(out, body_ids.astype(np.uint16))

    def test_geom_channel_discarded(self):
        h, w = 10, 10
        body_ids = np.ones((h, w), dtype=np.int32) * 5
        geom_ids = np.ones((h, w), dtype=np.int32) * 99
        raw = self._make_seg_raw(h, w, body_ids=body_ids, geom_ids=geom_ids)
        out = decode_seg(_encode_seg(raw), h, w)
        np.testing.assert_array_equal(out, 5)

    def test_encode_is_base64_ascii(self):
        raw = self._make_seg_raw(8, 8)
        b64 = _encode_seg(raw)
        import base64
        decoded = base64.b64decode(b64)
        assert len(decoded) == 8 * 8 * 2   # uint16 = 2 bytes each

    def test_roundtrip_large_array(self):
        h, w = 480, 640
        body_ids = np.ones((h, w), dtype=np.int32) * 7
        raw = self._make_seg_raw(h, w, body_ids=body_ids)
        out = decode_seg(_encode_seg(raw), h, w)
        assert out.shape == (h, w)
        np.testing.assert_array_equal(out, 7)

    def test_encode_different_body_ids_differ(self):
        raw_a = self._make_seg_raw(10, 10, body_ids=np.ones((10, 10), dtype=np.int32))
        raw_b = self._make_seg_raw(10, 10, body_ids=np.zeros((10, 10), dtype=np.int32))
        assert _encode_seg(raw_a) != _encode_seg(raw_b)

    def test_background_id_zero(self):
        raw = self._make_seg_raw(8, 8)   # all zeros = background
        out = decode_seg(_encode_seg(raw), 8, 8)
        assert out.min() == 0
        assert out.max() == 0
