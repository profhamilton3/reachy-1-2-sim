"""Issue #40: /status must report the backend that actually wrote the frames
currently on disk, not a hardcoded string.

_detect_backend() reads the sidecar camera_fixture.frame_file_writer and
mujoco_remote_backend._ingest_camera_frame each maintain for their own
backend; these tests only cover that read side.

A6 (follow-up review): "backend" (source) and "frames_stale" (freshness) are
deliberately separate signals -- TestFramesStale covers the one that answers
whether frames are actually still arriving, including the stalled-writer case
`_detect_backend()` cannot see: camera_fixture.py's `frame_file_writer`
re-stamps the sidecar's `wall_time_ns` every tick even when no new frame was
written, so a sidecar-only staleness check would read a stalled fixture
capture as live. `_frames_stale()` is deliberately keyed to the frame files'
own mtimes instead, which only change when a frame is actually written.
"""

import json
import os
import time
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../web"))

import camera_server  # noqa: E402


class TestDetectBackend:
    def test_reports_the_named_backend(self, tmp_path, monkeypatch):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps({"backend": "mujoco-remote",
                                    "wall_time_ns": 123}))
        monkeypatch.setattr(camera_server, "_FRAME_META_FILE", str(meta))
        assert camera_server._detect_backend() == "mujoco-remote"

    def test_reports_fixture_when_that_is_what_wrote_it(self, tmp_path, monkeypatch):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps({"backend": "fixture", "wall_time_ns": 1}))
        monkeypatch.setattr(camera_server, "_FRAME_META_FILE", str(meta))
        assert camera_server._detect_backend() == "fixture"

    def test_missing_sidecar_is_unknown_not_a_guess(self, tmp_path, monkeypatch):
        """No writer has ever run (or the field it names is gone): say so
        plainly rather than falling back to the old hardcoded lie."""
        monkeypatch.setattr(camera_server, "_FRAME_META_FILE",
                            str(tmp_path / "does_not_exist.json"))
        assert camera_server._detect_backend() == "unknown"

    def test_malformed_sidecar_is_unknown_not_a_crash(self, tmp_path, monkeypatch):
        meta = tmp_path / "meta.json"
        meta.write_text("not json")
        monkeypatch.setattr(camera_server, "_FRAME_META_FILE", str(meta))
        assert camera_server._detect_backend() == "unknown"

    def test_sidecar_missing_the_backend_key_is_unknown(self, tmp_path, monkeypatch):
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps({"wall_time_ns": 1}))
        monkeypatch.setattr(camera_server, "_FRAME_META_FILE", str(meta))
        assert camera_server._detect_backend() == "unknown"


class TestFramesStale:
    """A6: frame freshness, judged from the frame files' own mtimes -- never
    from the sidecar, which a stalled-but-still-running writer can keep
    touching (see module docstring)."""

    def test_fresh_frames_are_not_stale(self):
        assert camera_server._frames_stale(0, 0) is False
        assert camera_server._frames_stale(10, 10) is False

    def test_either_camera_past_the_threshold_is_stale(self):
        over = int(camera_server._STALE_THRESHOLD_MS) + 1
        assert camera_server._frames_stale(over, 0) is True
        assert camera_server._frames_stale(0, over) is True

    def test_a_missing_frame_file_is_stale_not_a_perfect_zero(self):
        """_frame_age_ms returns -1 for a file that does not exist (OSError);
        that must read as stale, not as fresher than every real reading."""
        assert camera_server._frames_stale(-1, 0) is True
        assert camera_server._frames_stale(0, -1) is True

    def test_a_stalled_writer_is_caught_even_with_a_fresh_sidecar(
            self, tmp_path, monkeypatch):
        """The scenario camera_fixture.py's per-tick sidecar stamp creates:
        the capture loop is still running (so wall_time_ns keeps advancing)
        but no new frame has actually been written (so the JPEG's mtime is
        old). Reading `_detect_backend()` alone here would say "fixture",
        unqualified -- `_frames_stale` must independently catch this from the
        frame file's own age, which is what this test asserts end to end."""
        left = tmp_path / "left.jpg"
        right = tmp_path / "right.jpg"
        left.write_bytes(b"stale-frame")
        right.write_bytes(b"stale-frame")
        old = time.time() - 5.0  # 5 s: far past _STALE_THRESHOLD_MS
        os.utime(left, (old, old))
        os.utime(right, (old, old))
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps({"backend": "fixture",
                                    "wall_time_ns": time.time_ns()}))

        monkeypatch.setattr(camera_server, "_LEFT_FILE", str(left))
        monkeypatch.setattr(camera_server, "_RIGHT_FILE", str(right))
        monkeypatch.setattr(camera_server, "_FRAME_META_FILE", str(meta))

        assert camera_server._detect_backend() == "fixture"
        left_age = camera_server._frame_age_ms(str(left))
        right_age = camera_server._frame_age_ms(str(right))
        assert camera_server._frames_stale(left_age, right_age) is True
