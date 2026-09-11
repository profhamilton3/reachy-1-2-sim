"""Issue #40: /status must report the backend that actually wrote the frames
currently on disk, not a hardcoded string.

_detect_backend() reads the sidecar camera_fixture.frame_file_writer and
mujoco_remote_backend._ingest_camera_frame each maintain for their own
backend; these tests only cover that read side.
"""

import json
import os
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
