"""Priority 1 (2026-09-15 matrix readiness, the matrix gate):
`native_mujoco/server.py::_build_recorder_manifest` must carry
`contacts_tracked`, sourced from `self._record_contacts` -- the same
attribute `native_mujoco/server.py:414` sets to `bool(record_dir)`. This is
the manifest field `scripts/e1_identity.py` and `scripts/link_e1_flight.py`
now refuse on when it isn't `True` (see test_e1_identity.py's
TestContactsTrackedManifestGate and test_link_e1_flight.py's
TestContactsTrackedManifestGate).

Offline: `_build_recorder_manifest` is called on a minimal stub carrying
only the attributes it reads -- no MuJoCo model, no server, no socket.
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "native_mujoco"))

import server as native_server  # noqa: E402


def _stub_server(*, record_contacts):
    """Enough attributes for `_build_recorder_manifest` to run without a
    real MuJoCo model or a live server."""
    return types.SimpleNamespace(
        _model_path="/tmp/does-not-need-to-exist.xml",
        _scene_path="/tmp/does-not-need-to-exist.yaml",
        _sim=types.SimpleNamespace(scene_revision="r1"),
        _calibration=None,
        _enable_depth=False,
        _enable_seg=False,
        _record_contacts=record_contacts,
        _effects=types.SimpleNamespace(
            blur_sigma=0.0, noise_std=0.0, drop_probability=0.0, latency_ms=0.0),
    )


def test_contacts_tracked_true_when_recording_contacts():
    stub = _stub_server(record_contacts=True)
    manifest = native_server.ReachyMujocoServer._build_recorder_manifest(stub)
    assert manifest["contacts_tracked"] is True


def test_contacts_tracked_false_when_not_recording_contacts():
    stub = _stub_server(record_contacts=False)
    manifest = native_server.ReachyMujocoServer._build_recorder_manifest(stub)
    assert manifest["contacts_tracked"] is False
