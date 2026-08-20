"""R12-801: Unit tests for the research context guard (learning safety)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from reachy_ai.experience.identity import (
    ResearchContextError,
    assert_research_context,
)
from reachy_ai.experience.models import SimulatorIdentity


def _valid_identity(**kw) -> SimulatorIdentity:
    defaults = dict(model_sha256="a" * 64, scene_sha256="b" * 64)
    defaults.update(kw)
    return SimulatorIdentity(**defaults)


# ---------------------------------------------------------------------------
# Physical execution guard
# ---------------------------------------------------------------------------

class TestPhysicalExecutionDenied:
    def test_passes_with_no_env(self, monkeypatch):
        monkeypatch.delenv("REACHY_ENABLE_MOTION", raising=False)
        assert_research_context()  # should not raise

    def test_passes_when_false(self, monkeypatch):
        monkeypatch.setenv("REACHY_ENABLE_MOTION", "false")
        assert_research_context()  # should not raise

    def test_passes_when_False_caps(self, monkeypatch):
        monkeypatch.setenv("REACHY_ENABLE_MOTION", "False")
        assert_research_context()  # case-insensitive

    def test_raises_when_true(self, monkeypatch):
        monkeypatch.setenv("REACHY_ENABLE_MOTION", "true")
        with pytest.raises(ResearchContextError, match="REACHY_ENABLE_MOTION=true"):
            assert_research_context()

    def test_raises_when_True_caps(self, monkeypatch):
        monkeypatch.setenv("REACHY_ENABLE_MOTION", "True")
        with pytest.raises(ResearchContextError):
            assert_research_context()

    def test_raises_when_true_no_identity(self, monkeypatch):
        monkeypatch.setenv("REACHY_ENABLE_MOTION", "true")
        with pytest.raises(ResearchContextError):
            assert_research_context(identity=None)

    def test_raises_when_true_valid_identity(self, monkeypatch):
        monkeypatch.setenv("REACHY_ENABLE_MOTION", "true")
        with pytest.raises(ResearchContextError):
            assert_research_context(identity=_valid_identity())


# ---------------------------------------------------------------------------
# Identity guard
# ---------------------------------------------------------------------------

class TestIdentityGuard:
    def test_passes_with_valid_identity(self, monkeypatch):
        monkeypatch.delenv("REACHY_ENABLE_MOTION", raising=False)
        assert_research_context(identity=_valid_identity())

    def test_passes_with_none_identity(self, monkeypatch):
        monkeypatch.delenv("REACHY_ENABLE_MOTION", raising=False)
        assert_research_context(identity=None)

    def test_raises_with_empty_model_sha(self, monkeypatch):
        monkeypatch.delenv("REACHY_ENABLE_MOTION", raising=False)
        bad = SimulatorIdentity(model_sha256="", scene_sha256="bbb")
        with pytest.raises(ResearchContextError, match="model_sha256"):
            assert_research_context(identity=bad)

    def test_raises_with_scene_path_but_no_hash(self, monkeypatch):
        monkeypatch.delenv("REACHY_ENABLE_MOTION", raising=False)
        bad = SimulatorIdentity(
            model_sha256="a" * 64,
            scene_sha256="",
            scene_source_path="/tmp/scene.yaml",
        )
        with pytest.raises(ResearchContextError, match="scene_sha256"):
            assert_research_context(identity=bad)

    def test_passes_with_empty_scene_sha_and_no_path(self, monkeypatch):
        """Empty scene hash is OK when there is no scene (model-only sim)."""
        monkeypatch.delenv("REACHY_ENABLE_MOTION", raising=False)
        ident = SimulatorIdentity(model_sha256="a" * 64, scene_sha256="")
        assert_research_context(identity=ident)  # no scene_source_path — OK

    def test_passes_with_full_scene_identity(self, monkeypatch):
        monkeypatch.delenv("REACHY_ENABLE_MOTION", raising=False)
        ident = _valid_identity(
            scene_source_path="/opt/scenes/control_panel.yaml",
        )
        assert_research_context(identity=ident)


# ---------------------------------------------------------------------------
# IdentityMismatchError propagation from the store
# ---------------------------------------------------------------------------

class TestIdentityMismatchStore:
    def test_empty_model_sha_raises_identity_mismatch(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore, IdentityMismatchError

        bad = SimulatorIdentity()  # model_sha256=""
        with ExperienceStore.open(tmp_path / "test.db") as store:
            with pytest.raises(IdentityMismatchError):
                store.query_compatible_trials(bad)

    def test_valid_identity_does_not_raise(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        ident = _valid_identity()
        with ExperienceStore.open(tmp_path / "test.db") as store:
            results = store.query_compatible_trials(ident)
        assert results == []
