"""R12-801: Unit tests for the SQLite ExperienceStore."""

from __future__ import annotations

import os
import pathlib
import sqlite3
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from reachy_ai.experience.models import (
    EpisodeConfig,
    EpisodeResult,
    EpisodeStatus,
    SimulatorIdentity,
    TaskSpec,
)
from reachy_ai.experience.store import (
    ExperienceStore,
    ExperienceStoreError,
    IdentityMismatchError,
)
from reachy_ai.motion.recipe import TrajectoryRecipe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _identity(**kw) -> SimulatorIdentity:
    defaults = dict(model_sha256="model_aaa", scene_sha256="scene_bbb")
    defaults.update(kw)
    return SimulatorIdentity(**defaults)


def _config(identity: SimulatorIdentity | None = None) -> EpisodeConfig:
    return EpisodeConfig(simulator_identity=identity or _identity(), seed=0)


def _task() -> TaskSpec:
    return TaskSpec(task_id="task-1", task_type="operate_control")


def _recipe_json() -> str:
    return TrajectoryRecipe(recipe_id="base_v1", task_type="operate_control").to_json()


def _succeeded_result(trial_id: str) -> EpisodeResult:
    return EpisodeResult(
        episode_id="ep-1",
        trial_id=trial_id,
        status=EpisodeStatus.SUCCEEDED,
        success=True,
        end_sim_step=1000,
        metrics={"clearance_m": 0.05},
    )


def _failed_result(trial_id: str) -> EpisodeResult:
    return EpisodeResult(
        episode_id="ep-2",
        trial_id=trial_id,
        status=EpisodeStatus.FAILED,
        success=False,
        hard_violations=["forbidden_contact"],
    )


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_experience.db"


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

class TestTrialLifecycle:
    def test_create_trial_returns_id(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
        assert isinstance(tid, str) and len(tid) == 32

    def test_create_starts_in_pending(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            row = store.get_trial(tid)
        assert row is not None
        assert row["status"] == EpisodeStatus.PENDING.value

    def test_start_transitions_to_running(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.start_trial(tid)
            row = store.get_trial(tid)
        assert row["status"] == EpisodeStatus.RUNNING.value
        assert row["started_at"] is not None

    def test_complete_success(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.start_trial(tid)
            store.complete_trial(tid, _succeeded_result(tid))
            row = store.get_trial(tid)
        assert row["status"] == EpisodeStatus.SUCCEEDED.value
        assert row["success"] == 1
        assert row["completed_at"] is not None

    def test_complete_failure(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.start_trial(tid)
            store.complete_trial(tid, _failed_result(tid))
            row = store.get_trial(tid)
        assert row["status"] == EpisodeStatus.FAILED.value
        assert row["success"] == 0

    def test_fail_trial(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.start_trial(tid)
            store.fail_trial(tid, "unexpected exception")
            row = store.get_trial(tid)
        assert row["status"] == EpisodeStatus.FAILED.value

    def test_abort_trial(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.start_trial(tid)
            store.abort_trial(tid, "reset timed out")
            row = store.get_trial(tid)
        assert row["status"] == EpisodeStatus.ABORTED.value

    def test_cancel_pending_trial(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.cancel_trial(tid)
            row = store.get_trial(tid)
        assert row["status"] == EpisodeStatus.CANCELLED.value

    def test_start_non_pending_raises(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.start_trial(tid)
            with pytest.raises(ExperienceStoreError):
                store.start_trial(tid)  # already RUNNING

    def test_complete_with_wrong_status_raises(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.start_trial(tid)
            bad_result = EpisodeResult(
                episode_id="e", trial_id=tid, status=EpisodeStatus.ABORTED
            )
            with pytest.raises(ExperienceStoreError, match="SUCCEEDED or FAILED"):
                store.complete_trial(tid, bad_result)

    def test_abort_already_terminal_raises(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.start_trial(tid)
            store.complete_trial(tid, _succeeded_result(tid))
            with pytest.raises(ExperienceStoreError):
                store.abort_trial(tid, "too late")

    def test_result_json_persisted(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.start_trial(tid)
            store.complete_trial(tid, _succeeded_result(tid))
            row = store.get_trial(tid)
        result = EpisodeResult.from_json(row["result_json"])
        assert result.metrics["clearance_m"] == pytest.approx(0.05)

    def test_identity_json_persisted(self, db_path):
        ident = _identity(mujoco_version="3.2.0", host_arch="arm64")
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config(ident))
            row = store.get_trial(tid)
        recovered = SimulatorIdentity.from_json(row["identity_json"])
        assert recovered.mujoco_version == "3.2.0"


# ---------------------------------------------------------------------------
# Orphan recovery
# ---------------------------------------------------------------------------

class TestOrphanRecovery:
    def test_running_trial_recovered_as_aborted(self, db_path):
        with ExperienceStore.open(db_path, recover_orphans=False) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.start_trial(tid)

        with ExperienceStore.open(db_path, orphan_threshold_s=0.0) as store:
            row = store.get_trial(tid)
        assert row["status"] == EpisodeStatus.ABORTED.value

    def test_succeeded_not_recovered(self, db_path):
        with ExperienceStore.open(db_path, recover_orphans=False) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.start_trial(tid)
            store.complete_trial(tid, _succeeded_result(tid))

        with ExperienceStore.open(db_path, orphan_threshold_s=0.0) as store:
            row = store.get_trial(tid)
        assert row["status"] == EpisodeStatus.SUCCEEDED.value

    def test_pending_not_recovered(self, db_path):
        with ExperienceStore.open(db_path, recover_orphans=False) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())

        with ExperienceStore.open(db_path, orphan_threshold_s=0.0) as store:
            row = store.get_trial(tid)
        assert row["status"] == EpisodeStatus.PENDING.value

    def test_fresh_running_not_recovered_by_high_threshold(self, db_path):
        with ExperienceStore.open(db_path, recover_orphans=False) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.start_trial(tid)

        with ExperienceStore.open(db_path, orphan_threshold_s=3600.0) as store:
            row = store.get_trial(tid)
        assert row["status"] == EpisodeStatus.RUNNING.value


# ---------------------------------------------------------------------------
# Compatible trial query
# ---------------------------------------------------------------------------

class TestCompatibleQuery:
    def test_returns_succeeded_with_matching_identity(self, db_path):
        ident = _identity()
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config(ident))
            store.start_trial(tid)
            store.complete_trial(tid, _succeeded_result(tid))
            results = store.query_compatible_trials(ident)
        assert any(r["trial_id"] == tid for r in results)

    def test_excludes_different_model_hash(self, db_path):
        ident_a = _identity(model_sha256="model_a")
        ident_b = _identity(model_sha256="model_b")
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config(ident_a))
            store.start_trial(tid)
            store.complete_trial(tid, _succeeded_result(tid))
            results = store.query_compatible_trials(ident_b)
        assert not any(r["trial_id"] == tid for r in results)

    def test_excludes_different_scene_hash(self, db_path):
        ident_a = _identity(scene_sha256="scene_a")
        ident_b = _identity(scene_sha256="scene_b")
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config(ident_a))
            store.start_trial(tid)
            store.complete_trial(tid, _succeeded_result(tid))
            results = store.query_compatible_trials(ident_b)
        assert not any(r["trial_id"] == tid for r in results)

    def test_excludes_failed_trials(self, db_path):
        ident = _identity()
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config(ident))
            store.start_trial(tid)
            store.complete_trial(tid, _failed_result(tid))
            results = store.query_compatible_trials(ident)
        assert not any(r["trial_id"] == tid for r in results)

    def test_task_type_filter(self, db_path):
        ident = _identity()
        task_cp = TaskSpec(task_id="cp1", task_type="operate_control")
        task_pp = TaskSpec(task_id="pp1", task_type="pick_and_place")
        with ExperienceStore.open(db_path) as store:
            tid_cp = store.create_trial("study-1", task_cp, _recipe_json(), _config(ident))
            tid_pp = store.create_trial("study-1", task_pp, _recipe_json(), _config(ident))
            store.start_trial(tid_cp)
            store.start_trial(tid_pp)
            store.complete_trial(tid_cp, _succeeded_result(tid_cp))
            store.complete_trial(tid_pp, _succeeded_result(tid_pp))
            results = store.query_compatible_trials(ident, task_type="operate_control")
        ids = [r["trial_id"] for r in results]
        assert tid_cp in ids
        assert tid_pp not in ids

    def test_empty_model_sha_raises_identity_mismatch(self, db_path):
        bad_ident = SimulatorIdentity()  # model_sha256 is ""
        with ExperienceStore.open(db_path) as store:
            with pytest.raises(IdentityMismatchError):
                store.query_compatible_trials(bad_ident)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

class TestArtifacts:
    def test_attach_and_retrieve(self, db_path, tmp_path):
        artifact = tmp_path / "trace.jsonl"
        artifact.write_text("line1\nline2\n")
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.attach_artifact(tid, "trace", str(artifact), kind="recorder")
            arts = store.get_artifacts(tid)
        assert len(arts) == 1
        assert arts[0]["name"] == "trace"
        assert arts[0]["kind"] == "recorder"
        assert arts[0]["sha256"] is not None

    def test_non_existent_file_no_sha(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.attach_artifact(tid, "missing", "/tmp/does_not_exist.bin")
            arts = store.get_artifacts(tid)
        assert arts[0]["sha256"] is None
        assert arts[0]["size_bytes"] is None


# ---------------------------------------------------------------------------
# WAL mode and migration
# ---------------------------------------------------------------------------

class TestStoreInfrastructure:
    def test_wal_mode_enabled(self, db_path):
        with ExperienceStore.open(db_path) as store:
            journal = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal.lower() == "wal"

    def test_schema_version_set(self, db_path):
        with ExperienceStore.open(db_path) as store:
            ver = store._conn.execute("PRAGMA user_version").fetchone()[0]
        assert ver >= 1

    def test_parent_directory_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "experience.db"
        with ExperienceStore.open(nested):
            pass
        assert nested.exists()

    def test_reopen_preserves_data(self, db_path):
        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial("study-1", _task(), _recipe_json(), _config())
        with ExperienceStore.open(db_path) as store:
            row = store.get_trial(tid)
        assert row is not None
        assert row["study_id"] == "study-1"

    def test_list_trials(self, db_path):
        with ExperienceStore.open(db_path) as store:
            store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.create_trial("study-1", _task(), _recipe_json(), _config())
            store.create_trial("study-2", _task(), _recipe_json(), _config())
            rows = store.list_trials("study-1")
        assert len(rows) == 2
