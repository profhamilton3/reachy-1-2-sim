"""R12-801: SQLite-backed experience store for Epic 8 trials.

One study can contain many trials.  Each trial binds a SimulatorIdentity,
TaskSpec, TrajectoryRecipe (as JSON), EpisodeConfig, and EpisodeResult into
a single durable record.

Design decisions
----------------
- Schema version via PRAGMA user_version; migrations run automatically on
  every open() so the file is always up-to-date.
- Large traces (recorder HDF5, image stacks) stay in artifact files; this
  store holds only metadata, paths, and indexes.
- WAL journal mode allows one writer and multiple concurrent readers without
  blocking.
- Orphan recovery marks stale RUNNING trials as ABORTED on startup — a
  crashed episode runner leaves them; the threshold is configurable via
  REACHY_STORE_ORPHAN_THRESHOLD_S (default 3600 s).
- All writes are transactional (conn as context manager commits or rolls back).

Thread safety: the store is not safe across OS processes without external
locking.  Within one process, SQLite WAL handles concurrent reads.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import pathlib
import sqlite3
import uuid
from typing import Any, Dict, Iterator, List, Optional

from .models import (
    EpisodeConfig,
    EpisodeResult,
    EpisodeStatus,
    SimulatorIdentity,
    TaskSpec,
    TrialRecord,
)

_SCHEMA_VERSION = 1
_DEFAULT_ORPHAN_THRESHOLD_S = 3600.0

_DDL = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id                TEXT PRIMARY KEY,
    study_id                TEXT NOT NULL,
    task_type               TEXT NOT NULL,
    task_spec_json          TEXT NOT NULL,
    recipe_json             TEXT NOT NULL,
    config_json             TEXT NOT NULL,
    result_json             TEXT NOT NULL,
    identity_json           TEXT NOT NULL,
    status                  TEXT NOT NULL,
    success                 INTEGER NOT NULL DEFAULT 0,
    model_sha256            TEXT NOT NULL DEFAULT '',
    scene_sha256            TEXT NOT NULL DEFAULT '',
    scene_revision          TEXT NOT NULL DEFAULT '',
    created_at              TEXT NOT NULL,
    started_at              TEXT,
    completed_at            TEXT,
    optimizer_metadata_json TEXT,
    parent_trial_id         TEXT NOT NULL DEFAULT '',
    warm_start_trial_id     TEXT NOT NULL DEFAULT '',
    promotion_state         TEXT NOT NULL DEFAULT 'unpromoted',
    review_metadata_json    TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id    TEXT    NOT NULL REFERENCES trials(trial_id),
    name        TEXT    NOT NULL,
    path        TEXT    NOT NULL,
    kind        TEXT    NOT NULL DEFAULT 'file',
    size_bytes  INTEGER,
    sha256      TEXT,
    created_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trials_study
    ON trials(study_id);
CREATE INDEX IF NOT EXISTS idx_trials_status
    ON trials(status, success);
CREATE INDEX IF NOT EXISTS idx_trials_identity
    ON trials(model_sha256, scene_sha256, status);
CREATE INDEX IF NOT EXISTS idx_artifacts_trial
    ON artifacts(trial_id);
"""

_TERMINAL = (
    EpisodeStatus.SUCCEEDED.value,
    EpisodeStatus.FAILED.value,
    EpisodeStatus.INVALID.value,
    EpisodeStatus.ABORTED.value,
    EpisodeStatus.CANCELLED.value,
)


class ExperienceStoreError(RuntimeError):
    """Base error for store operations."""


class IdentityMismatchError(ExperienceStoreError):
    """Raised when a query is attempted with an incompatible SimulatorIdentity."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _sha256_file(path: pathlib.Path) -> Optional[str]:
    try:
        if path.stat().st_size > 50 * 1024 * 1024:
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _migrate(conn: sqlite3.Connection) -> None:
    current: int = conn.execute("PRAGMA user_version").fetchone()[0]
    if current < _SCHEMA_VERSION:
        conn.executescript(_DDL)
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        conn.commit()


def _row_dict(cursor: sqlite3.Cursor, row: tuple) -> Dict[str, Any]:
    return {desc[0]: val for desc, val in zip(cursor.description, row)}


# ---------------------------------------------------------------------------
# ExperienceStore
# ---------------------------------------------------------------------------

class ExperienceStore:
    """Persistent SQLite store for trial records.

    Use as a context manager via open():

        with ExperienceStore.open(db_path) as store:
            tid = store.create_trial(study_id, task_spec, recipe_json, config)
            store.start_trial(tid)
            store.complete_trial(tid, result)
    """

    def __init__(self, conn: sqlite3.Connection, db_path: pathlib.Path) -> None:
        self._conn = conn
        self._db_path = db_path

    @classmethod
    @contextlib.contextmanager
    def open(
        cls,
        db_path: str | pathlib.Path,
        *,
        recover_orphans: bool = True,
        orphan_threshold_s: Optional[float] = None,
    ) -> Iterator["ExperienceStore"]:
        """Open (creating if necessary) the experience database.

        Runs schema migration and orphan recovery before yielding.
        Always closes the connection on exit, even on exception.
        """
        p = pathlib.Path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _migrate(conn)
        store = cls(conn, p)
        if recover_orphans:
            threshold = orphan_threshold_s if orphan_threshold_s is not None else float(
                os.environ.get("REACHY_STORE_ORPHAN_THRESHOLD_S", str(_DEFAULT_ORPHAN_THRESHOLD_S))
            )
            store._recover_orphans(threshold)
        try:
            yield store
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Trial lifecycle
    # ------------------------------------------------------------------

    def create_trial(
        self,
        study_id: str,
        task_spec: TaskSpec,
        recipe_json: str,
        config: EpisodeConfig,
        *,
        parent_trial_id: str = "",
        warm_start_trial_id: str = "",
        optimizer_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new trial in PENDING state. Returns the new trial_id."""
        trial_id = uuid.uuid4().hex
        identity = config.simulator_identity
        now = _now_iso()
        result = EpisodeResult(
            episode_id=uuid.uuid4().hex,
            trial_id=trial_id,
            status=EpisodeStatus.PENDING,
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO trials (
                    trial_id, study_id, task_type, task_spec_json, recipe_json,
                    config_json, result_json, identity_json, status, success,
                    model_sha256, scene_sha256, scene_revision,
                    created_at, optimizer_metadata_json,
                    parent_trial_id, warm_start_trial_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    trial_id,
                    study_id,
                    task_spec.task_type,
                    task_spec.to_json(),
                    recipe_json,
                    config.to_json(),
                    result.to_json(),
                    identity.to_json(),
                    EpisodeStatus.PENDING.value,
                    0,
                    identity.model_sha256,
                    identity.scene_sha256,
                    identity.scene_revision,
                    now,
                    json.dumps(optimizer_metadata or {}),
                    parent_trial_id,
                    warm_start_trial_id,
                ),
            )
        return trial_id

    def start_trial(self, trial_id: str) -> None:
        """Transition PENDING → RUNNING. Raises if trial is not in PENDING state."""
        now = _now_iso()
        with self._conn:
            n = self._conn.execute(
                "UPDATE trials SET status=?, started_at=? WHERE trial_id=? AND status=?",
                (EpisodeStatus.RUNNING.value, now, trial_id, EpisodeStatus.PENDING.value),
            ).rowcount
        if n == 0:
            raise ExperienceStoreError(
                f"Cannot start trial '{trial_id}': not found or not in PENDING state"
            )

    def complete_trial(self, trial_id: str, result: EpisodeResult) -> None:
        """Transition RUNNING → SUCCEEDED or FAILED (determined by result.status).

        result.status must be SUCCEEDED or FAILED; use fail_trial() or
        abort_trial() for other terminal transitions.
        """
        if result.status not in (EpisodeStatus.SUCCEEDED, EpisodeStatus.FAILED):
            raise ExperienceStoreError(
                f"complete_trial expects SUCCEEDED or FAILED, got {result.status.value}. "
                "Use fail_trial() / abort_trial() for other terminal transitions."
            )
        now = _now_iso()
        with self._conn:
            n = self._conn.execute(
                """
                UPDATE trials
                SET status=?, success=?, result_json=?, completed_at=?
                WHERE trial_id=? AND status=?
                """,
                (
                    result.status.value,
                    1 if result.success else 0,
                    result.to_json(),
                    now,
                    trial_id,
                    EpisodeStatus.RUNNING.value,
                ),
            ).rowcount
        if n == 0:
            raise ExperienceStoreError(
                f"Cannot complete trial '{trial_id}': not found or not in RUNNING state"
            )

    def fail_trial(
        self,
        trial_id: str,
        reason: str,
        result: Optional[EpisodeResult] = None,
    ) -> None:
        """Transition any non-terminal state → FAILED."""
        self._force_terminal(trial_id, EpisodeStatus.FAILED, reason, result)

    def abort_trial(self, trial_id: str, reason: str) -> None:
        """Transition any non-terminal state → ABORTED."""
        self._force_terminal(trial_id, EpisodeStatus.ABORTED, reason)

    def cancel_trial(self, trial_id: str) -> None:
        """Transition any non-terminal state → CANCELLED."""
        self._force_terminal(trial_id, EpisodeStatus.CANCELLED, "cancelled")

    def _force_terminal(
        self,
        trial_id: str,
        status: EpisodeStatus,
        reason: str,
        result: Optional[EpisodeResult] = None,
    ) -> None:
        now = _now_iso()
        placeholders = [status.value, now, trial_id, *_TERMINAL]
        set_clause = "status=?, completed_at=?"
        if result is not None:
            set_clause = "status=?, success=?, result_json=?, completed_at=?"
            placeholders = [
                status.value,
                1 if result.success else 0,
                result.to_json(),
                now,
                trial_id,
                *_TERMINAL,
            ]
        not_in = ",".join("?" * len(_TERMINAL))
        with self._conn:
            n = self._conn.execute(
                f"UPDATE trials SET {set_clause} WHERE trial_id=? AND status NOT IN ({not_in})",
                placeholders,
            ).rowcount
        if n == 0:
            raise ExperienceStoreError(
                f"Cannot transition trial '{trial_id}' to {status.value}: "
                "not found or already in a terminal state"
            )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def attach_artifact(
        self,
        trial_id: str,
        name: str,
        path: str,
        kind: str = "file",
    ) -> None:
        """Record an artifact file path against a trial.

        Files larger than 50 MiB are referenced but not hashed.
        The trial must exist (but need not be in a specific state).
        """
        p = pathlib.Path(path)
        size_bytes: Optional[int] = p.stat().st_size if p.exists() else None
        sha: Optional[str] = _sha256_file(p) if p.exists() else None
        now = _now_iso()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO artifacts (trial_id, name, path, kind, size_bytes, sha256, created_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (trial_id, name, path, kind, size_bytes, sha, now),
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_compatible_trials(
        self,
        identity: SimulatorIdentity,
        *,
        task_type: Optional[str] = None,
        study_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return SUCCEEDED trials whose model/scene hashes exactly match identity.

        Raises IdentityMismatchError if identity.model_sha256 is empty — an
        empty hash would match every trial, silently reusing incompatible data.
        """
        if not identity.model_sha256:
            raise IdentityMismatchError(
                "Cannot query compatible trials: identity.model_sha256 is empty. "
                "Build the identity from a real model file."
            )
        sql = (
            "SELECT * FROM trials "
            "WHERE model_sha256=? AND scene_sha256=? AND status=?"
        )
        params: List[Any] = [
            identity.model_sha256,
            identity.scene_sha256,
            EpisodeStatus.SUCCEEDED.value,
        ]
        if task_type:
            sql += " AND task_type=?"
            params.append(task_type)
        if study_id:
            sql += " AND study_id=?"
            params.append(study_id)
        sql += " ORDER BY completed_at DESC LIMIT ?"
        params.append(limit)

        cur = self._conn.execute(sql, params)
        rows = cur.fetchall()
        return [_row_dict(cur, r) for r in rows]

    def get_trial(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """Return a trial row as a dict, or None if not found."""
        cur = self._conn.execute("SELECT * FROM trials WHERE trial_id=?", (trial_id,))
        row = cur.fetchone()
        return _row_dict(cur, row) if row else None

    def list_trials(self, study_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Return all trials for a study, newest first."""
        cur = self._conn.execute(
            "SELECT * FROM trials WHERE study_id=? ORDER BY created_at DESC LIMIT ?",
            (study_id, limit),
        )
        rows = cur.fetchall()
        return [_row_dict(cur, r) for r in rows]

    def get_artifacts(self, trial_id: str) -> List[Dict[str, Any]]:
        """Return all artifact records for a trial."""
        cur = self._conn.execute(
            "SELECT * FROM artifacts WHERE trial_id=? ORDER BY id", (trial_id,)
        )
        rows = cur.fetchall()
        return [_row_dict(cur, r) for r in rows]

    # ------------------------------------------------------------------
    # Orphan recovery (called on open)
    # ------------------------------------------------------------------

    def patch_optimizer_metadata(self, trial_id: str, update: Dict[str, Any]) -> None:
        """Merge update dict into the trial's optimizer_metadata_json in-place."""
        row = self._conn.execute(
            "SELECT optimizer_metadata_json FROM trials WHERE trial_id=?", (trial_id,)
        ).fetchone()
        if row is None:
            raise ExperienceStoreError(f"Trial '{trial_id}' not found")
        existing = json.loads(row[0] or "{}")
        existing.update(update)
        with self._conn:
            self._conn.execute(
                "UPDATE trials SET optimizer_metadata_json=? WHERE trial_id=?",
                (json.dumps(existing), trial_id),
            )

    def _recover_orphans(self, threshold_s: float = _DEFAULT_ORPHAN_THRESHOLD_S) -> int:
        """Mark stale RUNNING trials as ABORTED. Returns the count recovered.

        A trial is orphaned when it has been in RUNNING state longer than
        threshold_s seconds (default 3600 s).  This handles crashed episode
        runners that never called complete/fail/abort.
        """
        cutoff = (
            datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=threshold_s)
        ).isoformat()
        now = _now_iso()
        with self._conn:
            n = self._conn.execute(
                """
                UPDATE trials
                SET status=?, completed_at=?
                WHERE status=? AND (started_at IS NULL OR started_at < ?)
                """,
                (EpisodeStatus.ABORTED.value, now, EpisodeStatus.RUNNING.value, cutoff),
            ).rowcount
        return n
