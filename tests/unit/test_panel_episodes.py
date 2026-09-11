"""Issue #88: one durable record per ability the panel flies.

Offline.  No robot, no simulator, no SDK — the executor's motion process is a
stub, and what is under test is the half that stays in the server: what gets
written down after the arm stops, and what must never happen because of it.

Builds its own scene, link and worker stubs rather than importing them from a
sibling test module.  This repo has no conftest.py, so a cross-test import
resolves only when the whole suite runs and leaves the file broken in
isolation.
"""

import dataclasses
import json
import os
import sys
import time
import types

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../web"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

from panel_episodes import STUDY_ID, EpisodeRecorder, build_recorder  # noqa: E402
from panel_executor import SimulatorExecutor  # noqa: E402
from panel_scene import apply_snapshot, scene_view_from_doc  # noqa: E402
from panel_sim_link import SimSnapshot  # noqa: E402
from tasks import Proposal  # noqa: E402

from reachy_ai.experience.identity import build_simulator_identity  # noqa: E402
from reachy_ai.experience.models import (EpisodeConfig, EpisodeStatus,  # noqa: E402
                                         SimulatorIdentity, TaskSpec)
from reachy_ai.experience.store import ExperienceStore  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _Cell:
    def __init__(self, name, x, y, reachable=True):
        self.name = name
        self.x = x
        self.y = y
        self.top_z = 0.80
        self.half_extent = 0.06
        self.reachable = reachable
        self.shoulder_distance_m = 0.4


def make_scene():
    cells = {f"r{r}c{c}": _Cell(f"r{r}c{c}", 0.20 + 0.15 * (r - 1),
                                0.15 * (c - 2))
             for r in (1, 2, 3) for c in (1, 2, 3)}
    doc = {"name": "TestScene",
           "objects": [{"id": "soda_can", "semantic_class": "can",
                        "tags": ["manipulable", "pickable"]}]}
    return scene_view_from_doc(doc, cells, placeable=["soda_can"])


def live_scene():
    scene = make_scene()
    c = scene.cells["r1c1"]
    apply_snapshot(scene, SimSnapshot(
        scene_revision="rev-1",
        objects={"soda_can": (c.x, c.y, c.top_z + 0.03)},
        received_at=time.monotonic(),
    ))
    return scene


class StubLink:
    def __init__(self, *, grant=True):
        self.supports_lease = True
        self._grant = grant
        self.released = 0

    def snapshot(self):
        return SimSnapshot(received_at=time.monotonic())

    def acquire_control(self, motion_client_id, *, reason="", ttl_s=120.0):
        return (True, "") if self._grant else (False, "held by someone else")

    def release_control(self, timeout=5.0):
        self.released += 1


class StubWorker:
    def __init__(self, result=None, phases=()):
        self.result = result or {"status": "moved", "flown": ["a", "b"],
                                 "start_posture": "rest",
                                 "final_posture": "home"}
        self.phases = list(phases)

    def run(self, job, *, on_phase=None, should_cancel=None):
        for name in self.phases:
            if on_phase is not None:
                on_phase(name)
        if should_cancel is not None and should_cancel():
            pass
        return self.result

    def close(self):
        pass


def _ability(**kw):
    base = dict(plan_id="plan-a", plan_version=3, task_type="stow_arm",
                target_id=None, destination=None, arm="right",
                route="STOW_ROUTE", route_version=1,
                expected_start_posture="rest", end_posture="home",
                brief_reason="STOW_ROUTE with my right arm",
                summary="stow the arm")
    base.update(kw)
    return Proposal(**base)


@pytest.fixture
def fake_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "reachy_sdk",
                        types.SimpleNamespace(ReachySDK=object))
    monkeypatch.setenv("REACHY_SIM_BACKEND", "mujoco-remote")
    # #82's footprint check reads a real scene FILE; every executor() call in
    # this module points scene_file at a placeholder ("scene.yaml") that was
    # never meant to touch disk, so stand in with an object-free model.
    from reachy_ai.scene.awareness import SceneModel
    monkeypatch.setattr(
        SceneModel, "from_yaml",
        staticmethod(lambda path: SceneModel("pedestal", [], None)))
    yield


@pytest.fixture
def validated(monkeypatch):
    from reachy_ai.motion import rig_routes as R
    monkeypatch.setattr(R, "check_route", lambda route, scene: (True, ""))
    yield


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "episodes.db")


def executor(db, worker=None, link=None, keep=100):
    return SimulatorExecutor(
        link or StubLink(), live_scene, "scene.yaml",
        worker=worker or StubWorker(),
        recorder=EpisodeRecorder("scene.yaml", db_path=db, keep=keep),
    )


def rows(db, study=STUDY_ID):
    with ExperienceStore.open(db) as store:
        return store.list_trials(study, limit=None)


# ---------------------------------------------------------------------------
# One record, per execution, whatever the outcome
# ---------------------------------------------------------------------------

def test_a_completed_ability_leaves_one_record(db, fake_sdk, validated):
    out = executor(db, StubWorker(phases=["leaving rest", "flying STOW_ROUTE"])
                   ).execute(_ability())
    assert out.status == "completed", out.detail

    row, = rows(db)
    assert row["task_type"] == "stow_arm"
    assert row["status"] == EpisodeStatus.SUCCEEDED.value
    assert row["success"] == 1
    assert row["live_interactive"] == 1

    meta = json.loads(row["optimizer_metadata_json"])
    assert meta["live_interactive"] is True
    assert meta["plan_id"] == "plan-a"
    assert meta["plan_version"] == 3
    # Measured, not assumed: the proposal said the route starts at rest, and
    # the arm reported where it actually was.
    assert meta["expected_start_posture"] == "rest"
    assert meta["start_posture"] == "rest"
    assert meta["final_posture"] == "home"
    assert [p["name"] for p in meta["phases"]] == ["leaving rest",
                                                   "flying STOW_ROUTE"]
    assert all(p["at_s"] >= 0 for p in meta["phases"])


def test_the_record_reproduces_what_the_operator_was_told(db, fake_sdk, validated):
    """A record that cannot reproduce the sentence the human read is a record
    of a different event."""
    worker = StubWorker({"status": "moved", "flown": [], "start_posture": "rest",
                         "final_posture": "tuck"})
    out = executor(db, worker).execute(_ability())
    assert out.status == "failed"

    row, = rows(db)
    result = json.loads(row["result_json"])
    assert row["status"] == EpisodeStatus.FAILED.value
    assert result["termination_reason"] == out.detail
    assert out.detail in result["warnings"]
    assert "task_failure" in result["hard_violations"]


def test_a_cancelled_episode_is_recorded_as_cancelled(db, fake_sdk, validated):
    worker = StubWorker({"status": "cancelled", "detail": "Stopped at TUCK.",
                         "evidence": {}})
    out = executor(db, worker).execute(_ability())
    assert out.status == "cancelled"

    row, = rows(db)
    assert row["status"] == EpisodeStatus.CANCELLED.value
    assert row["success"] == 0
    # A cancelled episode is not an empty one — where it stopped is the fact a
    # reader wants, so the result travels with the cancellation.
    assert json.loads(row["result_json"])["termination_reason"] == out.detail


def test_a_refusal_taken_after_the_lease_is_still_an_episode(db, fake_sdk,
                                                             validated):
    """The lease was granted and the board had moved: nothing flew, but the
    attempt reached the arm's front door and is worth a record."""
    scene = live_scene()

    def drifting():
        # A different object position on every read, so the executor's
        # post-lease revalidation sees a board that changed under it.
        c = scene.cells["r3c3"]
        apply_snapshot(scene, SimSnapshot(
            scene_revision="rev-2",
            objects={"soda_can": (c.x, c.y, c.top_z + 0.5)},
            received_at=time.monotonic()))
        return scene

    ex = SimulatorExecutor(StubLink(), drifting, "scene.yaml",
                           worker=StubWorker(),
                           recorder=EpisodeRecorder("scene.yaml", db_path=db))
    out = ex.execute(_ability())
    assert out.status in ("completed", "failed")
    assert len(rows(db)) == 1


def test_object_drift_is_recorded_as_a_violation(db, fake_sdk, validated):
    """The arm reached HOME having knocked the can across the board on the
    way.  The operator is told; the record says so too."""
    scene = live_scene()

    class Sweeps(StubWorker):
        def run(self, job, **kw):
            c = scene.cells["r3c3"]
            apply_snapshot(scene, SimSnapshot(
                scene_revision="rev-1",
                objects={"soda_can": (c.x, c.y + 0.25, c.top_z + 0.03)},
                received_at=time.monotonic()))
            return self.result

    ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml",
                           worker=Sweeps(),
                           recorder=EpisodeRecorder("scene.yaml", db_path=db))
    ex.execute(_ability())

    row, = rows(db)
    result = json.loads(row["result_json"])
    assert result["metrics"]["objects_disturbed"] >= 1.0
    assert result["metrics"]["max_object_drift_m"] > 0.02
    assert "forbidden_contact" in result["hard_violations"]


def test_pick_place_is_not_recorded_here(db, fake_sdk, validated):
    """Documented scope, not an oversight: pick-and-place has its own offline
    runner and recipe, and no success definition measured the way this path
    measures one."""
    pick = Proposal(plan_id="p", plan_version=0, task_type="pick_place",
                    target_id="soda_can", destination="cell:r2c2")
    executor(db).execute(pick)
    assert rows(db) == []


# ---------------------------------------------------------------------------
# It must not be able to cost a motion
# ---------------------------------------------------------------------------

def test_a_broken_store_costs_a_record_and_not_a_motion(db, fake_sdk, validated,
                                                        caplog):
    class Broken(EpisodeRecorder):
        def _record(self, *a, **kw):
            raise RuntimeError("disk is on fire")

    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml",
                           worker=StubWorker(),
                           recorder=Broken("scene.yaml", db_path=db))
    out = ex.execute(_ability())
    assert out.status == "completed"
    assert out.detail                       # unchanged, and still the operator's
    assert not os.path.exists(db)


def test_recording_happens_after_the_lease_is_released(db, fake_sdk, validated):
    """A slow disk must not hold the arm.  The lease is released before the
    recorder is called, so the release count is already final when it runs."""
    link = StubLink()
    seen = {}

    class Watching(EpisodeRecorder):
        def record(self, *a, **kw):
            seen["released"] = link.released
            return super().record(*a, **kw)

    ex = SimulatorExecutor(link, live_scene, "scene.yaml",
                           worker=StubWorker(),
                           recorder=Watching("scene.yaml", db_path=db))
    ex.execute(_ability())
    assert seen["released"] == 1


def test_no_executor_recorder_writes_nothing(db, fake_sdk, validated):
    """The default construction — every other test file's — records nothing."""
    ex = SimulatorExecutor(StubLink(), live_scene, "scene.yaml",
                           worker=StubWorker())
    assert ex.execute(_ability()).status == "completed"
    assert not os.path.exists(db)


def test_recording_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("REACHY_PANEL_EPISODES", "0")
    assert build_recorder("scene.yaml") is None
    monkeypatch.setenv("REACHY_PANEL_EPISODES", "1")
    assert build_recorder("scene.yaml") is not None


# ---------------------------------------------------------------------------
# What a live-interactive record may not do
# ---------------------------------------------------------------------------

def _offline_trial(store, identity, *, task_type="stow_arm"):
    tid = store.create_trial(
        "study-offline", TaskSpec(task_id="t", task_type=task_type),
        "{}", EpisodeConfig(simulator_identity=identity))
    store.start_trial(tid)
    from reachy_ai.experience.models import EpisodeResult
    store.complete_trial(tid, EpisodeResult(
        episode_id=tid, trial_id=tid, status=EpisodeStatus.SUCCEEDED,
        success=True))
    return tid


def test_live_episodes_are_excluded_from_compatible_trials(db, fake_sdk,
                                                           validated):
    executor(db).execute(_ability())
    identity = rows(db)[0]["identity_json"]
    identity = SimulatorIdentity.from_json(identity)
    # A real hash, so the query is not refused for a degenerate identity.
    identity = SimulatorIdentity.from_dict(
        {**identity.to_dict(), "model_sha256": "abc123"})

    with ExperienceStore.open(db) as store:
        store._conn.execute("UPDATE trials SET model_sha256='abc123'")
        store._conn.commit()
        offline = _offline_trial(store, identity)
        store._conn.execute(
            "UPDATE trials SET scene_sha256=? WHERE trial_id=?",
            (identity.scene_sha256, offline))
        store._conn.commit()

        default = store.query_compatible_trials(identity)
        included = store.query_compatible_trials(identity,
                                                 include_live_interactive=True)

    assert [r["live_interactive"] for r in default] == [0]
    assert sorted(r["live_interactive"] for r in included) == [0, 1]


def test_live_episodes_are_pruned_and_offline_ones_are_not(db, fake_sdk,
                                                           validated):
    ex = executor(db, keep=3)
    for i in range(6):
        ex.execute(_ability(plan_id=f"plan-{i}"))

    with ExperienceStore.open(db) as store:
        _offline_trial(store, build_simulator_identity())
        assert store.prune_live_interactive(3) == 0     # already at the cap
        live = [r for r in store.list_trials(STUDY_ID, limit=None)]
        offline = store.list_trials("study-offline", limit=None)

    assert len(live) == 3
    assert len(offline) == 1


# ---------------------------------------------------------------------------
# What must not end up in the file
# ---------------------------------------------------------------------------

def test_the_record_carries_no_host_port_or_absolute_path(db, fake_sdk,
                                                          validated):
    ex = SimulatorExecutor(StubLink(), live_scene,
                           os.path.join(os.getcwd(), "scenes", "scene.yaml"),
                           worker=StubWorker(),
                           sdk_host="10.0.0.7", sdk_port=50051,
                           recorder=EpisodeRecorder(
                               os.path.join(os.getcwd(), "scenes", "scene.yaml"),
                               db_path=db))
    ex.execute(_ability())

    blob = json.dumps(rows(db)[0])
    assert "10.0.0.7" not in blob
    assert "50051" not in blob
    assert os.path.expanduser("~") not in blob
    assert not any(part.startswith("/Users") or part.startswith("/home")
                   for part in blob.split('"'))


def test_the_scene_is_hashed_from_a_directory_that_is_not_the_repo(
        db, tmp_path, monkeypatch, fake_sdk, validated):
    """The container runs with CWD "/".  A repo-relative path handed to
    `build_simulator_identity` is opened relative to THAT, so the read fails,
    the hash comes back empty, and every record carries a scene_source_path
    with no scene_sha256 — the degenerate identity `assert_research_context`
    exists to refuse."""
    scene_file = tmp_path / "scene.yaml"
    scene_file.write_text("name: TestScene\n")
    monkeypatch.chdir(tmp_path / "..")

    ex = SimulatorExecutor(StubLink(), live_scene, str(scene_file),
                           worker=StubWorker(),
                           recorder=EpisodeRecorder(str(scene_file), db_path=db))
    ex.execute(_ability())

    identity = SimulatorIdentity.from_json(rows(db)[0]["identity_json"])
    assert identity.scene_sha256, "the scene was not hashed"
    # And the path recorded next to it is still not an absolute one.
    assert not identity.scene_source_path.startswith("/")


def test_the_revision_recorded_is_the_one_that_moves(db, fake_sdk, validated):
    """`SceneView` carries a name and a revision.  The name does not change
    when the board is edited; the revision does, and that is the whole reason
    these episodes are marked live-interactive."""
    executor(db).execute(_ability())

    row, = rows(db)
    identity = SimulatorIdentity.from_json(row["identity_json"])
    spec = json.loads(row["task_spec_json"])
    assert identity.scene_revision == "rev-1"
    assert spec["scene_revision"] == "rev-1"
    # The name is kept, just not where the revision belongs.
    assert json.loads(row["optimizer_metadata_json"])["scene_name"] == "TestScene"


def test_a_scene_that_will_not_load_costs_a_record_and_not_a_motion(
        db, fake_sdk, validated):
    """The scene read that feeds the recorder is the executor's, not the
    recorder's, so it needs the same wall around it."""
    state = {"flown": False, "reads_after": 0}

    class Flags(StubWorker):
        def run(self, job, **kw):
            state["flown"] = True
            return self.result

    def provider():
        # Anchored to the motion rather than to a raw call count: the reads
        # before it belong to availability and to the pre-lease snapshot, and
        # failing one of those is a refusal rather than the case under test.
        if state["flown"]:
            state["reads_after"] += 1
            if state["reads_after"] > 1:        # past _verify_posture's read
                raise RuntimeError("the scene file is gone")
        return live_scene()

    ex = SimulatorExecutor(StubLink(), provider, "scene.yaml",
                           worker=Flags(),
                           recorder=EpisodeRecorder("scene.yaml", db_path=db))
    out = ex.execute(_ability())
    assert out.status == "completed", out.detail
    assert state["reads_after"] > 1, "the recorder never reached its scene read"
    assert not os.path.exists(db)


def test_the_default_database_is_not_resolved_against_the_process_cwd(
        tmp_path, monkeypatch):
    """supervisord starts the panel with CWD "/", where a relative default
    writes into the container's ephemeral layer."""
    monkeypatch.chdir(tmp_path)
    recorder = EpisodeRecorder("scene.yaml")
    assert os.path.isabs(recorder._db_path)
    assert not recorder._db_path.startswith(str(tmp_path))
    assert recorder._db_path.endswith(os.path.join("runs", "panel_episodes.db"))


def test_the_board_the_episode_happened_on_is_recorded(db, fake_sdk, validated):
    """The reuse gate rejects a candidate that never recorded one, so an
    episode without it can never become evidence, however well it went."""
    executor(db).execute(_ability())
    meta = json.loads(rows(db)[0]["optimizer_metadata_json"])
    assert meta["obstacles"] == ["soda_can"]


def test_an_object_off_the_board_is_not_recorded_as_an_obstacle(db, fake_sdk,
                                                                validated):
    """`scene.objects` is the declared pool. An object sitting in the pool
    rather than on the table is not an obstacle, and recording it as one makes
    a later, correctly-computed board fail to match."""
    scene = live_scene()
    scene.objects["red_cube"] = dataclasses.replace(
        next(iter(scene.objects.values())), object_id="red_cube",
        on_board=False)

    ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml",
                           worker=StubWorker(),
                           recorder=EpisodeRecorder("scene.yaml", db_path=db))
    ex.execute(_ability())
    meta = json.loads(rows(db)[0]["optimizer_metadata_json"])
    assert meta["obstacles"] == ["soda_can"]


def test_a_board_nobody_observed_is_recorded_as_not_recorded(db, fake_sdk,
                                                             validated):
    """A scene with no snapshot has `on_board` None everywhere — unknown, not
    absent. Writing [] there would certify a route against an empty board that
    was never seen, which is the inversion the gate's rule exists to stop."""
    scene = make_scene()                    # never given a snapshot

    ex = SimulatorExecutor(StubLink(), lambda: scene, "scene.yaml",
                           worker=StubWorker(),
                           recorder=EpisodeRecorder("scene.yaml", db_path=db))
    ex.execute(_ability())
    meta = json.loads(rows(db)[0]["optimizer_metadata_json"])
    assert meta["obstacles"] is None

    # And the gate refuses it, rather than reading it as an empty board.
    from reachy_ai.experience.compatibility import ReuseCandidate
    assert ReuseCandidate.from_row(rows(db)[0]).obstacles is None


def test_an_episode_this_panel_recorded_is_refused_by_the_reuse_gate(
        db, fake_sdk, validated):
    """The two halves have to agree.  A wave that went perfectly is still one
    run observed by one person, and the gate says so twice over: it was
    live-interactive, and it was never promoted."""
    from reachy_ai.experience.compatibility import (ReuseCandidate,
                                                    ReuseRequest, check_reuse)

    executor(db).execute(_ability())
    row, = rows(db)
    assert row["status"] == EpisodeStatus.SUCCEEDED.value

    # A real model hash, so the gate judges the row rather than refusing to
    # answer about a degenerate identity.
    identity = SimulatorIdentity.from_dict(
        {**SimulatorIdentity.from_json(row["identity_json"]).to_dict(),
         "model_sha256": "m"})
    candidate = ReuseCandidate.from_row(
        {**row, "identity_json": identity.to_json()})

    decision = check_reuse(candidate, identity, ReuseRequest(
        task_type="stow_arm", arm="right", route="STOW_ROUTE",
        route_version=1, start_posture="rest",
        obstacles=frozenset({"soda_can"})))
    assert not decision.allowed
    assert "a person was driving" in decision.reason


def test_the_route_is_recorded_as_a_route_and_not_as_a_recipe(db, fake_sdk,
                                                              validated):
    """The abilities have no TrajectoryRecipe yet — that is #91 — and an empty
    one written here would claim a search artefact exists."""
    executor(db).execute(_ability())
    recipe = json.loads(rows(db)[0]["recipe_json"])
    assert recipe["kind"] == "ability_route"
    assert recipe["route"] == "STOW_ROUTE"
    assert recipe["route_version"] == 1


# ---------------------------------------------------------------------------
# The schema change
# ---------------------------------------------------------------------------

def _write_v1_database(path):
    """A real v1 file, built the way v1 built one.

    Not a v2 file with the column dropped: ALTER TABLE ... DROP COLUMN needs
    SQLite 3.35, and the project's own ros:foxy image ships 3.31, where that
    scaffolding would error before it tested anything.  Reconstructing the old
    schema also tests the migration against what v1 actually wrote rather than
    against an approximation of it.
    """
    import sqlite3

    from reachy_ai.experience.store import _DDL

    v1_ddl = _DDL.replace(",\n    live_interactive        INTEGER NOT NULL DEFAULT 0", "")
    v1_ddl = "\n".join(
        block for block in v1_ddl.split("\n\n")
        if "idx_trials_live\n" not in block + "\n"
    )
    assert "live_interactive" not in v1_ddl, v1_ddl

    conn = sqlite3.connect(path)
    conn.executescript(v1_ddl)
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()


def test_a_v1_database_migrates_and_keeps_its_rows(tmp_path):
    """Every row a v1 file holds was written by an offline runner, which is
    exactly what the new column's default says."""
    import sqlite3

    path = str(tmp_path / "v1.db")
    _write_v1_database(path)
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    columns = {row[1] for row in conn.execute("PRAGMA table_info(trials)")}
    assert "live_interactive" not in columns
    conn.close()

    # Opening it is the migration.
    with ExperienceStore.open(path) as store:
        _offline_trial(store, build_simulator_identity())
        rows_ = store.list_trials("study-offline", limit=None)

    assert len(rows_) == 1
    assert rows_[0]["live_interactive"] == 0

    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    conn.close()


def test_a_v1_database_keeps_the_rows_it_already_had(tmp_path):
    """The rows survive the ALTER, not just the schema."""
    import sqlite3

    path = str(tmp_path / "v1-with-rows.db")
    _write_v1_database(path)

    identity = build_simulator_identity()
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO trials (trial_id, study_id, task_type, task_spec_json, "
        "recipe_json, config_json, result_json, identity_json, status, "
        "success, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("old-1", "study-v1", "stow_arm", "{}", "{}", "{}", "{}",
         identity.to_json(), EpisodeStatus.SUCCEEDED.value, 1,
         "2026-01-01T00:00:00+00:00"))
    conn.commit()
    conn.close()

    with ExperienceStore.open(path) as store:
        row, = store.list_trials("study-v1", limit=None)

    assert row["trial_id"] == "old-1"
    assert row["live_interactive"] == 0
