"""One durable record per ability the panel flies (issue #88).

WHAT THIS IS
------------
The panel measured everything an episode record needs and then dropped it.
`SimulatorExecutor._verify_posture()` reads the final posture, the waypoints
actually flown, and how far every object on the board moved; the coordinator
renders that onto the task, and the task is discarded when `MAX_TASKS` rolls
over.  Meanwhile `reachy_ai.experience` is a durable store shaped exactly for
it that no panel module had ever imported.  This is the adapter between them.

WHAT A RECORD IS AND IS NOT
---------------------------
A record of a wave is MEMORY.  It is not improvement, and writing one does not
mean Reachy waves better next time.  Improvement needs an explicit evaluation
and search process over these records — #91 for the evaluators, #90 for the
retrieval, and #89 for the compatibility rules that must exist before anything
retrieved is allowed to move the arm.  Reports written from this store have to
keep that distinction visible.

WHY THEY ARE MARKED LIVE-INTERACTIVE
------------------------------------
These are not offline deterministic trials.  There is no fixed seed, no
controlled reset, a person in the loop, and a scene that may have been edited
between one episode and the next.  `live_interactive=1` says so in the store,
and `query_compatible_trials()` excludes them by default, so an episode
observed here cannot quietly join a search population.

BOUNDS THIS ACCEPTS
-------------------
- It never raises.  A store that cannot be opened or written costs the
  operator a record, never a motion.
- It runs after the execution lease and the motion lock are both released, so
  a slow disk cannot hold the arm.
- It prunes itself.  A container that runs for months writes one record per
  confirmed movement, and unbounded is not a policy.
- It records no host, no port, no credential, and no path outside the repo.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("panel.episodes")

#: Where the episodes go, in the search CLI's own convention (`runs/`).
#:
#: RESOLVED AGAINST THE REPO, NOT THE PROCESS CWD.  supervisord starts the
#: panel with CWD "/", and a relative default there writes `/runs/...` — the
#: container's ephemeral layer, outside every mounted volume and outside the
#: `.gitignore` rule that is supposed to cover it.  Set
#: REACHY_PANEL_EPISODE_DB to put it on a volume.
DEFAULT_DB = ""

#: How many live-interactive episodes to keep.  Roughly a month of steady
#: demonstration use; old ones are dropped newest-first-kept.
DEFAULT_KEEP = 2000

#: The study every panel episode belongs to.  A name, not a search run: it
#: exists so these rows are trivially separable from a real study's.
STUDY_ID = "panel-live"

#: Objects that moved less than this are the physics engine's own jitter, not
#: something the arm did.  The same number the executor judges drift by —
#: `panel_executor._verify_posture` — because two thresholds for one question
#: is how a record comes to disagree with what the operator was told.
DRIFT_TOLERANCE_M = 0.02


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def _ensure_paths() -> None:
    """Same two candidates as panel_executor: a checkout, or /opt in the image."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "..", "src"), "/opt/src"):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)


def _relative(path: str) -> str:
    """A repo-relative path, or the bare filename if it is outside the repo.

    Absolute paths from a developer's machine are provenance nobody else can
    use and a small leak of where this ran.  The scene file's identity is its
    hash, which `build_simulator_identity` records; the path is a label.
    """
    if not path:
        return ""
    try:
        return str(pathlib.Path(path).resolve().relative_to(_repo_root()))
    except (ValueError, OSError):
        return os.path.basename(path)


class EpisodeRecorder:
    """Writes one trial per executed ability.  Constructed once, reused."""

    def __init__(self, scene_file: str = "", *, db_path: str = "",
                 keep: int = DEFAULT_KEEP) -> None:
        self._scene_file = scene_file
        self._db_path = db_path or str(_repo_root() / "runs" / "panel_episodes.db")
        self._keep = keep
        #: Built from git and a file hash, so it is rebuilt only when the
        #: scene file changes rather than once per movement.
        self._identity = None
        self._identity_key: Optional[Tuple[Any, ...]] = None

    # -- the one public call ------------------------------------------------

    def record(self, proposal, result, *, phases: Sequence[Tuple[str, float]] = (),
               started_at: float = 0.0, ended_at: float = 0.0,
               scene_name: str = "", scene_revision: str = "",
               obstacles: Optional[Sequence[str]] = None) -> Optional[str]:
        """Record one episode.  Returns the trial id, or None if nothing was
        written — which is a log line and never an exception.

        Called with the locks released.  Everything it needs was captured
        while they were held.
        """
        try:
            return self._record(proposal, result, phases, started_at,
                                ended_at, scene_name, scene_revision,
                                obstacles)
        except Exception:
            # Deliberately broad.  The alternative to swallowing this is a
            # failed motion report for a movement that actually succeeded.
            log.exception("could not record the episode for plan %s",
                          getattr(proposal, "plan_id", "?"))
            return None

    # -- everything below may raise; `record` is the wall -------------------

    def _record(self, proposal, result, phases, started_at, ended_at,
                scene_name, scene_revision, obstacles=None) -> str:
        _ensure_paths()
        from reachy_ai.evaluation.base import ViolationKind
        from reachy_ai.experience.models import (EpisodeConfig, EpisodeResult,
                                                 EpisodeStatus, TaskSpec)
        from reachy_ai.experience.store import ExperienceStore

        # The executor's three words for an outcome, in the store's terms.
        # A status this does not know is FAILED rather than dropped: an
        # episode nobody can classify is still an episode that happened.
        _STATUS = {"completed": EpisodeStatus.SUCCEEDED,
                   "failed": EpisodeStatus.FAILED,
                   "cancelled": EpisodeStatus.CANCELLED}

        evidence = dict(getattr(result, "evidence", None) or {})
        status = getattr(result, "status", "failed")
        detail = getattr(result, "detail", "")
        drift = {k: float(v) for k, v in (evidence.get("object_drift") or {}).items()}
        flown = [str(w) for w in (evidence.get("waypoints_flown") or [])]

        spec = TaskSpec(
            task_id=f"panel:{proposal.task_type}",
            task_type=proposal.task_type,
            # The REVISION, which moves when the board is edited — not the
            # scene's name, which does not.  Two episodes either side of a
            # placement must not read as the same world.
            scene_revision=scene_revision,
            arm_policy=getattr(proposal, "arm", "") or "right",
            # The rule the executor actually applied, in the words it applied
            # it in.  A success definition invented here would describe a
            # judgement nobody made.
            success_definition=(
                f"reached {proposal.end_posture or 'the route endpoint'} and "
                f"moved no object by more than {DRIFT_TOLERANCE_M} m"
            ),
            forbidden_contact_policy="fail",
            randomization_profile="none",
        )

        config = EpisodeConfig(
            simulator_identity=self._identity_for(scene_revision),
            # Zero because there was no seed, not because the seed was zero.
            # `live_interactive` is what tells a reader to disregard it.
            seed=0,
            render_mode="off",
            record_trace=False,
            contact_sampling="none",
            state_sampling="none",
        )

        violations: List[str] = []
        if drift:
            violations.append(ViolationKind.FORBIDDEN_CONTACT.value)
        if status == "failed":
            violations.append(ViolationKind.TASK_FAILURE.value)

        with ExperienceStore.open(self._db_path) as store:
            trial_id = store.create_trial(
                STUDY_ID, spec, self._route_json(proposal), config,
                live_interactive=True,
                optimizer_metadata=self._metadata(proposal, phases, evidence,
                                                  scene_name, obstacles),
            )
            store.start_trial(trial_id)

            episode = EpisodeResult(
                episode_id=trial_id,
                trial_id=trial_id,
                status=_STATUS.get(status, EpisodeStatus.FAILED),
                termination_reason=detail,
                success=(status == "completed"),
                wall_duration_s=round(max(0.0, ended_at - started_at), 3),
                metrics={
                    "waypoints_flown": float(len(flown)),
                    "objects_disturbed": float(len(drift)),
                    "max_object_drift_m": max(drift.values()) if drift else 0.0,
                },
                hard_violations=violations,
                final_object_states={"drift_m": drift},
                # What the operator was shown, kept verbatim.  A record that
                # cannot reproduce the sentence the human read is a record of
                # a different event.
                warnings=[detail] if detail else [],
            )
            if status == "cancelled":
                store.cancel_trial(trial_id, episode)
            else:
                store.complete_trial(trial_id, episode)

            dropped = store.prune_live_interactive(self._keep)
        if dropped:
            log.info("pruned %d old panel episodes", dropped)
        return trial_id

    def _route_json(self, proposal) -> str:
        """What flew, in place of a recipe.

        NOT a TrajectoryRecipe.  The abilities are measured routes from the
        notebook, not parameterised recipes — those arrive with #91 — and
        writing an empty recipe here would claim a search artefact exists.
        `kind` is present so a reader that expects a recipe can tell.
        """
        return json.dumps({
            "kind": "ability_route",
            "route": getattr(proposal, "route", ""),
            "route_version": getattr(proposal, "route_version", 0),
            "arm": getattr(proposal, "arm", ""),
            "expected_start_posture": getattr(proposal, "expected_start_posture", ""),
            "end_posture": getattr(proposal, "end_posture", ""),
        })

    def _metadata(self, proposal, phases, evidence, scene_name,
                  obstacles) -> Dict[str, Any]:
        return {
            "live_interactive": True,
            "source": "panel",
            "scene_name": scene_name or getattr(proposal, "scene_name", ""),
            # WHICH OBJECTS WERE ON THE BOARD, or null for "nobody looked".
            # The reuse gate (#89) rejects a candidate that never recorded
            # one, because a route certified over an unknown board is
            # certified over nothing — and an empty list written where nothing
            # was observed would claim an empty board instead.
            "obstacles": None if obstacles is None else sorted(obstacles),
            "plan_id": getattr(proposal, "plan_id", ""),
            "plan_version": getattr(proposal, "plan_version", 0),
            "requested_cell": getattr(proposal, "cell", None),
            "requested_object": getattr(proposal, "object_id", None),
            "expected_start_posture": getattr(proposal, "expected_start_posture", ""),
            "start_posture": evidence.get("start_posture", ""),
            "final_posture": evidence.get("final_posture", ""),
            "end_posture": getattr(proposal, "end_posture", ""),
            "waypoints_flown": [str(w) for w in (evidence.get("waypoints_flown") or [])],
            "phases": [{"name": str(n), "at_s": round(float(t), 3)}
                       for n, t in phases],
            "recovery_needed": bool(evidence.get("recovery_needed", False)),
            "scene_changed": bool(evidence.get("scene_changed", False)),
        }

    def _identity_for(self, scene_revision: str):
        """The identity for this episode, cached on the scene file.

        HASHED WHERE THE FILE IS, RECORDED WHERE IT LIVES IN THE REPO.  Those
        are two different paths and conflating them cost the hash: passing the
        repo-relative path straight to `build_simulator_identity` makes it open
        that path relative to the PROCESS CWD, which under supervisord is "/".
        The read fails, `_sha256_path` returns "", and every record carries a
        scene_source_path with no scene_sha256 — the degenerate identity
        `assert_research_context` exists to refuse.
        """
        from reachy_ai.experience.identity import build_simulator_identity

        absolute = os.path.abspath(self._scene_file) if self._scene_file else ""
        try:
            stat = os.stat(absolute) if absolute else None
        except OSError:
            stat = None
        key = (absolute, stat.st_mtime_ns if stat else 0,
               stat.st_size if stat else 0)
        if self._identity is None or self._identity_key != key:
            # `build_simulator_identity` shells out to git twice.  Off the
            # motion path, and cached on the scene file, so a session of twenty
            # waves pays for it once.
            identity = build_simulator_identity(
                scene_path=absolute,
                backend_name="sdk_bridge",
            )
            self._identity = dataclasses.replace(
                identity, scene_source_path=_relative(self._scene_file))
            self._identity_key = key

        # The revision is the cheap half and the half that moves: it changes
        # every time the board is edited, and rebuilding the whole identity
        # (two git subprocesses) for it would be paying a lot for one string.
        return dataclasses.replace(self._identity,
                                   scene_revision=scene_revision)


def build_recorder(scene_file: str = "") -> Optional[EpisodeRecorder]:
    """The recorder for this deployment, or None if it is switched off.

    On by default, unlike live execution: recording observes, it does not act,
    and the reason to have it is that the episodes it would have caught are
    the ones nobody can get back.  `REACHY_PANEL_EPISODES=0` turns it off;
    `REACHY_PANEL_EPISODE_DB` moves the file.
    """
    if os.environ.get("REACHY_PANEL_EPISODES", "1").lower() in ("0", "false", "no", "off"):
        return None
    keep = DEFAULT_KEEP
    raw = os.environ.get("REACHY_PANEL_EPISODE_KEEP", "")
    if raw.strip().isdigit():
        keep = int(raw.strip())
    return EpisodeRecorder(scene_file,
                           db_path=os.environ.get("REACHY_PANEL_EPISODE_DB", ""),
                           keep=keep)
