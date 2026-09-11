"""R12-801: Shared data contracts for the Epic 8 experience system.

All types are plain dataclasses with explicit schema version fields and JSON
serialisation so stored records remain interpretable after code changes.
Schema versions are integers; bump when a field is removed or semantically
changed (adding optional fields with defaults is backwards-compatible).

Type hierarchy
--------------
  SimulatorIdentity — exact description of the compiled simulation world.
  TaskSpec          — what task to run and how to judge success.
  EpisodeConfig     — how to execute one episode (identity + runner settings).
  EpisodeResult     — what happened (status, metrics, violations, artifacts).
  TrialRecord       — one row in the experience store binding all of the above.

TrajectoryRecipe lives in src/reachy_ai/motion/recipe.py (Section 7.3) and
is stored here as a JSON string inside TrialRecord to avoid a cross-package
import cycle in models.
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class EpisodeStatus(str, Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED    = "FAILED"
    INVALID   = "INVALID"
    ABORTED   = "ABORTED"
    CANCELLED = "CANCELLED"

    def is_terminal(self) -> bool:
        return self in (
            EpisodeStatus.SUCCEEDED,
            EpisodeStatus.FAILED,
            EpisodeStatus.INVALID,
            EpisodeStatus.ABORTED,
            EpisodeStatus.CANCELLED,
        )


# ---------------------------------------------------------------------------
# SimulatorIdentity (Section 7.1)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SimulatorIdentity:
    """Exact description of the compiled simulation world (EPIC-8 Section 7.1).

    Two identities match (are physics-compatible) when their model_sha256,
    scene_sha256, and scene_schema_version are all equal.  All other fields
    are provenance metadata.
    """
    identity_version: int           = 1
    repository_git_sha: str         = ""
    working_tree_dirty: bool        = False
    model_source_path: str          = ""
    model_sha256: str               = ""
    compiled_model_sha256: str      = ""
    scene_source_path: str          = ""
    scene_sha256: str               = ""
    scene_revision: str             = ""
    scene_schema_version: str       = "1.0"
    scene_compiler_version: str     = ""
    physics_profile_id: str         = "default"
    protocol_version: int           = 1
    mujoco_version: str             = ""
    python_version: str             = ""
    host_os: str                    = ""
    host_arch: str                  = ""
    backend_name: str               = "native_mujoco"
    calibration_profile_id: str     = "default"
    sensor_effect_profile_id: str   = "default"

    def matches(self, other: "SimulatorIdentity") -> bool:
        """True when the physics world is identical (hashes + schema version match)."""
        return (
            self.model_sha256 == other.model_sha256
            and self.scene_sha256 == other.scene_sha256
            and self.scene_schema_version == other.scene_schema_version
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulatorIdentity":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "SimulatorIdentity":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# TaskSpec (Section 7.2)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TaskSpec:
    """Common task specification (EPIC-8 Section 7.2)."""
    task_id: str
    task_type: str
    task_schema_version: int        = 1
    scene_revision: str             = ""
    arm_policy: str                 = "auto"
    timeout_steps: int              = 5000
    success_definition: str         = ""
    forbidden_contact_policy: str   = "fail"
    randomization_profile: str      = "none"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskSpec":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "TaskSpec":
        return cls.from_dict(json.loads(s))


@dataclasses.dataclass
class ControlPanelTaskSpec(TaskSpec):
    """Control-panel task extension (EPIC-8 Section 7.2)."""
    control_id: str                         = ""
    requested_final_state: bool             = True
    allowed_contact_geoms: List[str]        = dataclasses.field(default_factory=list)
    neighbor_control_ids: List[str]         = dataclasses.field(default_factory=list)
    preferred_arm: str                      = "auto"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ControlPanelTaskSpec":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclasses.dataclass
class PickPlaceTaskSpec(TaskSpec):
    """Pick-and-place task extension (EPIC-8 Section 7.2)."""
    object_id: str                              = ""
    initial_pose_tolerance: float               = 0.005
    target_pose: List[float]                    = dataclasses.field(default_factory=list)
    target_pose_tolerance: float                = 0.01
    required_lift_height: float                 = 0.05
    settle_velocity_thresholds: List[float]     = dataclasses.field(
        default_factory=lambda: [0.01, 0.01]
    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PickPlaceTaskSpec":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclasses.dataclass
class PanelRouteTaskSpec(TaskSpec):
    """A measured rig route flown as an ability (#91).

    The five abilities the panel flies are not pick-and-place and not the
    control panel.  They are named routes through a rig whose geometry IS the
    safety argument, so what this spec carries is what an evaluator needs to
    decide whether the route was flown as measured — the route's name and
    version, the postures it claims to start and end at, and the board it was
    flown over.

    `initial_object_positions` is here because an EpisodeResult records where
    the objects ENDED and nothing records where they began.  "Nothing moved"
    is a comparison, and a spec that cannot supply the first half of it makes
    the evaluator say so rather than assume an undisturbed board.
    """
    ability: str                                = ""
    route: str                                  = ""
    route_version: int                          = 1
    expected_start_posture: str                 = ""
    expected_end_posture: str                   = ""
    #: object_id -> [x, y, z] in metres, read after reset and before the move.
    initial_object_positions: Dict[str, List[float]] = dataclasses.field(
        default_factory=dict)
    #: Contact this route is EXPECTED to make.  Resting the forearm on the
    #: table is the task, not a violation; an empty list means no contact with
    #: anything is intended.
    intended_contact_bodies: List[str]          = dataclasses.field(
        default_factory=list)
    #: Where a point was asked to hover, in metres, and how near counts as
    #: arrived.  Empty for the routes that do not point.
    hover_target_xyz: List[float]               = dataclasses.field(
        default_factory=list)
    hover_clearance_required_m: float           = 0.0
    #: How many wave cycles were asked for.  Zero for the routes that do not
    #: wave.
    expected_wave_cycles: int                   = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PanelRouteTaskSpec":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# EpisodeConfig (Section 7.4)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EpisodeConfig:
    """How to run one episode (EPIC-8 Section 7.4)."""
    simulator_identity: SimulatorIdentity
    seed: int                               = 0
    fixed_timestep: float                   = 0.002
    max_steps: int                          = 10000
    render_mode: str                        = "off"
    record_trace: bool                      = False
    contact_sampling: str                   = "all"
    state_sampling: str                     = "every_step"
    randomization_values: Dict[str, Any]    = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodeConfig":
        d = dict(d)
        identity = SimulatorIdentity.from_dict(d.pop("simulator_identity", {}))
        known = {f.name for f in dataclasses.fields(cls)} - {"simulator_identity"}
        return cls(simulator_identity=identity, **{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "EpisodeConfig":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# EpisodeResult (Section 7.5)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EpisodeResult:
    """What happened during an episode (EPIC-8 Section 7.5)."""
    episode_id: str
    trial_id: str
    status: EpisodeStatus                       = EpisodeStatus.PENDING
    termination_reason: str                     = ""
    success: bool                               = False
    start_sim_step: int                         = 0
    end_sim_step: int                           = 0
    sim_duration_s: float                       = 0.0
    wall_duration_s: float                      = 0.0
    metrics: Dict[str, float]                   = dataclasses.field(default_factory=dict)
    hard_violations: List[str]                  = dataclasses.field(default_factory=list)
    contact_summary: Dict[str, Any]             = dataclasses.field(default_factory=dict)
    control_state_changes: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    final_object_states: Dict[str, Any]         = dataclasses.field(default_factory=dict)
    final_control_states: Dict[str, Any]        = dataclasses.field(default_factory=dict)
    artifact_paths: List[str]                   = dataclasses.field(default_factory=list)
    warnings: List[str]                         = dataclasses.field(default_factory=list)
    #: Where the arm finished, joint name -> SDK degrees (#91).
    #:
    #: EMPTY IS NOT "AT ZERO".  A result from before this field existed, or
    #: from a path that does not record joints, deserialises to {} — and an
    #: evaluator that reads {} as a pose would certify arrival at HOME for
    #: every episode ever recorded.  The evaluators treat it as "I cannot tell
    #: whether the arm arrived", which fails the episode rather than passing it.
    final_joint_positions_deg: Dict[str, float] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["status"] = self.status.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodeResult":
        d = dict(d)
        d["status"] = EpisodeStatus(d.get("status", EpisodeStatus.PENDING.value))
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "EpisodeResult":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# TrialRecord (Section 7.6)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrialRecord:
    """One row in the experience store (EPIC-8 Section 7.6).

    recipe_json stores the TrajectoryRecipe as a JSON string so this module
    remains free of a runtime import from motion.recipe — callers serialise
    with recipe.to_json() and deserialise with TrajectoryRecipe.from_json().
    """
    study_id: str
    trial_id: str
    task_spec: TaskSpec
    recipe_json: str
    config: EpisodeConfig
    result: EpisodeResult
    simulator_identity: SimulatorIdentity
    created_at: str                         = ""
    started_at: str                         = ""
    completed_at: str                       = ""
    optimizer_metadata: Dict[str, Any]      = dataclasses.field(default_factory=dict)
    parent_trial_id: str                    = ""
    warm_start_trial_id: str               = ""
    promotion_state: str                    = "unpromoted"
    review_metadata: Dict[str, Any]         = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["result"]["status"] = self.result.status.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
