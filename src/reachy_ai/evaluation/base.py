"""R12-804: Evaluation foundation — policy, violations, and verdict types.

EvaluationPolicy is frozen and versioned so thresholds cannot change once a
study begins.  Every verdict carries a full explanation so the search engine
and human reviewers can understand why an episode succeeded or failed.
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from typing import Any, Dict, List, Optional


class ViolationKind(str, Enum):
    """Classification of a hard or soft failure."""
    FORBIDDEN_CONTACT    = "forbidden_contact"
    SELF_COLLISION       = "self_collision"
    NAN_IN_STATE         = "nan_in_state"
    JOINT_LIMIT          = "joint_limit"
    SATURATION           = "saturation"
    UNINTENDED_TOGGLE    = "unintended_toggle"
    TASK_TIMEOUT         = "task_timeout"
    IDENTITY_MISMATCH    = "identity_mismatch"
    BACKEND_DEGRADED     = "backend_degraded"
    INVALID_EPISODE      = "invalid_episode"
    TASK_FAILURE         = "task_failure"


@dataclasses.dataclass(frozen=True)
class EvaluationPolicy:
    """Immutable thresholds for one study.  Always increment policy_version
    if any threshold changes — do not modify an existing policy in-place.
    """
    policy_version: int                     = 1
    # Contact rules
    forbidden_contact_allowed: int          = 0   # any = fail
    # Joint limits
    joint_limit_tolerance_rad: float        = 0.05
    # Actuator saturation
    saturation_fraction_limit: float        = 0.8
    # Control-panel
    max_unintended_toggles: int             = 0
    settle_steps_required: int              = 5
    # Pick-place
    target_position_tolerance_m: float      = 0.02
    required_lift_height_m: float           = 0.05
    # Ranking
    clearance_weight: float                 = 1.0
    effort_weight: float                    = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluationPolicy":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "EvaluationPolicy":
        return cls.from_dict(json.loads(s))


@dataclasses.dataclass
class Violation:
    """One identified failure or safety concern in an episode."""
    kind: ViolationKind
    description: str
    step: Optional[int] = None
    severity: str = "hard"  # "hard" terminates candidate eligibility

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "description": self.description,
            "step": self.step,
            "severity": self.severity,
        }


@dataclasses.dataclass
class EpisodeVerdict:
    """Structured result of evaluating one episode against a task spec.

    Tier ordering (highest first) for lexicographic ranking:
      1. is_valid         — episode produced no NaN/INVALID state
      2. is_safe          — no forbidden contacts or hard violations
      3. is_successful    — task success criteria met
      4. accuracy_score   — task-specific accuracy [0, 1]
      5. clearance_score  — minimum robot-fixture clearance [0, 1]
      6. effort_score     — inverse of saturation fraction [0, 1]
      7. smoothness_score — jerk proxy (1.0 = no per-step data) [0, 1]
      8. duration_score   — shorter = better [0, 1]
    """
    episode_id: str
    trial_id: str
    task_type: str
    policy_version: int
    is_valid: bool
    is_safe: bool
    is_successful: bool
    violations: List[Violation]
    metrics: Dict[str, float]
    ranking_scores: Dict[str, float]
    explanation: str

    # Convenience tier tuple for sorting (higher = better, all tiers).
    @property
    def rank_key(self):
        s = self.ranking_scores
        return (
            float(self.is_valid),
            float(self.is_safe),
            float(self.is_successful),
            s.get("accuracy_score", 0.0),
            s.get("clearance_score", 0.0),
            s.get("effort_score", 0.0),
            s.get("smoothness_score", 0.0),
            s.get("duration_score", 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "trial_id": self.trial_id,
            "task_type": self.task_type,
            "policy_version": self.policy_version,
            "is_valid": self.is_valid,
            "is_safe": self.is_safe,
            "is_successful": self.is_successful,
            "violations": [v.to_dict() for v in self.violations],
            "metrics": self.metrics,
            "ranking_scores": self.ranking_scores,
            "explanation": self.explanation,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
