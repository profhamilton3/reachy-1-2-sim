"""R12-804: Safety evaluator and task objective library."""

from reachy_ai.evaluation.base import (
    EvaluationPolicy,
    EpisodeVerdict,
    Violation,
    ViolationKind,
)
from reachy_ai.evaluation.contacts import (
    check_forbidden_contacts,
    check_invalid_episode,
    check_joint_limits,
    check_saturation,
    compute_contact_metrics,
    compute_effort_metrics,
)
from reachy_ai.evaluation.control_panel import evaluate_control_panel
from reachy_ai.evaluation.pick_place import evaluate_pick_place
from reachy_ai.evaluation.ranking import (
    RankedCandidate,
    rank_candidates,
    best_candidate,
    dominated_by,
    explain_ranking,
)

__all__ = [
    "EvaluationPolicy",
    "EpisodeVerdict",
    "Violation",
    "ViolationKind",
    "check_forbidden_contacts",
    "check_invalid_episode",
    "check_joint_limits",
    "check_saturation",
    "compute_contact_metrics",
    "compute_effort_metrics",
    "evaluate_control_panel",
    "evaluate_pick_place",
    "RankedCandidate",
    "rank_candidates",
    "best_candidate",
    "dominated_by",
    "explain_ranking",
]
