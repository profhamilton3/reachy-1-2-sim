"""R12-804: Contact and joint-state violation detection.

All checks operate on EpisodeResult (always available) and do not require
re-running the episode.  Per-step snapshots are not yet threaded through;
metrics that need them are returned as None and documented.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from reachy_ai.experience.models import EpisodeResult, EpisodeStatus
from reachy_ai.evaluation.base import EvaluationPolicy, Violation, ViolationKind


# Pattern emitted by EpisodeRunner into result.hard_violations
_RE_FORBIDDEN = re.compile(
    r"forbidden_robot_fixture_contact:\s*(\d+)\s*occurrences?", re.IGNORECASE
)
_RE_JOINT_LIMIT = re.compile(
    r"joint_limit_violation:\s*(.+)", re.IGNORECASE
)


def check_forbidden_contacts(
    result: EpisodeResult,
    policy: EvaluationPolicy,
) -> List[Violation]:
    """Detect forbidden robot↔fixture contacts from EpisodeResult.

    Returns a list of Violation objects, one per distinct forbidden-contact
    entry in result.hard_violations plus one if contact_summary shows
    forbidden_total > policy.forbidden_contact_allowed.
    """
    violations: List[Violation] = []
    seen_count: Optional[int] = None

    for entry in result.hard_violations:
        m = _RE_FORBIDDEN.match(entry)
        if m:
            count = int(m.group(1))
            seen_count = count
            if count > policy.forbidden_contact_allowed:
                violations.append(Violation(
                    kind=ViolationKind.FORBIDDEN_CONTACT,
                    description=(
                        f"Robot-fixture contact detected: {count} occurrence(s). "
                        f"Policy allows at most {policy.forbidden_contact_allowed}."
                    ),
                    severity="hard",
                ))

    # Also check contact_summary in case the runner populated it differently.
    forbidden_total = result.contact_summary.get("forbidden_total", 0)
    if seen_count is None and forbidden_total > policy.forbidden_contact_allowed:
        violations.append(Violation(
            kind=ViolationKind.FORBIDDEN_CONTACT,
            description=(
                f"Forbidden contact total from contact_summary: {forbidden_total}. "
                f"Policy allows at most {policy.forbidden_contact_allowed}."
            ),
            severity="hard",
        ))

    return violations


def check_joint_limits(
    result: EpisodeResult,
    policy: EvaluationPolicy,
) -> List[Violation]:
    """Parse joint-limit violations from result.hard_violations."""
    violations: List[Violation] = []
    for entry in result.hard_violations:
        m = _RE_JOINT_LIMIT.match(entry)
        if m:
            joint_name = m.group(1).strip()
            violations.append(Violation(
                kind=ViolationKind.JOINT_LIMIT,
                description=f"Joint '{joint_name}' exceeded limit during episode.",
                severity="soft",  # soft: recorded but does not block success
            ))
    return violations


def check_saturation(
    result: EpisodeResult,
    policy: EvaluationPolicy,
) -> List[Violation]:
    """Flag episodes where actuator saturation fraction exceeds the policy limit."""
    violations: List[Violation] = []
    sat_count = result.metrics.get("saturated_joint_count", 0.0)
    total_steps = result.metrics.get("total_steps", 1.0) or 1.0
    # Saturation fraction: (number of saturated joints × 1) / total joints
    # Use saturated_joint_count from the FINAL snapshot as a proxy.
    from joint_map import NUM_JOINTS  # native_mujoco must be in sys.path
    sat_fraction = sat_count / max(1, NUM_JOINTS)
    if sat_fraction > policy.saturation_fraction_limit:
        violations.append(Violation(
            kind=ViolationKind.SATURATION,
            description=(
                f"Actuator saturation fraction {sat_fraction:.2f} exceeds "
                f"policy limit {policy.saturation_fraction_limit:.2f} "
                f"({int(sat_count)} of {NUM_JOINTS} joints saturated)."
            ),
            severity="soft",
        ))
    return violations


def check_invalid_episode(
    result: EpisodeResult,
    policy: EvaluationPolicy,
) -> List[Violation]:
    """Flag episodes that terminated in INVALID (NaN/Inf or backend error)."""
    violations: List[Violation] = []
    if result.status == EpisodeStatus.INVALID:
        violations.append(Violation(
            kind=ViolationKind.NAN_IN_STATE,
            description=(
                f"Episode terminated INVALID: {result.termination_reason or 'unknown reason'}. "
                "NaN/Inf or unstable simulation detected."
            ),
            severity="hard",
        ))
    elif result.status == EpisodeStatus.ABORTED:
        violations.append(Violation(
            kind=ViolationKind.BACKEND_DEGRADED,
            description=(
                f"Episode was ABORTED: {result.termination_reason or 'unknown reason'}."
            ),
            severity="hard",
        ))
    return violations


def compute_contact_metrics(result: EpisodeResult) -> Dict[str, float]:
    """Extract contact-related scalar metrics from EpisodeResult."""
    forbidden_total = float(
        result.contact_summary.get("forbidden_total", 0) or
        result.metrics.get("forbidden_contact_count", 0)
    )
    total_contacts = float(result.contact_summary.get("total_contacts", 0))
    return {
        "forbidden_contact_count": forbidden_total,
        "total_contact_events": total_contacts,
    }


def compute_effort_metrics(result: EpisodeResult) -> Dict[str, float]:
    """Extract actuator effort metrics available from EpisodeResult."""
    return {
        "saturated_joint_count": float(result.metrics.get("saturated_joint_count", 0)),
        "total_steps": float(result.metrics.get("total_steps", 0)),
        "wall_duration_s": float(result.wall_duration_s),
        "sim_duration_s": float(result.sim_duration_s),
    }
