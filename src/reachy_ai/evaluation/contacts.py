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


# Reachy 1.2 right-arm + head + antennas = 21 controllable joints (see
# native_mujoco/joint_map.py). The evaluator must run offline on EpisodeResult
# alone, so we resolve the count from the native module when it is importable
# and otherwise fall back to the known constant. The safety evaluator must
# never raise merely because the native MuJoCo package is not on sys.path.
_DEFAULT_NUM_JOINTS = 21


def _num_joints() -> int:
    try:
        from joint_map import NUM_JOINTS  # native_mujoco, when on sys.path
        return int(NUM_JOINTS)
    except Exception:
        return _DEFAULT_NUM_JOINTS


NUM_JOINTS = _num_joints()


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

    Fail-closed: the reported count is the MAX across every available evidence
    source (parsed hard_violations entries, contact_summary["forbidden_total"],
    and metrics["forbidden_contact_count"]). Sources may disagree — e.g. the
    runner's structured summary can report contacts that never made it into the
    free-text hard_violations list — so trusting only the first source that
    parses could let a real collision through. We take the worst case.

    Uses re.search (not match) so a step/timestamp prefix on the runner's
    string cannot silently defeat detection.
    """
    violations: List[Violation] = []
    counts: List[int] = []

    for entry in result.hard_violations:
        m = _RE_FORBIDDEN.search(entry)
        if m:
            counts.append(int(m.group(1)))

    try:
        counts.append(int(result.contact_summary.get("forbidden_total", 0) or 0))
    except (TypeError, ValueError):
        pass
    try:
        counts.append(int(result.metrics.get("forbidden_contact_count", 0) or 0))
    except (TypeError, ValueError):
        pass

    worst = max(counts) if counts else 0
    if worst > policy.forbidden_contact_allowed:
        violations.append(Violation(
            kind=ViolationKind.FORBIDDEN_CONTACT,
            description=(
                f"Robot-fixture contact detected: {worst} occurrence(s) "
                f"(max across hard_violations / contact_summary / metrics). "
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
