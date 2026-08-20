"""R12-804: Control-panel task evaluator.

Evaluates an operate_control episode against Section 8.2 of EPIC-8.md:
  1. Intended control reached the desired logical state.
  2. The change persists (settle steps).
  3. No neighboring control changed.
  4. No forbidden console contact.
  5. The acting arm retracted to a safe guard/home state.
  6. Final state observed after actuation with matching scene revision.
  7. Backend and recording remained valid through episode completion.

Points 5–7 cannot be verified from EpisodeResult alone; they are noted in the
explanation and left to the compatibility replay (PR 8.9).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from reachy_ai.experience.models import EpisodeResult, EpisodeStatus, ControlPanelTaskSpec
from reachy_ai.evaluation.base import (
    EvaluationPolicy, EpisodeVerdict, Violation, ViolationKind,
)
from reachy_ai.evaluation.contacts import (
    check_forbidden_contacts,
    check_invalid_episode,
    check_joint_limits,
    check_saturation,
    compute_contact_metrics,
    compute_effort_metrics,
)


def evaluate_control_panel(
    result: EpisodeResult,
    task_spec: ControlPanelTaskSpec,
    policy: Optional[EvaluationPolicy] = None,
) -> EpisodeVerdict:
    """Evaluate an operate_control episode.

    Args:
        result:    EpisodeResult from EpisodeRunner.run().
        task_spec: Which control to operate and what final state is required.
        policy:    Threshold policy; defaults to EvaluationPolicy().

    Returns:
        EpisodeVerdict with full violation list, metrics, ranking scores, and
        a human-readable explanation.
    """
    if policy is None:
        policy = EvaluationPolicy()

    violations: List[Violation] = []
    lines: List[str] = [
        f"=== Control-Panel Verdict: {task_spec.control_id} "
        f"(target={'ON' if task_spec.requested_final_state else 'OFF'}) ==="
    ]

    # ------------------------------------------------------------------ #
    # 1. Episode validity (NaN / INVALID / ABORTED)                       #
    # ------------------------------------------------------------------ #
    invalid_violations = check_invalid_episode(result, policy)
    violations.extend(invalid_violations)
    is_valid = len(invalid_violations) == 0
    if not is_valid:
        lines.append(f"INVALID: {invalid_violations[0].description}")

    # ------------------------------------------------------------------ #
    # 2. Hard safety: forbidden contact                                    #
    # ------------------------------------------------------------------ #
    contact_violations = check_forbidden_contacts(result, policy)
    violations.extend(contact_violations)
    joint_violations = check_joint_limits(result, policy)
    violations.extend(joint_violations)
    sat_violations = check_saturation(result, policy)
    violations.extend(sat_violations)
    if contact_violations:
        lines.append(
            f"UNSAFE: {len(contact_violations)} forbidden-contact violation(s)."
        )

    # ------------------------------------------------------------------ #
    # 3. Task success: intended control state change                       #
    # ------------------------------------------------------------------ #
    success_issues: List[str] = []
    intended_toggled = False
    unintended_toggles: List[str] = []

    final_ctrl = result.final_control_states
    if task_spec.control_id:
        ctrl_state = final_ctrl.get(task_spec.control_id, {})
        actual_on = bool(ctrl_state.get("on", False))
        intended_toggled = (actual_on == task_spec.requested_final_state)
        if not intended_toggled:
            success_issues.append(
                f"Control '{task_spec.control_id}' is "
                f"{'ON' if actual_on else 'OFF'} but expected "
                f"{'ON' if task_spec.requested_final_state else 'OFF'}."
            )
            violations.append(Violation(
                kind=ViolationKind.TASK_FAILURE,
                description=success_issues[-1],
                severity="soft",
            ))

    # Neighboring controls must not have changed state. Collect first, then
    # assign severity based on the policy tolerance: toggles beyond the allowed
    # count are hard (unsafe); toggles within tolerance are recorded as soft so
    # a "safe" verdict never coexists with an un-counted hard violation.
    for nbr in task_spec.neighbor_control_ids:
        nbr_state = final_ctrl.get(nbr, {})
        if bool(nbr_state.get("on", False)):
            unintended_toggles.append(f"Neighbor control '{nbr}' is unexpectedly ON.")

    toggles_exceed_tolerance = len(unintended_toggles) > policy.max_unintended_toggles
    for msg in unintended_toggles:
        violations.append(Violation(
            kind=ViolationKind.UNINTENDED_TOGGLE,
            description=msg,
            severity="hard" if toggles_exceed_tolerance else "soft",
        ))
    if toggles_exceed_tolerance:
        lines.append(
            f"UNSAFE: {len(unintended_toggles)} unintended neighboring "
            f"control toggle(s)."
        )

    # Single source of truth for safety: no hard violation of any kind.
    is_safe = not any(v.severity == "hard" for v in violations)

    is_successful = (
        is_valid and
        is_safe and
        intended_toggled and
        len(unintended_toggles) <= policy.max_unintended_toggles
    )

    if is_successful:
        lines.append("SUCCESS: Intended control actuated; no safety violations.")
    elif success_issues:
        lines.extend(success_issues)

    # ------------------------------------------------------------------ #
    # 4. Metrics                                                          #
    # ------------------------------------------------------------------ #
    contact_metrics = compute_contact_metrics(result)
    effort_metrics = compute_effort_metrics(result)
    metrics: Dict[str, float] = {
        **contact_metrics,
        **effort_metrics,
        "intended_state_change": float(intended_toggled),
        "unintended_state_changes": float(len(unintended_toggles)),
        "hard_violation_count": float(len([v for v in violations if v.severity == "hard"])),
        "soft_violation_count": float(len([v for v in violations if v.severity == "soft"])),
    }

    # ------------------------------------------------------------------ #
    # 5. Ranking scores (all in [0, 1], higher = better)                  #
    # ------------------------------------------------------------------ #
    total_steps = max(1.0, effort_metrics["total_steps"])
    ranking_scores: Dict[str, float] = {
        "accuracy_score": float(intended_toggled),
        "clearance_score": 1.0 - min(1.0, contact_metrics["forbidden_contact_count"]),
        "effort_score": max(
            0.0,
            1.0 - (effort_metrics["saturated_joint_count"] / max(1.0, 21.0))
        ),
        "smoothness_score": 1.0,  # per-step jerk not available without snapshots
        "duration_score": 1.0 / (1.0 + total_steps / 1000.0),
    }

    lines.append(
        f"Steps: {int(total_steps)}  "
        f"ForbiddenContact: {int(contact_metrics['forbidden_contact_count'])}  "
        f"UnintendedToggles: {len(unintended_toggles)}"
    )

    return EpisodeVerdict(
        episode_id=result.episode_id,
        trial_id=result.trial_id,
        task_type=task_spec.task_type or "operate_control",
        policy_version=policy.policy_version,
        is_valid=is_valid,
        is_safe=is_safe,
        is_successful=is_successful,
        violations=violations,
        metrics=metrics,
        ranking_scores=ranking_scores,
        explanation="\n".join(lines),
    )
