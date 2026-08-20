"""R12-804: Pick-and-place task evaluator.

Evaluates a pick_and_place episode against Section 8.3 of EPIC-8.md.

Success requires (all verified from EpisodeResult):
  1. No forbidden contact.
  2. Final object state is within target_pose_tolerance of target_pose.
  3. Gripper is open at the end (object released).
  4. Object velocity within settle threshold.
  5. Right gripper pinched the target object at least once during the episode.
  6. Object reached required_lift_height (m) above table during transit.

Checks 5 and 6 require per-step metrics written by EpisodeRunner (issue #18).
When those metrics are absent (kinematic backend), the checks default to True.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from reachy_ai.experience.models import EpisodeResult, EpisodeStatus, PickPlaceTaskSpec
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

_GRIPPER_OPEN_THRESHOLD_RAD = -0.4     # r_gripper < this value → open
_GRIPPER_OPEN_FORCE_THRESHOLD_N = 0.1  # grip_force_n ≤ this → gripper open


def _euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _object_position(obj_state: Dict) -> Optional[List[float]]:
    """Extract [x, y, z] from an object state dict produced by ObjectTracker."""
    for key in ("pos", "position", "xyz", "pos_xyz"):
        val = obj_state.get(key)
        if val is not None and len(val) >= 3:
            return list(val[:3])
    return None


def _object_velocity(obj_state: Dict) -> Optional[float]:
    """Return scalar linear velocity from object state dict."""
    for key in ("vel", "velocity", "lin_vel"):
        val = obj_state.get(key)
        if val is not None:
            if isinstance(val, (int, float)):
                return float(val)
            if hasattr(val, "__len__") and len(val) >= 3:
                return math.sqrt(sum(v ** 2 for v in val[:3]))
    return None


def evaluate_pick_place(
    result: EpisodeResult,
    task_spec: PickPlaceTaskSpec,
    policy: Optional[EvaluationPolicy] = None,
) -> EpisodeVerdict:
    """Evaluate a pick_and_place episode.

    Args:
        result:    EpisodeResult from EpisodeRunner.run().
        task_spec: Object, target pose, and tolerance.
        policy:    Threshold policy; defaults to EvaluationPolicy().

    Returns:
        EpisodeVerdict with violations, metrics, ranking scores, and explanation.
    """
    if policy is None:
        policy = EvaluationPolicy()

    violations: List[Violation] = []
    lines: List[str] = [
        f"=== Pick-Place Verdict: {task_spec.object_id} "
        f"→ target {task_spec.target_pose} ==="
    ]

    # ------------------------------------------------------------------ #
    # 1. Episode validity                                                  #
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
    violations.extend(check_joint_limits(result, policy))
    violations.extend(check_saturation(result, policy))
    hard_violations = [v for v in violations if v.severity == "hard"]
    is_safe = len(hard_violations) == 0
    if contact_violations:
        lines.append(f"UNSAFE: {len(contact_violations)} forbidden-contact violation(s).")

    # ------------------------------------------------------------------ #
    # 3. Task success                                                      #
    # ------------------------------------------------------------------ #
    obj_state = result.final_object_states.get(task_spec.object_id, {})
    obj_pos = _object_position(obj_state)
    obj_vel = _object_velocity(obj_state)
    target_pos = list(task_spec.target_pose[:3]) if len(task_spec.target_pose) >= 3 else None

    position_ok = False
    position_error = None
    if obj_pos and target_pos:
        position_error = _euclidean_distance(obj_pos, target_pos)
        position_ok = position_error <= (
            task_spec.target_pose_tolerance or policy.target_position_tolerance_m
        )
        if not position_ok:
            msg = (
                f"Object '{task_spec.object_id}' is {position_error:.4f} m from "
                f"target (tolerance {task_spec.target_pose_tolerance:.4f} m)."
            )
            violations.append(Violation(
                kind=ViolationKind.TASK_FAILURE,
                description=msg,
                severity="soft",
            ))
            lines.append(f"TASK FAIL: {msg}")
    elif not obj_pos:
        lines.append(
            f"NOTE: Object '{task_spec.object_id}' not found in final_object_states — "
            "position success cannot be verified."
        )

    # Velocity settle check
    velocity_ok = True
    if obj_vel is not None:
        thresholds = task_spec.settle_velocity_thresholds or [0.01]
        max_v = max(thresholds)
        if obj_vel > max_v:
            velocity_ok = False
            violations.append(Violation(
                kind=ViolationKind.TASK_FAILURE,
                description=(
                    f"Object '{task_spec.object_id}' velocity {obj_vel:.4f} m/s "
                    f"exceeds settle threshold {max_v:.4f} m/s."
                ),
                severity="soft",
            ))

    # ------------------------------------------------------------------ #
    # 3b. Gripper open at end (object released)                           #
    # ------------------------------------------------------------------ #
    _final_force = result.metrics.get("final_grip_force_n")
    if _final_force is None:
        gripper_open = True  # kinematic backend — not checkable
    else:
        gripper_open = float(_final_force) <= _GRIPPER_OPEN_FORCE_THRESHOLD_N
        if not gripper_open:
            msg = (
                f"Gripper not open at episode end "
                f"(grip_force={_final_force:.3f} N > "
                f"{_GRIPPER_OPEN_FORCE_THRESHOLD_N:.3f} N)."
            )
            violations.append(Violation(
                kind=ViolationKind.TASK_FAILURE,
                description=msg,
                severity="soft",
            ))
            lines.append(f"TASK FAIL: {msg}")

    # ------------------------------------------------------------------ #
    # 3c. Verified grasp (right gripper pinched the object)               #
    # ------------------------------------------------------------------ #
    _grasp_steps = result.metrics.get("grasp_step_count")
    if _grasp_steps is None:
        grasped = True  # kinematic backend — not checkable
    else:
        grasped = float(_grasp_steps) > 0
        if not grasped:
            msg = (
                f"No verified grasp: right gripper never pinched "
                f"'{task_spec.object_id}' during the episode."
            )
            violations.append(Violation(
                kind=ViolationKind.TASK_FAILURE,
                description=msg,
                severity="soft",
            ))
            lines.append(f"TASK FAIL: {msg}")

    # ------------------------------------------------------------------ #
    # 3d. Lift height (object reached required height during transit)      #
    # ------------------------------------------------------------------ #
    _peak_z = result.metrics.get(f"object_{task_spec.object_id}_peak_z_m")
    if _peak_z is None:
        lift_ok = True  # kinematic backend — not checkable
        lines.append(
            f"NOTE: peak_z not tracked for '{task_spec.object_id}' "
            "(lift height cannot be verified)."
        )
    elif task_spec.required_lift_height > 0.0:
        lift_ok = float(_peak_z) >= task_spec.required_lift_height
        if not lift_ok:
            msg = (
                f"Object '{task_spec.object_id}' peak height {_peak_z:.4f} m "
                f"below required lift {task_spec.required_lift_height:.4f} m."
            )
            violations.append(Violation(
                kind=ViolationKind.TASK_FAILURE,
                description=msg,
                severity="soft",
            ))
            lines.append(f"TASK FAIL: {msg}")
    else:
        lift_ok = True  # required_lift_height == 0 means no lift required

    is_successful = (
        is_valid and is_safe
        and position_ok and velocity_ok
        and gripper_open and grasped and lift_ok
    )
    if is_successful:
        lines.append(
            f"SUCCESS: Object at target (error={position_error:.4f} m), "
            f"grasped={grasped}, lift_ok={lift_ok}, gripper_open={gripper_open}."
        )
    elif not position_ok and obj_pos and target_pos:
        pass  # already logged above

    # ------------------------------------------------------------------ #
    # 4. Metrics                                                          #
    # ------------------------------------------------------------------ #
    contact_metrics = compute_contact_metrics(result)
    effort_metrics = compute_effort_metrics(result)
    metrics: Dict[str, float] = {
        **contact_metrics,
        **effort_metrics,
        "position_error_m": float(position_error) if position_error is not None else -1.0,
        "object_velocity_m_s": float(obj_vel) if obj_vel is not None else -1.0,
        "hard_violation_count": float(len([v for v in violations if v.severity == "hard"])),
        "soft_violation_count": float(len([v for v in violations if v.severity == "soft"])),
        "peak_object_z_m": float(_peak_z) if _peak_z is not None else -1.0,
        "grasp_step_count": float(_grasp_steps) if _grasp_steps is not None else -1.0,
        "gripper_open_at_end": float(gripper_open),
    }

    # ------------------------------------------------------------------ #
    # 5. Ranking scores                                                    #
    # ------------------------------------------------------------------ #
    total_steps = max(1.0, effort_metrics["total_steps"])
    tol = task_spec.target_pose_tolerance or policy.target_position_tolerance_m
    accuracy = 0.0
    if position_error is not None:
        accuracy = max(0.0, 1.0 - position_error / max(tol, 0.001))

    ranking_scores: Dict[str, float] = {
        "accuracy_score": accuracy if position_ok else 0.0,
        "effort_score": max(
            0.0,
            1.0 - effort_metrics["saturated_joint_count"] / 21.0
        ),
        "duration_score": 1.0 / (1.0 + total_steps / 1000.0),
        # clearance_score and smoothness_score require per-step snapshot data
        # not available in EpisodeResult; omitted until EpisodeRunner exposes
        # snapshots (tracked in issue #17).
    }

    lines.append(
        f"Steps: {int(total_steps)}  "
        f"PositionError: {position_error:.4f} m"
        if position_error is not None
        else f"Steps: {int(total_steps)}"
    )

    return EpisodeVerdict(
        episode_id=result.episode_id,
        trial_id=result.trial_id,
        task_type=task_spec.task_type or "pick_and_place",
        policy_version=policy.policy_version,
        is_valid=is_valid,
        is_safe=is_safe,
        is_successful=is_successful,
        violations=violations,
        metrics=metrics,
        ranking_scores=ranking_scores,
        explanation="\n".join(lines),
    )
