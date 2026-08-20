"""R12-808: Unit tests for pick-and-place success criteria (issue #18).

Verifies that evaluate_pick_place() correctly:
- Requires verified grasp (right gripper pinched the object at least once).
- Requires lift height (object peak z >= task_spec.required_lift_height).
- Requires gripper open at episode end (final_grip_force_n <= threshold).
- Defaults to True for all three checks when physics metrics are absent
  (kinematic backend / legacy EpisodeResult).
- Each check independently gates is_successful.

All tests are purely unit-level (synthetic EpisodeResult) — offline-safe,
no native MuJoCo required.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reachy_ai.experience.models import (
    EpisodeResult, EpisodeStatus, PickPlaceTaskSpec,
)
from reachy_ai.evaluation.base import EvaluationPolicy, ViolationKind
from reachy_ai.evaluation.pick_place import evaluate_pick_place


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OBJECT_ID = "red_cube"
_TARGET = [0.3, 0.0, 0.05]
_TOLERANCE = 0.02
_REQUIRED_LIFT = 0.05


def _spec(
    object_id: str = _OBJECT_ID,
    target_pose: list = None,
    target_pose_tolerance: float = _TOLERANCE,
    required_lift_height: float = _REQUIRED_LIFT,
) -> PickPlaceTaskSpec:
    return PickPlaceTaskSpec(
        task_id="t0",
        task_type="pick_and_place",
        object_id=object_id,
        target_pose=target_pose or _TARGET,
        target_pose_tolerance=target_pose_tolerance,
        required_lift_height=required_lift_height,
        settle_velocity_thresholds=[0.01],
    )


def _result(
    object_pos: list = None,
    object_vel: float = 0.0,
    # physics metrics (None = kinematic backend / not available)
    peak_z: float | None = 0.1,
    grasp_step_count: float | None = 10.0,
    final_grip_force_n: float | None = 0.0,
    forbidden_total: int = 0,
) -> EpisodeResult:
    """Build a synthetic EpisodeResult for the pick-and-place evaluator."""
    pos = object_pos if object_pos is not None else _TARGET
    metrics: dict = {
        "total_steps": 300.0,
        "saturated_joint_count": 0.0,
        "forbidden_contact_count": float(forbidden_total),
    }
    if peak_z is not None:
        metrics[f"object_{_OBJECT_ID}_peak_z_m"] = peak_z
    if grasp_step_count is not None:
        metrics["grasp_step_count"] = grasp_step_count
    if final_grip_force_n is not None:
        metrics["final_grip_force_n"] = final_grip_force_n

    hard_v = []
    if forbidden_total > 0:
        hard_v.append(
            f"forbidden_robot_fixture_contact: {forbidden_total} occurrences"
        )
    return EpisodeResult(
        episode_id="ep",
        trial_id="tr",
        status=EpisodeStatus.SUCCEEDED if forbidden_total == 0 else EpisodeStatus.FAILED,
        hard_violations=hard_v,
        contact_summary={"forbidden_total": forbidden_total},
        metrics=metrics,
        final_object_states={
            _OBJECT_ID: {
                "object_id": _OBJECT_ID,
                "pos_xyz": pos,
                "vel": object_vel,
            }
        },
    )


_POLICY = EvaluationPolicy()


# ---------------------------------------------------------------------------
# Baseline: all checks pass
# ---------------------------------------------------------------------------

class TestAllCheckPass:
    def test_full_success_all_criteria_met(self):
        result = _result(
            object_pos=_TARGET,
            peak_z=0.12,
            grasp_step_count=20.0,
            final_grip_force_n=0.0,
        )
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        assert verdict.is_successful
        assert verdict.is_safe
        assert verdict.is_valid

    def test_success_metrics_recorded(self):
        result = _result(peak_z=0.12, grasp_step_count=20.0, final_grip_force_n=0.0)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        assert verdict.metrics["peak_object_z_m"] == pytest.approx(0.12)
        assert verdict.metrics["grasp_step_count"] == pytest.approx(20.0)
        assert verdict.metrics["gripper_open_at_end"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Grasp verification
# ---------------------------------------------------------------------------

class TestGraspVerification:
    def test_zero_grasp_steps_fails(self):
        result = _result(grasp_step_count=0.0)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        assert not verdict.is_successful

    def test_zero_grasp_steps_produces_task_failure_violation(self):
        result = _result(grasp_step_count=0.0)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        grasp_v = [
            v for v in verdict.violations
            if v.kind == ViolationKind.TASK_FAILURE
            and "grasp" in v.description.lower()
        ]
        assert len(grasp_v) >= 1

    def test_grasp_violation_is_soft(self):
        result = _result(grasp_step_count=0.0)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        grasp_v = [
            v for v in verdict.violations
            if v.kind == ViolationKind.TASK_FAILURE
            and "grasp" in v.description.lower()
        ]
        assert all(v.severity == "soft" for v in grasp_v)

    def test_nonzero_grasp_steps_passes(self):
        result = _result(grasp_step_count=1.0)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        grasp_v = [
            v for v in verdict.violations
            if "grasp" in v.description.lower()
        ]
        assert grasp_v == []

    def test_absent_grasp_metric_defaults_to_pass(self):
        """kinematic backend: grasp_step_count not in metrics → no grasp violation."""
        result = _result(grasp_step_count=None)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        grasp_v = [
            v for v in verdict.violations
            if "grasp" in v.description.lower()
        ]
        assert grasp_v == []

    def test_grasp_metric_recorded_as_minus_one_when_absent(self):
        result = _result(grasp_step_count=None)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        assert verdict.metrics["grasp_step_count"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Lift height verification
# ---------------------------------------------------------------------------

class TestLiftHeight:
    def test_peak_z_below_required_fails(self):
        result = _result(peak_z=0.02)  # required=0.05
        verdict = evaluate_pick_place(result, _spec(required_lift_height=0.05), _POLICY)
        assert not verdict.is_successful

    def test_peak_z_below_required_produces_violation(self):
        result = _result(peak_z=0.02)
        verdict = evaluate_pick_place(result, _spec(required_lift_height=0.05), _POLICY)
        lift_v = [
            v for v in verdict.violations
            if v.kind == ViolationKind.TASK_FAILURE
            and "lift" in v.description.lower()
        ]
        assert len(lift_v) >= 1

    def test_lift_violation_is_soft(self):
        result = _result(peak_z=0.02)
        verdict = evaluate_pick_place(result, _spec(required_lift_height=0.05), _POLICY)
        lift_v = [
            v for v in verdict.violations
            if "lift" in v.description.lower()
        ]
        assert all(v.severity == "soft" for v in lift_v)

    def test_peak_z_exactly_at_required_passes(self):
        result = _result(peak_z=0.05)
        verdict = evaluate_pick_place(result, _spec(required_lift_height=0.05), _POLICY)
        lift_v = [v for v in verdict.violations if "lift" in v.description.lower()]
        assert lift_v == []

    def test_peak_z_above_required_passes(self):
        result = _result(peak_z=0.15)
        verdict = evaluate_pick_place(result, _spec(required_lift_height=0.05), _POLICY)
        lift_v = [v for v in verdict.violations if "lift" in v.description.lower()]
        assert lift_v == []

    def test_absent_peak_z_metric_defaults_to_pass(self):
        result = _result(peak_z=None)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        lift_v = [v for v in verdict.violations if "lift" in v.description.lower()]
        assert lift_v == []

    def test_peak_z_metric_recorded_as_minus_one_when_absent(self):
        result = _result(peak_z=None)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        assert verdict.metrics["peak_object_z_m"] == pytest.approx(-1.0)

    def test_zero_required_lift_skips_lift_check(self):
        """required_lift_height=0 should never produce a lift violation."""
        result = _result(peak_z=0.0)
        verdict = evaluate_pick_place(result, _spec(required_lift_height=0.0), _POLICY)
        lift_v = [v for v in verdict.violations if "lift" in v.description.lower()]
        assert lift_v == []


# ---------------------------------------------------------------------------
# Gripper open at end
# ---------------------------------------------------------------------------

class TestGripperOpenAtEnd:
    def test_high_final_force_fails(self):
        result = _result(final_grip_force_n=2.0)  # still gripping
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        assert not verdict.is_successful

    def test_high_final_force_produces_violation(self):
        result = _result(final_grip_force_n=2.0)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        grip_v = [
            v for v in verdict.violations
            if v.kind == ViolationKind.TASK_FAILURE
            and "gripper" in v.description.lower()
        ]
        assert len(grip_v) >= 1

    def test_grip_violation_is_soft(self):
        result = _result(final_grip_force_n=2.0)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        grip_v = [v for v in verdict.violations if "gripper" in v.description.lower()]
        assert all(v.severity == "soft" for v in grip_v)

    def test_zero_final_force_passes(self):
        result = _result(final_grip_force_n=0.0)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        grip_v = [v for v in verdict.violations if "gripper" in v.description.lower()]
        assert grip_v == []

    def test_force_at_threshold_passes(self):
        """final_grip_force_n == threshold (0.1 N) should pass (≤)."""
        result = _result(final_grip_force_n=0.1)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        grip_v = [v for v in verdict.violations if "gripper" in v.description.lower()]
        assert grip_v == []

    def test_absent_final_force_metric_defaults_to_open(self):
        result = _result(final_grip_force_n=None)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        grip_v = [v for v in verdict.violations if "gripper" in v.description.lower()]
        assert grip_v == []

    def test_gripper_open_metric_recorded(self):
        result = _result(final_grip_force_n=0.0)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        assert verdict.metrics["gripper_open_at_end"] == pytest.approx(1.0)

    def test_gripper_closed_metric_recorded(self):
        result = _result(final_grip_force_n=1.5)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        assert verdict.metrics["gripper_open_at_end"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Independence of checks: position passes but grasp/lift/grip fail
# ---------------------------------------------------------------------------

class TestCheckIndependence:
    def test_position_ok_but_no_grasp_is_failure(self):
        result = _result(object_pos=_TARGET, grasp_step_count=0.0)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        assert not verdict.is_successful

    def test_position_ok_but_no_lift_is_failure(self):
        result = _result(object_pos=_TARGET, peak_z=0.01)
        verdict = evaluate_pick_place(result, _spec(required_lift_height=0.05), _POLICY)
        assert not verdict.is_successful

    def test_position_ok_but_gripper_not_open_is_failure(self):
        result = _result(object_pos=_TARGET, final_grip_force_n=5.0)
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        assert not verdict.is_successful

    def test_all_three_fail_produces_three_violations(self):
        result = _result(
            object_pos=_TARGET,
            grasp_step_count=0.0,
            peak_z=0.01,
            final_grip_force_n=5.0,
        )
        verdict = evaluate_pick_place(result, _spec(required_lift_height=0.05), _POLICY)
        task_v = [v for v in verdict.violations if v.kind == ViolationKind.TASK_FAILURE]
        assert len(task_v) >= 3


# ---------------------------------------------------------------------------
# Kinematic backend: all physics metrics absent → optimistic defaults
# ---------------------------------------------------------------------------

class TestKinematicBackendDefaults:
    def test_all_absent_metrics_object_at_target_succeeds(self):
        result = _result(
            object_pos=_TARGET,
            peak_z=None,
            grasp_step_count=None,
            final_grip_force_n=None,
        )
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        assert verdict.is_successful

    def test_all_absent_metrics_recorded_as_minus_one(self):
        result = _result(
            object_pos=_TARGET,
            peak_z=None,
            grasp_step_count=None,
            final_grip_force_n=None,
        )
        verdict = evaluate_pick_place(result, _spec(), _POLICY)
        assert verdict.metrics["peak_object_z_m"] == pytest.approx(-1.0)
        assert verdict.metrics["grasp_step_count"] == pytest.approx(-1.0)
