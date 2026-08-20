"""R12-804: Unintended neighboring control toggle detection tests.

Verifies that evaluate_control_panel() correctly:
- Detects when a neighbor control changes state unexpectedly.
- Classifies unintended toggles as hard violations (unsafe).
- Ensures an unintended-toggle episode can never outrank a clean episode.
- Handles edge cases: empty neighbor list, all neighbors unchanged, etc.

These tests are purely unit-level (synthetic EpisodeResult) so they run
offline without native MuJoCo.
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
    EpisodeResult, EpisodeStatus, ControlPanelTaskSpec,
)
from reachy_ai.evaluation.base import EvaluationPolicy, ViolationKind
from reachy_ai.evaluation.control_panel import evaluate_control_panel
from reachy_ai.evaluation.ranking import RankedCandidate, rank_candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    control_states: dict,
    forbidden_total: int = 0,
    total_steps: float = 300.0,
) -> EpisodeResult:
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
        metrics={
            "total_steps": total_steps,
            "saturated_joint_count": 0.0,
            "forbidden_contact_count": float(forbidden_total),
        },
        final_control_states=control_states,
    )


def _spec(
    control_id: str = "btn_red",
    requested: bool = True,
    neighbors: list = None,
) -> ControlPanelTaskSpec:
    return ControlPanelTaskSpec(
        task_id="t0",
        task_type="operate_control",
        control_id=control_id,
        requested_final_state=requested,
        neighbor_control_ids=neighbors or [],
    )


_POLICY = EvaluationPolicy()


# ---------------------------------------------------------------------------
# Basic toggle detection
# ---------------------------------------------------------------------------

class TestIntendedToggle:
    def test_intended_on_succeeds(self):
        result = _make_result({"btn_red": {"id": "btn_red", "on": True}})
        verdict = evaluate_control_panel(result, _spec("btn_red", True), _POLICY)
        assert verdict.is_successful
        assert verdict.is_safe

    def test_intended_off_from_on_succeeds(self):
        result = _make_result({"btn_red": {"id": "btn_red", "on": False}})
        verdict = evaluate_control_panel(result, _spec("btn_red", False), _POLICY)
        assert verdict.is_successful
        assert verdict.is_safe

    def test_wrong_final_state_fails(self):
        result = _make_result({"btn_red": {"id": "btn_red", "on": False}})
        verdict = evaluate_control_panel(result, _spec("btn_red", True), _POLICY)
        assert not verdict.is_successful
        assert verdict.is_safe  # still safe — just didn't succeed

    def test_control_absent_from_final_states_fails(self):
        result = _make_result({})  # no control state recorded
        verdict = evaluate_control_panel(result, _spec("btn_red", True), _POLICY)
        assert not verdict.is_successful

    def test_task_failure_violation_is_soft(self):
        result = _make_result({"btn_red": {"id": "btn_red", "on": False}})
        verdict = evaluate_control_panel(result, _spec("btn_red", True), _POLICY)
        task_violations = [
            v for v in verdict.violations if v.kind == ViolationKind.TASK_FAILURE
        ]
        assert len(task_violations) >= 1
        assert all(v.severity == "soft" for v in task_violations)


# ---------------------------------------------------------------------------
# Neighboring control toggle detection
# ---------------------------------------------------------------------------

class TestNeighborToggleDetection:
    def test_unchanged_neighbor_no_violation(self):
        result = _make_result({
            "btn_red": {"id": "btn_red", "on": True},
            "btn_green": {"id": "btn_green", "on": False},
        })
        verdict = evaluate_control_panel(
            result,
            _spec("btn_red", True, neighbors=["btn_green"]),
            _POLICY,
        )
        unintended = [v for v in verdict.violations
                      if v.kind == ViolationKind.UNINTENDED_TOGGLE]
        assert unintended == []
        assert verdict.is_safe

    def test_toggled_neighbor_produces_hard_violation(self):
        result = _make_result({
            "btn_red": {"id": "btn_red", "on": True},
            "btn_green": {"id": "btn_green", "on": True},  # should remain OFF
        })
        verdict = evaluate_control_panel(
            result,
            _spec("btn_red", True, neighbors=["btn_green"]),
            _POLICY,
        )
        unintended = [v for v in verdict.violations
                      if v.kind == ViolationKind.UNINTENDED_TOGGLE]
        assert len(unintended) == 1
        assert unintended[0].severity == "hard"

    def test_toggled_neighbor_makes_episode_unsafe(self):
        result = _make_result({
            "btn_red": {"id": "btn_red", "on": True},
            "btn_green": {"id": "btn_green", "on": True},
        })
        verdict = evaluate_control_panel(
            result,
            _spec("btn_red", True, neighbors=["btn_green"]),
            _POLICY,
        )
        assert not verdict.is_safe
        assert not verdict.is_successful

    def test_multiple_toggled_neighbors_all_flagged(self):
        result = _make_result({
            "btn_red": {"id": "btn_red", "on": True},
            "sw_left": {"id": "sw_left", "on": True},
            "sw_right": {"id": "sw_right", "on": True},
        })
        verdict = evaluate_control_panel(
            result,
            _spec("btn_red", True, neighbors=["sw_left", "sw_right"]),
            _POLICY,
        )
        unintended = [v for v in verdict.violations
                      if v.kind == ViolationKind.UNINTENDED_TOGGLE]
        assert len(unintended) == 2

    def test_empty_neighbor_list_no_violation(self):
        result = _make_result({"btn_red": {"id": "btn_red", "on": True}})
        verdict = evaluate_control_panel(
            result,
            _spec("btn_red", True, neighbors=[]),
            _POLICY,
        )
        assert verdict.is_safe
        assert verdict.is_successful

    def test_absent_neighbor_not_flagged(self):
        """Neighbor not present in final_control_states should not cause violation."""
        result = _make_result({"btn_red": {"id": "btn_red", "on": True}})
        # btn_green not in final_control_states — treat as OFF (no toggle)
        verdict = evaluate_control_panel(
            result,
            _spec("btn_red", True, neighbors=["btn_green"]),
            _POLICY,
        )
        unintended = [v for v in verdict.violations
                      if v.kind == ViolationKind.UNINTENDED_TOGGLE]
        assert unintended == []


# ---------------------------------------------------------------------------
# Combined: forbidden contact + unintended toggle
# ---------------------------------------------------------------------------

class TestCombinedViolations:
    def test_both_contact_and_toggle_produce_distinct_violations(self):
        result = _make_result(
            {"btn_red": {"id": "btn_red", "on": True},
             "btn_green": {"id": "btn_green", "on": True}},
            forbidden_total=2,
        )
        verdict = evaluate_control_panel(
            result,
            _spec("btn_red", True, neighbors=["btn_green"]),
            _POLICY,
        )
        contact_v = [v for v in verdict.violations
                     if v.kind == ViolationKind.FORBIDDEN_CONTACT]
        toggle_v = [v for v in verdict.violations
                    if v.kind == ViolationKind.UNINTENDED_TOGGLE]
        assert len(contact_v) >= 1
        assert len(toggle_v) >= 1
        assert not verdict.is_safe
        assert not verdict.is_successful


# ---------------------------------------------------------------------------
# Ranking: unintended-toggle episode cannot outrank clean episode
# ---------------------------------------------------------------------------

class TestRankingWithToggleViolations:
    def test_clean_beats_unintended_toggle_episode(self):
        clean_result = _make_result(
            {"btn_red": {"id": "btn_red", "on": True}},
            total_steps=5000.0,
        )
        toggle_result = _make_result(
            {"btn_red": {"id": "btn_red", "on": True},
             "btn_green": {"id": "btn_green", "on": True}},
            total_steps=100.0,  # faster
        )
        spec_clean = _spec("btn_red", True)
        spec_toggle = _spec("btn_red", True, neighbors=["btn_green"])
        clean_verdict = evaluate_control_panel(clean_result, spec_clean, _POLICY)
        toggle_verdict = evaluate_control_panel(toggle_result, spec_toggle, _POLICY)

        clean_cand = RankedCandidate(verdict=clean_verdict, trial_id="clean")
        toggle_cand = RankedCandidate(verdict=toggle_verdict, trial_id="toggle")
        ranked = rank_candidates([toggle_cand, clean_cand])
        assert ranked[0].trial_id == "clean", (
            "Episode with unintended toggle must never outrank a clean episode"
        )

    def test_fastest_clean_beats_fastest_toggle(self):
        fast_clean = _make_result(
            {"btn_red": {"id": "btn_red", "on": True}},
            total_steps=50.0,
        )
        fast_toggle = _make_result(
            {"btn_red": {"id": "btn_red", "on": True},
             "sw_left": {"id": "sw_left", "on": True}},
            total_steps=10.0,
        )
        spec_clean = _spec("btn_red", True)
        spec_toggle = _spec("btn_red", True, neighbors=["sw_left"])
        clean_v = evaluate_control_panel(fast_clean, spec_clean, _POLICY)
        toggle_v = evaluate_control_panel(fast_toggle, spec_toggle, _POLICY)

        ranked = rank_candidates([
            RankedCandidate(verdict=toggle_v, trial_id="toggle"),
            RankedCandidate(verdict=clean_v, trial_id="clean"),
        ])
        assert ranked[0].trial_id == "clean"


# ---------------------------------------------------------------------------
# Metrics presence
# ---------------------------------------------------------------------------

class TestMetricsPopulated:
    def test_verdict_metrics_contain_expected_keys(self):
        result = _make_result({"btn_red": {"id": "btn_red", "on": True}})
        verdict = evaluate_control_panel(result, _spec(), _POLICY)
        for key in ("intended_state_change", "unintended_state_changes",
                    "total_steps", "forbidden_contact_count"):
            assert key in verdict.metrics, f"Missing metric: {key}"

    def test_ranking_scores_contain_expected_keys(self):
        result = _make_result({"btn_red": {"id": "btn_red", "on": True}})
        verdict = evaluate_control_panel(result, _spec(), _POLICY)
        for key in ("accuracy_score", "clearance_score", "effort_score",
                    "smoothness_score", "duration_score"):
            assert key in verdict.ranking_scores, f"Missing ranking score: {key}"

    def test_verdict_serialises_to_json(self):
        import json
        result = _make_result({"btn_red": {"id": "btn_red", "on": True}})
        verdict = evaluate_control_panel(result, _spec(), _POLICY)
        data = json.loads(verdict.to_json())
        assert data["is_successful"] is True
        assert data["policy_version"] == 1
        assert isinstance(data["violations"], list)
