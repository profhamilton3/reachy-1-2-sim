"""R12-804: Lexicographic ranking unit tests.

Verifies that the tier ordering is correct:
  1. valid > invalid
  2. safe > unsafe
  3. successful > failed
  4. accurate > inaccurate
  5. high clearance > low clearance
  6. low saturation > high saturation
  7. shorter > longer

Critical invariant: a faster collision path can NEVER outrank a safe path
regardless of speed tier advantage.
"""

from __future__ import annotations

import json

import pytest

from reachy_ai.evaluation.base import (
    EvaluationPolicy, EpisodeVerdict, Violation, ViolationKind,
)
from reachy_ai.evaluation.ranking import (
    RankedCandidate, rank_candidates, best_candidate, dominated_by,
    explain_ranking,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic verdicts
# ---------------------------------------------------------------------------

def _verdict(
    episode_id: str = "ep0",
    trial_id: str = "tr0",
    task_type: str = "operate_control",
    is_valid: bool = True,
    is_safe: bool = True,
    is_successful: bool = True,
    accuracy: float = 1.0,
    clearance: float = 1.0,
    effort: float = 1.0,
    smoothness: float = 1.0,
    duration: float = 0.5,
    violations: list = None,
) -> EpisodeVerdict:
    return EpisodeVerdict(
        episode_id=episode_id,
        trial_id=trial_id,
        task_type=task_type,
        policy_version=1,
        is_valid=is_valid,
        is_safe=is_safe,
        is_successful=is_successful,
        violations=violations or [],
        metrics={},
        ranking_scores={
            "accuracy_score": accuracy,
            "clearance_score": clearance,
            "effort_score": effort,
            "smoothness_score": smoothness,
            "duration_score": duration,
        },
        explanation="test verdict",
    )


def _candidate(verdict: EpisodeVerdict, trial_id: str = "tr0") -> RankedCandidate:
    return RankedCandidate(verdict=verdict, trial_id=trial_id)


# ---------------------------------------------------------------------------
# Tier 1: valid > invalid
# ---------------------------------------------------------------------------

class TestTier1Validity:
    def test_valid_beats_invalid(self):
        valid = _candidate(_verdict(is_valid=True), "valid")
        invalid = _candidate(_verdict(is_valid=False, is_safe=False,
                                      is_successful=False), "invalid")
        ranked = rank_candidates([invalid, valid])
        assert ranked[0].trial_id == "valid"

    def test_valid_beats_invalid_even_if_invalid_is_faster(self):
        valid = _candidate(_verdict(is_valid=True, duration=0.1), "valid")
        invalid = _candidate(
            _verdict(is_valid=False, is_safe=False, is_successful=False,
                     duration=1.0),  # even with worst duration
            "invalid"
        )
        ranked = rank_candidates([invalid, valid])
        assert ranked[0].trial_id == "valid"


# ---------------------------------------------------------------------------
# Tier 2: safe > unsafe  (THE CRITICAL INVARIANT)
# ---------------------------------------------------------------------------

class TestTier2Safety:
    def test_safe_beats_unsafe(self):
        safe = _candidate(_verdict(is_safe=True, is_successful=True, duration=0.1), "safe")
        unsafe = _candidate(_verdict(is_safe=False, is_successful=False,
                                     duration=0.9), "unsafe")
        ranked = rank_candidates([unsafe, safe])
        assert ranked[0].trial_id == "safe"

    def test_fast_collision_cannot_outrank_safe_path(self):
        """Core invariant from EPIC-8.md Section 5.3."""
        # Unsafe is faster (higher duration_score) but has forbidden contact.
        collision_fast = _candidate(
            _verdict(is_safe=False, is_successful=False, duration=0.95),
            "collision_fast"
        )
        safe_slow = _candidate(
            _verdict(is_safe=True, is_successful=True, duration=0.1),
            "safe_slow"
        )
        ranked = rank_candidates([collision_fast, safe_slow])
        assert ranked[0].trial_id == "safe_slow", (
            "A faster collision trajectory must NEVER outrank a slower safe path"
        )

    def test_unsafe_successful_loses_to_safe_failed(self):
        """A task 'success' with forbidden contact loses to a safe task failure."""
        unsafe_success = _candidate(
            _verdict(is_safe=False, is_successful=True),
            "unsafe_success"
        )
        safe_fail = _candidate(
            _verdict(is_safe=True, is_successful=False),
            "safe_fail"
        )
        ranked = rank_candidates([unsafe_success, safe_fail])
        assert ranked[0].trial_id == "safe_fail"

    def test_multiple_unsafe_all_ranked_below_any_safe(self):
        unsafe_a = _candidate(_verdict(is_safe=False, duration=0.99), "ua")
        unsafe_b = _candidate(_verdict(is_safe=False, accuracy=0.99, duration=0.99), "ub")
        safe = _candidate(_verdict(is_safe=True, is_successful=False, duration=0.01), "s")
        ranked = rank_candidates([unsafe_a, unsafe_b, safe])
        assert ranked[0].trial_id == "s"


# ---------------------------------------------------------------------------
# Tier 3: successful > failed
# ---------------------------------------------------------------------------

class TestTier3Success:
    def test_success_beats_failure(self):
        success = _candidate(_verdict(is_successful=True), "success")
        failure = _candidate(_verdict(is_successful=False), "failure")
        ranked = rank_candidates([failure, success])
        assert ranked[0].trial_id == "success"

    def test_fast_failure_cannot_outrank_slow_success(self):
        fast_fail = _candidate(
            _verdict(is_successful=False, duration=0.99), "fast_fail"
        )
        slow_success = _candidate(
            _verdict(is_successful=True, duration=0.01), "slow_success"
        )
        ranked = rank_candidates([fast_fail, slow_success])
        assert ranked[0].trial_id == "slow_success"


# ---------------------------------------------------------------------------
# Tier 4–8: within safe+successful, fine-grained ordering
# ---------------------------------------------------------------------------

class TestFineTiers:
    def test_higher_accuracy_wins(self):
        hi = _candidate(_verdict(accuracy=0.9), "hi_acc")
        lo = _candidate(_verdict(accuracy=0.1), "lo_acc")
        ranked = rank_candidates([lo, hi])
        assert ranked[0].trial_id == "hi_acc"

    def test_higher_clearance_wins_when_accuracy_equal(self):
        hi = _candidate(_verdict(accuracy=1.0, clearance=0.9), "hi_cl")
        lo = _candidate(_verdict(accuracy=1.0, clearance=0.1), "lo_cl")
        ranked = rank_candidates([lo, hi])
        assert ranked[0].trial_id == "hi_cl"

    def test_lower_saturation_wins_when_clearance_equal(self):
        lo_sat = _candidate(_verdict(clearance=1.0, effort=0.9), "lo_sat")
        hi_sat = _candidate(_verdict(clearance=1.0, effort=0.1), "hi_sat")
        ranked = rank_candidates([hi_sat, lo_sat])
        assert ranked[0].trial_id == "lo_sat"

    def test_shorter_duration_wins_at_lowest_tier(self):
        fast = _candidate(_verdict(effort=1.0, smoothness=1.0, duration=0.9), "fast")
        slow = _candidate(_verdict(effort=1.0, smoothness=1.0, duration=0.1), "slow")
        ranked = rank_candidates([slow, fast])
        assert ranked[0].trial_id == "fast"


# ---------------------------------------------------------------------------
# best_candidate helper
# ---------------------------------------------------------------------------

class TestBestCandidate:
    def test_empty_returns_none(self):
        assert best_candidate([]) is None

    def test_single_item_returned(self):
        c = _candidate(_verdict())
        assert best_candidate([c]) is c

    def test_returns_safest_fastest(self):
        safe_fast = _candidate(_verdict(is_safe=True, duration=0.9), "sf")
        safe_slow = _candidate(_verdict(is_safe=True, duration=0.1), "ss")
        unsafe = _candidate(_verdict(is_safe=False), "u")
        best = best_candidate([unsafe, safe_slow, safe_fast])
        assert best.trial_id == "sf"


# ---------------------------------------------------------------------------
# dominated_by
# ---------------------------------------------------------------------------

class TestDominatedBy:
    def test_worse_is_dominated(self):
        strong = _candidate(_verdict(accuracy=0.9, duration=0.9), "s")
        weak = _candidate(_verdict(accuracy=0.1, duration=0.1), "w")
        assert dominated_by(weak, strong)

    def test_equal_not_dominated(self):
        a = _candidate(_verdict(), "a")
        b = _candidate(_verdict(), "b")
        assert not dominated_by(a, b)

    def test_better_not_dominated_by_worse(self):
        strong = _candidate(_verdict(accuracy=0.9, duration=0.9), "s")
        weak = _candidate(_verdict(accuracy=0.1, duration=0.1), "w")
        assert not dominated_by(strong, weak)


# ---------------------------------------------------------------------------
# rank_candidates — stable sort and serialisation
# ---------------------------------------------------------------------------

class TestRankCandidates:
    def test_empty_list_returns_empty(self):
        assert rank_candidates([]) == []

    def test_single_item_preserved(self):
        c = _candidate(_verdict())
        assert rank_candidates([c]) == [c]

    def test_does_not_mutate_input(self):
        cs = [_candidate(_verdict(duration=0.1), "a"),
              _candidate(_verdict(duration=0.9), "b")]
        original_order = [c.trial_id for c in cs]
        rank_candidates(cs)
        assert [c.trial_id for c in cs] == original_order

    def test_explain_ranking_returns_string(self):
        cs = [_candidate(_verdict(is_safe=True), "safe"),
              _candidate(_verdict(is_safe=False), "unsafe")]
        text = explain_ranking(cs)
        assert "safe" in text
        assert isinstance(text, str)

    def test_verdict_serialises_to_json(self):
        v = _verdict()
        data = json.loads(v.to_json())
        assert data["is_valid"] is True
        assert "ranking_scores" in data
        assert "violations" in data

    def test_policy_serialisation_round_trip(self):
        import json
        p = EvaluationPolicy(policy_version=3, forbidden_contact_allowed=0)
        p2 = EvaluationPolicy.from_json(p.to_json())
        assert p2.policy_version == 3
        assert p2.forbidden_contact_allowed == 0

    def test_policy_is_frozen(self):
        p = EvaluationPolicy()
        with pytest.raises((TypeError, AttributeError)):
            p.policy_version = 99  # type: ignore[misc]
