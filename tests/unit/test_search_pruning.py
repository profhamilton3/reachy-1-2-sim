"""R12-805: Pruning strategy unit tests.

Verifies DominancePruner, BudgetPruner, and CompositePruner against the
lexicographic ranking invariants.

All tests are offline (no native_mujoco, no physics) — they work entirely
with synthetic EpisodeVerdict objects.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reachy_ai.evaluation.base import EpisodeVerdict
from reachy_ai.evaluation.ranking import RankedCandidate, rank_candidates
from reachy_ai.search.pruning import (
    BudgetPruner, CompositePruner, DominancePruner, NoPruner, PruningResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verdict(
    trial_id: str = "t0",
    is_valid: bool = True,
    is_safe: bool = True,
    is_successful: bool = True,
    accuracy: float = 1.0,
    clearance: float = 1.0,
    effort: float = 1.0,
    smoothness: float = 1.0,
    duration: float = 0.5,
) -> EpisodeVerdict:
    return EpisodeVerdict(
        episode_id=trial_id,
        trial_id=trial_id,
        task_type="test",
        policy_version=1,
        is_valid=is_valid,
        is_safe=is_safe,
        is_successful=is_successful,
        violations=[],
        metrics={},
        ranking_scores={
            "accuracy_score": accuracy,
            "clearance_score": clearance,
            "effort_score": effort,
            "smoothness_score": smoothness,
            "duration_score": duration,
        },
        explanation="test",
    )


def _cand(trial_id: str, **kw) -> RankedCandidate:
    return RankedCandidate(verdict=_verdict(trial_id=trial_id, **kw), trial_id=trial_id)


# ---------------------------------------------------------------------------
# NoPruner
# ---------------------------------------------------------------------------

class TestNoPruner:
    def test_keeps_all(self):
        cs = [_cand("a"), _cand("b"), _cand("c")]
        result = NoPruner().prune(cs)
        assert len(result.kept) == 3
        assert len(result.pruned) == 0

    def test_empty_input(self):
        result = NoPruner().prune([])
        assert result.kept == []
        assert result.pruned == []


# ---------------------------------------------------------------------------
# DominancePruner
# ---------------------------------------------------------------------------

class TestDominancePruner:
    def test_removes_dominated_candidate(self):
        # 'better' dominates 'worse' in every tier.
        better = _cand("better", accuracy=0.9, duration=0.9)
        worse = _cand("worse", accuracy=0.1, duration=0.1)
        result = DominancePruner().prune([better, worse])
        ids = {c.trial_id for c in result.kept}
        assert "better" in ids
        assert "worse" not in ids

    def test_keeps_pareto_front(self):
        # a is better on accuracy, b is better on duration — neither dominates.
        a = _cand("a", accuracy=0.9, duration=0.1)
        b = _cand("b", accuracy=0.1, duration=0.9)
        result = DominancePruner().prune([a, b])
        assert len(result.kept) == 2
        assert len(result.pruned) == 0

    def test_best_is_never_pruned(self):
        best = _cand("best", accuracy=1.0, duration=1.0)
        others = [_cand(f"c{i}", accuracy=0.1, duration=0.1) for i in range(5)]
        result = DominancePruner().prune([best] + others)
        kept_ids = [c.trial_id for c in result.kept]
        assert "best" in kept_ids

    def test_unsafe_loses_to_safe_and_is_pruned(self):
        # safe must be >= unsafe in every tier and > in at least one for Pareto
        # dominance to hold.  duration=1.0 on safe ensures the last tier
        # (duration_score) doesn't let unsafe escape pruning.
        safe = _cand("safe", is_safe=True, is_successful=True, duration=1.0)
        unsafe = _cand("unsafe", is_safe=False, is_successful=False, duration=0.99)
        result = DominancePruner().prune([safe, unsafe])
        # safe dominates unsafe (all tiers >= and tier 2 strictly >)
        assert "safe" in {c.trial_id for c in result.kept}
        assert "unsafe" not in {c.trial_id for c in result.kept}

    def test_empty_input(self):
        result = DominancePruner().prune([])
        assert result.kept == []
        assert result.pruned == []

    def test_single_candidate_kept(self):
        c = _cand("only")
        result = DominancePruner().prune([c])
        assert len(result.kept) == 1
        assert len(result.pruned) == 0

    def test_all_equal_all_kept(self):
        # Identical rank_key → no one dominates anyone.
        cs = [_cand(f"c{i}") for i in range(4)]
        result = DominancePruner().prune(cs)
        assert len(result.kept) == 4
        assert len(result.pruned) == 0

    def test_dominated_count_matches_pruned(self):
        # 'top' dominates all three 'bot' candidates.
        top = _cand("top", accuracy=1.0, clearance=1.0, duration=1.0)
        bots = [_cand(f"b{i}", accuracy=0.0, clearance=0.0, duration=0.0) for i in range(3)]
        result = DominancePruner().prune([top] + bots)
        assert len(result.pruned) == 3

    def test_pruned_plus_kept_equals_input(self):
        cs = [_cand(f"c{i}", accuracy=float(i) / 10) for i in range(8)]
        result = DominancePruner().prune(cs)
        assert len(result.kept) + len(result.pruned) == 8


# ---------------------------------------------------------------------------
# BudgetPruner
# ---------------------------------------------------------------------------

class TestBudgetPruner:
    def test_caps_at_max_kept(self):
        cs = [_cand(f"c{i}", duration=float(i) / 10) for i in range(10)]
        result = BudgetPruner(max_kept=3).prune(cs)
        assert len(result.kept) == 3
        assert len(result.pruned) == 7

    def test_best_three_are_kept(self):
        # c9 > c8 > c7 > ... by duration_score.
        cs = [_cand(f"c{i}", duration=float(i) / 10) for i in range(10)]
        result = BudgetPruner(max_kept=3).prune(cs)
        kept_ids = {c.trial_id for c in result.kept}
        assert "c9" in kept_ids
        assert "c8" in kept_ids
        assert "c7" in kept_ids

    def test_fewer_candidates_than_budget_keeps_all(self):
        cs = [_cand(f"c{i}") for i in range(2)]
        result = BudgetPruner(max_kept=5).prune(cs)
        assert len(result.kept) == 2
        assert len(result.pruned) == 0

    def test_max_kept_one_keeps_only_best(self):
        best = _cand("best", duration=0.99)
        worst = _cand("worst", duration=0.01)
        result = BudgetPruner(max_kept=1).prune([worst, best])
        assert result.kept[0].trial_id == "best"

    def test_invalid_max_kept_raises(self):
        with pytest.raises(ValueError):
            BudgetPruner(max_kept=0)

    def test_empty_input(self):
        result = BudgetPruner(max_kept=3).prune([])
        assert result.kept == []
        assert result.pruned == []


# ---------------------------------------------------------------------------
# CompositePruner
# ---------------------------------------------------------------------------

class TestCompositePruner:
    def test_applies_both_strategies(self):
        # Create candidates where dominance prunes some and budget caps the rest.
        top = _cand("top", accuracy=1.0, duration=1.0)
        mid_a = _cand("mid_a", accuracy=0.5, duration=0.9)
        mid_b = _cand("mid_b", accuracy=0.9, duration=0.5)  # pareto with mid_a
        bots = [_cand(f"bot{i}", accuracy=0.0, duration=0.0) for i in range(5)]
        all_cands = [top, mid_a, mid_b] + bots

        result = CompositePruner([DominancePruner(), BudgetPruner(max_kept=2)]).prune(all_cands)
        assert len(result.kept) <= 2

    def test_total_equals_input_size(self):
        cs = [_cand(f"c{i}", accuracy=float(i) / 10) for i in range(6)]
        result = CompositePruner([DominancePruner(), BudgetPruner(max_kept=2)]).prune(cs)
        assert len(result.kept) + len(result.pruned) == 6

    def test_no_pruner_in_composite_is_passthrough(self):
        cs = [_cand(f"c{i}") for i in range(4)]
        result = CompositePruner([NoPruner()]).prune(cs)
        assert len(result.kept) == 4

    def test_empty_strategies_keeps_all(self):
        cs = [_cand(f"c{i}") for i in range(3)]
        result = CompositePruner([]).prune(cs)
        assert len(result.kept) == 3

    def test_best_survives_composite(self):
        # The globally best candidate must survive any composite strategy.
        best = _cand("best", accuracy=1.0, clearance=1.0, duration=1.0)
        others = [_cand(f"c{i}", accuracy=float(i) / 10) for i in range(5)]
        result = CompositePruner([DominancePruner(), BudgetPruner(max_kept=1)]).prune([best] + others)
        assert len(result.kept) == 1
        assert result.kept[0].trial_id == "best"

    def test_safety_invariant_through_composite(self):
        safe = _cand("safe", is_safe=True, is_successful=True, duration=0.9)
        collision_fast = _cand(
            "collision_fast", is_safe=False, is_successful=False, duration=0.99
        )
        result = CompositePruner([DominancePruner(), BudgetPruner(max_kept=1)]).prune(
            [collision_fast, safe]
        )
        assert result.kept[0].trial_id == "safe"
