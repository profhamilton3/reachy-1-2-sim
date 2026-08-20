"""Tests for multi-seed evaluation (issue #20 — C3: single deterministic seed).

These tests verify that:
  - evaluate_fn is called once per seed per search point
  - Seeds are passed correctly to evaluate_fn
  - Boolean gates (is_valid/is_safe/is_successful) require all seeds to pass (AND)
  - Numeric ranking scores are averaged across seeds
  - "pessimistic" aggregation subtracts one std from each score
  - eval_seed_count and per-score std are recorded in metrics
  - Single-seed config (the default) returns the verdict unchanged
  - SearchConfig.eval_seeds defaults to [0]
"""

from __future__ import annotations

import math
import uuid
from typing import List, Tuple

import pytest

from reachy_ai.evaluation.base import EpisodeVerdict, Violation, ViolationKind
from reachy_ai.motion.recipe import TrajectoryRecipe
from reachy_ai.search.runner import (
    SearchConfig,
    SearchRunner,
    _aggregate_verdicts,
    _evaluate_multi_seed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recipe(*, bounded=None):
    return TrajectoryRecipe(
        recipe_id="r",
        task_type="test",
        bounded_parameters=bounded or {
            "x": {"value": 0.0, "min": 0.0, "max": 2.0, "step": 1.0},
        },
    )


def _verdict(
    *,
    is_valid: bool = True,
    is_safe: bool = True,
    is_successful: bool = True,
    accuracy: float = 0.5,
    metrics: dict = None,
) -> EpisodeVerdict:
    eid = uuid.uuid4().hex
    return EpisodeVerdict(
        episode_id=eid,
        trial_id=eid,
        task_type="test",
        policy_version=1,
        is_valid=is_valid,
        is_safe=is_safe,
        is_successful=is_successful,
        violations=[],
        metrics=metrics or {},
        ranking_scores={
            "accuracy_score": accuracy,
            "clearance_score": 0.8,
            "effort_score": 1.0,
            "smoothness_score": 1.0,
            "duration_score": 0.5,
        },
        explanation="test",
    )


# ---------------------------------------------------------------------------
# SearchConfig defaults
# ---------------------------------------------------------------------------

class TestSearchConfigDefaults:
    def test_eval_seeds_defaults_to_seed_zero(self):
        config = SearchConfig(study_id="s", baseline_recipe=_recipe())
        assert config.eval_seeds == [0]

    def test_eval_aggregation_defaults_to_mean(self):
        config = SearchConfig(study_id="s", baseline_recipe=_recipe())
        assert config.eval_aggregation == "mean"

    def test_custom_seeds_accepted(self):
        config = SearchConfig(study_id="s", baseline_recipe=_recipe(), eval_seeds=[0, 1, 2])
        assert config.eval_seeds == [0, 1, 2]

    def test_pessimistic_aggregation_accepted(self):
        config = SearchConfig(study_id="s", baseline_recipe=_recipe(), eval_aggregation="pessimistic")
        assert config.eval_aggregation == "pessimistic"


# ---------------------------------------------------------------------------
# _aggregate_verdicts — unit tests
# ---------------------------------------------------------------------------

class TestAggregateVerdicts:
    def test_single_verdict_returned_unchanged(self):
        v = _verdict(accuracy=0.7)
        result = _aggregate_verdicts([v], [0], "mean")
        assert result is v

    def test_mean_aggregation_averages_scores(self):
        v1 = _verdict(accuracy=0.4)
        v2 = _verdict(accuracy=0.8)
        result = _aggregate_verdicts([v1, v2], [0, 1], "mean")
        assert result.ranking_scores["accuracy_score"] == pytest.approx(0.6)

    def test_pessimistic_aggregation_subtracts_std(self):
        v1 = _verdict(accuracy=0.4)
        v2 = _verdict(accuracy=0.8)
        result = _aggregate_verdicts([v1, v2], [0, 1], "pessimistic")
        mean = 0.6
        std = math.sqrt(((0.4 - mean) ** 2 + (0.8 - mean) ** 2) / 2)
        assert result.ranking_scores["accuracy_score"] == pytest.approx(mean - std)

    def test_is_valid_requires_all_seeds(self):
        v1 = _verdict(is_valid=True)
        v2 = _verdict(is_valid=False)
        result = _aggregate_verdicts([v1, v2], [0, 1], "mean")
        assert result.is_valid is False

    def test_is_valid_true_when_all_pass(self):
        v1 = _verdict(is_valid=True)
        v2 = _verdict(is_valid=True)
        result = _aggregate_verdicts([v1, v2], [0, 1], "mean")
        assert result.is_valid is True

    def test_is_safe_requires_all_seeds(self):
        v1 = _verdict(is_safe=True)
        v2 = _verdict(is_safe=False)
        result = _aggregate_verdicts([v1, v2], [0, 1], "mean")
        assert result.is_safe is False

    def test_is_successful_requires_all_seeds(self):
        v1 = _verdict(is_successful=True)
        v2 = _verdict(is_successful=False)
        result = _aggregate_verdicts([v1, v2], [0, 1], "mean")
        assert result.is_successful is False

    def test_is_successful_true_when_all_pass(self):
        v1 = _verdict(is_successful=True)
        v2 = _verdict(is_successful=True)
        result = _aggregate_verdicts([v1, v2], [0, 1], "mean")
        assert result.is_successful is True

    def test_eval_seed_count_in_metrics(self):
        v1 = _verdict()
        v2 = _verdict()
        v3 = _verdict()
        result = _aggregate_verdicts([v1, v2, v3], [0, 1, 2], "mean")
        assert result.metrics["eval_seed_count"] == pytest.approx(3.0)

    def test_score_std_recorded_in_metrics(self):
        v1 = _verdict(accuracy=0.2)
        v2 = _verdict(accuracy=0.8)
        result = _aggregate_verdicts([v1, v2], [0, 1], "mean")
        assert "accuracy_score_std" in result.metrics
        expected_std = math.sqrt(((0.2 - 0.5) ** 2 + (0.8 - 0.5) ** 2) / 2)
        assert result.metrics["accuracy_score_std"] == pytest.approx(expected_std)

    def test_zero_std_when_all_scores_identical(self):
        v1 = _verdict(accuracy=0.5)
        v2 = _verdict(accuracy=0.5)
        result = _aggregate_verdicts([v1, v2], [0, 1], "mean")
        assert result.metrics["accuracy_score_std"] == pytest.approx(0.0)

    def test_explanation_mentions_seed_count(self):
        v1 = _verdict()
        v2 = _verdict()
        result = _aggregate_verdicts([v1, v2], [10, 20], "mean")
        assert "2" in result.explanation
        assert "seed=10" in result.explanation
        assert "seed=20" in result.explanation

    def test_metrics_averaged_across_seeds(self):
        v1 = _verdict(metrics={"wall_duration_s": 1.0})
        v2 = _verdict(metrics={"wall_duration_s": 3.0})
        result = _aggregate_verdicts([v1, v2], [0, 1], "mean")
        assert result.metrics["wall_duration_s"] == pytest.approx(2.0)

    def test_violations_deduplicated(self):
        viol = Violation(kind=ViolationKind.FORBIDDEN_CONTACT, description="hit table")
        v1 = _verdict()
        v1.violations.append(viol)
        v2 = _verdict()
        v2.violations.append(viol)
        result = _aggregate_verdicts([v1, v2], [0, 1], "mean")
        assert len(result.violations) == 1


# ---------------------------------------------------------------------------
# _evaluate_multi_seed — unit tests
# ---------------------------------------------------------------------------

class TestEvaluateMultiSeed:
    def test_seeds_passed_to_evaluate_fn(self):
        seen_seeds: List[int] = []

        def ev(r, seed):
            seen_seeds.append(seed)
            return _verdict()

        _evaluate_multi_seed(ev, _recipe(), [7, 13, 42], "mean")
        assert seen_seeds == [7, 13, 42]

    def test_single_seed_returns_verdict_from_fn(self):
        expected = _verdict(accuracy=0.99)

        result = _evaluate_multi_seed(lambda r, seed: expected, _recipe(), [0], "mean")
        assert result is expected

    def test_multi_seed_aggregates(self):
        call_n = [0]

        def ev(r, seed):
            call_n[0] += 1
            return _verdict(accuracy=float(seed) * 0.1)

        result = _evaluate_multi_seed(ev, _recipe(), [2, 8], "mean")
        # seed=2 → acc=0.2, seed=8 → acc=0.8 → mean=0.5
        assert result.ranking_scores["accuracy_score"] == pytest.approx(0.5)
        assert call_n[0] == 2


# ---------------------------------------------------------------------------
# SearchRunner multi-seed integration tests
# ---------------------------------------------------------------------------

class TestSearchRunnerMultiSeed:
    def test_evaluate_fn_called_once_per_seed_per_trial(self):
        call_log: List[Tuple[str, int]] = []

        def ev(r, seed):
            call_log.append((r.recipe_id, seed))
            return _verdict()

        recipe = _recipe(bounded={"x": {"value": 0.0, "min": 0.0, "max": 1.0, "step": 1.0}})
        config = SearchConfig(
            study_id="s",
            baseline_recipe=recipe,
            budget=2,
            eval_seeds=[0, 1, 2],
        )
        SearchRunner(config).run(ev)
        # 2 search points × 3 seeds = 6 calls
        assert len(call_log) == 6
        seeds_used = {seed for _, seed in call_log}
        assert seeds_used == {0, 1, 2}

    def test_single_seed_default_unchanged_behaviour(self):
        seeds_seen: List[int] = []

        def ev(r, seed):
            seeds_seen.append(seed)
            return _verdict(accuracy=0.7)

        config = SearchConfig(study_id="s", baseline_recipe=_recipe(), budget=3)
        result = SearchRunner(config).run(ev)
        assert result.trials_run == 3
        assert all(s == 0 for s in seeds_seen)

    def test_multi_seed_gate_fails_when_any_seed_unsafe(self):
        call_n = [0]

        def ev(r, seed):
            call_n[0] += 1
            # seed=1 is always unsafe
            return _verdict(is_safe=(seed != 1), is_successful=(seed != 1))

        config = SearchConfig(
            study_id="s",
            baseline_recipe=_recipe(),
            budget=3,
            eval_seeds=[0, 1],
        )
        result = SearchRunner(config).run(ev)
        # All candidates unsafe across seeds → best should be None or unsafe
        for c in result.ranked:
            assert c.verdict.is_safe is False

    def test_eval_seed_count_in_result_metrics(self):
        config = SearchConfig(
            study_id="s",
            baseline_recipe=_recipe(),
            budget=1,
            eval_seeds=[0, 5, 10],
        )
        result = SearchRunner(config).run(lambda r, _seed: _verdict())
        assert result.best is not None
        assert result.best.verdict.metrics["eval_seed_count"] == pytest.approx(3.0)

    def test_pessimistic_mode_penalises_high_variance(self):
        """A recipe with score variance 0 beats a recipe with identical mean but high variance
        under pessimistic aggregation (mean - std)."""
        # Recipe A (x=0): seed 0 → acc=1.0, seed 1 → acc=0.0  (mean=0.5, std=0.5) → pessimistic=0.0
        # Recipe B (x=1): seed 0 → acc=0.5, seed 1 → acc=0.5  (mean=0.5, std=0.0) → pessimistic=0.5

        def ev(r, seed):
            x = float(r.bounded_parameters["x"]["value"])
            if x == 0.0:
                acc = 1.0 if seed == 0 else 0.0
            else:
                acc = 0.5
            return _verdict(accuracy=acc)

        recipe = _recipe(bounded={"x": {"value": 0.0, "min": 0.0, "max": 1.0, "step": 1.0}})
        config = SearchConfig(
            study_id="s",
            baseline_recipe=recipe,
            budget=2,
            eval_seeds=[0, 1],
            eval_aggregation="pessimistic",
        )
        result = SearchRunner(config).run(ev)
        assert result.best is not None
        # Stable recipe (x=1, acc always 0.5) should beat the volatile one
        best_x = result.best_search_point["x"] if result.best_search_point else None
        assert best_x == pytest.approx(1.0)

    def test_mean_mode_does_not_penalise_variance(self):
        """With eval_aggregation='mean', a zero-variance 0.5 and a high-variance 0.5 tie."""

        def ev(r, seed):
            x = float(r.bounded_parameters["x"]["value"])
            if x == 0.0:
                acc = 1.0 if seed == 0 else 0.0
            else:
                acc = 0.5
            return _verdict(accuracy=acc)

        recipe = _recipe(bounded={"x": {"value": 0.0, "min": 0.0, "max": 1.0, "step": 1.0}})
        config = SearchConfig(
            study_id="s",
            baseline_recipe=recipe,
            budget=2,
            eval_seeds=[0, 1],
            eval_aggregation="mean",
        )
        result = SearchRunner(config).run(ev)
        # Both recipes average to acc=0.5; ranking should be a tie at accuracy tier
        assert len(result.ranked) == 2
        acc_scores = [c.verdict.ranking_scores["accuracy_score"] for c in result.ranked]
        assert all(abs(a - 0.5) < 1e-9 for a in acc_scores)
