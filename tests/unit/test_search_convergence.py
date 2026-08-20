"""R12-806: Tests for ConvergenceChecker and SearchRunner early-stopping integration."""

from __future__ import annotations

import uuid
from typing import List

import pytest

from reachy_ai.evaluation.base import EpisodeVerdict
from reachy_ai.search.convergence import ConvergenceChecker, ConvergenceResult
from reachy_ai.search.runner import SearchConfig, SearchResult, SearchRunner

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _rk(*floats: float) -> tuple:
    """Construct a rank_key tuple from positional floats."""
    return tuple(floats)


def _verdict(
    *,
    is_valid: bool = True,
    is_safe: bool = True,
    is_successful: bool = True,
    accuracy: float = 0.5,
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
        metrics={},
        ranking_scores={
            "accuracy_score": accuracy,
            "clearance_score": 0.5,
            "effort_score": 1.0,
            "smoothness_score": 1.0,
            "duration_score": 0.5,
        },
        explanation="test",
    )


def _recipe(*, bounded=None):
    from reachy_ai.motion.recipe import TrajectoryRecipe
    return TrajectoryRecipe(
        recipe_id="test_recipe",
        task_type="test",
        bounded_parameters=bounded or {
            "x": {"value": 0.0, "min": 0.0, "max": 2.0, "step": 1.0},
        },
    )


# ---------------------------------------------------------------------------
# ConvergenceChecker unit tests
# ---------------------------------------------------------------------------

class TestConvergenceCheckerInit:
    def test_defaults(self):
        cc = ConvergenceChecker()
        assert cc.window == 10
        assert cc.min_trials == 5

    def test_custom_params(self):
        cc = ConvergenceChecker(window=3, min_trials=2)
        assert cc.window == 3
        assert cc.min_trials == 2

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="window"):
            ConvergenceChecker(window=0)

    def test_invalid_min_trials(self):
        with pytest.raises(ValueError, match="min_trials"):
            ConvergenceChecker(min_trials=0)


class TestConvergenceCheckerBehavior:
    def test_not_converged_initially(self):
        cc = ConvergenceChecker(window=3, min_trials=2)
        assert not cc.is_converged()
        assert cc.trials_since_improvement() == 0

    def test_not_converged_below_min_trials(self):
        """Even if the window is satisfied, min_trials must be met first."""
        cc = ConvergenceChecker(window=2, min_trials=5)
        for _ in range(4):
            cc.update(_rk(1.0, 1.0, 1.0))
        assert not cc.is_converged()

    def test_not_converged_below_window(self):
        """min_trials met but fewer than window non-improving updates: not converged."""
        cc = ConvergenceChecker(window=5, min_trials=2)
        cc.update(_rk(1.0,))
        cc.update(_rk(1.0,))  # min_trials met
        cc.update(_rk(1.0,))  # 2 non-improving after the initial best
        cc.update(_rk(1.0,))  # 3 non-improving
        # window=5 requires 5 non-improving; only 3 so far
        assert not cc.is_converged()

    def test_converges_after_window_without_improvement(self):
        cc = ConvergenceChecker(window=3, min_trials=2)
        cc.update(_rk(1.0,))  # improvement — sets best
        cc.update(_rk(1.0,))  # no improvement #1 (total=2 → min_trials met)
        cc.update(_rk(1.0,))  # no improvement #2
        cc.update(_rk(1.0,))  # no improvement #3 → window reached
        assert cc.is_converged()
        assert cc.trials_since_improvement() == 3

    def test_improvement_resets_counter(self):
        cc = ConvergenceChecker(window=3, min_trials=1)
        cc.update(_rk(0.5,))  # best = 0.5
        cc.update(_rk(0.5,))  # +1 without improvement
        cc.update(_rk(0.5,))  # +2
        cc.update(_rk(0.8,))  # improvement! resets counter
        assert not cc.is_converged()
        assert cc.trials_since_improvement() == 0

    def test_then_converges_after_reset(self):
        cc = ConvergenceChecker(window=2, min_trials=1)
        cc.update(_rk(0.5,))
        cc.update(_rk(0.8,))  # improvement
        assert cc.trials_since_improvement() == 0
        cc.update(_rk(0.8,))  # no improvement #1
        cc.update(_rk(0.8,))  # no improvement #2 → converged
        assert cc.is_converged()

    def test_strict_improvement_required(self):
        """Equal rank_key does NOT count as an improvement."""
        cc = ConvergenceChecker(window=2, min_trials=1)
        cc.update(_rk(1.0, 0.5))
        cc.update(_rk(1.0, 0.5))  # equal, not improvement
        assert cc.trials_since_improvement() == 1

    def test_all_failures_no_candidates_no_update(self):
        """_maybe_update_convergence is a no-op when candidates list is empty."""
        from reachy_ai.search.runner import _maybe_update_convergence
        cc = ConvergenceChecker(window=2, min_trials=1)
        _maybe_update_convergence(cc, [])
        assert cc.trials_since_improvement() == 0
        assert not cc.is_converged()

    def test_none_checker_is_noop(self):
        """_maybe_update_convergence with None checker is always safe."""
        from reachy_ai.search.runner import _maybe_update_convergence
        _maybe_update_convergence(None, [])  # must not raise


class TestConvergenceResult:
    def test_result_before_convergence(self):
        cc = ConvergenceChecker(window=5, min_trials=3)
        cc.update(_rk(0.5,))
        r = cc.result()
        assert isinstance(r, ConvergenceResult)
        assert not r.is_converged
        assert "Still searching" in r.message

    def test_result_after_convergence(self):
        cc = ConvergenceChecker(window=2, min_trials=1)
        cc.update(_rk(0.9,))
        cc.update(_rk(0.9,))
        cc.update(_rk(0.9,))
        r = cc.result()
        assert r.is_converged
        assert r.trials_since_improvement == 2
        assert r.best_rank_key == (0.9,)
        assert "Converged" in r.message

    def test_result_before_any_update(self):
        cc = ConvergenceChecker()
        r = cc.result()
        assert not r.is_converged
        assert r.best_rank_key == ()


# ---------------------------------------------------------------------------
# SearchRunner early-stopping integration tests
# ---------------------------------------------------------------------------

class TestSearchRunnerConvergence:
    """Integration tests using a tiny 3-point grid so the budget is never the bottleneck."""

    def _make_config(self, budget: int = 20) -> SearchConfig:
        recipe = _recipe(bounded={
            "x": {"value": 0.0, "min": 0.0, "max": 2.0, "step": 1.0},
        })
        return SearchConfig(study_id="conv_study", baseline_recipe=recipe, budget=budget)

    def test_no_checker_runs_full_budget(self):
        """Without a checker, the grid stops only when exhausted (3 points)."""
        config = self._make_config(budget=100)
        runner = SearchRunner(config)

        def ev(r):
            return _verdict(accuracy=0.5)

        result = runner.run(ev)
        assert result.trials_run == 3  # grid exhausted, not budget
        assert not result.converged
        assert result.trials_since_improvement == 0

    def test_early_stopping_fires_before_budget(self):
        """With checker window=1, min_trials=1: stops after 2nd non-improving trial."""
        config = self._make_config(budget=100)
        runner = SearchRunner(config)
        checker = ConvergenceChecker(window=1, min_trials=1)

        call_count = 0

        def ev(r):
            nonlocal call_count
            call_count += 1
            return _verdict(accuracy=0.5)  # constant quality → converges quickly

        result = runner.run(ev, convergence_checker=checker)
        # First trial: sets best; subsequent trials don't improve → converges at trial 2
        assert result.trials_run < 100
        assert result.converged

    def test_convergence_result_fields_populated(self):
        config = self._make_config(budget=50)
        runner = SearchRunner(config)
        checker = ConvergenceChecker(window=2, min_trials=1)

        result = runner.run(lambda r: _verdict(accuracy=0.5), convergence_checker=checker)

        assert result.converged is True
        assert result.trials_since_improvement >= 2

    def test_converged_false_when_still_improving(self):
        """If each trial is strictly better, the checker never fires."""
        recipe = _recipe(bounded={
            "x": {"value": 0.0, "min": 0.0, "max": 9.0, "step": 1.0},
        })
        config = SearchConfig(study_id="s", baseline_recipe=recipe, budget=10)
        runner = SearchRunner(config)
        checker = ConvergenceChecker(window=5, min_trials=3)

        call_count = 0

        def ev(r):
            nonlocal call_count
            call_count += 1
            # Each trial gets a strictly increasing accuracy score
            acc = call_count * 0.1
            return _verdict(accuracy=min(acc, 1.0))

        result = runner.run(ev, convergence_checker=checker)
        # 10 grid points, each better than the last — should NOT converge early
        assert not result.converged
        assert result.trials_run == 10

    def test_convergence_with_resume(self):
        """resume() forwards convergence_checker through to run()."""
        recipe = _recipe(bounded={
            "x": {"value": 0.0, "min": 0.0, "max": 4.0, "step": 1.0},
        })
        config = SearchConfig(study_id="s", baseline_recipe=recipe, budget=1)
        runner = SearchRunner(config)

        # Run 1 trial to get a prior
        prior_result = runner.run(lambda r: _verdict(accuracy=0.5))
        prior = [({"x": 0.0}, prior_result.ranked[0].verdict)]

        # Resume with checker that should fire quickly
        checker = ConvergenceChecker(window=1, min_trials=1)
        resumed = runner.resume(
            lambda r: _verdict(accuracy=0.5),
            prior,
            extra_budget=10,
            convergence_checker=checker,
        )
        assert resumed.converged

    def test_failed_trials_count_toward_convergence(self):
        """An exception in evaluate_fn marks the point done; checker sees no improvement."""
        recipe = _recipe(bounded={
            "x": {"value": 0.0, "min": 0.0, "max": 5.0, "step": 1.0},
        })
        config = SearchConfig(study_id="s", baseline_recipe=recipe, budget=20)
        runner = SearchRunner(config)
        checker = ConvergenceChecker(window=2, min_trials=1)

        call_count = 0

        def ev(r):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _verdict(accuracy=0.8)  # first trial succeeds — sets best
            raise RuntimeError("simulated failure")  # all subsequent fail

        result = runner.run(ev, convergence_checker=checker)
        # After the first success, all failures don't improve → converges quickly
        assert result.converged

    def test_search_result_converged_defaults_false(self):
        """SearchResult has converged=False and trials_since_improvement=0 by default."""
        config = self._make_config(budget=1)
        runner = SearchRunner(config)
        result = runner.run(lambda r: _verdict())
        assert result.converged is False
        assert result.trials_since_improvement == 0
