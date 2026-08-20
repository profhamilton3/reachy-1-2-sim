"""R12-808: Tests for AdaptiveSampler and SearchRunner adaptive integration."""

from __future__ import annotations

import uuid
from typing import List

import pytest

from reachy_ai.evaluation.base import EpisodeVerdict
from reachy_ai.motion.recipe import TrajectoryRecipe
from reachy_ai.search.runner import SearchConfig, SearchRunner
from reachy_ai.search.samplers import AdaptiveSampler
from reachy_ai.search.space import SearchSpace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _space(*, discrete=True) -> SearchSpace:
    from reachy_ai.motion.recipe import TrajectoryRecipe
    if discrete:
        recipe = TrajectoryRecipe(
            recipe_id="r",
            task_type="test",
            bounded_parameters={
                "x": {"value": 0.0, "min": 0.0, "max": 4.0, "step": 1.0},
                "y": {"value": 0.0, "min": 0.0, "max": 4.0, "step": 1.0},
            },
        )
    else:
        recipe = TrajectoryRecipe(
            recipe_id="r",
            task_type="test",
            bounded_parameters={
                "a": {"value": 0.5, "min": 0.0, "max": 1.0},   # continuous
                "b": {"value": 2.0, "min": 0.0, "max": 4.0, "step": 1.0},  # discrete
            },
        )
    return SearchSpace.from_recipe(recipe)


def _verdict(accuracy: float = 0.5) -> EpisodeVerdict:
    eid = uuid.uuid4().hex
    return EpisodeVerdict(
        episode_id=eid,
        trial_id=eid,
        task_type="test",
        policy_version=1,
        is_valid=True,
        is_safe=True,
        is_successful=True,
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


def _recipe(bounded=None) -> TrajectoryRecipe:
    return TrajectoryRecipe(
        recipe_id="r",
        task_type="test",
        bounded_parameters=bounded or {
            "x": {"value": 0.0, "min": 0.0, "max": 4.0, "step": 1.0},
        },
    )


# ---------------------------------------------------------------------------
# AdaptiveSampler unit tests
# ---------------------------------------------------------------------------

class TestAdaptiveSamplerInit:
    def test_defaults(self):
        s = AdaptiveSampler([])
        assert s._exploration_weight == pytest.approx(0.3)
        assert s._perturbation_sigma == pytest.approx(0.15)

    def test_invalid_exploration_weight_low(self):
        with pytest.raises(ValueError, match="exploration_weight"):
            AdaptiveSampler([], exploration_weight=-0.1)

    def test_invalid_exploration_weight_high(self):
        with pytest.raises(ValueError, match="exploration_weight"):
            AdaptiveSampler([], exploration_weight=1.1)

    def test_invalid_perturbation_sigma(self):
        with pytest.raises(ValueError, match="perturbation_sigma"):
            AdaptiveSampler([], perturbation_sigma=0.0)

    def test_top_k_slices_top_points(self):
        pts = [{"x": float(i)} for i in range(10)]
        s = AdaptiveSampler(pts, top_k=3)
        assert len(s._top_points) == 3


class TestAdaptiveSamplerSuggest:
    def test_returns_valid_points_in_bounds(self):
        space = _space(discrete=True)
        top = [{"x": 2.0, "y": 2.0}]
        sampler = AdaptiveSampler(top, exploration_weight=0.0, seed=0)
        pts = sampler.suggest(space, [], n=20)
        for pt in pts:
            assert space.validate_point(pt), f"out-of-bounds: {pt}"

    def test_never_repeats_already_done(self):
        space = _space(discrete=True)
        top = [{"x": 0.0, "y": 0.0}]
        sampler = AdaptiveSampler(top, seed=42)
        done = [{"x": 0.0, "y": 0.0}]
        done_keys = {space.point_key(p) for p in done}
        pts = sampler.suggest(space, done, n=10)
        for pt in pts:
            assert space.point_key(pt) not in done_keys

    def test_no_duplicates_in_batch(self):
        space = _space(discrete=True)
        sampler = AdaptiveSampler([], exploration_weight=1.0, seed=7)
        pts = sampler.suggest(space, [], n=5)
        keys = [space.point_key(p) for p in pts]
        assert len(keys) == len(set(keys))

    def test_empty_top_points_falls_back_to_random(self):
        space = _space(discrete=True)
        sampler = AdaptiveSampler([], exploration_weight=0.0, seed=0)
        pts = sampler.suggest(space, [], n=5)
        assert len(pts) > 0
        for pt in pts:
            assert space.validate_point(pt)

    def test_pure_exploitation_clusters_near_anchor(self):
        """With exploration_weight=0 and a single anchor, all points should be
        within ±1 step of the anchor on each discrete axis."""
        space = _space(discrete=True)
        anchor = {"x": 2.0, "y": 2.0}
        sampler = AdaptiveSampler([anchor], exploration_weight=0.0, seed=0)
        pts = sampler.suggest(space, [], n=15)
        for pt in pts:
            assert abs(pt["x"] - 2.0) <= 1.0 + 1e-9, f"x too far: {pt}"
            assert abs(pt["y"] - 2.0) <= 1.0 + 1e-9, f"y too far: {pt}"

    def test_continuous_axis_stays_in_bounds(self):
        space = _space(discrete=False)
        anchor = {"a": 0.5, "b": 2.0}
        sampler = AdaptiveSampler([anchor], exploration_weight=0.0, seed=1)
        pts = sampler.suggest(space, [], n=30)
        for pt in pts:
            assert 0.0 <= pt["a"] <= 1.0, f"a out of bounds: {pt}"
            assert 0.0 <= pt["b"] <= 4.0, f"b out of bounds: {pt}"

    def test_pure_exploration_spreads_across_space(self):
        """With exploration_weight=1.0, x values should not all equal anchor."""
        space = _space(discrete=True)
        anchor = {"x": 0.0, "y": 0.0}
        sampler = AdaptiveSampler([anchor], exploration_weight=1.0, seed=99)
        pts = sampler.suggest(space, [], n=25)
        x_vals = {pt["x"] for pt in pts}
        assert len(x_vals) > 1  # should span more than just 0.0

    def test_returns_fewer_than_n_when_exploitation_exhausted(self):
        """Pure exploitation on a 2-point discrete space produces at most 2 unique
        grid-aligned points; once both are in already_done, suggest returns []."""
        from reachy_ai.motion.recipe import TrajectoryRecipe
        recipe = TrajectoryRecipe(
            recipe_id="r",
            task_type="test",
            bounded_parameters={
                "x": {"value": 0.0, "min": 0.0, "max": 1.0, "step": 1.0},
            },
        )
        space = SearchSpace.from_recipe(recipe)
        # Pure exploitation: _perturb always snaps to grid values {0.0, 1.0}
        anchor = {"x": 0.0}
        sampler = AdaptiveSampler([anchor], exploration_weight=0.0, seed=0)
        # Mark both grid points as done; sampler should give up quickly
        done = [{"x": 0.0}, {"x": 1.0}]
        pts = sampler.suggest(space, done, n=5)
        assert pts == []

    def test_returns_empty_when_all_done(self):
        from reachy_ai.motion.recipe import TrajectoryRecipe
        recipe = TrajectoryRecipe(
            recipe_id="r",
            task_type="test",
            bounded_parameters={
                "x": {"value": 0.0, "min": 0.0, "max": 0.0, "step": 1.0},
            },
        )
        space = SearchSpace.from_recipe(recipe)
        sampler = AdaptiveSampler([], exploration_weight=1.0, seed=0)
        done = [{"x": 0.0}]
        pts = sampler.suggest(space, done, n=3)
        assert pts == []


# ---------------------------------------------------------------------------
# SearchRunner adaptive integration tests
# ---------------------------------------------------------------------------

class TestSearchRunnerAdaptive:
    def _make_prior(self, n: int = 3):
        """Build n synthetic (SearchPoint, EpisodeVerdict) pairs."""
        from reachy_ai.search.runner import PriorResults
        prior: PriorResults = []
        for i in range(n):
            v = _verdict(accuracy=float(i) / n)
            prior.append(({"x": float(i)}, v))
        return prior

    def test_adaptive_sampler_used_when_prior_given(self):
        """With sampler='adaptive' and prior, AdaptiveSampler is instantiated."""
        recipe = _recipe(bounded={
            "x": {"value": 0.0, "min": 0.0, "max": 10.0, "step": 1.0},
        })
        config = SearchConfig(
            study_id="s",
            baseline_recipe=recipe,
            budget=5,
            sampler="adaptive",
        )
        runner = SearchRunner(config)
        prior = self._make_prior(3)

        results = []
        def ev(r, _seed):
            results.append(dict(r.bounded_parameters))
            return _verdict(accuracy=float(r.bounded_parameters["x"]["value"]) / 10)

        result = runner.run(ev, prior_results=prior)
        assert result.trials_run == 5
        # All suggested points must be valid
        space = runner.space
        for r in results:
            val = r["x"]["value"]
            assert 0.0 <= val <= 10.0

    def test_adaptive_falls_back_to_random_without_prior(self):
        """sampler='adaptive' with no prior behaves like RandomSampler."""
        recipe = _recipe(bounded={
            "x": {"value": 0.0, "min": 0.0, "max": 10.0},  # continuous
        })
        config = SearchConfig(
            study_id="s",
            baseline_recipe=recipe,
            budget=5,
            sampler="adaptive",
            random_seed=42,
        )
        runner = SearchRunner(config)
        result = runner.run(lambda r, _seed: _verdict())
        assert result.trials_run == 5

    def test_adaptive_respects_top_k(self):
        """adaptive_top_k=1 means only the best prior point is exploited."""
        recipe = _recipe(bounded={
            "x": {"value": 0.0, "min": 0.0, "max": 4.0, "step": 1.0},
        })
        config = SearchConfig(
            study_id="s",
            baseline_recipe=recipe,
            budget=10,
            sampler="adaptive",
            adaptive_top_k=1,
            adaptive_exploration_weight=0.0,  # pure exploitation
            random_seed=0,
        )
        runner = SearchRunner(config)
        prior = [({"x": 4.0}, _verdict(accuracy=0.99))]  # best = x=4

        suggested_x = []
        def ev(r, _seed):
            suggested_x.append(float(r.bounded_parameters["x"]["value"]))
            return _verdict()

        runner.run(ev, prior_results=prior)
        # All suggestions should be near x=4 (within ±1 step = 3, 4)
        for x in suggested_x:
            assert x >= 3.0, f"Expected near 4.0, got {x}"

    def test_adaptive_exploration_weight_one_avoids_anchor(self):
        """exploration_weight=1.0 means pure random — x should vary widely."""
        recipe = _recipe(bounded={
            "x": {"value": 0.0, "min": 0.0, "max": 9.0, "step": 1.0},
        })
        config = SearchConfig(
            study_id="s",
            baseline_recipe=recipe,
            budget=10,
            sampler="adaptive",
            adaptive_exploration_weight=1.0,
            random_seed=77,
        )
        runner = SearchRunner(config)
        prior = [({"x": 0.0}, _verdict(accuracy=0.99))]

        x_vals = []
        def ev(r, _seed):
            x_vals.append(float(r.bounded_parameters["x"]["value"]))
            return _verdict()

        runner.run(ev, prior_results=prior)
        assert len(set(x_vals)) > 1  # not all identical

    def test_adaptive_config_fields_default_sensibly(self):
        config = SearchConfig(study_id="s", baseline_recipe=_recipe())
        assert config.adaptive_top_k == 3
        assert config.adaptive_exploration_weight == pytest.approx(0.3)

    def test_adaptive_with_store_persists_and_resumes(self, tmp_path):
        """AdaptiveSampler integrates with store persistence end-to-end."""
        from reachy_ai.experience.store import ExperienceStore

        recipe = _recipe(bounded={
            "x": {"value": 0.0, "min": 0.0, "max": 9.0, "step": 1.0},
        })
        db = str(tmp_path / "adaptive.db")

        # First run: plain grid to build a prior
        with ExperienceStore.open(db) as store:
            config = SearchConfig(study_id="study", baseline_recipe=recipe, budget=5)
            SearchRunner(config).run(lambda r, _seed: _verdict(accuracy=0.5), store=store)

        # Second run: adaptive resume
        with ExperienceStore.open(db) as store:
            prior = SearchRunner.load_prior_from_store(store, "study")
            config2 = SearchConfig(
                study_id="study",
                baseline_recipe=recipe,
                budget=3,
                sampler="adaptive",
            )
            result = SearchRunner(config2).run(
                lambda r, _seed: _verdict(accuracy=0.7),
                prior_results=prior,
                store=store,
            )
        assert result.trials_skipped == len(prior)
        assert result.trials_run == 3


# ---------------------------------------------------------------------------
# CLI --sampler adaptive smoke test
# ---------------------------------------------------------------------------

class TestCliAdaptive:
    def test_cli_run_adaptive_accepted(self, tmp_path, capsys):
        """--sampler adaptive is a valid CLI choice (dry-run, no prior)."""
        from reachy_ai.search.cli import main
        from reachy_ai.motion.recipe import TrajectoryRecipe
        import json, tempfile, os

        recipe = TrajectoryRecipe(
            recipe_id="r",
            task_type="test",
            bounded_parameters={
                "x": {"value": 0.5, "min": 0.0, "max": 1.0, "step": 0.5},
            },
        )
        recipe_file = str(tmp_path / "recipe.json")
        with open(recipe_file, "w") as f:
            f.write(recipe.to_json())

        rc = main([
            "run",
            "--baseline", recipe_file,
            "--study-id", "cli_adaptive",
            "--budget", "3",
            "--sampler", "adaptive",
        ])
        assert rc == 0
