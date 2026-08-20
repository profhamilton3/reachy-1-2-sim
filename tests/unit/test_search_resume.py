"""R12-805: SearchRunner unit tests — space, samplers, runner, and resume.

All tests are offline:
- No native_mujoco import required (the search engine has no native dependency).
- evaluate_fn is a deterministic synthetic function.
- Store integration tests use an in-memory SQLite path (:memory: routed via
  a temp file so ExperienceStore.open() can use the context manager protocol).

Critical invariants verified:
1. The safety tier of the lexicographic ranking is preserved through the
   search engine — a faster collision trajectory never outranks a safe one.
2. A resumed search never re-evaluates a prior search point.
3. Verdict round-trip through the store produces identical rank_key.
4. A search with budget 0 returns a SearchResult with no new trials run.
5. An exhausted grid halts gracefully before the budget is consumed.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import uuid

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reachy_ai.evaluation.base import EpisodeVerdict, EvaluationPolicy
from reachy_ai.motion.recipe import PrimitiveStep, TrajectoryRecipe
from reachy_ai.search.runner import SearchConfig, SearchRunner
from reachy_ai.search.samplers import GridSampler, RandomSampler
from reachy_ai.search.space import ParameterBound, SearchSpace


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _recipe(
    rid: str = "baseline",
    bounded: dict = None,
    steps: list = None,
) -> TrajectoryRecipe:
    return TrajectoryRecipe(
        recipe_id=rid,
        task_type="operate_control",
        primitive_sequence=steps or [],
        bounded_parameters=bounded or {},
    )


def _small_recipe() -> TrajectoryRecipe:
    """2-parameter recipe with a 3×2 grid (6 total grid points)."""
    return _recipe(
        bounded={
            "approach_height": {"value": 0.12, "min": 0.05, "max": 0.15, "step": 0.05},
            "press_force":     {"value": 0.5,  "min": 0.3,  "max": 0.5,  "step": 0.1},
        }
    )


def _make_verdict(
    trial_id: str = None,
    is_safe: bool = True,
    is_successful: bool = True,
    duration: float = 0.5,
) -> EpisodeVerdict:
    tid = trial_id or uuid.uuid4().hex
    return EpisodeVerdict(
        episode_id=tid,
        trial_id=tid,
        task_type="test",
        policy_version=1,
        is_valid=True,
        is_safe=is_safe,
        is_successful=is_successful,
        violations=[],
        metrics={},
        ranking_scores={
            "accuracy_score": 1.0 if is_successful else 0.0,
            "clearance_score": 1.0 if is_safe else 0.0,
            "effort_score": 1.0,
            "smoothness_score": 1.0,
            "duration_score": duration,
        },
        explanation="test",
    )


_call_log: list = []


def _counting_evaluate(recipe: TrajectoryRecipe) -> EpisodeVerdict:
    """Deterministic synthetic evaluator that records each call."""
    _call_log.append(recipe.recipe_id)
    return _make_verdict(is_safe=True, is_successful=True, duration=0.5)


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------

class TestSearchSpace:
    def test_from_recipe_extracts_bounds(self):
        recipe = _small_recipe()
        space = SearchSpace.from_recipe(recipe)
        names = {b.name for b in space.bounds}
        assert "approach_height" in names
        assert "press_force" in names

    def test_fixed_scalars_excluded(self):
        recipe = _recipe(bounded={"x": 1.0, "y": {"value": 0.5, "min": 0.0, "max": 1.0, "step": 0.5}})
        space = SearchSpace.from_recipe(recipe)
        names = {b.name for b in space.bounds}
        assert "y" in names
        assert "x" not in names  # plain scalar — not searchable

    def test_grid_size_is_product_of_axes(self):
        recipe = _small_recipe()
        space = SearchSpace.from_recipe(recipe)
        # approach_height: [0.05, 0.10, 0.15] → 3 values
        # press_force:     [0.3, 0.4, 0.5]   → 3 values
        assert space.num_grid_points() == 9

    def test_baseline_point_uses_recipe_values(self):
        recipe = _small_recipe()
        space = SearchSpace.from_recipe(recipe)
        pt = space.baseline_point()
        assert abs(pt["approach_height"] - 0.12) < 1e-9
        assert abs(pt["press_force"] - 0.5) < 1e-9

    def test_validate_point_accepts_in_bounds(self):
        space = SearchSpace.from_recipe(_small_recipe())
        assert space.validate_point({"approach_height": 0.10, "press_force": 0.4})

    def test_validate_point_rejects_out_of_bounds(self):
        space = SearchSpace.from_recipe(_small_recipe())
        assert not space.validate_point({"approach_height": 0.99, "press_force": 0.4})

    def test_clip_point_clamps_values(self):
        space = SearchSpace.from_recipe(_small_recipe())
        pt = space.clip_point({"approach_height": 9.99, "press_force": -1.0})
        assert pt["approach_height"] == pytest.approx(0.15)
        assert pt["press_force"] == pytest.approx(0.3)

    def test_empty_space_grid_yields_one_empty_dict(self):
        space = SearchSpace.from_recipe(_recipe())
        pts = list(space.grid_points())
        assert pts == [{}]

    def test_point_key_is_deterministic(self):
        space = SearchSpace.from_recipe(_small_recipe())
        pt = {"approach_height": 0.05, "press_force": 0.3}
        assert space.point_key(pt) == space.point_key(pt)

    def test_different_points_have_different_keys(self):
        space = SearchSpace.from_recipe(_small_recipe())
        k1 = space.point_key({"approach_height": 0.05, "press_force": 0.3})
        k2 = space.point_key({"approach_height": 0.10, "press_force": 0.3})
        assert k1 != k2

    def test_parameter_bound_lo_gt_hi_raises(self):
        with pytest.raises(ValueError, match="lo"):
            ParameterBound(name="x", lo=1.0, hi=0.0, baseline=0.5)

    def test_parameter_bound_negative_step_raises(self):
        with pytest.raises(ValueError, match="step"):
            ParameterBound(name="x", lo=0.0, hi=1.0, baseline=0.5, step=-0.1)

    def test_continuous_axis_grid_values_is_baseline(self):
        b = ParameterBound(name="x", lo=0.0, hi=1.0, baseline=0.5, step=0.0)
        assert b.grid_values() == [0.5]

    def test_discrete_axis_grid_values_correct(self):
        b = ParameterBound(name="x", lo=0.0, hi=1.0, baseline=0.5, step=0.5)
        vals = b.grid_values()
        assert vals == pytest.approx([0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# GridSampler
# ---------------------------------------------------------------------------

class TestGridSampler:
    def test_suggests_all_grid_points_sequentially(self):
        space = SearchSpace.from_recipe(_small_recipe())
        sampler = GridSampler()
        total = space.num_grid_points()
        done = []
        all_pts = []
        for _ in range(total):
            pts = sampler.suggest(space, done)
            assert len(pts) == 1
            all_pts.append(pts[0])
            done.append(pts[0])
        assert len(all_pts) == total

    def test_exhausted_grid_returns_empty(self):
        space = SearchSpace.from_recipe(_small_recipe())
        sampler = GridSampler()
        done = list(space.grid_points())
        pts = sampler.suggest(space, done)
        assert pts == []

    def test_skips_already_done(self):
        space = SearchSpace.from_recipe(_small_recipe())
        sampler = GridSampler()
        first = list(space.grid_points())[:5]
        pts = sampler.suggest(space, first)
        assert len(pts) == 1
        assert space.point_key(pts[0]) not in {space.point_key(p) for p in first}

    def test_remaining_decreases_with_done(self):
        space = SearchSpace.from_recipe(_small_recipe())
        sampler = GridSampler()
        total = space.num_grid_points()
        done = list(space.grid_points())[:4]
        assert sampler.remaining(space, done) == total - 4

    def test_batch_suggest_returns_n_points(self):
        space = SearchSpace.from_recipe(_small_recipe())
        sampler = GridSampler()
        pts = sampler.suggest(space, [], n=3)
        assert len(pts) == 3
        keys = {space.point_key(p) for p in pts}
        assert len(keys) == 3  # all distinct


# ---------------------------------------------------------------------------
# RandomSampler
# ---------------------------------------------------------------------------

class TestRandomSampler:
    def test_produces_in_bounds_values(self):
        space = SearchSpace.from_recipe(_small_recipe())
        sampler = RandomSampler(seed=42)
        pts = sampler.suggest(space, [], n=20)
        for pt in pts:
            assert space.validate_point(pt)

    def test_reproducible_with_same_seed(self):
        space = SearchSpace.from_recipe(_small_recipe())
        pts_a = RandomSampler(seed=7).suggest(space, [], n=5)
        pts_b = RandomSampler(seed=7).suggest(space, [], n=5)
        for a, b in zip(pts_a, pts_b):
            assert space.point_key(a) == space.point_key(b)

    def test_different_seeds_produce_different_points(self):
        space = SearchSpace.from_recipe(_small_recipe())
        pts_a = RandomSampler(seed=1).suggest(space, [], n=5)
        pts_b = RandomSampler(seed=2).suggest(space, [], n=5)
        keys_a = {space.point_key(p) for p in pts_a}
        keys_b = {space.point_key(p) for p in pts_b}
        assert keys_a != keys_b


# ---------------------------------------------------------------------------
# SearchRunner — basic run
# ---------------------------------------------------------------------------

class TestSearchRunnerBasic:
    def test_run_respects_budget(self):
        recipe = _small_recipe()
        config = SearchConfig(study_id="s1", baseline_recipe=recipe, budget=3)
        runner = SearchRunner(config)
        result = runner.run(_counting_evaluate)
        assert result.trials_run == 3

    def test_run_exhausted_grid_halts_early(self):
        # budget larger than grid; should stop when grid is exhausted.
        recipe = _recipe(
            bounded={"x": {"value": 0.5, "min": 0.0, "max": 1.0, "step": 0.5}}
        )  # grid size = 3 (0.0, 0.5, 1.0)
        config = SearchConfig(study_id="s2", baseline_recipe=recipe, budget=100)
        runner = SearchRunner(config)
        result = runner.run(_counting_evaluate)
        assert result.trials_run == 3  # grid exhausted at 3 points

    def test_run_budget_zero_no_trials(self):
        recipe = _small_recipe()
        config = SearchConfig(study_id="s3", baseline_recipe=recipe, budget=0)
        runner = SearchRunner(config)
        result = runner.run(_counting_evaluate)
        assert result.trials_run == 0
        assert result.best is None

    def test_best_is_top_ranked(self):
        # Evaluator returns the same verdict for all points; best should exist.
        recipe = _small_recipe()
        config = SearchConfig(study_id="s4", baseline_recipe=recipe, budget=4)
        runner = SearchRunner(config)
        result = runner.run(_counting_evaluate)
        assert result.best is not None
        assert result.best.verdict.is_safe
        assert result.best.verdict.is_successful

    def test_all_candidates_in_ranked(self):
        recipe = _small_recipe()
        config = SearchConfig(study_id="s5", baseline_recipe=recipe, budget=5)
        runner = SearchRunner(config)
        result = runner.run(_counting_evaluate)
        assert len(result.ranked) == result.trials_run

    def test_empty_space_single_trial(self):
        recipe = _recipe()  # no bounded parameters
        config = SearchConfig(study_id="s6", baseline_recipe=recipe, budget=5)
        runner = SearchRunner(config)
        result = runner.run(_counting_evaluate)
        # Grid for an empty space yields one point; budget cap is 5 but
        # only 1 distinct point exists.
        assert result.trials_run == 1

    def test_evaluate_fn_exception_does_not_crash_runner(self):
        def bad_fn(recipe: TrajectoryRecipe) -> EpisodeVerdict:
            raise RuntimeError("evaluation failed")

        recipe = _small_recipe()
        config = SearchConfig(study_id="s7", baseline_recipe=recipe, budget=3)
        runner = SearchRunner(config)
        result = runner.run(bad_fn)
        # All trials attempted; none produced candidates.
        assert result.trials_run == 3
        assert result.best is None

    def test_ranked_is_sorted_best_first(self):
        # Evaluator returns varied duration scores based on parameter value.
        def eval_fn(recipe: TrajectoryRecipe) -> EpisodeVerdict:
            bp = recipe.bounded_parameters.get("x", {})
            val = float(bp.get("value", 0.5)) if isinstance(bp, dict) else float(bp)
            return _make_verdict(duration=val)

        recipe = _recipe(bounded={"x": {"value": 0.5, "min": 0.0, "max": 1.0, "step": 0.5}})
        config = SearchConfig(study_id="s8", baseline_recipe=recipe, budget=10)
        runner = SearchRunner(config)
        result = runner.run(eval_fn)
        keys = [c.rank_key for c in result.ranked]
        assert keys == sorted(keys, reverse=True)


# ---------------------------------------------------------------------------
# SearchRunner — safety invariant
# ---------------------------------------------------------------------------

class TestSearchRunnerSafetyInvariant:
    def test_safe_always_beats_unsafe_in_search_result(self):
        """The safety tier must hold even if an unsafe path had a higher score."""
        call_n = [0]

        def eval_fn(recipe: TrajectoryRecipe) -> EpisodeVerdict:
            n = call_n[0]
            call_n[0] += 1
            # First evaluation: safe but slow.
            # Second: unsafe but fast.
            is_safe = n == 0
            duration = 0.1 if is_safe else 0.99
            return _make_verdict(is_safe=is_safe, is_successful=is_safe, duration=duration)

        recipe = _recipe(bounded={"x": {"value": 0.0, "min": 0.0, "max": 1.0, "step": 1.0}})
        config = SearchConfig(study_id="safe_test", baseline_recipe=recipe, budget=2)
        runner = SearchRunner(config)
        result = runner.run(eval_fn)
        assert result.best is not None
        assert result.best.verdict.is_safe, (
            "A faster collision path must never outrank a safe path in search results"
        )


# ---------------------------------------------------------------------------
# SearchRunner — prior_results / resume
# ---------------------------------------------------------------------------

class TestSearchRunnerResume:
    def test_prior_points_not_re_evaluated(self):
        recipe = _recipe(bounded={"x": {"value": 0.5, "min": 0.0, "max": 1.0, "step": 0.5}})
        # grid = [0.0, 0.5, 1.0]
        space = SearchSpace.from_recipe(recipe)
        first_pt = list(space.grid_points())[0]

        prior = [(first_pt, _make_verdict(duration=0.8))]
        evals = [0]

        def eval_fn(recipe: TrajectoryRecipe) -> EpisodeVerdict:
            evals[0] += 1
            return _make_verdict(duration=0.5)

        config = SearchConfig(study_id="resume_test", baseline_recipe=recipe, budget=3)
        runner = SearchRunner(config)
        result = runner.run(eval_fn, prior_results=prior)

        # Only 2 new evaluations (grid has 3 total; 1 already done)
        assert result.trials_run == 2
        assert evals[0] == 2
        assert result.trials_skipped == 1

    def test_prior_verdicts_appear_in_ranked(self):
        recipe = _recipe(bounded={"x": {"value": 0.5, "min": 0.0, "max": 1.0, "step": 0.5}})
        space = SearchSpace.from_recipe(recipe)
        first_pt = list(space.grid_points())[0]
        prior_verdict = _make_verdict(duration=0.99)
        prior = [(first_pt, prior_verdict)]

        config = SearchConfig(study_id="resume_test2", baseline_recipe=recipe, budget=2)
        runner = SearchRunner(config)
        result = runner.run(_counting_evaluate, prior_results=prior)

        # Total ranked = 1 prior + 2 new = 3
        assert len(result.ranked) == 3

    def test_resume_helper_delegates_correctly(self):
        recipe = _recipe(bounded={"x": {"value": 0.5, "min": 0.0, "max": 1.0, "step": 0.5}})
        space = SearchSpace.from_recipe(recipe)
        first_pt = list(space.grid_points())[0]
        prior = [(first_pt, _make_verdict())]

        config = SearchConfig(study_id="resume_helper", baseline_recipe=recipe, budget=99)
        runner = SearchRunner(config)
        result = runner.resume(_counting_evaluate, prior, extra_budget=2)
        assert result.trials_run == 2


# ---------------------------------------------------------------------------
# SearchRunner — store integration (in-process, temp SQLite file)
# ---------------------------------------------------------------------------

class TestSearchRunnerStoreIntegration:
    def _run_with_store(self, db_path: str, budget: int = 3):
        from reachy_ai.experience.store import ExperienceStore

        recipe = _recipe(bounded={"x": {"value": 0.0, "min": 0.0, "max": 1.0, "step": 0.5}})
        config = SearchConfig(study_id="store_study", baseline_recipe=recipe, budget=budget)
        runner = SearchRunner(config)
        with ExperienceStore.open(db_path) as store:
            result = runner.run(_counting_evaluate, store=store)
        return result

    def test_store_records_trials(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "search.db")
        self._run_with_store(db, budget=2)
        with ExperienceStore.open(db) as store:
            rows = store.list_trials("store_study")
        assert len(rows) == 2

    def test_store_verdict_round_trip(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore
        import json

        db = str(tmp_path / "search.db")
        self._run_with_store(db, budget=2)
        with ExperienceStore.open(db) as store:
            rows = store.list_trials("store_study")
            for row in rows:
                meta = json.loads(row.get("optimizer_metadata_json") or "{}")
                assert "search_point" in meta
                assert "verdict_json" in meta
                verdict = EpisodeVerdict.from_json(meta["verdict_json"])
                assert verdict.is_safe

    def test_load_prior_from_store_returns_prior(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "search.db")
        self._run_with_store(db, budget=2)
        with ExperienceStore.open(db) as store:
            prior = SearchRunner.load_prior_from_store(store, "store_study")
        assert len(prior) == 2
        for pt, verdict in prior:
            assert isinstance(pt, dict)
            assert verdict.is_valid

    def test_resume_from_store_no_re_evaluation(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore

        db = str(tmp_path / "search.db")
        self._run_with_store(db, budget=2)

        recipe = _recipe(bounded={"x": {"value": 0.0, "min": 0.0, "max": 1.0, "step": 0.5}})
        config = SearchConfig(study_id="store_study", baseline_recipe=recipe, budget=1)
        runner = SearchRunner(config)

        evals = [0]
        def counting_fn(r: TrajectoryRecipe) -> EpisodeVerdict:
            evals[0] += 1
            return _make_verdict()

        with ExperienceStore.open(db) as store:
            prior = SearchRunner.load_prior_from_store(store, "store_study")
            result = runner.run(counting_fn, prior_results=prior, store=store)

        # 2 already done (grid size = 3): only 1 new evaluation.
        assert evals[0] == 1
        assert result.trials_skipped == 2

    def test_store_rank_key_survives_round_trip(self, tmp_path):
        from reachy_ai.experience.store import ExperienceStore
        import json

        db = str(tmp_path / "search.db")
        result_orig = self._run_with_store(db, budget=2)
        orig_keys = {c.rank_key for c in result_orig.ranked}

        with ExperienceStore.open(db) as store:
            prior = SearchRunner.load_prior_from_store(store, "store_study")

        restored_keys = {v.rank_key for _, v in prior}
        assert orig_keys == restored_keys

    def test_load_prior_recovers_more_than_default_list_limit(self, tmp_path):
        """Regression: load_prior_from_store must not truncate at list_trials'
        default 500-row cap.  A study with >500 recorded trials would otherwise
        drop prior points from already_done and re-evaluate them on resume."""
        from reachy_ai.experience.store import ExperienceStore

        # 24 x 24 = 576 grid points, safely above the 500 default limit.
        recipe = _recipe(
            bounded={
                "a": {"value": 0.0, "min": 0.0, "max": 23.0, "step": 1.0},
                "b": {"value": 0.0, "min": 0.0, "max": 23.0, "step": 1.0},
            }
        )
        n_points = 24 * 24
        config = SearchConfig(
            study_id="big_study", baseline_recipe=recipe, budget=n_points
        )
        runner = SearchRunner(config)

        db = str(tmp_path / "big.db")
        with ExperienceStore.open(db) as store:
            result = runner.run(_counting_evaluate, store=store)
            assert result.trials_run == n_points

            # Default limit truncates; None must not.
            capped = store.list_trials("big_study")
            assert len(capped) == 500
            full = store.list_trials("big_study", limit=None)
            assert len(full) == n_points

            prior = SearchRunner.load_prior_from_store(store, "big_study")

        assert len(prior) == n_points

        # Resuming with the full prior re-samples nothing: the grid is exhausted.
        resume_cfg = SearchConfig(
            study_id="big_study", baseline_recipe=recipe, budget=10
        )
        resumed = SearchRunner(resume_cfg).run(
            lambda r: _make_verdict(), prior_results=prior
        )
        assert resumed.trials_run == 0
        assert resumed.trials_skipped == n_points
