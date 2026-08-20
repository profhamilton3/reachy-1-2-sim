"""Research-validity regression tests for the Epic 8 search engine.

These cover two validity bugs found in the Epic 8 review that the existing
suite missed because it only exercised the synthetic dry-run evaluator (which
mints a unique trial_id per verdict):

  C2 — real evaluators (EpisodeRunner) emit a CONSTANT trial_id
       (`scene_revision or "standalone"`).  The search engine must not key the
       trial->search_point map on verdict.trial_id, or every trial collapses
       to one identity and the exported "best" recipe carries the wrong params.

  C1 — the default `grid` sampler silently evaluates only the baseline when
       searchable axes have no `step`, turning a "study" into N=1.  The engine
       must fail loudly instead.
"""

from __future__ import annotations

import uuid

import pytest

from reachy_ai.evaluation.base import EpisodeVerdict
from reachy_ai.motion.recipe import TrajectoryRecipe
from reachy_ai.search.report import generate_report
from reachy_ai.search.runner import (
    SearchConfig,
    SearchRunner,
    apply_best_to_recipe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONST_TRIAL_ID = "standalone"  # what EpisodeRunner emits when scene_revision==""


def _gridded_recipe(recipe_id="baseline_r"):
    """A recipe whose single axis x has a step, so grid explores {0,1,2,3,4}."""
    return TrajectoryRecipe(
        recipe_id=recipe_id,
        task_type="test",
        bounded_parameters={"x": {"value": 0.0, "min": 0.0, "max": 4.0, "step": 1.0}},
    )


def _continuous_recipe(recipe_id="baseline_c"):
    """A recipe whose axes have NO step — mirrors recipes/pick_place/baseline_v1."""
    return TrajectoryRecipe(
        recipe_id=recipe_id,
        task_type="test",
        bounded_parameters={
            "home_steps": {"value": 40, "min": 15, "max": 150},
            "lift_steps": {"value": 30, "min": 15, "max": 120},
        },
    )


def _sim_like_verdict(accuracy: float) -> EpisodeVerdict:
    """A verdict shaped like the real evaluator output: unique episode_id but a
    CONSTANT trial_id, exactly as EpisodeRunner + evaluate_pick_place produce."""
    return EpisodeVerdict(
        episode_id=uuid.uuid4().hex,
        trial_id=_CONST_TRIAL_ID,          # <-- the collapse trigger
        task_type="test",
        policy_version=1,
        is_valid=True,
        is_safe=True,
        is_successful=True,
        violations=[],
        metrics={},
        ranking_scores={
            "accuracy_score": accuracy,
            "clearance_score": 1.0,
            "effort_score": 1.0,
            "smoothness_score": 1.0,
            "duration_score": 0.5,
        },
        explanation="sim-like",
    )


# ---------------------------------------------------------------------------
# C2 — constant trial_id must not collapse trial identity
# ---------------------------------------------------------------------------

class TestConstantTrialIdDoesNotCollapse:
    def test_best_search_point_matches_best_verdict_in_memory(self):
        """x=2 is the highest-accuracy point (peak in the MIDDLE, so it is not
        the last grid point evaluated); best_search_point must be x=2 even
        though every verdict shares the same constant trial_id."""
        baseline = _gridded_recipe()
        config = SearchConfig(study_id="s", baseline_recipe=baseline, budget=5)
        # accuracy peaks at x=2 -> best point is x=2.0 (not the last, x=4)
        result = SearchRunner(config).run(
            lambda r, _seed: _sim_like_verdict(1.0 - abs(float(r.bounded_parameters["x"]["value"]) - 2.0) / 4.0)
        )
        assert result.best_search_point is not None
        assert result.best_search_point["x"] == pytest.approx(2.0)

    def test_exported_recipe_carries_best_params_not_last(self):
        """apply_best_to_recipe must export the winning params (x=2), not the
        params of whichever trial happened to run last (x=4)."""
        baseline = _gridded_recipe()
        config = SearchConfig(study_id="s", baseline_recipe=baseline, budget=5)
        result = SearchRunner(config).run(
            lambda r, _seed: _sim_like_verdict(1.0 - abs(float(r.bounded_parameters["x"]["value"]) - 2.0) / 4.0)
        )
        exported = apply_best_to_recipe(result, baseline)
        assert exported is not None
        assert exported.bounded_parameters["x"]["value"] == pytest.approx(2.0)

    def test_candidates_have_distinct_trial_ids(self):
        """Each trial must land in the map under a distinct id despite the
        constant verdict.trial_id."""
        baseline = _gridded_recipe()
        config = SearchConfig(study_id="s", baseline_recipe=baseline, budget=5)
        result = SearchRunner(config).run(
            lambda r, _seed: _sim_like_verdict(1.0 - abs(float(r.bounded_parameters["x"]["value"]) - 2.0) / 4.0)
        )
        ids = [c.trial_id for c in result.ranked]
        assert len(ids) == len(set(ids)) == 5

    def test_store_roundtrip_report_attributes_correct_point(self, tmp_path):
        """Persist a study with sim-like (constant-trial_id) verdicts, then
        reload via the report path and confirm the best trial's search_point is
        the winning one — not a collapsed/aliased point."""
        from reachy_ai.experience.store import ExperienceStore

        baseline = _gridded_recipe()
        db = str(tmp_path / "study.db")
        with ExperienceStore.open(db) as store:
            config = SearchConfig(study_id="st", baseline_recipe=baseline, budget=5)
            SearchRunner(config).run(
                lambda r, _seed: _sim_like_verdict(
                    1.0 - abs(float(r.bounded_parameters["x"]["value"]) - 2.0) / 4.0
                ),
                store=store,
            )
            report = generate_report(store, "st", top_n=5)

        assert report.total_trials == 5
        assert report.best is not None
        # Best reloaded trial must carry x=2.0 in its search_point.
        assert report.best.search_point.get("x") == pytest.approx(2.0)
        # And every reloaded trial has a distinct identity.
        ids = [t.trial_id for t in report.top]
        assert len(ids) == len(set(ids)) == 5


# ---------------------------------------------------------------------------
# C1 — degenerate grid must fail loudly, not silently run N=1
# ---------------------------------------------------------------------------

class TestDegenerateGridRejected:
    def test_grid_with_stepless_axes_raises(self):
        baseline = _continuous_recipe()
        config = SearchConfig(study_id="pp", baseline_recipe=baseline, budget=20,
                              sampler="grid")
        with pytest.raises(ValueError, match="grid sampler would evaluate only the baseline"):
            SearchRunner(config).run(lambda r, _seed: _sim_like_verdict(0.5))

    def test_error_names_the_offending_axes(self):
        baseline = _continuous_recipe()
        config = SearchConfig(study_id="pp", baseline_recipe=baseline, sampler="grid")
        with pytest.raises(ValueError, match="home_steps"):
            SearchRunner(config).run(lambda r, _seed: _sim_like_verdict(0.5))

    def test_random_sampler_is_unaffected(self):
        """Stepless axes are fine for random sampling — no raise, real search."""
        baseline = _continuous_recipe()
        config = SearchConfig(study_id="pp", baseline_recipe=baseline, budget=8,
                              sampler="random", random_seed=1)
        result = SearchRunner(config).run(lambda r, _seed: _sim_like_verdict(0.5))
        assert result.trials_run == 8

    def test_grid_with_stepped_axes_still_runs(self):
        """A recipe that DOES define steps grids normally."""
        baseline = _gridded_recipe()
        config = SearchConfig(study_id="ok", baseline_recipe=baseline, budget=5,
                              sampler="grid")
        result = SearchRunner(config).run(lambda r, _seed: _sim_like_verdict(0.5))
        assert result.trials_run == 5
