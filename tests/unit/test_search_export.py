"""R12-809: Tests for best-recipe export — apply_best_to_recipe and CLI export."""

from __future__ import annotations

import json
import uuid
from typing import List

import pytest

from reachy_ai.evaluation.base import EpisodeVerdict
from reachy_ai.motion.recipe import TrajectoryRecipe
from reachy_ai.search.runner import (
    SearchConfig,
    SearchResult,
    SearchRunner,
    apply_best_to_recipe,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recipe(*, bounded=None, recipe_id="baseline_r"):
    return TrajectoryRecipe(
        recipe_id=recipe_id,
        task_type="test",
        bounded_parameters=bounded or {
            "x": {"value": 0.0, "min": 0.0, "max": 4.0, "step": 1.0},
            "y": {"value": 0.0, "min": 0.0, "max": 2.0, "step": 1.0},
        },
    )


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


def _run(recipe=None, budget=None, evaluate_fn=None) -> SearchResult:
    r = recipe or _recipe()
    config = SearchConfig(
        study_id="s",
        baseline_recipe=r,
        budget=budget or 50,
    )
    ev = evaluate_fn or (lambda rec: _verdict(
        accuracy=float(rec.bounded_parameters["x"]["value"]) / 4.0
    ))
    return SearchRunner(config).run(ev)


# ---------------------------------------------------------------------------
# apply_best_to_recipe — unit tests
# ---------------------------------------------------------------------------

class TestApplyBestToRecipe:
    def test_returns_none_when_no_best(self):
        baseline = _recipe()
        result = SearchResult(
            study_id="s", best=None, ranked=[], kept=[], pruned=[],
            trials_run=0, trials_skipped=0,
        )
        assert apply_best_to_recipe(result, baseline) is None

    def test_returns_none_when_no_search_point(self):
        from reachy_ai.evaluation.ranking import RankedCandidate
        baseline = _recipe()
        v = _verdict()
        rc = RankedCandidate(verdict=v, trial_id=v.trial_id)
        result = SearchResult(
            study_id="s", best=rc, ranked=[rc], kept=[rc], pruned=[],
            trials_run=1, trials_skipped=0, best_search_point=None,
        )
        assert apply_best_to_recipe(result, baseline) is None

    def test_recipe_id_encodes_baseline_and_trial(self):
        result = _run()
        recipe = apply_best_to_recipe(result, _recipe(recipe_id="my_base"))
        assert recipe is not None
        assert recipe.recipe_id.startswith("my_base_best_")
        assert len(recipe.recipe_id) > len("my_base_best_")

    def test_source_is_search_best(self):
        result = _run()
        recipe = apply_best_to_recipe(result, _recipe())
        assert recipe is not None
        assert recipe.source == "search_best"

    def test_parent_recipe_id_is_baseline(self):
        baseline = _recipe(recipe_id="orig")
        result = _run(recipe=baseline)
        exported = apply_best_to_recipe(result, baseline)
        assert exported is not None
        assert exported.parent_recipe_id == "orig"

    def test_best_parameter_values_applied(self):
        """The best search point's values must appear in bounded_parameters."""
        baseline = _recipe()
        result = _run(recipe=baseline)
        assert result.best_search_point is not None
        exported = apply_best_to_recipe(result, baseline)
        assert exported is not None
        for name, val in result.best_search_point.items():
            spec = exported.bounded_parameters[name]
            assert isinstance(spec, dict)
            assert spec["value"] == pytest.approx(val)

    def test_non_bounded_parameters_preserved(self):
        """Plain scalar parameters that aren't in the search space must survive."""
        baseline = TrajectoryRecipe(
            recipe_id="r",
            task_type="test",
            bounded_parameters={
                "x": {"value": 0.0, "min": 0.0, "max": 2.0, "step": 1.0},
                "fixed": "constant_value",   # not searchable
            },
        )
        result = _run(recipe=baseline)
        exported = apply_best_to_recipe(result, baseline)
        assert exported is not None
        assert exported.bounded_parameters["fixed"] == "constant_value"

    def test_primitive_sequence_preserved(self):
        from reachy_ai.motion.recipe import PrimitiveStep
        baseline = TrajectoryRecipe(
            recipe_id="r",
            task_type="test",
            bounded_parameters={"x": {"value": 0.0, "min": 0.0, "max": 2.0, "step": 1.0}},
            primitive_sequence=[PrimitiveStep(primitive="move_to", parameters={"joint": "elbow"})],
        )
        result = _run(recipe=baseline, budget=3)
        exported = apply_best_to_recipe(result, baseline)
        assert exported is not None
        assert len(exported.primitive_sequence) == 1
        assert exported.primitive_sequence[0].primitive == "move_to"

    def test_best_search_point_populated_after_run(self):
        """SearchResult.best_search_point is set by run()."""
        result = _run()
        assert result.best_search_point is not None
        assert isinstance(result.best_search_point, dict)

    def test_best_search_point_populated_from_prior(self):
        """best_search_point is also set when best comes from prior_results."""
        baseline = _recipe()
        config = SearchConfig(study_id="s", baseline_recipe=baseline, budget=0)
        runner = SearchRunner(config)
        prior = [({"x": 3.0, "y": 1.0}, _verdict(accuracy=0.9))]
        result = runner.run(lambda r: _verdict(), prior_results=prior)
        assert result.best_search_point == {"x": 3.0, "y": 1.0}

    def test_highest_ranked_point_selected(self):
        """apply_best_to_recipe uses the highest-ranked candidate's point."""
        baseline = _recipe()
        # x=4 gets acc=1.0 (best), x=0 gets acc=0.0 (worst)
        result = _run(recipe=baseline, evaluate_fn=lambda r: _verdict(
            accuracy=float(r.bounded_parameters["x"]["value"]) / 4.0
        ))
        exported = apply_best_to_recipe(result, baseline)
        assert exported is not None
        assert exported.bounded_parameters["x"]["value"] == pytest.approx(4.0)

    def test_json_serialization_of_exported_recipe(self):
        baseline = _recipe()
        result = _run(recipe=baseline)
        exported = apply_best_to_recipe(result, baseline)
        assert exported is not None
        raw = exported.to_json()
        reloaded = TrajectoryRecipe.from_json(raw)
        assert reloaded.recipe_id == exported.recipe_id
        assert reloaded.source == "search_best"


# ---------------------------------------------------------------------------
# SearchResult.best_search_point across different sampler types
# ---------------------------------------------------------------------------

class TestBestSearchPointAcrossSamplers:
    def _check(self, sampler: str, prior=None):
        baseline = _recipe()
        config = SearchConfig(study_id="s", baseline_recipe=baseline, budget=5, sampler=sampler)
        runner = SearchRunner(config)
        result = runner.run(lambda r: _verdict(accuracy=0.5), prior_results=prior)
        assert result.best_search_point is not None
        assert runner.space.validate_point(result.best_search_point)

    def test_grid_sampler(self):
        self._check("grid")

    def test_random_sampler(self):
        self._check("random")

    def test_adaptive_sampler_with_prior(self):
        prior = [({"x": 2.0, "y": 1.0}, _verdict(accuracy=0.8))]
        self._check("adaptive", prior=prior)

    def test_adaptive_sampler_without_prior(self):
        self._check("adaptive")


# ---------------------------------------------------------------------------
# CLI export subcommand
# ---------------------------------------------------------------------------

class TestCliExport:
    def _populate_db(self, tmp_path, study_id="exp_study"):
        """1D recipe x∈{0,1,2,3,4}: budget=5 covers the full grid."""
        from reachy_ai.experience.store import ExperienceStore
        baseline = _recipe(bounded={
            "x": {"value": 0.0, "min": 0.0, "max": 4.0, "step": 1.0},
        })
        db = str(tmp_path / "exp.db")
        with ExperienceStore.open(db) as store:
            config = SearchConfig(study_id=study_id, baseline_recipe=baseline, budget=5)
            SearchRunner(config).run(
                lambda r: _verdict(accuracy=float(r.bounded_parameters["x"]["value"]) / 4.0),
                store=store,
            )
        return db, baseline

    def test_export_prints_json_to_stdout(self, tmp_path, capsys):
        from reachy_ai.search.cli import main
        db, _ = self._populate_db(tmp_path)
        rc = main(["export", "--study-id", "exp_study", "--db", db])
        assert rc == 0
        out = capsys.readouterr().out
        d = json.loads(out)
        assert d["source"] == "search_best"
        assert "recipe_id" in d

    def test_export_writes_to_file(self, tmp_path, capsys):
        from reachy_ai.search.cli import main
        import pathlib
        db, _ = self._populate_db(tmp_path)
        out_path = str(tmp_path / "best.json")
        rc = main(["export", "--study-id", "exp_study", "--db", db, "--output", out_path])
        assert rc == 0
        text = pathlib.Path(out_path).read_text()
        d = json.loads(text)
        assert d["source"] == "search_best"

    def test_export_with_explicit_baseline(self, tmp_path, capsys):
        from reachy_ai.search.cli import main
        import pathlib
        db, baseline = self._populate_db(tmp_path)
        baseline_file = str(tmp_path / "baseline.json")
        pathlib.Path(baseline_file).write_text(baseline.to_json())
        rc = main([
            "export", "--study-id", "exp_study",
            "--db", db, "--baseline", baseline_file,
        ])
        assert rc == 0

    def test_export_requires_db(self, capsys):
        from reachy_ai.search.cli import main
        rc = main(["export", "--study-id", "x"])
        assert rc != 0

    def test_export_empty_study_returns_nonzero(self, tmp_path, capsys):
        from reachy_ai.experience.store import ExperienceStore
        from reachy_ai.search.cli import main
        db = str(tmp_path / "empty.db")
        with ExperienceStore.open(db) as store:
            pass
        rc = main(["export", "--study-id", "ghost", "--db", db])
        assert rc != 0

    def test_export_best_x_is_maximum(self, tmp_path, capsys):
        """The exported recipe should carry x=4.0 (the highest-accuracy point)."""
        from reachy_ai.search.cli import main
        db, _ = self._populate_db(tmp_path)
        rc = main(["export", "--study-id", "exp_study", "--db", db])
        assert rc == 0
        out = capsys.readouterr().out
        d = json.loads(out)
        x_spec = d["bounded_parameters"]["x"]
        assert x_spec["value"] == pytest.approx(4.0)
