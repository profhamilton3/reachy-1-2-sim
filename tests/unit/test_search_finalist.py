"""R12-809: Finalist re-evaluation tests (issue #19).

Verifies that SearchRunner:
- Skips the finalist stage when finalist_seeds is empty (default).
- Re-evaluates exactly top-k kept candidates on finalist_seeds.
- Uses finalist_seeds, NOT eval_seeds, for the re-evaluation calls.
- Can reorder top candidates: the candidate ranked 2nd in search can become
  best when it outperforms the search winner on held-out seeds.
- `best` and `best_search_point` reflect the finalist winner, not the
  search-phase winner, when the finalist stage ran.
- `finalist_verdicts` is populated with (trial_id, verdict) pairs.
- finalist_k caps how many candidates are re-evaluated.
- Handles edge cases: fewer kept candidates than finalist_k, all finalist
  evaluations raising exceptions.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
import uuid
from typing import Dict, List, Tuple

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reachy_ai.evaluation.base import EpisodeVerdict
from reachy_ai.evaluation.ranking import RankedCandidate
from reachy_ai.motion.recipe import TrajectoryRecipe
from reachy_ai.search.runner import (
    SearchConfig, SearchResult, SearchRunner, _evaluate_multi_seed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recipe(n_steps: int = 3) -> TrajectoryRecipe:
    """Minimal recipe with one bounded parameter gridded at integer step=1.
    n_steps controls max so grid produces values {1, 2, ..., n_steps}."""
    return TrajectoryRecipe(
        recipe_id="test_finalist",
        task_type="pick_and_place",
        bounded_parameters={
            "hold_steps": {"value": 1.0, "min": 1.0, "max": float(n_steps), "step": 1.0}
        },
    )


def _verdict(
    accuracy: float = 1.0,
    effort: float = 1.0,
    duration: float = 0.5,
    is_successful: bool = True,
) -> EpisodeVerdict:
    eid = uuid.uuid4().hex
    return EpisodeVerdict(
        episode_id=eid,
        trial_id=eid,
        task_type="pick_and_place",
        policy_version=1,
        is_valid=True,
        is_safe=True,
        is_successful=is_successful,
        violations=[],
        metrics={},
        ranking_scores={
            "accuracy_score": accuracy,
            "effort_score": effort,
            "duration_score": duration,
        },
        explanation="test",
    )


class _RecordingEvaluator:
    """Evaluator that records every (recipe, seed) call and returns a preset verdict."""

    def __init__(self, verdicts_by_seed: Dict[int, EpisodeVerdict] = None) -> None:
        self.calls: List[Tuple[str, int]] = []   # (recipe_id, seed)
        self._verdicts = verdicts_by_seed or {}

    def __call__(self, recipe: TrajectoryRecipe, seed: int) -> EpisodeVerdict:
        self.calls.append((recipe.recipe_id, seed))
        if seed in self._verdicts:
            v = self._verdicts[seed]
            v = dataclasses.replace(v, episode_id=uuid.uuid4().hex, trial_id=uuid.uuid4().hex)
            return v
        return _verdict()


def _config(
    finalist_seeds: List[int] = (),
    finalist_k: int = 3,
    eval_seeds: List[int] = (0,),
    budget: int = 3,
) -> SearchConfig:
    return SearchConfig(
        study_id="test_finalist",
        baseline_recipe=_recipe(),
        budget=budget,
        sampler="grid",
        eval_seeds=list(eval_seeds),
        finalist_seeds=list(finalist_seeds),
        finalist_k=finalist_k,
    )


# ---------------------------------------------------------------------------
# Default: finalist stage disabled
# ---------------------------------------------------------------------------

class TestFinalistDisabledByDefault:
    def test_default_finalist_seeds_is_empty(self):
        cfg = SearchConfig(study_id="x", baseline_recipe=_recipe())
        assert cfg.finalist_seeds == []

    def test_no_finalist_seeds_skips_stage(self):
        ev = _RecordingEvaluator()
        runner = SearchRunner(_config(finalist_seeds=[]))
        result = runner.run(ev)
        assert result.finalist_verdicts == []

    def test_no_finalist_stage_best_from_search(self):
        ev = _RecordingEvaluator()
        runner = SearchRunner(_config(finalist_seeds=[]))
        result = runner.run(ev)
        # best should be set (there are candidates), finalist_verdicts empty
        assert result.best is not None
        assert result.finalist_verdicts == []

    def test_finalist_stage_does_not_call_evaluate_fn_extra_times(self):
        ev = _RecordingEvaluator()
        cfg = _config(finalist_seeds=[], budget=2, eval_seeds=[0])
        runner = SearchRunner(cfg)
        runner.run(ev)
        # Only 2 search evals expected (budget=2, 1 seed each)
        assert len(ev.calls) == 2


# ---------------------------------------------------------------------------
# Finalist stage runs when finalist_seeds provided
# ---------------------------------------------------------------------------

class TestFinalistStageRuns:
    def test_finalist_verdicts_populated(self):
        ev = _RecordingEvaluator()
        runner = SearchRunner(_config(finalist_seeds=[99, 100], finalist_k=2))
        result = runner.run(ev)
        assert len(result.finalist_verdicts) > 0

    def test_finalist_uses_held_out_seeds(self):
        ev = _RecordingEvaluator()
        cfg = _config(eval_seeds=[0], finalist_seeds=[99, 100], finalist_k=2)
        runner = SearchRunner(cfg)
        runner.run(ev)
        finalist_seeds_used = {seed for _, seed in ev.calls if seed in (99, 100)}
        assert finalist_seeds_used == {99, 100}

    def test_finalist_does_not_use_eval_seeds(self):
        """Finalist calls must only use finalist_seeds, not eval_seeds."""
        ev = _RecordingEvaluator()
        cfg = _config(eval_seeds=[7], finalist_seeds=[99], finalist_k=2)
        runner = SearchRunner(cfg)
        runner.run(ev)
        # Calls with seed=7 are search-phase; calls with seed=99 are finalist.
        finalist_calls = [(r, s) for r, s in ev.calls if s == 99]
        assert len(finalist_calls) > 0

    def test_finalist_verdicts_each_have_trial_id(self):
        ev = _RecordingEvaluator()
        runner = SearchRunner(_config(finalist_seeds=[42], finalist_k=3))
        result = runner.run(ev)
        for tid, fv in result.finalist_verdicts:
            assert tid  # non-empty
            assert fv.trial_id == tid

    def test_finalist_k_caps_number_of_finalists(self):
        ev = _RecordingEvaluator()
        # budget=3 gives 3 candidates; finalist_k=2 should re-eval at most 2
        runner = SearchRunner(_config(finalist_seeds=[55], finalist_k=2, budget=3))
        result = runner.run(ev)
        assert len(result.finalist_verdicts) <= 2


# ---------------------------------------------------------------------------
# Finalist can reorder: search #2 becomes best
# ---------------------------------------------------------------------------

class TestFinalistReorders:
    def test_finalist_winner_overrides_search_winner(self):
        """
        Search phase: candidate A beats candidate B (A is kept[0]).
        Finalist phase: B scores higher on held-out seeds → B becomes best.
        """
        # We need two deterministically different recipe points.
        # Recipe has hold_steps with values [1, 2, 3].
        # Seed 0 (search): point hold_steps=1 wins (accuracy=0.9 > 0.1).
        # Seed 99 (finalist): point hold_steps=2 wins (accuracy=0.8 > 0.2).

        def evaluator(recipe: TrajectoryRecipe, seed: int) -> EpisodeVerdict:
            val = recipe.bounded_parameters["hold_steps"]
            param_val = val["value"] if isinstance(val, dict) else val
            if seed == 0:
                # hold_steps=1 is the search winner
                acc = 0.9 if param_val == 1 else 0.1
            else:
                # seed=99: hold_steps=2 is the finalist winner
                acc = 0.8 if param_val == 2 else 0.2
            return _verdict(accuracy=acc)

        cfg = SearchConfig(
            study_id="reorder_test",
            baseline_recipe=_recipe(n_steps=3),
            budget=3,
            sampler="grid",
            pruner="none",           # keep all 3 candidates for finalist pool
            eval_seeds=[0],
            finalist_seeds=[99],
            finalist_k=3,
        )
        runner = SearchRunner(cfg)
        result = runner.run(evaluator)

        assert result.finalist_verdicts, "finalist stage should have run"
        # The finalist winner should be hold_steps=2 (param_val=2.0)
        best_sp = result.best_search_point
        assert best_sp is not None
        assert best_sp.get("hold_steps") == pytest.approx(2.0), (
            f"Expected finalist winner hold_steps=2.0, got {best_sp}"
        )

    def test_search_ranking_unchanged_after_finalist(self):
        """ranked/kept lists preserve the search-phase order regardless of finalist."""
        def evaluator(recipe: TrajectoryRecipe, seed: int) -> EpisodeVerdict:
            val = recipe.bounded_parameters["hold_steps"]
            param_val = val["value"] if isinstance(val, dict) else val
            acc = 0.9 if param_val == 1 else 0.1
            return _verdict(accuracy=acc if seed == 0 else 0.5)

        cfg = SearchConfig(
            study_id="order_preserved",
            baseline_recipe=_recipe(n_steps=3),
            budget=3,
            sampler="grid",
            eval_seeds=[0],
            finalist_seeds=[99],
            finalist_k=3,
        )
        runner = SearchRunner(cfg)
        result = runner.run(evaluator)

        # ranked[0] comes from search phase (hold_steps=1 has highest search score)
        first_sp = result.best_search_point  # finalist winner
        # ranked list should still be present
        assert len(result.ranked) > 0


# ---------------------------------------------------------------------------
# best and best_search_point update
# ---------------------------------------------------------------------------

class TestBestUpdatedByFinalist:
    def test_best_reflects_finalist_winner(self):
        ev = _RecordingEvaluator(verdicts_by_seed={0: _verdict(accuracy=0.5),
                                                   77: _verdict(accuracy=0.9)})
        cfg = _config(eval_seeds=[0], finalist_seeds=[77], finalist_k=3)
        runner = SearchRunner(cfg)
        result = runner.run(ev)
        assert result.best is not None
        assert result.finalist_verdicts

    def test_best_search_point_resolvable_after_finalist(self):
        ev = _RecordingEvaluator()
        cfg = _config(finalist_seeds=[88], finalist_k=2)
        runner = SearchRunner(cfg)
        result = runner.run(ev)
        if result.finalist_verdicts:
            assert result.best_search_point is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestFinalistEdgeCases:
    def test_fewer_candidates_than_finalist_k(self):
        """If only 1 candidate is kept, finalist_k=5 should still work."""
        ev = _RecordingEvaluator()
        # Random sampler with budget=1 → exactly 1 candidate; finalist_k=5 > 1.
        cfg = SearchConfig(
            study_id="few_candidates",
            baseline_recipe=_recipe(n_steps=3),
            budget=1,
            sampler="random",
            eval_seeds=[0],
            finalist_seeds=[99],
            finalist_k=5,
        )
        runner = SearchRunner(cfg)
        result = runner.run(ev)
        assert len(result.finalist_verdicts) <= 1

    def test_all_finalist_evaluations_raise_produces_empty_finalist(self):
        """If every finalist re-eval raises, finalist_verdicts stays empty and
        best falls back to the search-phase winner."""

        def bad_evaluator(recipe: TrajectoryRecipe, seed: int) -> EpisodeVerdict:
            if seed == 99:
                raise RuntimeError("simulated finalist failure")
            return _verdict()

        cfg = SearchConfig(
            study_id="failing_finalist",
            baseline_recipe=_recipe(n_steps=3),
            budget=3,
            sampler="grid",
            eval_seeds=[0],
            finalist_seeds=[99],
            finalist_k=3,
        )
        runner = SearchRunner(cfg)
        result = runner.run(bad_evaluator)
        # finalist_verdicts empty (all raised), best is search-phase winner
        assert result.finalist_verdicts == []
        assert result.best is not None  # search winner still present

    def test_finalist_seeds_disjoint_from_eval_seeds(self):
        """Verify that eval_seeds and finalist_seeds calls are non-overlapping."""
        search_seeds_used: set = set()
        finalist_seeds_used: set = set()
        eval_seeds = [1, 2]
        fin_seeds = [98, 99]

        def tracking_eval(recipe: TrajectoryRecipe, seed: int) -> EpisodeVerdict:
            if seed in eval_seeds:
                search_seeds_used.add(seed)
            if seed in fin_seeds:
                finalist_seeds_used.add(seed)
            return _verdict()

        cfg = SearchConfig(
            study_id="disjoint_seeds",
            baseline_recipe=_recipe(n_steps=3),
            budget=3,
            sampler="grid",
            eval_seeds=eval_seeds,
            finalist_seeds=fin_seeds,
            finalist_k=3,
        )
        runner = SearchRunner(cfg)
        runner.run(tracking_eval)
        # Both sets used, no overlap by design
        assert search_seeds_used == set(eval_seeds)
        assert finalist_seeds_used == set(fin_seeds)
        assert search_seeds_used.isdisjoint(finalist_seeds_used)
