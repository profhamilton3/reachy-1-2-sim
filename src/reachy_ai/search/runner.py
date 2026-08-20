"""R12-805: SearchRunner — bounded parameter search with optional store persistence.

SearchRunner iterates over search points suggested by a Sampler, calls a
caller-supplied evaluate_fn for each, ranks all candidates lexicographically,
and applies a PruningStrategy to the result.

Store integration is opt-in.  When an ExperienceStore is provided, each trial
is recorded with the search_point and verdict stored in optimizer_metadata so
the search can be resumed later via load_prior_from_store() + run().

evaluate_fn contract
--------------------
    def evaluate_fn(recipe: TrajectoryRecipe) -> EpisodeVerdict:
        ...

The function receives a candidate recipe (bounded_parameters updated with the
current search point) and must return an EpisodeVerdict.  It may run a real
simulation or return a synthetic verdict for dry-run use.  Exceptions from
evaluate_fn are caught: the trial is marked failed in the store (if any) and
the search point is added to already_done so it is not re-tried.

Resume workflow
---------------
    prior = SearchRunner.load_prior_from_store(store, study_id)
    result = runner.run(evaluate_fn, prior_results=prior, store=store)
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from reachy_ai.evaluation.base import EpisodeVerdict
from reachy_ai.evaluation.ranking import RankedCandidate, rank_candidates
from reachy_ai.motion.recipe import TrajectoryRecipe
from reachy_ai.search.convergence import ConvergenceChecker
from reachy_ai.search.pruning import (
    BudgetPruner, CompositePruner, DominancePruner, NoPruner, PruningStrategy,
)
from reachy_ai.search.samplers import AdaptiveSampler, GridSampler, RandomSampler, Sampler
from reachy_ai.search.space import SearchPoint, SearchSpace

EvaluateFn = Callable[[TrajectoryRecipe], EpisodeVerdict]
PriorResults = List[Tuple[SearchPoint, EpisodeVerdict]]


@dataclasses.dataclass
class SearchConfig:
    """Parameters that govern one search run."""
    study_id: str
    baseline_recipe: TrajectoryRecipe
    budget: int = 20
    sampler: str = "grid"     # "grid" | "random" | "adaptive"
    random_seed: int = 0
    pruner: str = "dominance" # "dominance" | "budget" | "none"
    max_kept: int = 10        # used by "budget" pruner
    # Adaptive sampler settings (ignored for grid/random)
    adaptive_top_k: int = 3
    adaptive_exploration_weight: float = 0.3


@dataclasses.dataclass
class SearchResult:
    """Output of one search run (or resume)."""
    study_id: str
    best: Optional[RankedCandidate]
    ranked: List[RankedCandidate]   # all candidates before pruning, best-first
    kept: List[RankedCandidate]     # after pruning
    pruned: List[RankedCandidate]   # discarded by pruner
    trials_run: int                 # evaluations performed this run
    trials_skipped: int             # prior results reused without re-evaluation
    converged: bool = False         # True if convergence checker fired early
    trials_since_improvement: int = 0  # consecutive trailing non-improving trials
    best_search_point: Optional[SearchPoint] = None  # search point of the best candidate


class SearchRunner:
    """Bounded parameter search engine."""

    def __init__(self, config: SearchConfig) -> None:
        self._config = config
        self._space = SearchSpace.from_recipe(config.baseline_recipe)
        self._sampler: Sampler = _make_sampler(config)
        self._pruner: PruningStrategy = _make_pruner(config)

    @property
    def space(self) -> SearchSpace:
        return self._space

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        evaluate_fn: EvaluateFn,
        *,
        prior_results: Optional[PriorResults] = None,
        store: Optional[Any] = None,        # ExperienceStore (imported lazily)
        task_spec: Optional[Any] = None,    # TaskSpec for store recording
        episode_config: Optional[Any] = None,  # EpisodeConfig for store recording
        convergence_checker: Optional[ConvergenceChecker] = None,
    ) -> SearchResult:
        """Run the search, optionally extending prior results.

        Args:
            evaluate_fn:          Caller-provided evaluator — recipe → EpisodeVerdict.
            prior_results:        (search_point, verdict) pairs from a previous run or
                                  loaded via load_prior_from_store().  These seed the
                                  already_done set so their points are not re-sampled.
            store:                Optional ExperienceStore.  When provided, each trial
                                  is persisted with search_point and verdict_json in
                                  optimizer_metadata.
            task_spec:            Passed to store.create_trial(); defaults to a minimal
                                  synthetic TaskSpec derived from the baseline recipe.
            episode_config:       Passed to store.create_trial(); defaults to a minimal
                                  synthetic EpisodeConfig.
            convergence_checker:  Optional ConvergenceChecker.  When provided, the
                                  search loop calls checker.update() after every trial
                                  and breaks early when checker.is_converged() is True.
                                  Pass a freshly constructed checker; do not share
                                  instances across run() calls.
        """
        # Guard against a silently-degenerate grid search.  When sampler="grid"
        # and every searchable axis is continuous (no "step"), grid_points()
        # yields only the baseline, so the "study" is really N=1 — a common and
        # invisible source of invalid conclusions.  Fail loudly with an
        # actionable message instead of producing a one-trial study.
        if (
            self._config.sampler == "grid"
            and not self._space.is_empty()
            and self._space.num_grid_points() <= 1
        ):
            cont = self._space.continuous_axes()
            detail = (
                f"axes {list(cont)} have no 'step' and cannot be gridded"
                if cont
                else "all axes collapse to a single value"
            )
            raise ValueError(
                f"grid sampler would evaluate only the baseline for study "
                f"'{self._config.study_id}' ({self._space.num_grid_points()} "
                f"grid point(s)): {detail}. Add a 'step' to each searchable "
                f"parameter, or use sampler='random' or sampler='adaptive'."
            )

        prior = list(prior_results or [])
        already_done: List[SearchPoint] = [pt for pt, _ in prior]
        candidates: List[RankedCandidate] = [
            RankedCandidate(verdict=v, trial_id=v.trial_id or v.episode_id)
            for _, v in prior
        ]
        # Map trial_id → search_point for both prior and new trials so
        # apply_best_to_recipe() can materialise the winning parameters.
        trial_point_map: Dict[str, SearchPoint] = {
            (v.trial_id or v.episode_id): pt for pt, v in prior
        }

        # When sampler="adaptive" and prior is available, build AdaptiveSampler
        # from the ranked prior so exploitation targets the best-known points.
        # Falls back to the pre-built sampler (RandomSampler) when prior is empty.
        sampler: Sampler = self._sampler
        if self._config.sampler == "adaptive" and prior:
            ranked_prior = rank_candidates(candidates)
            point_map = {
                (v.trial_id or v.episode_id): pt for pt, v in prior
            }
            top_points = [
                point_map[rc.trial_id]
                for rc in ranked_prior
                if rc.trial_id in point_map
            ]
            sampler = AdaptiveSampler(
                top_points,
                top_k=self._config.adaptive_top_k,
                exploration_weight=self._config.adaptive_exploration_weight,
                seed=self._config.random_seed,
            )

        trials_run = 0
        budget = self._config.budget

        while trials_run < budget:
            points = sampler.suggest(self._space, already_done, n=1)
            if not points:
                break  # grid fully exhausted or sampler gave up

            pt = points[0]
            recipe = _apply_point(self._config.baseline_recipe, pt)

            store_trial_id: Optional[str] = None
            if store is not None:
                store_trial_id = _store_create(
                    store, self._config.study_id, recipe, pt, task_spec, episode_config
                )

            try:
                verdict = evaluate_fn(recipe)
            except Exception as exc:
                if store is not None and store_trial_id is not None:
                    _store_fail(store, store_trial_id, str(exc))
                already_done.append(pt)
                trials_run += 1
                _maybe_update_convergence(convergence_checker, candidates)
                if convergence_checker is not None and convergence_checker.is_converged():
                    break
                continue

            if store is not None and store_trial_id is not None:
                _store_complete(store, store_trial_id, verdict, pt)

            already_done.append(pt)
            # Trial identity MUST be unique per trial.  verdict.trial_id comes
            # from the evaluator/simulator and may be a constant — EpisodeRunner
            # sets it to `scene_revision or "standalone"` — which would collapse
            # every trial to one key in trial_point_map and make
            # best_search_point resolve to the wrong recipe.  Prefer the
            # store-assigned trial_id (guaranteed unique); fall back to a fresh
            # uuid when running store-less.  We do NOT trust verdict.trial_id.
            trial_id = store_trial_id or uuid.uuid4().hex
            verdict.trial_id = trial_id  # keep the verdict consistent downstream
            candidates.append(RankedCandidate(verdict=verdict, trial_id=trial_id))
            trial_point_map[trial_id] = pt
            trials_run += 1

            _maybe_update_convergence(convergence_checker, candidates)
            if convergence_checker is not None and convergence_checker.is_converged():
                break

        ranked = rank_candidates(candidates)
        pruning_result = self._pruner.prune(ranked)

        best_candidate = pruning_result.kept[0] if pruning_result.kept else None
        best_sp = trial_point_map.get(best_candidate.trial_id) if best_candidate else None

        return SearchResult(
            study_id=self._config.study_id,
            best=best_candidate,
            ranked=ranked,
            kept=pruning_result.kept,
            pruned=pruning_result.pruned,
            trials_run=trials_run,
            trials_skipped=len(prior),
            converged=convergence_checker.is_converged() if convergence_checker else False,
            trials_since_improvement=(
                convergence_checker.trials_since_improvement()
                if convergence_checker else 0
            ),
            best_search_point=best_sp,
        )

    # ------------------------------------------------------------------
    # Resume helper
    # ------------------------------------------------------------------

    def resume(
        self,
        evaluate_fn: EvaluateFn,
        prior_results: PriorResults,
        *,
        extra_budget: int,
        store: Optional[Any] = None,
        task_spec: Optional[Any] = None,
        episode_config: Optional[Any] = None,
        convergence_checker: Optional[ConvergenceChecker] = None,
    ) -> SearchResult:
        """Continue search from prior results with extra_budget new trials."""
        extended = dataclasses.replace(self._config, budget=extra_budget)
        runner = SearchRunner(extended)
        return runner.run(
            evaluate_fn,
            prior_results=prior_results,
            store=store,
            task_spec=task_spec,
            episode_config=episode_config,
            convergence_checker=convergence_checker,
        )

    # ------------------------------------------------------------------
    # Store-backed prior loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_prior_from_store(store: Any, study_id: str) -> PriorResults:
        """Read completed search trials from the store and rebuild prior results.

        Only SUCCEEDED and FAILED trials with both search_point and verdict_json
        in optimizer_metadata are included.  Trials recorded by other means
        (without optimizer_metadata populated by the runner) are silently skipped.

        limit=None requests every trial: truncating here would drop prior points
        from already_done and cause them to be re-evaluated on resume, defeating
        the dedup guarantee this method exists to provide.
        """
        rows = store.list_trials(study_id, limit=None)
        results: PriorResults = []
        for row in rows:
            status = row.get("status", "")
            if status not in ("SUCCEEDED", "FAILED"):
                continue
            try:
                metadata = json.loads(row.get("optimizer_metadata_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            search_point = metadata.get("search_point")
            verdict_json = metadata.get("verdict_json")
            if not search_point or not verdict_json:
                continue
            try:
                verdict = EpisodeVerdict.from_json(verdict_json)
            except Exception:
                continue
            # Canonicalise trial identity to the store's unique trial_id.  A
            # verdict persisted from the simulator may carry a constant
            # trial_id ("standalone"); using it would collapse every trial to
            # one identity in ranking, report, and export.  The store row's
            # trial_id is the authoritative per-trial key.
            row_trial_id = row.get("trial_id")
            if row_trial_id:
                verdict.trial_id = row_trial_id
            results.append((search_point, verdict))
        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_point(baseline: TrajectoryRecipe, point: SearchPoint) -> TrajectoryRecipe:
    """Return a new recipe with bounded_parameters values updated from point."""
    new_bp: Dict[str, Any] = {}
    for name, spec in baseline.bounded_parameters.items():
        if name in point:
            if isinstance(spec, dict):
                new_bp[name] = {**spec, "value": point[name]}
            else:
                new_bp[name] = point[name]
        else:
            new_bp[name] = spec
    return dataclasses.replace(
        baseline,
        recipe_id=f"{baseline.recipe_id}_s{uuid.uuid4().hex[:6]}",
        bounded_parameters=new_bp,
        source="search_candidate",
        parent_recipe_id=baseline.recipe_id,
    )


def _make_sampler(config: SearchConfig) -> Sampler:
    if config.sampler == "random":
        return RandomSampler(seed=config.random_seed)
    if config.sampler == "adaptive":
        # No prior at construction time; run() overrides with AdaptiveSampler
        # once prior_results are available.  Fall back to random when no prior.
        return RandomSampler(seed=config.random_seed)
    return GridSampler()


def _make_pruner(config: SearchConfig) -> PruningStrategy:
    if config.pruner == "budget":
        return BudgetPruner(max_kept=config.max_kept)
    if config.pruner == "none":
        return NoPruner()
    return DominancePruner()


def _store_create(
    store: Any,
    study_id: str,
    recipe: TrajectoryRecipe,
    point: SearchPoint,
    task_spec: Optional[Any],
    episode_config: Optional[Any],
) -> str:
    """Create a PENDING trial in the store. Returns store-assigned trial_id."""
    from reachy_ai.experience.models import (
        EpisodeConfig, SimulatorIdentity, TaskSpec,
    )
    ts = task_spec or TaskSpec(task_id=recipe.recipe_id, task_type=recipe.task_type or "search")
    cfg = episode_config or EpisodeConfig(simulator_identity=SimulatorIdentity())
    trial_id = store.create_trial(
        study_id,
        ts,
        recipe.to_json(),
        cfg,
        optimizer_metadata={"search_point": point},
    )
    store.start_trial(trial_id)
    return trial_id


def _store_complete(
    store: Any,
    trial_id: str,
    verdict: EpisodeVerdict,
    point: SearchPoint,
) -> None:
    """Persist a completed verdict to the store trial."""
    from reachy_ai.experience.models import EpisodeResult, EpisodeStatus
    status = (
        EpisodeStatus.SUCCEEDED
        if (verdict.is_valid and verdict.is_safe and verdict.is_successful)
        else EpisodeStatus.FAILED
    )
    result = EpisodeResult(
        episode_id=verdict.episode_id or trial_id,
        trial_id=trial_id,
        status=status,
        success=verdict.is_successful,
        metrics=verdict.metrics,
    )
    # Status + verdict_json are written in one transaction so a crash can never
    # leave a completed trial without its verdict (which load_prior_from_store
    # would silently skip, forcing a re-evaluation on resume).
    try:
        store.complete_trial(
            trial_id, result, optimizer_metadata={"verdict_json": verdict.to_json()}
        )
    except Exception:
        store.fail_trial(trial_id, "complete_trial failed")


def _store_fail(store: Any, trial_id: str, reason: str) -> None:
    try:
        store.fail_trial(trial_id, reason)
    except Exception:
        pass


def apply_best_to_recipe(
    result: "SearchResult",
    baseline: TrajectoryRecipe,
) -> Optional[TrajectoryRecipe]:
    """Return a new recipe with bounded_parameters updated from result.best_search_point.

    Returns None if result has no best candidate or no search_point recorded
    (e.g. all trials failed before any candidate was accepted).

    The returned recipe has:
    - recipe_id = "<baseline_id>_best_<trial_id[:8]>"
    - source = "search_best"
    - parent_recipe_id = baseline.recipe_id

    All other fields (primitive_sequence, limits, arm, etc.) are copied from
    the baseline unchanged.
    """
    if result.best is None or result.best_search_point is None:
        return None
    applied = _apply_point(baseline, result.best_search_point)
    return dataclasses.replace(
        applied,
        recipe_id=f"{baseline.recipe_id}_best_{result.best.trial_id[:8]}",
        source="search_best",
    )


def _maybe_update_convergence(
    checker: Optional[ConvergenceChecker],
    candidates: List[RankedCandidate],
) -> None:
    """Push the current best rank_key into the checker (no-op if no checker or
    no candidates yet)."""
    if checker is None or not candidates:
        return
    best_rk = max(c.rank_key for c in candidates)
    checker.update(best_rk)
