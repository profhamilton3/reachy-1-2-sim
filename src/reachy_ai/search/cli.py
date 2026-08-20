"""R12-805: CLI entry point for the bounded parameter search engine.

Usage
-----
Dry-run (no physics, synthetic evaluator):

    python -m reachy_ai.search.cli run \\
        --baseline recipes/baseline_control.json \\
        --study-id study_001 \\
        --budget 10 \\
        --sampler grid

Resume a previous search (loading prior verdicts from the store):

    python -m reachy_ai.search.cli resume \\
        --study-id study_001 \\
        --db runs/study_001.db \\
        --budget 5

Show the current ranking for a study:

    python -m reachy_ai.search.cli show \\
        --study-id study_001 \\
        --db runs/study_001.db

All subcommands default to dry-run mode: the built-in synthetic evaluator
returns a verdict with quality scores derived from the normalised parameter
values.  For real evaluation, integrate SearchRunner.run() directly and
supply a task-specific evaluate_fn.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Optional

from reachy_ai.evaluation.base import EpisodeVerdict
from reachy_ai.evaluation.ranking import explain_ranking
from reachy_ai.motion.recipe import TrajectoryRecipe
from reachy_ai.search.convergence import ConvergenceChecker
from reachy_ai.search.report import compare_studies, generate_report
from reachy_ai.search.runner import SearchConfig, SearchRunner, apply_best_to_recipe


# ---------------------------------------------------------------------------
# Built-in dry-run evaluator
# ---------------------------------------------------------------------------

def _dry_run_evaluate(recipe: TrajectoryRecipe) -> EpisodeVerdict:
    """Synthetic evaluator: always succeeds; quality derived from param values."""
    scores: list[float] = []
    for spec in recipe.bounded_parameters.values():
        if not isinstance(spec, dict):
            continue
        lo = float(spec.get("min", 0.0))
        hi = float(spec.get("max", 1.0))
        val = float(spec.get("value", (lo + hi) / 2.0))
        if hi > lo:
            scores.append((val - lo) / (hi - lo))
    avg = sum(scores) / len(scores) if scores else 0.5

    eid = uuid.uuid4().hex
    return EpisodeVerdict(
        episode_id=eid,
        trial_id=eid,
        task_type=recipe.task_type or "dry_run",
        policy_version=1,
        is_valid=True,
        is_safe=True,
        is_successful=True,
        violations=[],
        metrics={"dry_run": 1.0, "avg_param_score": avg},
        ranking_scores={
            "accuracy_score": avg,
            "clearance_score": avg,
            "effort_score": 1.0,
            "smoothness_score": 1.0,
            "duration_score": avg,
        },
        explanation=f"[dry-run] recipe={recipe.recipe_id} avg_score={avg:.3f}",
    )


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> int:
    recipe = TrajectoryRecipe.load(args.baseline)
    config = SearchConfig(
        study_id=args.study_id,
        baseline_recipe=recipe,
        budget=args.budget,
        sampler=args.sampler,
        random_seed=args.seed,
        pruner=args.pruner,
        max_kept=args.max_kept,
        adaptive_top_k=args.adaptive_top_k,
        adaptive_exploration_weight=args.adaptive_exploration_weight,
    )
    runner = SearchRunner(config)

    store = _open_store(args.db) if args.db else None
    ctx = store.__enter__() if store is not None else None
    checker = _make_checker(args)

    try:
        result = runner.run(_dry_run_evaluate, store=ctx, convergence_checker=checker)
    finally:
        if store is not None:
            store.__exit__(None, None, None)

    _print_result(result)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: resume
# ---------------------------------------------------------------------------

def _cmd_resume(args: argparse.Namespace) -> int:
    if not args.db:
        print("ERROR: --db is required for resume", file=sys.stderr)
        return 1

    recipe_path: Optional[str] = getattr(args, "baseline", None)

    store = _open_store(args.db)
    with store as s:
        prior = SearchRunner.load_prior_from_store(s, args.study_id)
        if not prior:
            print(
                f"No prior search results found in '{args.db}' for "
                f"study '{args.study_id}'.",
                file=sys.stderr,
            )
            return 1

        if recipe_path:
            recipe = TrajectoryRecipe.load(recipe_path)
        else:
            # Reconstruct from first prior trial's recipe_json in the store
            rows = s.list_trials(args.study_id, limit=1)
            if not rows:
                print("ERROR: no trials in store to recover baseline recipe.", file=sys.stderr)
                return 1
            recipe = TrajectoryRecipe.from_json(rows[0]["recipe_json"])

        config = SearchConfig(
            study_id=args.study_id,
            baseline_recipe=recipe,
            budget=args.budget,
            sampler=args.sampler,
            random_seed=args.seed,
            pruner=args.pruner,
            max_kept=args.max_kept,
            adaptive_top_k=args.adaptive_top_k,
            adaptive_exploration_weight=args.adaptive_exploration_weight,
        )
        runner = SearchRunner(config)
        checker = _make_checker(args)
        result = runner.resume(
            _dry_run_evaluate, prior,
            extra_budget=args.budget,
            store=s,
            convergence_checker=checker,
        )

    _print_result(result)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: show
# ---------------------------------------------------------------------------

def _cmd_show(args: argparse.Namespace) -> int:
    if not args.db:
        print("ERROR: --db is required for show", file=sys.stderr)
        return 1

    store = _open_store(args.db)
    with store as s:
        prior = SearchRunner.load_prior_from_store(s, args.study_id)

    if not prior:
        print(f"No completed search trials for study '{args.study_id}'.")
        return 0

    from reachy_ai.evaluation.ranking import RankedCandidate, rank_candidates
    candidates = [
        RankedCandidate(verdict=v, trial_id=v.trial_id or v.episode_id)
        for _, v in prior
    ]
    ranked = rank_candidates(candidates)
    print(explain_ranking(ranked))
    if ranked:
        best = ranked[0].verdict
        print(
            f"\nBest: trial={ranked[0].trial_id[:8]}  "
            f"valid={best.is_valid}  safe={best.is_safe}  "
            f"success={best.is_successful}  "
            f"scores={json.dumps(best.ranking_scores, separators=(',', ':'))}"
        )
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Subcommand: export
# ---------------------------------------------------------------------------

def _cmd_export(args: argparse.Namespace) -> int:
    if not args.db:
        print("ERROR: --db is required for export", file=sys.stderr)
        return 1

    store = _open_store(args.db)
    with store as s:
        prior = SearchRunner.load_prior_from_store(s, args.study_id)
        if not prior:
            print(
                f"No prior search results found for study '{args.study_id}'.",
                file=sys.stderr,
            )
            return 1

        baseline_path: Optional[str] = getattr(args, "baseline", None)
        if baseline_path:
            baseline = TrajectoryRecipe.load(baseline_path)
        else:
            rows = s.list_trials(args.study_id, limit=1)
            if not rows:
                print("ERROR: no trials in store to recover baseline recipe.", file=sys.stderr)
                return 1
            baseline = TrajectoryRecipe.from_json(rows[0]["recipe_json"])

    # Reconstruct result enough to call apply_best_to_recipe
    from reachy_ai.evaluation.ranking import RankedCandidate, rank_candidates
    from reachy_ai.search.runner import SearchResult

    candidates = [
        RankedCandidate(verdict=v, trial_id=v.trial_id or v.episode_id)
        for _, v in prior
    ]
    ranked = rank_candidates(candidates)
    point_map = {(v.trial_id or v.episode_id): pt for pt, v in prior}

    best_candidate = ranked[0] if ranked else None
    best_sp = point_map.get(best_candidate.trial_id) if best_candidate else None

    result = SearchResult(
        study_id=args.study_id,
        best=best_candidate,
        ranked=ranked,
        kept=ranked,
        pruned=[],
        trials_run=0,
        trials_skipped=len(prior),
        best_search_point=best_sp,
    )

    recipe = apply_best_to_recipe(result, baseline)
    if recipe is None:
        print("No best recipe available (no valid candidates).", file=sys.stderr)
        return 1

    fmt = getattr(args, "format", "json")
    text = recipe.to_json() if fmt == "json" else recipe.to_yaml()

    output_path: Optional[str] = getattr(args, "output", None)
    if output_path:
        import pathlib
        pathlib.Path(output_path).write_text(text)
        print(f"Best recipe written to: {output_path}")
        print(f"  recipe_id={recipe.recipe_id}")
        if recipe.bounded_parameters:
            for name, spec in recipe.bounded_parameters.items():
                val = spec.get("value") if isinstance(spec, dict) else spec
                print(f"  {name}={val}")
    else:
        print(text)

    return 0


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------

def _cmd_report(args: argparse.Namespace) -> int:
    if not args.db:
        print("ERROR: --db is required for report", file=sys.stderr)
        return 1
    store = _open_store(args.db)
    with store as s:
        report = generate_report(s, args.study_id, top_n=args.top_n)
    fmt = getattr(args, "format", "text")
    if fmt == "json":
        print(report.to_json())
    else:
        print(report.summary_text())
    return 0


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------

def _cmd_compare(args: argparse.Namespace) -> int:
    if not args.db:
        print("ERROR: --db is required for compare", file=sys.stderr)
        return 1
    store = _open_store(args.db)
    with store as s:
        result = compare_studies(s, args.study_a, args.study_b, top_n=args.top_n)
    fmt = getattr(args, "format", "text")
    if fmt == "json":
        print(result.to_json())
    else:
        print(result.summary_text())
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checker(args: argparse.Namespace) -> Optional[ConvergenceChecker]:
    """Build a ConvergenceChecker from CLI args, or None if not requested."""
    window = getattr(args, "convergence_window", None)
    if not window:
        return None
    min_trials = getattr(args, "min_trials", 5) or 5
    return ConvergenceChecker(window=window, min_trials=min_trials)


def _open_store(db_path: str):
    from reachy_ai.experience.store import ExperienceStore
    return ExperienceStore.open(db_path)


def _print_result(result) -> None:
    conv_tag = ""
    if result.converged:
        conv_tag = f"  converged=True(+{result.trials_since_improvement})"
    print(
        f"study={result.study_id}  "
        f"trials_run={result.trials_run}  "
        f"skipped={result.trials_skipped}  "
        f"kept={len(result.kept)}  "
        f"pruned={len(result.pruned)}"
        f"{conv_tag}"
    )
    if result.best:
        v = result.best.verdict
        print(
            f"Best: trial={result.best.trial_id[:8]}  "
            f"valid={v.is_valid}  safe={v.is_safe}  success={v.is_successful}"
        )
        print(f"  Ranking scores: {json.dumps(v.ranking_scores, separators=(',', ':'))}")
        print(f"  Explanation: {v.explanation}")
    else:
        print("No candidates produced.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reachy_ai.search.cli",
        description="Bounded parameter search engine for trajectory recipes.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--study-id", required=True, help="Unique study identifier")
    shared.add_argument("--db", default=None, help="ExperienceStore SQLite path")
    shared.add_argument("--budget", type=int, default=20, help="Max new trials to run")
    shared.add_argument("--sampler", choices=["grid", "random", "adaptive"], default="grid")
    shared.add_argument("--seed", type=int, default=0, help="RNG seed for random sampler")
    shared.add_argument("--pruner", choices=["dominance", "budget", "none"], default="dominance")
    shared.add_argument("--max-kept", type=int, default=10, dest="max_kept")
    shared.add_argument(
        "--adaptive-top-k",
        type=int,
        default=3,
        dest="adaptive_top_k",
        metavar="K",
        help="Top-K prior points to exploit when --sampler=adaptive (default: 3).",
    )
    shared.add_argument(
        "--adaptive-exploration",
        type=float,
        default=0.3,
        dest="adaptive_exploration_weight",
        metavar="W",
        help=(
            "Exploration fraction for adaptive sampler: 0.0=pure exploit, "
            "1.0=pure explore (default: 0.3)."
        ),
    )
    shared.add_argument(
        "--convergence-window",
        type=int,
        default=None,
        dest="convergence_window",
        metavar="N",
        help=(
            "Stop early when the best rank_key has not improved for N consecutive "
            "trials.  Omit to disable early stopping (default: run full budget)."
        ),
    )
    shared.add_argument(
        "--min-trials",
        type=int,
        default=5,
        dest="min_trials",
        metavar="M",
        help="Minimum trials before convergence is checked (default: 5).",
    )

    run_p = sub.add_parser("run", parents=[shared], help="Start a new search run")
    run_p.add_argument("--baseline", required=True, help="Path to baseline recipe JSON/YAML")

    resume_p = sub.add_parser("resume", parents=[shared], help="Resume from store")
    resume_p.add_argument("--baseline", default=None, help="Override baseline recipe (optional)")

    show_p = sub.add_parser("show", parents=[shared], help="Show ranking from store")

    # export: reconstruct the best recipe from the store and write it
    export_p = sub.add_parser(
        "export",
        help="Export the best-performing recipe from a study to a file or stdout",
    )
    export_p.add_argument("--study-id", required=True, dest="study_id")
    export_p.add_argument("--db", default=None, help="ExperienceStore SQLite path")
    export_p.add_argument(
        "--baseline", default=None,
        help="Baseline recipe path (recovered from store if omitted)",
    )
    export_p.add_argument(
        "--output", default=None, metavar="PATH",
        help="Write recipe to this file (print to stdout if omitted)",
    )
    export_p.add_argument(
        "--format", choices=["json", "yaml"], default="json",
        help="Output format (default: json)",
    )

    # report and compare do not use the convergence flags from shared
    _report_shared = argparse.ArgumentParser(add_help=False)
    _report_shared.add_argument("--db", default=None, help="ExperienceStore SQLite path")
    _report_shared.add_argument(
        "--top-n", type=int, default=5, dest="top_n",
        help="Number of top candidates to include (default: 5)",
    )
    _report_shared.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text)",
    )

    report_p = sub.add_parser(
        "report", parents=[_report_shared],
        help="Generate a structured report for one study",
    )
    report_p.add_argument("--study-id", required=True, dest="study_id")

    compare_p = sub.add_parser(
        "compare", parents=[_report_shared],
        help="Compare the best candidates from two studies",
    )
    compare_p.add_argument("--study-a", required=True, dest="study_a")
    compare_p.add_argument("--study-b", required=True, dest="study_b")

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.subcommand == "run":
        return _cmd_run(args)
    if args.subcommand == "resume":
        return _cmd_resume(args)
    if args.subcommand == "show":
        return _cmd_show(args)
    if args.subcommand == "export":
        return _cmd_export(args)
    if args.subcommand == "report":
        return _cmd_report(args)
    if args.subcommand == "compare":
        return _cmd_compare(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
