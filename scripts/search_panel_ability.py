#!/usr/bin/env python3
"""Search one panel ability's bounded parameters, offline (#91).

    python3 scripts/search_panel_ability.py wave \
        --scene scenes/FWDCenterLabSivaPool.yaml \
        --budget 8 --eval-seeds 0,1,2 --finalist-seeds 7,8 \
        --db runs/panel_wave_study.db --export runs/wave_best.yaml

Runs entirely offline: no SDK, no simulator server, no network.  Each trial
builds the candidate recipe's commands, flies them in a headless MuJoCo core,
and scores the episode with the ability's evaluator from
`reachy_ai.evaluation.panel_routes`.

WHAT THIS PRINTS, AND WHY IT IS NOT ONE NUMBER
----------------------------------------------
Every candidate is evaluated on several seeds and the spread across them is
reported next to the mean, because #20 and #19 were both single-seed bugs and a
best recipe picked from one rollout is a recipe picked from one rollout.  The
finalist stage then re-evaluates the top candidates on seeds the search never
saw, and the winner of THAT is what gets exported — the standard answer to the
winner's curse, and the reason `--finalist-seeds` defaults to non-empty here.

FAILED TRIALS ARE KEPT.  A failed trial is evidence about the bound, and the
store records it like any other.  Nothing in this script prunes them out.

NOTHING HERE RUNS FROM THE PANEL, and the panel cannot reach it: no web module
imports `reachy_ai.search`, which `tests/unit/test_panel_recipes.py` enforces
across the whole package.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
for _p in (_REPO / "src", _REPO / "native_mujoco"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reachy_ai.evaluation.panel_offline import OfflineRouteRunner, executable
from reachy_ai.evaluation.panel_routes import (ABILITY_ROUTES,
                                                 align_to_parameters)
from reachy_ai.motion.recipe import TrajectoryRecipe
from reachy_ai.search.runner import SearchConfig, SearchRunner, apply_best_to_recipe

DEFAULT_MODEL = _REPO / "native_mujoco" / "model" / "reachy_1_2.xml"
DEFAULT_SCENE = _REPO / "scenes" / "FWDCenterLabSivaPool.yaml"
RECIPES = _REPO / "recipes" / "panel"


def _seeds(raw: str) -> list:
    return [int(s) for s in raw.split(",") if s.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ability", choices=sorted(ABILITY_ROUTES),
                   help="which ability to search")
    p.add_argument("--baseline", default="",
                   help="recipe to start from (default: the ability's baseline)")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--scene", default=str(DEFAULT_SCENE))
    p.add_argument("--study-id", default="", dest="study_id")
    p.add_argument("--db", default="", help="ExperienceStore SQLite path")
    p.add_argument("--budget", type=int, default=8)
    p.add_argument("--sampler", choices=["grid", "random", "adaptive"],
                   default="random")
    p.add_argument("--seed", type=int, default=0, help="sampler RNG seed")
    p.add_argument("--eval-seeds", default="0,1,2", dest="eval_seeds",
                   help="simulation seeds each candidate is evaluated on")
    p.add_argument("--finalist-seeds", default="11,12", dest="finalist_seeds",
                   help="held-out seeds for the finalist re-evaluation; empty "
                        "disables the stage, which re-opens the winner's curse")
    p.add_argument("--finalist-k", type=int, default=3, dest="finalist_k")
    p.add_argument("--export", default="",
                   help="write the winning recipe to this YAML path")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    ok, why = executable(args.ability)
    if not ok:
        print(f"cannot search {args.ability}: {why}", file=sys.stderr)
        return 2

    baseline_path = args.baseline or str(
        RECIPES / f"{args.ability}_baseline_v1.yaml")
    baseline = TrajectoryRecipe.load(baseline_path)
    study_id = args.study_id or f"panel-{args.ability}"

    runner = OfflineRouteRunner(args.ability, args.model, args.scene,
                                recipe_arm=baseline.arm or "right")

    config = SearchConfig(
        study_id=study_id,
        baseline_recipe=baseline,
        budget=args.budget,
        sampler=args.sampler,
        random_seed=args.seed,
        eval_seeds=_seeds(args.eval_seeds),
        # The spread across seeds is what decides, not the best rollout.
        eval_aggregation="pessimistic",
        finalist_k=args.finalist_k,
        finalist_seeds=_seeds(args.finalist_seeds),
    )
    search = SearchRunner(config)

    store = None
    ctx = None
    if args.db:
        from reachy_ai.experience.store import ExperienceStore
        os.makedirs(os.path.dirname(os.path.abspath(args.db)) or ".",
                    exist_ok=True)
        store = ExperienceStore.open(args.db)
        ctx = store.__enter__()

    try:
        result = search.run(runner.evaluate_fn(), store=ctx)
    finally:
        if store is not None:
            store.__exit__(None, None, None)

    _report(args, result, baseline)
    return 0


def _report(args, result, baseline) -> None:
    print(f"\n=== {args.ability}: {result.trials_run} trial(s) run, "
          f"{result.trials_skipped} reused ===")
    print(f"scene   : {args.scene}")
    print(f"seeds   : eval {args.eval_seeds}  finalist {args.finalist_seeds or 'none'}")

    if result.best is None:
        print("\nNo candidate was valid, safe and successful.  That is a "
              "result, not an error: the failed trials are in the store and "
              "each one is evidence about a bound.")
        return

    verdict = result.best.verdict
    print(f"\nbest    : {result.best.trial_id[:8]}  "
          f"successful={verdict.is_successful}")
    print(f"          {dict(result.best_search_point or {})}")

    # THE SPREAD, NOT JUST THE MEAN.  `_aggregate_verdicts` records a `_std`
    # for every ranking score; a mean with no spread beside it is how a
    # single-rollout winner comes to look like a finding.
    spreads = {k: v for k, v in verdict.metrics.items() if k.endswith("_std")}
    print("\nacross seeds:")
    for key in sorted(verdict.ranking_scores):
        mean = verdict.ranking_scores[key]
        std = spreads.get(f"{key}_std")
        print(f"  {key:16s} {mean:.4f}"
              + (f"  +/- {std:.4f}" if std is not None
                 else "  (single seed — no spread to report)"))

    if result.finalist_verdicts:
        print(f"\nfinalist re-evaluation on held-out seeds "
              f"({len(result.finalist_verdicts)} candidate(s)):")
        for trial_id, fv in result.finalist_verdicts:
            print(f"  {trial_id[:8]}  successful={fv.is_successful}  "
                  f"accuracy={fv.ranking_scores.get('accuracy_score', 0.0):.4f}")
    else:
        print("\nNO FINALIST STAGE RAN, so the winner was chosen on the same "
              "seeds it was searched on and the winner's curse is unaddressed.")

    failures = [c for c in result.ranked if not c.verdict.is_successful]
    print(f"\n{len(failures)} of {len(result.ranked)} candidate(s) failed, and "
          "all of them are kept.")

    if args.export:
        winner = apply_best_to_recipe(result, baseline)
        if winner is None:
            print("\nnothing to export: no candidate won.")
            return
        # The step list follows the winning parameters, or the exported file
        # would declare a cycle count its own sequence does not have.
        winner = align_to_parameters(winner)
        pathlib.Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.export).write_text(winner.to_yaml())
        print(f"\nexported: {args.export}")
        print("NOT PROMOTED.  Promotion is a separate, deliberate act; the "
              "panel retrieves promoted trials only, so nothing in this file "
              "can reach an operator until somebody promotes it.")


if __name__ == "__main__":
    raise SystemExit(main())
