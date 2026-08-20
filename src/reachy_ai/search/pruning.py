"""R12-805: Candidate pruning strategies for the bounded search engine.

DominancePruner discards candidates that are strictly worse than some
already-kept candidate in every ranking tier (Pareto dominance).  It
preserves the full Pareto front — candidates that are better in at least
one tier and worse in none of the others are always kept.

BudgetPruner is a memory-cap: after ranking, only the top max_kept candidates
are retained regardless of dominance relationships.

CompositePruner chains strategies so the output of one feeds as input to the
next.  Typical use: DominancePruner first, BudgetPruner second.
"""

from __future__ import annotations

import dataclasses
from typing import List

from reachy_ai.evaluation.ranking import RankedCandidate, dominated_by, rank_candidates


@dataclasses.dataclass
class PruningResult:
    kept: List[RankedCandidate]
    pruned: List[RankedCandidate]


class PruningStrategy:
    def prune(self, candidates: List[RankedCandidate]) -> PruningResult:
        raise NotImplementedError


class NoPruner(PruningStrategy):
    def prune(self, candidates: List[RankedCandidate]) -> PruningResult:
        return PruningResult(kept=list(candidates), pruned=[])


class DominancePruner(PruningStrategy):
    """Remove candidates dominated by any higher-ranked (already-kept) candidate.

    The input is re-ranked before pruning so the dominance check is always
    made against candidates that are objectively better, never worse.  The
    best candidate is unconditionally kept regardless of how it compares to
    others (nothing can dominate the top-ranked entry from above).
    """

    def prune(self, candidates: List[RankedCandidate]) -> PruningResult:
        ranked = rank_candidates(candidates)
        kept: List[RankedCandidate] = []
        pruned: List[RankedCandidate] = []
        for candidate in ranked:
            if any(dominated_by(candidate, ref) for ref in kept):
                pruned.append(candidate)
            else:
                kept.append(candidate)
        return PruningResult(kept=kept, pruned=pruned)


class BudgetPruner(PruningStrategy):
    """Keep at most max_kept candidates (highest-ranked first)."""

    def __init__(self, max_kept: int) -> None:
        if max_kept < 1:
            raise ValueError(f"BudgetPruner.max_kept must be >= 1, got {max_kept}")
        self.max_kept = max_kept

    def prune(self, candidates: List[RankedCandidate]) -> PruningResult:
        ranked = rank_candidates(candidates)
        kept = ranked[: self.max_kept]
        pruned = ranked[self.max_kept :]
        return PruningResult(kept=kept, pruned=pruned)


class CompositePruner(PruningStrategy):
    """Apply multiple strategies in sequence; total pruned = union of each stage."""

    def __init__(self, strategies: List[PruningStrategy]) -> None:
        self.strategies = strategies

    def prune(self, candidates: List[RankedCandidate]) -> PruningResult:
        current = list(candidates)
        all_pruned: List[RankedCandidate] = []
        for strategy in self.strategies:
            result = strategy.prune(current)
            current = result.kept
            all_pruned.extend(result.pruned)
        return PruningResult(kept=current, pruned=all_pruned)
