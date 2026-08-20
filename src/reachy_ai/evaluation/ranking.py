"""R12-804: Lexicographic candidate ranking.

Active tier order (highest first):
  1. is_valid       — NaN/INVALID episodes are strictly worse than any valid one
  2. is_safe        — any forbidden contact or unintended toggle beats every unsafe
  3. is_successful  — task success beats failure
  4. accuracy_score — task-specific accuracy [0, 1]
  5. effort_score   — inverse saturation fraction [0, 1]
  6. duration_score — shorter episode = better [0, 1]

clearance_score and smoothness_score are excluded from rank_key because they
require per-step snapshot data not yet available from EpisodeResult alone.

A faster collision can never outrank a slower safe path because safety (tier 2)
is always evaluated before speed (tier 6).
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

from reachy_ai.evaluation.base import EvaluationPolicy, EpisodeVerdict


@dataclasses.dataclass
class RankedCandidate:
    """An episode verdict paired with its trial identifier for ranking."""
    verdict: EpisodeVerdict
    trial_id: str
    recipe_id: Optional[str] = None

    @property
    def rank_key(self):
        """Tuple for sorted() / max() — higher value = better candidate."""
        return self.verdict.rank_key


def rank_candidates(
    candidates: List[RankedCandidate],
    policy: Optional[EvaluationPolicy] = None,
) -> List[RankedCandidate]:
    """Return candidates sorted best-first by lexicographic tier ordering.

    Ties within a tier are broken by the next tier; identical tier tuples
    preserve the input order (stable sort).

    Args:
        candidates: Evaluated episodes to rank.
        policy:     Unused currently; reserved for policy-specific tie-breaking.

    Returns:
        New list sorted best → worst.
    """
    return sorted(candidates, key=lambda c: c.rank_key, reverse=True)


def best_candidate(
    candidates: List[RankedCandidate],
    policy: Optional[EvaluationPolicy] = None,
) -> Optional[RankedCandidate]:
    """Return the single best candidate, or None if the list is empty."""
    if not candidates:
        return None
    return rank_candidates(candidates, policy)[0]


def dominated_by(
    candidate: RankedCandidate,
    reference: RankedCandidate,
) -> bool:
    """True if reference is strictly better than candidate in every tier.

    Used for early pruning in the search engine (PR 8.5): if a new candidate
    is dominated by the current best, it can be discarded without further
    evaluation.
    """
    ck = candidate.rank_key
    rk = reference.rank_key
    return all(r >= c for r, c in zip(rk, ck)) and any(r > c for r, c in zip(rk, ck))


def explain_ranking(candidates: List[RankedCandidate]) -> str:
    """Return a human-readable ranking explanation for a sorted candidate list."""
    ranked = rank_candidates(candidates)
    lines = [f"Ranking ({len(ranked)} candidate(s)):"]
    for i, rc in enumerate(ranked, 1):
        v = rc.verdict
        lines.append(
            f"  {i}. trial={rc.trial_id[:8]}  "
            f"valid={v.is_valid}  safe={v.is_safe}  success={v.is_successful}  "
            f"accuracy={v.ranking_scores.get('accuracy_score', 0):.3f}  "
            f"duration_score={v.ranking_scores.get('duration_score', 0):.3f}  "
            f"violations={len(v.violations)}"
        )
    return "\n".join(lines)
