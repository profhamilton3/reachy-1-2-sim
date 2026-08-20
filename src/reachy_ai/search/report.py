"""R12-807: Study report generation and cross-study comparison.

generate_report() loads all completed search trials for a study and produces
a structured StudyReport with ranked trial summaries.  compare_studies()
produces a ComparisonResult showing which of two studies found the better
candidate.

Both functions work from an ExperienceStore; they use
SearchRunner.load_prior_from_store() internally so the result is always
consistent with the ranking logic used by the search engine.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from reachy_ai.evaluation.base import EpisodeVerdict
from reachy_ai.evaluation.ranking import RankedCandidate, rank_candidates

RankKey = Tuple[float, ...]


# ---------------------------------------------------------------------------
# TrialSummary
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrialSummary:
    """Condensed view of one search trial, suitable for tabular display."""
    trial_id: str
    rank: int                       # 1-based, best-first
    is_valid: bool
    is_safe: bool
    is_successful: bool
    rank_key: RankKey
    ranking_scores: Dict[str, float]
    search_point: Dict[str, float]  # empty dict if metadata not present
    violations_count: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def one_line(self) -> str:
        scores = self.ranking_scores
        acc = scores.get("accuracy_score", 0.0)
        dur = scores.get("duration_score", 0.0)
        pt_str = (
            "  ".join(f"{k}={v:.3g}" for k, v in sorted(self.search_point.items()))
            if self.search_point
            else "(no search point)"
        )
        return (
            f"  #{self.rank}  trial={self.trial_id[:8]}"
            f"  valid={self.is_valid}  safe={self.is_safe}  ok={self.is_successful}"
            f"  acc={acc:.3f}  dur={dur:.3f}"
            f"  violations={self.violations_count}"
            f"  [{pt_str}]"
        )


# ---------------------------------------------------------------------------
# StudyReport
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StudyReport:
    """Aggregate summary of all completed search trials for one study."""
    study_id: str
    total_trials: int       # all trials recovered from the store
    safe_count: int         # trials where is_safe=True
    successful_count: int   # trials where is_successful=True
    best: Optional[TrialSummary]
    top: List[TrialSummary] # top N ranked, best-first
    generated_at: str       # ISO-8601 UTC timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_id": self.study_id,
            "generated_at": self.generated_at,
            "total_trials": self.total_trials,
            "safe_count": self.safe_count,
            "successful_count": self.successful_count,
            "best": self.best.to_dict() if self.best else None,
            "top": [t.to_dict() for t in self.top],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary_text(self) -> str:
        lines = [
            f"Study: {self.study_id}",
            f"Generated: {self.generated_at}",
            f"Trials: {self.total_trials}  safe: {self.safe_count}"
            f"  successful: {self.successful_count}",
        ]
        if not self.top:
            lines.append("  (no completed search trials found)")
            return "\n".join(lines)
        lines.append(f"Top {len(self.top)} candidate(s):")
        for ts in self.top:
            lines.append(ts.one_line())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ComparisonResult:
    """Side-by-side comparison of the best candidates from two studies."""
    study_a: str
    study_b: str
    report_a: StudyReport
    report_b: StudyReport
    winner: Optional[str]           # "a", "b", or None (tie / no candidates)
    rank_delta: Optional[RankKey]   # best_a.rank_key − best_b.rank_key, element-wise
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_a": self.study_a,
            "study_b": self.study_b,
            "winner": self.winner,
            "rank_delta": list(self.rank_delta) if self.rank_delta is not None else None,
            "summary": self.summary,
            "report_a": self.report_a.to_dict(),
            "report_b": self.report_b.to_dict(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary_text(self) -> str:
        lines = [
            f"Comparison: '{self.study_a}'  vs  '{self.study_b}'",
            f"Winner: {self.winner or 'none (tie or no candidates)'}",
        ]
        if self.rank_delta is not None:
            tier_names = [
                "is_valid", "is_safe", "is_successful",
                "accuracy", "effort", "duration",
            ]
            delta_parts = [
                f"{name}={delta:+.3f}"
                for name, delta in zip(tier_names, self.rank_delta)
                if delta != 0.0
            ]
            lines.append("Rank delta (A − B): " + (", ".join(delta_parts) or "equal"))
        lines.append("")
        lines.append("--- Study A ---")
        lines.append(self.report_a.summary_text())
        lines.append("")
        lines.append("--- Study B ---")
        lines.append(self.report_b.summary_text())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(store: Any, study_id: str, *, top_n: int = 5) -> StudyReport:
    """Build a StudyReport for study_id from the given ExperienceStore.

    Args:
        store:    An open ExperienceStore (or anything exposing list_trials()).
        study_id: Identifier of the study to report on.
        top_n:    Maximum number of top candidates to include in the report.

    Returns:
        StudyReport with ranked summaries; top is empty if the study has no
        completed search trials with recoverable metadata.
    """
    from reachy_ai.search.runner import SearchRunner

    prior = SearchRunner.load_prior_from_store(store, study_id)
    candidates = [
        RankedCandidate(verdict=v, trial_id=v.trial_id or v.episode_id)
        for _, v in prior
    ]
    ranked = rank_candidates(candidates)

    safe_count = sum(1 for _, v in prior if v.is_safe)
    successful_count = sum(1 for _, v in prior if v.is_successful)

    point_map: Dict[str, Dict[str, float]] = {
        (v.trial_id or v.episode_id): pt
        for pt, v in prior
    }

    summaries: List[TrialSummary] = []
    for rank, rc in enumerate(ranked, 1):
        v = rc.verdict
        summaries.append(TrialSummary(
            trial_id=rc.trial_id,
            rank=rank,
            is_valid=v.is_valid,
            is_safe=v.is_safe,
            is_successful=v.is_successful,
            rank_key=v.rank_key,
            ranking_scores=dict(v.ranking_scores),
            search_point=dict(point_map.get(rc.trial_id, {})),
            violations_count=len(v.violations),
        ))

    top = summaries[:top_n]
    best = summaries[0] if summaries else None

    return StudyReport(
        study_id=study_id,
        total_trials=len(prior),
        safe_count=safe_count,
        successful_count=successful_count,
        best=best,
        top=top,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def compare_studies(
    store: Any,
    study_a: str,
    study_b: str,
    *,
    top_n: int = 5,
) -> ComparisonResult:
    """Compare the best candidates from two studies in the same store.

    Args:
        store:   An open ExperienceStore.
        study_a: First study identifier.
        study_b: Second study identifier.
        top_n:   Passed to generate_report() for each study.

    Returns:
        ComparisonResult with winner ("a", "b", or None) and element-wise
        rank_delta (best_a.rank_key − best_b.rank_key).
    """
    report_a = generate_report(store, study_a, top_n=top_n)
    report_b = generate_report(store, study_b, top_n=top_n)

    best_a = report_a.best
    best_b = report_b.best

    if best_a is None and best_b is None:
        winner = None
        rank_delta = None
        summary = "Both studies have no completed search trials."
    elif best_a is None:
        winner = "b"
        rank_delta = None
        summary = f"Study '{study_b}' wins — study '{study_a}' has no candidates."
    elif best_b is None:
        winner = "a"
        rank_delta = None
        summary = f"Study '{study_a}' wins — study '{study_b}' has no candidates."
    else:
        rk_a = best_a.rank_key
        rk_b = best_b.rank_key
        rank_delta = tuple(a - b for a, b in zip(rk_a, rk_b))
        if rk_a > rk_b:
            winner = "a"
            summary = f"Study '{study_a}' wins with a better best candidate."
        elif rk_b > rk_a:
            winner = "b"
            summary = f"Study '{study_b}' wins with a better best candidate."
        else:
            winner = None
            summary = "Both studies produced equally-ranked best candidates."

    return ComparisonResult(
        study_a=study_a,
        study_b=study_b,
        report_a=report_a,
        report_b=report_b,
        winner=winner,
        rank_delta=rank_delta,
        summary=summary,
    )
