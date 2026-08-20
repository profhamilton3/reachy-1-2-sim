"""R12-806: Pareto-front convergence detection for bounded parameter search.

A ConvergenceChecker tracks the best rank_key seen so far and signals early
stopping when the Pareto front has not improved for `window` consecutive
trials.  The checker is stateful and should not be shared across runs.

Typical usage inside SearchRunner.run():

    checker = ConvergenceChecker(window=10, min_trials=5)
    ...
    # after each trial (success or failure) that updates the candidate set:
    checker.update(current_best_rank_key)
    if checker.is_converged():
        break
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

RankKey = Tuple[float, ...]


@dataclasses.dataclass
class ConvergenceResult:
    """Point-in-time snapshot of the checker state."""
    is_converged: bool
    trials_since_improvement: int
    best_rank_key: RankKey
    message: str


@dataclasses.dataclass
class ConvergenceChecker:
    """Detects stagnation in the best observed rank_key.

    Args:
        window:     Consecutive non-improving trials required to declare
                    convergence.  Must be >= 1.
        min_trials: Minimum number of updates before convergence can be
                    declared.  Prevents premature stopping at the start of
                    a run where the Pareto front is still building up.

    State is mutable; create a fresh instance for each search run.
    """
    window: int = 10
    min_trials: int = 5

    _best_so_far: Optional[RankKey] = dataclasses.field(
        default=None, init=False, repr=False
    )
    _trials_without_improvement: int = dataclasses.field(
        default=0, init=False, repr=False
    )
    _total_updates: int = dataclasses.field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"ConvergenceChecker.window must be >= 1, got {self.window}")
        if self.min_trials < 1:
            raise ValueError(
                f"ConvergenceChecker.min_trials must be >= 1, got {self.min_trials}"
            )

    def update(self, rank_key: RankKey) -> None:
        """Record the best rank_key seen after the latest trial.

        Call once per completed trial (even failed ones, using the current
        best from the surviving candidate set).  A rank_key tuple compares
        element-wise with higher = better, matching EpisodeVerdict.rank_key.
        """
        self._total_updates += 1
        if self._best_so_far is None or rank_key > self._best_so_far:
            self._best_so_far = rank_key
            self._trials_without_improvement = 0
        else:
            self._trials_without_improvement += 1

    def is_converged(self) -> bool:
        """True when min_trials updates have occurred and the last `window`
        consecutive updates produced no improvement over the recorded best."""
        if self._total_updates < self.min_trials:
            return False
        return self._trials_without_improvement >= self.window

    def trials_since_improvement(self) -> int:
        """Number of consecutive trials since the last improvement."""
        return self._trials_without_improvement

    def result(self) -> ConvergenceResult:
        """Return a point-in-time ConvergenceResult snapshot."""
        converged = self.is_converged()
        if converged:
            msg = (
                f"Converged: no improvement in last {self._trials_without_improvement} "
                f"trial(s) (window={self.window})."
            )
        else:
            msg = (
                f"Still searching: {self._total_updates} update(s), "
                f"{self._trials_without_improvement} without improvement "
                f"(window={self.window}, min_trials={self.min_trials})."
            )
        return ConvergenceResult(
            is_converged=converged,
            trials_since_improvement=self._trials_without_improvement,
            best_rank_key=self._best_so_far or (),
            message=msg,
        )
