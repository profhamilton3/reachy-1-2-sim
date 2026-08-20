"""R12-805/R12-808: Search point samplers.

GridSampler enumerates all Cartesian grid points in insertion order, skipping
any that have already been evaluated.  RandomSampler draws uniform samples
within the bounds using a seeded RNG for reproducibility.  AdaptiveSampler
(R12-808) biases suggestions toward the neighbourhood of top-ranked prior
results while falling back to uniform random for a configurable exploration
fraction.

Both/all samplers return up to n new points that are not in already_done.
When exhausted or unable to find a fresh point within max_attempts, they
silently return fewer than n points; callers check for an empty list.
"""

from __future__ import annotations

import abc
import random
from typing import List, Optional, Set, Tuple

from reachy_ai.search.space import ParameterBound, SearchPoint, SearchSpace


class Sampler(abc.ABC):
    @abc.abstractmethod
    def suggest(
        self,
        space: SearchSpace,
        already_done: List[SearchPoint],
        n: int = 1,
    ) -> List[SearchPoint]:
        """Return up to n new points not in already_done."""
        ...


class GridSampler(Sampler):
    """Enumerate grid points in insertion order, skipping already-sampled ones."""

    def suggest(
        self,
        space: SearchSpace,
        already_done: List[SearchPoint],
        n: int = 1,
    ) -> List[SearchPoint]:
        done_keys: Set[str] = {space.point_key(p) for p in already_done}
        result: List[SearchPoint] = []
        for pt in space.grid_points():
            if len(result) >= n:
                break
            if space.point_key(pt) not in done_keys:
                result.append(pt)
        return result

    def remaining(self, space: SearchSpace, already_done: List[SearchPoint]) -> int:
        """Count unevaluated grid points."""
        done_keys = {space.point_key(p) for p in already_done}
        return sum(
            1 for pt in space.grid_points()
            if space.point_key(pt) not in done_keys
        )


class RandomSampler(Sampler):
    """Uniform random sampling within parameter bounds (seeded, reproducible)."""

    def __init__(self, seed: int = 0, max_attempts: int = 1000) -> None:
        self._rng = random.Random(seed)
        self._max_attempts = max_attempts

    def suggest(
        self,
        space: SearchSpace,
        already_done: List[SearchPoint],
        n: int = 1,
    ) -> List[SearchPoint]:
        done_keys: Set[str] = {space.point_key(p) for p in already_done}
        result: List[SearchPoint] = []
        attempts = 0
        while len(result) < n and attempts < self._max_attempts:
            pt: SearchPoint = {
                b.name: self._rng.uniform(b.lo, b.hi)
                for b in space.bounds
            }
            key = space.point_key(pt)
            if key not in done_keys:
                done_keys.add(key)
                result.append(pt)
            attempts += 1
        return result


class AdaptiveSampler(Sampler):
    """Exploit top-ranked prior results while exploring at a configurable rate.

    On each suggestion the sampler flips a biased coin:
    - Exploitation (probability = 1 - exploration_weight): pick a random
      point from the top-k prior results and perturb each axis by ±1 step
      (discrete) or Gaussian noise with σ = perturbation_sigma × (hi − lo)
      (continuous), then clamp to [lo, hi].
    - Exploration (probability = exploration_weight): draw a uniform random
      point (same as RandomSampler).

    The sampler is initialised with ranked prior results at construction time.
    It does not update as new results arrive during a run; create a fresh
    instance for each call to SearchRunner.run() (done automatically when
    sampler="adaptive" is set in SearchConfig).

    When prior is empty or top_k produces no usable points, every suggestion
    falls back to uniform random sampling.

    Args:
        top_points:         Ordered list of prior SearchPoints to exploit,
                            best-first.  Pass the best-first slice you want;
                            AdaptiveSampler picks uniformly among them.
        top_k:              Maximum number of top points to exploit from.
        exploration_weight: Fraction of suggestions drawn uniformly at random
                            (0.0 = pure exploitation, 1.0 = pure exploration).
        perturbation_sigma: Gaussian σ as a fraction of axis range for
                            continuous parameters (default 0.15 = 15 %).
        seed:               RNG seed for reproducibility.
        max_attempts:       Rejection-sampling limit before giving up.
    """

    def __init__(
        self,
        top_points: List[SearchPoint],
        *,
        top_k: int = 3,
        exploration_weight: float = 0.3,
        perturbation_sigma: float = 0.15,
        seed: int = 0,
        max_attempts: int = 1000,
    ) -> None:
        if not 0.0 <= exploration_weight <= 1.0:
            raise ValueError(
                f"exploration_weight must be in [0, 1], got {exploration_weight}"
            )
        if perturbation_sigma <= 0.0:
            raise ValueError(
                f"perturbation_sigma must be > 0, got {perturbation_sigma}"
            )
        self._top_points: List[SearchPoint] = top_points[:top_k]
        self._exploration_weight = exploration_weight
        self._perturbation_sigma = perturbation_sigma
        self._rng = random.Random(seed)
        self._max_attempts = max_attempts

    def suggest(
        self,
        space: SearchSpace,
        already_done: List[SearchPoint],
        n: int = 1,
    ) -> List[SearchPoint]:
        done_keys: Set[str] = {space.point_key(p) for p in already_done}
        result: List[SearchPoint] = []
        attempts = 0
        while len(result) < n and attempts < self._max_attempts:
            pt = self._draw(space)
            key = space.point_key(pt)
            if key not in done_keys:
                done_keys.add(key)
                result.append(pt)
            attempts += 1
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw(self, space: SearchSpace) -> SearchPoint:
        """Draw one point: exploitation or exploration based on the coin flip."""
        if self._top_points and self._rng.random() >= self._exploration_weight:
            anchor = self._rng.choice(self._top_points)
            return {b.name: self._perturb(b, anchor.get(b.name, b.baseline))
                    for b in space.bounds}
        return {b.name: self._rng.uniform(b.lo, b.hi) for b in space.bounds}

    def _perturb(self, bound: ParameterBound, center: float) -> float:
        """Perturb center within bound: ±1 step for discrete, Gaussian for continuous."""
        if bound.step > 0.0:
            delta = self._rng.choice([-bound.step, 0.0, bound.step])
            raw = center + delta
        else:
            sigma = self._perturbation_sigma * (bound.hi - bound.lo)
            raw = self._rng.gauss(center, sigma if sigma > 0 else 1e-9)
        return max(bound.lo, min(bound.hi, raw))
