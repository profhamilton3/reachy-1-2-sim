"""R12-805: Search point samplers.

GridSampler enumerates all Cartesian grid points in insertion order, skipping
any that have already been evaluated.  RandomSampler draws uniform samples
within the bounds using a seeded RNG for reproducibility.

Both samplers return up to n new points that are not in already_done.  When
the grid is exhausted or the random sampler cannot find a fresh point within
max_attempts, it silently returns fewer than n points; callers check for an
empty list to detect exhaustion.
"""

from __future__ import annotations

import abc
import random
from typing import List, Set

from reachy_ai.search.space import SearchPoint, SearchSpace


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
