"""R12-805: Search space definition for bounded parameter search.

A SearchSpace is derived from a TrajectoryRecipe's bounded_parameters dict.
Only parameters with explicit "min" and "max" keys are searchable; plain
scalar values and non-numeric specs are left fixed by the search engine.

bounded_parameters format expected in the recipe:

    {
        "approach_height": {"value": 0.12, "min": 0.05, "max": 0.20, "step": 0.01},
        "grip_force":       {"value": 0.5,  "min": 0.1,  "max": 1.0},   # continuous
        "object_id":        "red_cube",                                   # fixed
    }

Parameters with step > 0 are discretised for grid sampling.  Parameters with
step == 0 (or missing) are treated as continuous and appear in random sampling
but only at their baseline value for grid sampling.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

from reachy_ai.motion.recipe import TrajectoryRecipe

SearchPoint = Dict[str, float]


@dataclasses.dataclass(frozen=True)
class ParameterBound:
    """One axis in the search space with inclusive [lo, hi] bounds."""
    name: str
    lo: float
    hi: float
    baseline: float
    step: float = 0.0  # grid resolution; 0.0 = continuous (random only)

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(
                f"ParameterBound '{self.name}': lo ({self.lo}) > hi ({self.hi})"
            )
        if self.step < 0.0:
            raise ValueError(
                f"ParameterBound '{self.name}': step must be >= 0, got {self.step}"
            )

    def grid_values(self) -> List[float]:
        """All discrete values along this axis.

        If step == 0 (continuous), returns [baseline] so the grid still
        produces one value per axis without enumerating infinitely.
        """
        if self.step == 0.0:
            return [self.baseline]
        span = self.hi - self.lo
        vals: List[float] = []
        v = self.lo
        epsilon = 1e-9 * (span + 1.0)
        while v <= self.hi + epsilon:
            vals.append(round(v, 12))
            v += self.step
        return vals if vals else [self.baseline]

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi


@dataclasses.dataclass(frozen=True)
class SearchSpace:
    """Immutable set of ParameterBound axes for one TrajectoryRecipe."""
    bounds: Tuple[ParameterBound, ...]

    @classmethod
    def from_recipe(cls, recipe: TrajectoryRecipe) -> "SearchSpace":
        """Extract searchable dimensions from recipe.bounded_parameters."""
        axes: List[ParameterBound] = []
        for name, spec in recipe.bounded_parameters.items():
            if not isinstance(spec, dict):
                continue
            raw_lo = spec.get("min")
            raw_hi = spec.get("max")
            if raw_lo is None or raw_hi is None:
                continue
            try:
                lo = float(raw_lo)
                hi = float(raw_hi)
            except (TypeError, ValueError):
                continue
            raw_val = spec.get("value")
            baseline = float(raw_val) if raw_val is not None else (lo + hi) / 2.0
            step = float(spec.get("step", 0.0))
            axes.append(
                ParameterBound(name=name, lo=lo, hi=hi, baseline=baseline, step=step)
            )
        return cls(bounds=tuple(axes))

    def baseline_point(self) -> SearchPoint:
        """Return the baseline search point (recipe's current values)."""
        return {b.name: b.baseline for b in self.bounds}

    def validate_point(self, point: SearchPoint) -> bool:
        """True if every bounded parameter's value is within [lo, hi]."""
        return all(b.contains(point.get(b.name, b.baseline)) for b in self.bounds)

    def clip_point(self, point: SearchPoint) -> SearchPoint:
        """Clamp each value to its [lo, hi] range."""
        return {
            b.name: max(b.lo, min(b.hi, point.get(b.name, b.baseline)))
            for b in self.bounds
        }

    def grid_points(self) -> Iterator[SearchPoint]:
        """Yield all Cartesian grid points across all axes.

        Continuous axes (step == 0) contribute exactly one value (baseline).
        When bounds is empty, yields a single empty dict representing the
        baseline (no parameters to vary).
        """
        if not self.bounds:
            yield {}
            return
        names = [b.name for b in self.bounds]
        dim_lists = [b.grid_values() for b in self.bounds]
        for combo in itertools.product(*dim_lists):
            yield dict(zip(names, combo))

    def num_grid_points(self) -> int:
        """Total Cartesian grid size (product of per-axis grid sizes)."""
        total = 1
        for b in self.bounds:
            total *= len(b.grid_values())
        return total

    def continuous_axes(self) -> Tuple[str, ...]:
        """Names of bounded axes with no grid step (step == 0).

        The grid sampler cannot explore these axes — grid_values() pins them at
        their baseline value — so a search space made up entirely of continuous
        axes collapses to a single grid point (the baseline).
        """
        return tuple(b.name for b in self.bounds if b.step == 0.0)

    def point_key(self, point: SearchPoint) -> str:
        """Canonical JSON key for exact deduplication of search points."""
        return json.dumps(
            {b.name: point.get(b.name, b.baseline) for b in self.bounds},
            sort_keys=True,
            separators=(',', ':'),
        )

    def is_empty(self) -> bool:
        return len(self.bounds) == 0
