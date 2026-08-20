"""R12-801: Typed trajectory recipe for deterministic episode execution.

A TrajectoryRecipe is an immutable description of a sequence of named
primitive steps with typed, bounded parameters. No executable Python is
stored — only strings and numbers that the executor (PR 8.3) maps to
primitives.py calls.

YAML round-trip is supported when PyYAML is installed. JSON is always
available via the stdlib.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, Dict, List

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

RECIPE_SCHEMA_VERSION = 1


@dataclasses.dataclass
class PrimitiveStep:
    """One step in a trajectory: a named primitive and its typed parameters."""
    primitive: str
    parameters: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, str) or not self.primitive:
            raise ValueError("PrimitiveStep.primitive must be a non-empty string")
        for k, v in self.parameters.items():
            if callable(v) or isinstance(v, type):
                raise ValueError(
                    f"Parameter '{k}' must not be callable or a type — "
                    "recipes may not store executable objects"
                )


@dataclasses.dataclass
class TrajectoryRecipe:
    """Serialisable trajectory recipe (Section 7.3 of EPIC-8.md).

    Each primitive in primitive_sequence maps to a function in
    motion/primitives.py via its name.  bounded_parameters documents the
    tunable scalar offsets that the search engine may vary; their values here
    are the baseline defaults.
    """
    recipe_id: str
    recipe_version: int                     = RECIPE_SCHEMA_VERSION
    task_type: str                          = ""
    arm: str                                = "right"
    primitive_sequence: List[PrimitiveStep] = dataclasses.field(default_factory=list)
    bounded_parameters: Dict[str, Any]      = dataclasses.field(default_factory=dict)
    joint_limits: Dict[str, float]          = dataclasses.field(default_factory=dict)
    velocity_limits: Dict[str, float]       = dataclasses.field(default_factory=dict)
    effort_limits: Dict[str, float]         = dataclasses.field(default_factory=dict)
    source: str                             = "baseline"
    parent_recipe_id: str                   = ""

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML serialisation (pip install pyyaml)")
        return _yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryRecipe":
        d = dict(d)
        raw_steps = d.pop("primitive_sequence", [])
        steps = [
            PrimitiveStep(**s) if isinstance(s, dict) else s
            for s in raw_steps
        ]
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(primitive_sequence=steps, **{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "TrajectoryRecipe":
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_yaml(cls, s: str) -> "TrajectoryRecipe":
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML deserialisation (pip install pyyaml)")
        return cls.from_dict(_yaml.safe_load(s))

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "TrajectoryRecipe":
        """Load a recipe from a JSON or YAML file."""
        p = pathlib.Path(path)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            return cls.from_yaml(text)
        return cls.from_json(text)
