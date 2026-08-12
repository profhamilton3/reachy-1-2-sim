"""
R12-603: Simulation replayer.

Reads a run directory produced by recorder.py and yields commands in their
original sim_step order for deterministic replay.

Usage (standalone replay against a running server):
  rp = Replayer(run_dir)
  print(rp.manifest())
  for cmd in rp.commands():
      # send cmd to server at the right sim_step
      ...

The replayer does NOT manage timing — the caller decides whether to replay
as fast as possible (benchmark) or at real-time rate.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Iterator


class Replayer:
    """Read a recorded run directory."""

    def __init__(self, run_dir: str | pathlib.Path) -> None:
        self._dir = pathlib.Path(run_dir)
        if not self._dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {self._dir}")
        if not (self._dir / "manifest.json").exists():
            raise FileNotFoundError(f"manifest.json missing in {self._dir}")

    @property
    def run_dir(self) -> pathlib.Path:
        return self._dir

    def manifest(self) -> Dict[str, Any]:
        return json.loads((self._dir / "manifest.json").read_text())

    def states(self) -> Iterator[Dict[str, Any]]:
        """Yield state snapshot dicts in recording order."""
        path = self._dir / "states.jsonl"
        if not path.exists():
            return
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def commands(self) -> Iterator[Dict[str, Any]]:
        """Yield command dicts in recording order."""
        path = self._dir / "commands.jsonl"
        if not path.exists():
            return
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def joint_commands(self) -> Iterator[Dict[str, Any]]:
        """Yield only joint_command entries (exclude reset entries)."""
        for cmd in self.commands():
            if cmd.get("type") == "joint_command":
                yield cmd

    def resets(self) -> Iterator[Dict[str, Any]]:
        """Yield only reset entries."""
        for cmd in self.commands():
            if cmd.get("type") == "reset":
                yield cmd

    def summary(self) -> str:
        m = self.manifest()
        return (
            f"Run: {self._dir.name}\n"
            f"  started_at:   {m.get('started_at', 'unknown')}\n"
            f"  model:        {m.get('model_path', 'unknown')}\n"
            f"  scene:        {m.get('scene_path', 'none')}\n"
            f"  total_steps:  {m.get('total_steps', '?')}\n"
            f"  duration_s:   {m.get('total_duration_s', '?'):.1f}\n"
            f"  states:       {m.get('state_count', '?')}\n"
            f"  commands:     {m.get('command_count', '?')}\n"
            f"  seeds:        {m.get('seeds', [])}\n"
        )
