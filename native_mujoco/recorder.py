"""
R12-603: Simulation recorder.

Writes three JSONL streams + a manifest to a timestamped run directory:

  runs/run_<YYYYMMDD_HHMMSS>/
    manifest.json     — metadata: model, scene, calibration, seeds, SW version
    states.jsonl      — one state dict per line (from protocol.State messages)
    commands.jsonl    — one command dict per line (joint_command + reset messages)

Usage:
  rec = Recorder.new(run_root, config)
  rec.record_state(state_dict)
  rec.record_command(cmd_dict)
  rec.finalize(total_steps, duration_s)

All writes are done on the caller's thread; the caller must ensure this is
the sim thread (not the asyncio event loop thread) to avoid lock contention.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import time
from typing import Any, Dict, Optional


_FORMAT_VERSION = 1


class Recorder:
    """Records sim state and commands to a run directory."""

    def __init__(self, run_dir: pathlib.Path, manifest_meta: Dict[str, Any]) -> None:
        self._dir = run_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._state_f = (run_dir / "states.jsonl").open("w", buffering=1)
        self._cmd_f   = (run_dir / "commands.jsonl").open("w", buffering=1)
        self._state_count = 0
        self._cmd_count   = 0
        self._started_at  = time.time()
        self._seeds: list = []

        # Write initial manifest (will be updated at finalize)
        self._manifest_path = run_dir / "manifest.json"
        self._manifest_meta = {
            "format_version": _FORMAT_VERSION,
            "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
            **manifest_meta,
        }
        self._flush_manifest(total_steps=None, duration_s=None)

    @classmethod
    def new(
        cls,
        run_root: str | pathlib.Path,
        manifest_meta: Optional[Dict[str, Any]] = None,
    ) -> "Recorder":
        """Create a new timestamped run directory under run_root."""
        ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
        run_dir = pathlib.Path(run_root) / f"run_{ts}"
        return cls(run_dir, manifest_meta or {})

    @property
    def run_dir(self) -> pathlib.Path:
        return self._dir

    def record_state(self, state_dict: Dict[str, Any]) -> None:
        """Append a state snapshot dict to states.jsonl."""
        self._state_f.write(json.dumps(state_dict, separators=(",", ":")) + "\n")
        self._state_count += 1

    def record_command(self, cmd_dict: Dict[str, Any]) -> None:
        """Append a command dict to commands.jsonl."""
        self._cmd_f.write(json.dumps(cmd_dict, separators=(",", ":")) + "\n")
        self._cmd_count += 1

    def record_reset(self, seed: Optional[int], sim_step: int) -> None:
        """Record a reset event (stored in commands.jsonl, seeds list updated)."""
        if seed is not None:
            self._seeds.append(seed)
        entry = {"type": "reset", "seed": seed, "sim_step": sim_step,
                 "wall_time_s": time.time() - self._started_at}
        self._cmd_f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self._cmd_count += 1

    def finalize(self, total_steps: int, duration_s: float) -> None:
        """Close file handles and write the final manifest."""
        self._state_f.close()
        self._cmd_f.close()
        self._flush_manifest(total_steps=total_steps, duration_s=duration_s)

    def _flush_manifest(
        self,
        total_steps: Optional[int],
        duration_s: Optional[float],
    ) -> None:
        manifest = {
            **self._manifest_meta,
            "seeds": list(self._seeds),
            "total_steps": total_steps,
            "total_duration_s": duration_s,
            "state_count": self._state_count,
            "command_count": self._cmd_count,
        }
        tmp = self._manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, separators=(",", ": ")))
        tmp.replace(self._manifest_path)
