"""Offline comparison of the withdrawal options for #173.

Rebuilds the release state of recorded crane lifts in the offline world of
#172 (`SimulationCore`, table and rails present) and flies each option there:
`record` extracts what a recording holds, `rebuild` reconstructs the plan and
the physics, `options` runs base / O1 / O2 / O1+O2.  Nothing here talks to a
simulator server; nothing here is imported by the panel.

Run: ``python -m tools.withdrawal_replay --out report.md NAME=DIR ...``
"""

import pathlib as _pathlib
import sys as _sys

_REPO = _pathlib.Path(__file__).resolve().parents[2]
for _p in (_REPO / "src", _REPO / "native_mujoco", _REPO / "tests" / "unit"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
