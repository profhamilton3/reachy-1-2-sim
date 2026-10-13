"""Offline feasibility sweep for placement (#174): every ordered pair of
reachable board cells, for each box-shaped object, planned and preflighted
with `crane_place_live.plan_crane_place` and nothing else -- no motion, no
simulator.  The IK is the host mirror of the simulator's own
(`tests/unit/host_ik_arm.py`); the scene is FWDCenterLabSivaPool with the
object standing upright at the origin cell's centre and the board otherwise
empty.

Run: ``python -m tools.place_sweep --out REPORT.md [--json ROWS.json]``
"""

import pathlib as _pathlib
import sys as _sys

_REPO = _pathlib.Path(__file__).resolve().parents[2]
for _p in (_REPO / "src", _REPO / "native_mujoco", _REPO / "tests" / "unit"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
